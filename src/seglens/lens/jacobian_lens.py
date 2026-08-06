"""J-lens: Jacobiano promediado por píxel entre features de etapa y logits finales.

Análogo directo del *Jacobian lens* del paper de Anthropic, pero adaptado:

- En transformers, ``J_l = E[∂h_final/∂h_l]`` es ``d_model × d_model``.
- En segmentación, ``∂y/∂f_l`` por píxel es ``C_out × C_l`` (por píxel del
  espacio de f_l). Promediamos sobre imágenes y posiciones de píxel para
  obtener ``J_l_bar ∈ R^{K × C_l}``.
- Aproximación de **bloque diagonal espacial**: la dependencia entre píxeles
  distintos se ignora (consistente con campos receptivos locales en CNN).
  En la práctica, el ``J_l`` se aplica píxel a píxel: ``y_p ≈ J_l · f_{l,p}``.

Notas de implementación
-----------------------
PyTorch 2.x tiene una restricción práctica: ``torch.autograd.grad`` con
``grad_outputs`` falla cuando los inputs son **vistas** (``permute``,
``reshape``, ``F.interpolate``) de un tensor capturado por un hook en
ciertas versiones (error: *"One of the differentiated Tensors appears
to not have been used in the graph"*). La solución es trabajar con el
tensor **crudo** y usar un output **escalar** (``y[b, k, h_y, w_y]``) por
llamada de backward. Esto da un gradiente de la forma ``(B, C_l, H_l, W_l)``;
promediamos sobre las dimensiones espaciales y de batch para obtener
``J_l ∈ R^{K × C_l}``.

El "per-pixel" se interpreta en la **resolución nativa** de la feature
(fiel al paper, donde cada posición del residual stream es un índice
discreto). Para aplicar ``J_l`` a la feature hacemos ``J_l · f_l`` por
píxel y luego upsampleamos el logit resultante a la resolución del target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn.functional as F
from torch import Tensor

from seglens.lens.base import LensOutput


@dataclass
class _Jacobian:
    weight: Tensor  # (C_out, C_l)
    n_pixels: int = 0


class JacobianLens:
    """Lente 3: Jacobiano promediado por píxel (análogo del J-lens).

    Uso::

        lens = JacobianLens(target_hw=(H, W), out_channels=1)
        lens.calibrate(adapter, dataloader, n_images=200, device="cuda")
        out = lens("up4", features_up4)
    """

    name = "jacobian"

    def __init__(self, target_hw: tuple[int, int], out_channels: int = 1):
        self.target_hw = target_hw
        self.out_channels = out_channels
        self._jacobians: dict[str, _Jacobian] = {}

    @property
    def stages(self) -> list[str]:
        return list(self._jacobians.keys())

    def has(self, stage: str) -> bool:
        return stage in self._jacobians

    def _accumulate_per_image(
        self,
        adapter,
        x: Tensor,
        accumulators: dict[str, tuple[Tensor, int]],
        max_pixels_per_image: int | None = 64,
    ) -> None:
        """Captura features y calcula el Jacobiano promediado por píxel."""
        model = adapter.model
        stages = adapter.stages
        x = x.detach().clone()

        captured: dict[str, Tensor] = {}

        def make_hook(name: str):
            def fn(_m, _i, out):
                captured[name] = out

            return fn

        handles = [
            model.get_submodule(stage).register_forward_hook(make_hook(stage))
            for stage in stages
        ]
        try:
            with torch.enable_grad():
                model.zero_grad(set_to_none=True)
                y = model(x)  # (B, K, H_y, W_y)
                B, K, H_y, W_y = y.shape

                # Para cada stage, calculamos J_l iterando sobre píxeles
                # del output. Para K=1 (binario), el bucle es sobre (h_y, w_y).
                for s_idx, stage in enumerate(stages):
                    f_l = captured[stage]  # (B, C_l, H_l, W_l) — RAW
                    _, C_l, H_l, W_l = f_l.shape

                    jac = torch.zeros(K, C_l, device=f_l.device, dtype=f_l.dtype)
                    n_pix = 0
                    is_last_stage = s_idx == len(stages) - 1

                    # Sub-muestreo espacial para acelerar
                    pix_iter: Iterator[tuple[int, int]] = (
                        (hy, wy) for hy in range(H_y) for wy in range(W_y)
                    )
                    if max_pixels_per_image is not None:
                        step = max(1, (H_y * W_y) // max_pixels_per_image)
                        pix_iter = (
                            (hy, wy)
                            for hy in range(H_y)
                            for wy in range(W_y)
                            if (hy * W_y + wy) % step == 0
                        )

                    for k in range(K):
                        for hy, wy in pix_iter:
                            y_scalar = y[0, k, hy, wy]
                            grads = torch.autograd.grad(
                                outputs=y_scalar,
                                inputs=f_l,
                                retain_graph=True,
                                only_inputs=True,
                            )[0]  # (B, C_l, H_l, W_l)
                            # El Jacobiano "diagonal espacial" lo leemos
                            # en la posición (0, :, hy_up, wy_up) del gradiente,
                            # donde (hy_up, wy_up) es la coordenada en el
                            # espacio de f_l. Para un campo receptivo local,
                            # usamos una asignación por escala:
                            if (H_l, W_l) != (H_y, W_y):
                                hy_l = min(int(hy * H_l / H_y), H_l - 1)
                                wy_l = min(int(wy * W_l / W_y), W_l - 1)
                            else:
                                hy_l, wy_l = hy, wy
                            g_at_pix = grads[0, :, hy_l, wy_l]  # (C_l,)
                            jac[k] += g_at_pix
                            n_pix += 1
                    if n_pix > 0:
                        jac = jac / n_pix

                    if is_last_stage and K == 1 and n_pix > 0:
                        # Hint para liberar memoria; no afecta correctness
                        del y

                    if stage not in accumulators:
                        accumulators[stage] = (jac.clone(), 1)
                    else:
                        prev_jac, prev_n = accumulators[stage]
                        n_new = 1
                        n_total = prev_n + n_new
                        new_jac = prev_jac + (jac - prev_jac) * (n_new / n_total)
                        accumulators[stage] = (new_jac, n_total)
        finally:
            for h in handles:
                h.remove()

    def calibrate(
        self,
        adapter,
        dataloader,
        n_images: int = 200,
        device: str = "cpu",
        max_pixels_per_image: int | None = 64,
    ) -> "JacobianLens":
        """Calcula ``J_l`` promediado por píxel sobre ``n_images`` imágenes.

        ``max_pixels_per_image``: sub-muestreo espacial para acelerar. Si
        ``None``, se usan todos los píxeles (más preciso pero más lento).
        Para imágenes de 256×256 se recomienda 64-256 píxeles de calibración.
        """
        adapter.model.eval()
        for stage in adapter.stages:
            for p in adapter.model.get_submodule(stage).parameters(recurse=True):
                p.requires_grad_(True)

        accumulators: dict[str, tuple[Tensor, int]] = {}
        seen = 0
        for batch in dataloader:
            if seen >= n_images:
                break
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = adapter.preprocess(x).to(device)
            b = x.shape[0]
            take = min(b, n_images - seen)
            if take < b:
                x = x[:take]
            self._accumulate_per_image(adapter, x, accumulators, max_pixels_per_image)
            seen += take

        for stage, (jac, n) in accumulators.items():
            self._jacobians[stage] = _Jacobian(weight=jac, n_pixels=n)
        return self

    def save(self, path: str) -> None:
        state = {stage: j.weight for stage, j in self._jacobians.items()}
        torch.save(state, path)

    def load(self, path: str) -> "JacobianLens":
        state = torch.load(path, map_location="cpu")
        for stage, w in state.items():
            self._jacobians[stage] = _Jacobian(weight=w, n_pixels=-1)
        return self

    @torch.no_grad()
    def __call__(self, stage: str, features: Tensor) -> LensOutput:
        if stage not in self._jacobians:
            raise KeyError(f"JacobianLens sin Jacobiano calibrado para stage '{stage}'")
        jac = self._jacobians[stage].weight.to(features.device)  # (K, C_l)
        B, C, H, W = features.shape
        flat = features.permute(0, 2, 3, 1).reshape(-1, C)
        out = flat @ jac.T  # (B*H*W, K)
        K = out.shape[-1]
        logits = out.reshape(B, H, W, K).permute(0, 3, 1, 2).contiguous()
        # Upsample a target_hw si hace falta
        if tuple(logits.shape[-2:]) != self.target_hw:
            logits = F.interpolate(
                logits, size=self.target_hw, mode="bilinear", align_corners=False
            )
        return LensOutput(
            logits=logits,
            stage=stage,
            lens=self.name,
            extra={"n_pixels": self._jacobians[stage].n_pixels},
        )

    def jacobian_for(self, stage: str) -> Tensor:
        """Devuelve la matriz Jacobiana ``(K, C_l)`` de una etapa."""
        return self._jacobians[stage].weight
