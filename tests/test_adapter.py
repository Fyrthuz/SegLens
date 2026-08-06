"""Tests para los Adapters."""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from seglens.adapter import UNetAdapter, UniVerSegAdapter, make_adapter
from seglens.third_party import UNet


class TestUNetAdapter:
    def test_default_stages(self):
        m = UNet(in_channels=3, out_channels=1)
        a = UNetAdapter(m)
        assert a.name == "unet"
        assert a.out_channels == 1
        assert set(a.stages) == {
            "inc", "down1", "down2", "down3", "down4",
            "up1", "up2", "up3", "up4", "outc",
        }

    def test_head_is_1x1(self):
        m = UNet()
        a = UNetAdapter(m)
        h = a.head()
        assert isinstance(h, nn.Module)
        # OutputConv contiene una Conv2d 1x1
        conv = h.conv if hasattr(h, "conv") else h
        assert isinstance(conv, nn.Conv2d)
        assert conv.kernel_size == (1, 1)
        assert conv.in_channels == 64
        assert conv.out_channels == 1

    def test_segment_runs(self):
        m = UNet()
        a = UNetAdapter(m).eval()
        x = torch.randn(2, 3, 64, 64)
        y = a.segment(x)
        assert y.shape == (2, 1, 64, 64)

    def test_features_returns_all_stages(self):
        m = UNet()
        a = UNetAdapter(m).eval()
        x = torch.randn(1, 3, 64, 64)
        feats = a.features(x)
        assert set(feats.keys()) == set(a.stages)
        # Verificar shapes esperados (bilinear=True → factor=2 → down4=512)
        assert feats["inc"].shape[1] == 64 and feats["inc"].shape[-2:] == (64, 64)
        assert feats["down4"].shape[1] == 512 and feats["down4"].shape[-2:] == (4, 4)
        assert feats["outc"].shape == (1, 1, 64, 64)

    def test_features_with_transpose_decoder(self):
        """Con ``bilinear=False`` el decoder usa ConvTranspose2d y down4 tiene 1024 canales."""
        m = UNet(bilinear=False)
        a = UNetAdapter(m).eval()
        x = torch.randn(1, 3, 64, 64)
        feats = a.features(x)
        assert feats["down4"].shape[1] == 1024 and feats["down4"].shape[-2:] == (4, 4)

    def test_invalid_stage_raises(self):
        m = UNet()
        with pytest.raises(ValueError):
            UNetAdapter(m, stages=["inc", "fake_layer"])

    def test_preprocess_is_identity_for_unet(self):
        m = UNet()
        a = UNetAdapter(m)
        x = torch.randn(2, 3, 64, 64)
        out = a.preprocess(x)
        assert torch.equal(out, x)


class TestUniVerSegAdapterDiscovery:
    def test_discover_stages_with_typical_universeg(self):
        """Simula un universeg con módulos nombrados down_/up_/cross_/final."""

        class FakeBlock(nn.Module):
            def __init__(self, c):
                super().__init__()
                self.conv = nn.Conv2d(c, c, 3, padding=1)

            def forward(self, x):
                return self.conv(x)

        class FakeUniVerSeg(nn.Module):
            def __init__(self):
                super().__init__()
                self.down_0 = FakeBlock(1)
                self.down_1 = FakeBlock(1)
                self.up_2 = FakeBlock(1)
                self.up_3 = FakeBlock(1)
                self.cross_4 = FakeBlock(1)
                self.final = nn.Conv2d(1, 1, 1)

            def forward(self, image, ctx_imgs, ctx_masks):
                return self.final(image)

        m = FakeUniVerSeg()
        a = UniVerSegAdapter(m)
        assert "down_0" in a.stages
        assert "down_1" in a.stages
        assert "up_2" in a.stages
        assert "up_3" in a.stages
        assert "cross_4" in a.stages

    def test_discover_stages_fallback(self):
        class FlatModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 3, padding=1)
                self.linear = nn.Linear(1, 1)

            def forward(self, image, ctx_imgs, ctx_masks):
                return self.conv(image)

        a = UniVerSegAdapter(FlatModel())
        # sin matches con keywords, fallback a stages=[]
        assert a.stages == []
        # features() debe devolver dict vacío sin error
        feats = a.features(torch.randn(1, 1, 32, 32))
        assert feats == {}

    def test_segment_requires_context(self):
        class FlatModel(nn.Module):
            def forward(self, image, ctx_imgs, ctx_masks):
                return image

        a = UniVerSegAdapter(FlatModel())
        with pytest.raises(RuntimeError, match="set_context"):
            a.segment(torch.randn(1, 1, 32, 32))

    def test_preprocess_grayscales_and_resizes(self):
        class FlatModel(nn.Module):
            def forward(self, image, ctx_imgs, ctx_masks):
                return image

        a = UniVerSegAdapter(FlatModel(), input_size=(64, 64))
        x = torch.randn(2, 3, 128, 128)
        out = a.preprocess(x)
        assert out.shape == (2, 1, 64, 64)

    def test_preprocess_passes_through_grayscale(self):
        class FlatModel(nn.Module):
            def forward(self, image, ctx_imgs, ctx_masks):
                return image

        a = UniVerSegAdapter(FlatModel(), input_size=(64, 64))
        x = torch.randn(2, 1, 64, 64)
        out = a.preprocess(x)
        assert out.shape == (2, 1, 64, 64)


class TestMakeAdapterFactory:
    def test_recognizes_unet(self):
        m = UNet()
        a = make_adapter(m)
        assert isinstance(a, UNetAdapter)

    def test_recognizes_universeg_like(self):
        class ThreeArgModel(nn.Module):
            def forward(self, image, ctx_imgs, ctx_masks):
                return image

        a = make_adapter(ThreeArgModel())
        assert isinstance(a, UniVerSegAdapter)
