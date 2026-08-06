"""Re-implementación local del UNet 2D (equivalente a ``src.models.unet.UNet``
del TFM). Permite cargar checkpoints sin acoplar SegLens al TFM.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 1,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        dilation: int = 1,
        inplace: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2 if isinstance(kernel_size, int) else [x // 2 for x in kernel_size]
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.norm(self.conv(x)))


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        mid = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            ConvBNReLU(in_channels, mid, kernel_size=3),
            ConvBNReLU(mid, out_channels, kernel_size=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )
        return self.conv(torch.cat([x2, x1], dim=1))


class OutputConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        init_features: int = 64,
        bilinear: bool = True,
    ):
        super().__init__()
        f = init_features
        self.inc = DoubleConv(in_channels, f)
        self.down1 = Down(f, f * 2)
        self.down2 = Down(f * 2, f * 4)
        self.down3 = Down(f * 4, f * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(f * 8, f * 16 // factor)
        self.up1 = Up(f * 16, f * 8 // factor, bilinear)
        self.up2 = Up(f * 8, f * 4 // factor, bilinear)
        self.up3 = Up(f * 4, f * 2 // factor, bilinear)
        self.up4 = Up(f * 2, f, bilinear)
        self.outc = OutputConv(f, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
