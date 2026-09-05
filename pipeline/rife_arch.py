"""
pipeline/rife_arch.py
Standard IFNet architecture for RIFE v4 (Real-Time Intermediate Flow Estimation).
Used for frame interpolation when flownet.pkl contains a state dict.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    B, _, H, W = img.size()
    xx = torch.arange(0, W, device=img.device).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, device=img.device).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    vgrid = grid + flow
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    return F.grid_sample(img, vgrid, mode="bilinear", padding_mode="border", align_corners=True)


class ResConv(nn.Module):
    def __init__(self, c: int, dilation: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1),
            nn.PReLU(),
            nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1),
        )
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    def __init__(self, in_planes: int, c: int = 64):
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_planes, c // 2, 3, 2, 1),
            nn.PReLU(),
            nn.Conv2d(c // 2, c, 3, 2, 1),
            nn.PReLU(),
        )
        self.convblock = nn.Sequential(
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
        )
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4 * 6, 4, 2, 1),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor, flow: torch.Tensor = None, scale: float = 1.0):
        if scale != 1.0:
            x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            if scale != 1.0:
                flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False) * 1.0 / scale
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        if scale != 1.0:
            tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        flow_out = tmp[:, :4] * scale
        mask_out = tmp[:, 4:5]
        return flow_out, mask_out


class IFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = IFBlock(7 + 1, c=192)
        self.block1 = IFBlock(8 + 4 + 1, c=128)
        self.block2 = IFBlock(8 + 4 + 1, c=96)
        self.block3 = IFBlock(8 + 4 + 1, c=64)

    def forward(self, x: torch.Tensor, timestep: float = 0.5, scale_list=(8, 4, 2, 1)) -> torch.Tensor:
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        b, _, h, w = img0.shape
        flow = None
        mask = None
        timestep_plane = torch.ones((b, 1, h, w), device=x.device, dtype=x.dtype) * timestep

        for i, block in enumerate([self.block0, self.block1, self.block2, self.block3]):
            if flow is None:
                flow, mask = block(torch.cat((img0, img1, timestep_plane), 1), None, scale=scale_list[i])
            else:
                f0 = warp(img0, flow[:, :2])
                f1 = warp(img1, flow[:, 2:4])
                f_, m_ = block(torch.cat((img0, img1, f0, f1, mask), 1), flow, scale=scale_list[i])
                flow = flow + f_
                mask = mask + m_

        mask = torch.sigmoid(mask)
        warped0 = warp(img0, flow[:, :2])
        warped1 = warp(img1, flow[:, 2:4])
        return warped0 * mask + warped1 * (1 - mask)

    def inference(self, img0: torch.Tensor, img1: torch.Tensor, timestep: float = 0.5) -> torch.Tensor:
        return self.forward(torch.cat((img0, img1), 1), timestep=timestep)
