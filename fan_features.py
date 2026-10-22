"""Frozen official 2D-FAN, with locally differentiable heatmap coordinates.

Only dimensionless eyelid/lip opening ratios are used as attribute targets.
Full facial shape/landmark coordinates are never matched to the source face.
"""
from runtime import ROOT
import torch
from torch import nn
from torch.nn import functional as F
from runtime import load_module


class OpeningFeatures(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        module = load_module('_attribute_fan', ROOT / 'third_party/fan/fan.py')
        self.net = module.FAN(4)
        weights = torch.load(ROOT / 'models/2DFAN4-11f355bf06.pth.tar', map_location='cpu', weights_only=True)
        self.net.load_state_dict(weights, strict=True)
        self.eval().requires_grad_(False).to(device)

    @staticmethod
    def grid(face, device='cuda', scale_factor=1.0):
        # Official FAN crop convention, applied with differentiable grid_sample.
        x, y, width, height = map(float, face[:4])
        cx = x + width/2
        cy = y + height/2 - .12*height
        side = (width + height)/195*200*scale_factor
        v = torch.arange(256, device=device).float() + .5
        yy, xx = torch.meshgrid(v, v, indexing='ij')
        xx = cx + (xx/256-.5)*side
        yy = cy + (yy/256-.5)*side
        return torch.stack(((2*xx+1)/256-1, (2*yy+1)/256-1), -1)[None]

    @staticmethod
    def coordinates(heatmaps):
        # Select a 9x9 local support using detached maxima. Within this support,
        # normalized positive heatmap mass produces subpixel differentiable xy.
        b, n, h, w = heatmaps.shape
        indices = heatmaps.detach().flatten(2).argmax(-1)
        px, py = indices % w, indices // w
        yy, xx = torch.meshgrid(torch.arange(h, device=heatmaps.device),
                                torch.arange(w, device=heatmaps.device), indexing='ij')
        support = ((xx[None,None]-px[:,:,None,None]).abs() <= 4) & ((yy[None,None]-py[:,:,None,None]).abs() <= 4)
        mass = heatmaps.relu().square()*support
        denominator = mass.sum((-1,-2)).clamp_min(1e-8)
        coords = torch.stack(((mass*(xx+.5)).sum((-1,-2))/denominator,
                              (mass*(yy+.5)).sum((-1,-2))/denominator), -1)
        confidence = heatmaps.detach().flatten(2).amax(-1)
        return coords, confidence

    @staticmethod
    def ratios(points):
        def distance(a,b):
            return torch.linalg.vector_norm(points[:,a]-points[:,b], dim=-1)
        left = (distance(37,41)+distance(38,40))/(2*distance(36,39).clamp_min(1e-4))
        right = (distance(43,47)+distance(44,46))/(2*distance(42,45).clamp_min(1e-4))
        mouth = (distance(61,67)+distance(62,66)+distance(63,65))/(3*distance(60,64).clamp_min(1e-4))
        return torch.stack((left,right,mouth), -1)

    def forward(self, image, grid, details=False):
        crop = F.grid_sample((image+1)/2, grid.to(image), mode='bilinear',
                             padding_mode='zeros', align_corners=False)
        maps = self.net(crop)[-1]
        points, confidence = self.coordinates(maps)
        ratios = self.ratios(points)
        if details:
            return ratios, points, confidence, crop
        return ratios
