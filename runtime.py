"""Frozen pretrained models; differentiable paths and local-only assets."""
import importlib.util
import math
import pickle
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def package(name, path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def fused_leaky_relu(x, bias=None, negative_slope=.2, scale=math.sqrt(2)):
    if bias is not None:
        x = x + bias.view(1, -1, *([1] * (x.ndim - 2)))
    return F.leaky_relu(x, negative_slope=negative_slope) * scale


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, bias=True, negative_slope=.2, scale=math.sqrt(2)):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel)) if bias else None
        self.negative_slope, self.scale = negative_slope, scale

    def forward(self, x):
        return fused_leaky_relu(x, self.bias, self.negative_slope, self.scale)


def upfirdn2d(x, kernel, up=1, down=1, pad=(0, 0)):
    """Native tensor equivalent of upstream zero-insert/filter/decimate op."""
    b, c, h, w = x.shape
    x = x.reshape(b * c, h, 1, w, 1)
    x = F.pad(x, (0, up - 1, 0, 0, 0, up - 1)).reshape(b * c, 1, h * up, w * up)
    p0, p1 = pad
    x = F.pad(x, (max(p0, 0), max(p1, 0), max(p0, 0), max(p1, 0)))
    x = x[:, :, max(-p0, 0):x.shape[2] - max(-p1, 0),
          max(-p0, 0):x.shape[3] - max(-p1, 0)]
    k = kernel.to(x).flip((0, 1)).reshape(1, 1, *kernel.shape)
    x = F.conv2d(x, k)[:, :, ::down, ::down]
    return x.reshape(b, c, *x.shape[-2:])


def prepare_e4e():
    path = str(ROOT / 'third_party/e4e')
    if path not in sys.path:
        sys.path.insert(0, path)
    op = types.ModuleType('models.stylegan2.op')
    op.FusedLeakyReLU = FusedLeakyReLU
    op.fused_leaky_relu = fused_leaky_relu
    op.upfirdn2d = upfirdn2d
    sys.modules[op.__name__] = op


class StyleGAN:
    def __init__(self, device='cuda'):
        prepare_e4e()
        from models.encoders.psp_encoders import Encoder4Editing
        from models.stylegan2.model import Generator
        state = torch.load(ROOT / 'models/e4e_ffhq_encode.pt', map_location='cpu', weights_only=True)
        self.opts = state['opts']
        opts = SimpleNamespace(**self.opts)
        self.encoder = Encoder4Editing(50, 'ir_se', opts)
        self.decoder = Generator(1024, 512, 8, channel_multiplier=2)
        for name, model in [('encoder', self.encoder), ('decoder', self.decoder)]:
            weights = {k[len(name) + 1:]: v for k, v in state['state_dict'].items() if k.startswith(name + '.')}
            model.load_state_dict(weights, strict=True)
            model.eval().requires_grad_(False)
        self.avg = state['latent_avg'].to(device)
        self.decoder.to(device)
        # The encoder only runs once per input, on CPU to keep more GPU memory free.
        self.encoder.cpu()
        self.device = device
        self.info = {'generator': 'e4e bundled FFHQ StyleGAN2, 1024x1024',
            'strict_loading': True, 'latent_shape': [1, 18, 512],
            'encoder_type': opts.encoder_type, 'start_from_latent_avg': opts.start_from_latent_avg,
            'operator_backend': 'PyTorch native, fixed checkpoint noise, checkpointed synthesis',
            'generator_parameters': sum(p.numel() for p in self.decoder.parameters())}

    @torch.no_grad()
    def invert(self, rgb):
        image = Image.fromarray(rgb).resize((256, 256), Image.Resampling.BILINEAR)
        x = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)[None].float() / 127.5 - 1
        w = self.encoder(x).to(self.device)
        if self.opts['start_from_latent_avg']:
            w = w + self.avg.reshape(1, -1, 512)
        return w

    def generate(self, w, checkpointed=True):
        g = self.decoder
        noises = [getattr(g.noises, f'noise_{i}') for i in range(g.num_layers)]
        out = g.conv1(g.input(w), w[:, 0], noise=noises[0])
        skip = g.to_rgb1(out, w[:, 1])
        i = 1
        for conv1, conv2, rgb in zip(g.convs[::2], g.convs[1::2], g.to_rgbs):
            # Bind closures: backward recomputes after this loop has completed.
            def block(x, s, styles, a=conv1, b=conv2, t=rgb, n1=noises[i], n2=noises[i + 1]):
                x = a(x, styles[:, 0], noise=n1)
                x = b(x, styles[:, 1], noise=n2)
                return x, t(x, styles[:, 2], s)
            if checkpointed and torch.is_grad_enabled() and w.requires_grad:
                out, skip = checkpoint(block, out, skip, w[:, i:i + 3], use_reentrant=False)
            else:
                out, skip = block(out, skip, w[:, i:i + 3])
            i += 2
        return skip


def tensor(rgb, device='cuda', size=256):
    x = torch.from_numpy(np.array(rgb, copy=True, order='C')).permute(2, 0, 1)[None].float().to(device) / 127.5 - 1
    return F.interpolate(x, (size, size), mode='area')


def to_rgb(x, size=None):
    if size:
        x = F.interpolate(x, (size, size), mode='area')
    return ((x.detach().float().cpu()[0].permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).round().astype(np.uint8)


class ArcFace(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        folder = ROOT / 'third_party'
        package('_pilot_arc', folder)
        load_module('_pilot_arc.config', folder / 'config.py')
        arc = load_module('_pilot_arc.arcface_models', folder / 'arcface_models.py')
        allowed = [nn.PReLU, nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.BatchNorm1d,
                   nn.BatchNorm2d, nn.Sequential, nn.AdaptiveAvgPool2d, nn.Dropout, nn.Sigmoid]
        allowed += [(getattr(arc, name), f'models.arcface_models.{name}') for name in ['SEBlock', 'IRBlock', 'ResNet']]
        with torch.serialization.safe_globals(allowed):
            self.model = torch.load(ROOT / 'models/arcface_checkpoint.tar', map_location='cpu', weights_only=True)
        if not isinstance(self.model, arc.ResNet):
            raise TypeError('Unexpected ArcFace checkpoint')
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))
        self.eval().requires_grad_(False).to(device)

    def forward(self, x, grid):
        face = F.grid_sample((x + 1) / 2, grid.to(x), align_corners=False, padding_mode='zeros')
        return F.normalize(self.model((face - self.mean) / self.std), dim=1)


class ArrayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {('numpy', 'ndarray'): np.ndarray, ('numpy', 'dtype'): np.dtype,
                   ('numpy.core.multiarray', '_reconstruct'): np._core.multiarray._reconstruct,
                   ('numpy._core.multiarray', '_reconstruct'): np._core.multiarray._reconstruct}
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f'Unexpected pickle type: {module}.{name}')
        return allowed[module, name]


def rotation(camera):
    a = F.normalize(camera[:, 0], dim=1)
    b = camera[:, 1] - (camera[:, 1] * a).sum(1, keepdim=True) * a
    b = F.normalize(b, dim=1)
    return torch.stack([a, b, torch.linalg.cross(a, b, dim=1)], dim=1)


class Attributes(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        folder = ROOT / 'third_party/tddfa'
        module = load_module('_pilot_mobilenet', folder / 'models/mobilenet_v1.py')
        self.model = module.mobilenet(widen_factor=1., num_classes=62)
        raw = torch.load(folder / 'weights/mb1_120x120.pth', map_location='cpu', weights_only=True)['state_dict']
        raw = {k.removeprefix('module.').replace('fc_param.', 'fc.'): v for k, v in raw.items()}
        extra = set(raw) - set(self.model.state_dict())
        if any(not k.startswith('fc_lm.') for k in extra):
            raise ValueError(f'Unexpected 3DDFA keys {extra}')
        self.model.load_state_dict({k: v for k, v in raw.items() if k not in extra}, strict=True)
        with (folder / 'configs/param_mean_std_62d_120x120.pkl').open('rb') as f:
            norm = ArrayUnpickler(f).load()
        self.register_buffer('mean', torch.as_tensor(norm['mean']).float().view(1, 62))
        self.register_buffer('std', torch.as_tensor(norm['std']).float().view(1, 62))
        self.eval().requires_grad_(False).to(device)

    def forward(self, x, roi):
        sx, sy, ex, ey = [int(round(v)) for v in roi]
        h, w = x.shape[-2:]
        crop = x[:, :, max(sy, 0):min(ey, h), max(sx, 0):min(ex, w)]
        crop = F.pad(crop, (max(-sx, 0), max(ex - w, 0), max(-sy, 0), max(ey - h, 0)), value=-1)
        crop = F.interpolate(crop, (120, 120), mode='bilinear', align_corners=False)
        pred = self.model(crop[:, [2, 1, 0]] * (127.5 / 128))
        actual = pred * self.std + self.mean
        return {'expression': pred[:, 52:62], 'rotation': rotation(actual[:, :12].view(-1, 3, 4)[:, :, :3])}


class Vision:
    def __init__(self):
        self.detector = cv2.FaceDetectorYN.create('onnx', np.frombuffer((ROOT / 'models/yunet.onnx').read_bytes(), np.uint8),
                                                  np.empty(0, np.uint8), (256, 256), .7, .3, 5000)

    def detect(self, rgb):
        if rgb.shape[:2] != (256, 256):
            rgb = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
        _, faces = self.detector.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if faces is None:
            return None
        return max(faces, key=lambda f: f[2] * f[3])

    @staticmethod
    def roi(face):
        x, y, w, h = map(float, face[:4])
        old = (w + h) / 2
        side = int(old * 1.58)
        cx, cy = x + w / 2, y + h / 2 + old * .14
        return [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2]

    @staticmethod
    def grid(face, device='cuda'):
        target = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                           [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
        source = np.asarray(face[4:14], np.float32).reshape(5, 2)
        # Deterministic least-squares similarity transform, without RANSAC.
        a, b = [], []
        for (x, y), (u, v) in zip(source, target):
            a += [[x, -y, 1, 0], [y, x, 0, 1]]
            b += [u, v]
        s, t, tx, ty = np.linalg.lstsq(np.asarray(a), np.asarray(b), rcond=None)[0]
        matrix = np.array([[s, -t, tx], [t, s, ty]], np.float32)
        inverse = cv2.invertAffineTransform(matrix)
        yy, xx = np.mgrid[:112, :112].astype(np.float32)
        xy = np.stack([xx, yy, np.ones_like(xx)], -1) @ inverse.T
        return torch.from_numpy((2 * xy + 1) / 256 - 1)[None].float().to(device)

def pose_degrees(a, b):
    relative = a @ b.transpose(-1, -2)
    value = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    return float(torch.acos(value).mean() * (180 / math.pi))
