"""Standalone V3 inference: frozen networks; two per-image latent branches."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
for key, relative in {'TORCH_HOME': 'torch_home', 'HF_HOME': '.cache/huggingface', 'XDG_CACHE_HOME': '.cache',
                      'TORCH_EXTENSIONS_DIR': '.cache/torch_extensions', 'MPLCONFIGDIR': '.cache/matplotlib',
                      'CUDA_CACHE_PATH': '.cache/cuda', 'TRITON_CACHE_DIR': '.cache/triton',
                      'TEMP': 'tmp', 'TMP': 'tmp'}.items():
    os.environ[key] = str(ROOT/relative)
(ROOT/'tmp').mkdir(exist_ok=True)

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from runtime import ArcFace, Attributes, StyleGAN, Vision, pose_degrees, tensor, to_rgb
from fan_features import OpeningFeatures
from gaze_features import GazeFeatures
from losses import V3Losses


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def model_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def check_models():
    manifest = json.loads((ROOT/'model_manifest.json').read_text(encoding='utf-8'))
    problems = []
    for entry in manifest['files']:
        path = ROOT/entry['path']
        if not path.is_file():
            problems.append('Missing: '+entry['path'])
        elif sha(path) != entry['sha256']:
            problems.append('SHA256 mismatch: '+entry['path'])
    if problems:
        raise FileNotFoundError('\n'.join(problems)+'\nSee README.md for model download paths.')
    return manifest


def load_rgb(path):
    with Image.open(path) as image:
        if image.width != image.height or image.width < 64:
            raise ValueError('Input must be a clear square face crop, at least 64 pixels wide.')
        return np.asarray(image.convert('RGB')).copy()


class Anonymizer(V3Losses):
    def __init__(self, device='auto'):
        torch.set_num_threads(4)
        torch.manual_seed(20261020)
        np.random.seed(20261020)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device not in {'cuda', 'cpu'}:
            raise ValueError('device must be auto, cuda or cpu')
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; install CUDA PyTorch or select --device cpu.')
        self.device = device
        self.options = json.loads((ROOT/'model_config.json').read_text(encoding='utf-8'))
        check_models()
        self.g = StyleGAN(device)
        self.arc = ArcFace(device)
        self.attr = Attributes(device)
        self.openings = OpeningFeatures(device)
        self.gaze = GazeFeatures(device)
        self.vision = Vision()
        self.networks = {'e4e': self.g.encoder, 'generator': self.g.decoder, 'arcface': self.arc,
                         '3ddfa': self.attr, 'fan': self.openings, 'gaze': self.gaze}
        self.assert_frozen()

    def assert_frozen(self):
        for name, model in self.networks.items():
            if model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
                raise RuntimeError('Network must stay frozen: '+name)

    @torch.no_grad()
    def context(self, rgb):
        src = tensor(rgb, device=self.device)
        face = self.vision.detect(to_rgb(src))
        if face is None:
            raise ValueError('No face detected. Supply a clear square face crop.')
        grid, roi = self.vision.grid(face, device=self.device), self.vision.roi(face)
        return {'src': src, 'face': face, 'grid': grid, 'roi': roi,
                'arc': self.arc(src, grid), 'attr': self.attr(src, roi)}

    @staticmethod
    def seed_for(path):
        stem = Path(path).stem
        value = int(stem) if stem.isascii() and stem.isdigit() else int(sha(path)[:8], 16)
        return (20261020+value) % (2**63-1)

    @torch.no_grad()
    def candidate_metrics(self, rgb, ctx):
        x = tensor(rgb, device=self.device)
        attr = self.attr(x, ctx['roi'])
        small = to_rgb(x)
        face = self.vision.detect(small)
        return {'arcface_cosine_fixed': float((self.arc(x, ctx['grid'])*ctx['arc']).sum()),
                'expression_mae': float((attr['expression']-ctx['attr']['expression']).abs().mean()),
                'pose_degrees': pose_degrees(attr['rotation'], ctx['attr']['rotation']),
                'face_detected': face is not None,
                'detection_confidence': float(face[-1]) if face is not None else 0.,
                'clipped_pixel_fraction': float(((small <= 1) | (small >= 254)).mean())}

    @torch.no_grad()
    def select_teacher(self, w0, ctx, seed):
        options = self.options['candidate']
        rng = torch.Generator(device=self.device).manual_seed(seed)
        records, best, fallback = [], None, None
        for donor in range(options['donors']):
            prior = self.g.decoder.style(torch.randn((1, 512), device=self.device, generator=rng))
            avg = self.g.avg.reshape(1, -1, 512)[:, 0]
            donor_w = avg+options['truncation']*(prior-avg)
            for alpha in options['alphas']:
                w = w0.clone()
                a, b = options['layer_slice']
                w[:, a:b] = (1-alpha)*w0[:, a:b]+alpha*donor_w[:, None]
                rgb = to_rgb(self.g.generate(w))
                metric = self.candidate_metrics(rgb, ctx)
                metric.update(index=len(records), donor=donor, alpha=alpha)
                metric['score'] = -5*metric['expression_mae']-metric['pose_degrees']/15-2*metric['arcface_cosine_fixed']+.1*metric['detection_confidence']-.25*metric['clipped_pixel_fraction']
                metric['qualified'] = bool(metric['face_detected'] and metric['arcface_cosine_fixed'] <= .30 and metric['expression_mae'] <= .15 and metric['pose_degrees'] <= 5)
                records.append(metric)
                entry = (metric, rgb, w.detach().clone())
                if fallback is None or metric['score'] > fallback[0]['score']:
                    fallback = entry
                if metric['qualified'] and (best is None or metric['score'] > best[0]['score']):
                    best = entry
        selected, rgb, latent = best or fallback
        return rgb, latent, {'mode': 'generated_local_layer_mixing', 'seed': seed, 'records': records,
                             'selected': selected['index'], 'qualified_count': sum(r['qualified'] for r in records),
                             'target_qualified': selected['qualified'], 'retained_original_latent': True}

    def load_latent(self, path):
        latent = torch.load(path, map_location=self.device, weights_only=True)
        if not isinstance(latent, torch.Tensor) or tuple(latent.shape) != (1, 18, 512) or not torch.isfinite(latent).all():
            raise ValueError('Expected a finite W+ tensor with shape [1,18,512]')
        return latent.detach()

    def optimize(self, initial, ctx, reference, folder, branch):
        folder.mkdir()
        started = time.monotonic()
        w = initial.detach().clone().requires_grad_(True)
        options = self.options
        optimizer = torch.optim.Adam([w], lr=options['lr_start'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=options['steps'], eta_min=options['lr_end'])
        history, face, detection_failures = [], None, 0
        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            Image.fromarray(to_rgb(self.g.generate(w))).save(folder/'initial.png')
        for step in range(options['steps']):
            optimizer.zero_grad(set_to_none=True)
            raw = self.g.generate(w)
            if step % options['roi_update_interval'] == 0:
                current = self.vision.detect(to_rgb(self.small(raw)))
                if current is None:
                    detection_failures += 1
                    if face is None:
                        raise ValueError('No face detected at branch initialization')
                else:
                    face = current
            terms = self.losses(raw, w, ctx, reference, face, branch)
            terms['total'].backward()
            if w.grad is None or not torch.isfinite(w.grad).all() or not all(torch.isfinite(v) for v in terms.values()):
                raise FloatingPointError(f'Invalid loss/gradient at {branch} step {step+1}')
            self.assert_frozen()
            norm = torch.nn.utils.clip_grad_norm_([w], options['gradient_clip_norm'])
            record = {'step': step+1, 'lr': optimizer.param_groups[0]['lr'],
                      **{key: float(v.detach()) for key, v in terms.items()}, 'gradient_norm_before_clip': float(norm)}
            history.append(record)
            optimizer.step()
            scheduler.step()
            if step % 25 == 0 or step+1 == options['steps']:
                write_json(folder/'progress.json', record)
                print(f'{branch}: {step+1}/{options["steps"]}, loss={record["total"]:.4f}', flush=True)
        with torch.no_grad():
            raw = self.g.generate(w)
            result = to_rgb(raw)
            current = self.vision.detect(to_rgb(self.small(raw)))
            terms = self.losses(raw, w, ctx, reference, current if current is not None else face, branch)
        Image.fromarray(result).save(folder/'final.png')
        torch.save(w.detach().cpu(), folder/'latent.pt')
        write_json(folder/'history.json', history)
        report = {'steps': options['steps'], 'seconds': time.monotonic()-started, 'checkpoint': 'fixed final step',
                  'roi_detection_failures': detection_failures, 'latent_gradient_nonzero_steps': sum(h['gradient_norm_before_clip'] > 0 for h in history),
                  'final_terms': {k: float(v.detach()) for k, v in terms.items()},
                  'peak_allocated_gpu_gb': torch.cuda.max_memory_allocated()/1024**3 if self.device == 'cuda' else None}
        write_json(folder/'optimization.json', report)
        return result, report

    def anonymize(self, input_path, output_dir, *, seed=None, initial_latent=None, teacher_path=None, teacher_latent=None):
        if bool(teacher_path) != bool(teacher_latent):
            raise ValueError('Saved teacher image and its original latent must be supplied together; no re-encoding.')
        folder = Path(output_dir).resolve()
        if folder.exists():
            raise FileExistsError('Choose a new output directory: '+str(folder))
        rgb = load_rgb(input_path)
        ctx = self.context(rgb)
        folder.mkdir(parents=True)
        started = time.monotonic()
        try:
            before = {name: model_hash(model) for name, model in self.networks.items()}
            Image.fromarray(rgb).save(folder/'source.png')
            w0 = self.load_latent(initial_latent) if initial_latent else self.g.invert(rgb)
            torch.save(w0.cpu(), folder/'initial_latent.pt')
            if teacher_path:
                teacher_rgb = load_rgb(teacher_path)
                teacher_w = self.load_latent(teacher_latent)
                with torch.no_grad():
                    difference = np.abs(to_rgb(self.g.generate(teacher_w)).astype(int)-teacher_rgb.astype(int))
                if difference.max() > 1:
                    raise ValueError('Provided teacher PNG does not match its original latent (max difference > 1 LSB).')
                info = {'mode': 'provided_image_and_original_latent', 'teacher_sha256': sha(teacher_path),
                        'teacher_latent_sha256': sha(teacher_latent), 'retained_original_latent': True}
            else:
                donor_seed = self.seed_for(input_path) if seed is None else int(seed)
                if not 0 <= donor_seed < 2**63:
                    raise ValueError('seed must be between 0 and 2**63-1')
                teacher_rgb, teacher_w, info = self.select_teacher(w0, ctx, donor_seed)
            Image.fromarray(teacher_rgb).save(folder/'teacher.png')
            torch.save(teacher_w.cpu(), folder/'teacher_latent.pt')
            write_json(folder/'candidates.json', info)
            teacher = tensor(teacher_rgb, device=self.device)
            reference = self.targets(ctx, teacher, teacher_w)
            torch.manual_seed(self.options['seed'])
            np.random.seed(self.options['seed'])
            reports = {}
            for branch, initial, filename in [('baseline', w0, 'baseline.png'), ('optimized', teacher_w, 'final.png')]:
                result, reports[branch] = self.optimize(initial, ctx, reference, folder/branch, branch)
                Image.fromarray(result).save(folder/filename)
            # Convenient top-level latent of the final image.
            torch.save(self.load_latent(folder/'optimized/latent.pt').cpu(), folder/'latent.pt')
            after = {name: model_hash(model) for name, model in self.networks.items()}
            self.assert_frozen()
            if before != after:
                raise RuntimeError('Frozen model state changed')
            report = {'status': 'completed', 'method': self.options['version'], 'input_sha256': sha(input_path),
                      'device': self.device, 'seconds': time.monotonic()-started, 'steps_per_branch': self.options['steps'],
                      'network_weight_updates': 0, 'model_states_before': before, 'model_states_after': after,
                      'baseline_initialization': 'e4e w0', 'optimized_initialization': 'retained teacher latent',
                      'common_latent_regularizer_anchor': 'teacher latent', 'body_loss': False,
                      'columns': ['source.png', 'baseline.png', 'teacher.png', 'final.png'], 'optimization': reports}
            write_json(folder/'result.json', report)
            self.comparison(folder)
            return report
        except Exception as error:
            write_json(folder/'failure.json', {'error': type(error).__name__, 'message': str(error)})
            raise

    @staticmethod
    def comparison(folder):
        labels = ['原图', '基线图', '监督图', '优化图']
        font = None
        for path in ['C:/Windows/Fonts/msyh.ttc', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
            if Path(path).exists():
                font = ImageFont.truetype(path, 24)
                break
        if font is None:
            labels = ['Original', 'Baseline', 'Teacher', 'Optimized']
            font = ImageFont.load_default(size=20)
        canvas = Image.new('RGB', (4*336+16, 376), 'white')
        draw = ImageDraw.Draw(canvas)
        for col, (name, label) in enumerate(zip(['source.png', 'baseline.png', 'teacher.png', 'final.png'], labels)):
            x = 16+col*336
            with Image.open(folder/name) as image:
                canvas.paste(image.convert('RGB').resize((320, 320), Image.Resampling.LANCZOS), (x, 44))
            draw.text((x+160, 7), label, font=font, fill='black', anchor='mt')
        canvas.save(folder/'comparison.jpg', quality=97, subsampling=0)

    @torch.no_grad()
    def render_latent(self, path, output):
        output = Path(output)
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(to_rgb(self.g.generate(self.load_latent(path)))).save(output)
