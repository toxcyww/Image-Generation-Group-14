"""Frozen joint latent optimization with an independent non-ID baseline."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
for key, relative in {'TORCH_HOME': 'torch_home', 'HF_HOME': '.cache/huggingface',
                      'XDG_CACHE_HOME': '.cache', 'TORCH_EXTENSIONS_DIR': '.cache/torch_extensions',
                      'MPLCONFIGDIR': '.cache/matplotlib', 'CUDA_CACHE_PATH': '.cache/cuda',
                      'TRITON_CACHE_DIR': '.cache/triton', 'TEMP': 'tmp', 'TMP': 'tmp'}.items():
    os.environ[key] = str(ROOT / relative)
(ROOT / 'tmp').mkdir(exist_ok=True)

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from runtime import ArcFace, Attributes, StyleGAN, Vision, perceptual, tensor, to_rgb
from round2_core import JointLoss, patch_mask


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def model_hash(model):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def load_rgb(path):
    with Image.open(path) as image:
        if image.width != image.height:
            raise ValueError('Input must be a square, cropped face image, like the course examples.')
        return np.asarray(image.convert('RGB')).copy()


class Detector(Vision):
    """Same YuNet detector/ROI/grid as round two; no evaluation models needed."""
    def __init__(self):
        self.detector = cv2.FaceDetectorYN.create(
            'onnx', np.frombuffer((ROOT / 'models/yunet.onnx').read_bytes(), np.uint8),
            np.empty(0, np.uint8), (256, 256), .7, .3, 5000)


class Anonymizer(JointLoss):
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
            raise RuntimeError('CUDA PyTorch and an NVIDIA GPU are required for --device cuda.')
        self.device = device
        # Fail locally instead of letting torchvision download missing backbone weights.
        if not (ROOT / 'torch_home/hub/checkpoints/squeezenet1_1-b8a52dc0.pth').is_file():
            raise FileNotFoundError('Missing torch_home/hub/checkpoints/squeezenet1_1-b8a52dc0.pth. See README.md for weight download and placement instructions.')
        self.config = json.loads((ROOT / 'model_config.json').read_text(encoding='utf-8'))
        self.g = StyleGAN(device)
        self.arc = ArcFace(device)
        self.attr = Attributes(device)
        self.lpips = perceptual(device)
        self.vision = Detector()
        self.networks = {'e4e': self.g.encoder, 'generator': self.g.decoder,
                         'arcface': self.arc, 'attributes': self.attr, 'lpips': self.lpips}
        self.assert_frozen()

    def assert_frozen(self):
        for name, model in self.networks.items():
            if model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
                raise RuntimeError(f'Network must stay frozen: {name}')

    @torch.no_grad()
    def context(self, rgb):
        src = tensor(rgb, device=self.device)
        face = self.vision.detect(to_rgb(src))
        if face is None:
            raise ValueError('No face detected in input. Supply a clear, square face crop.')
        grid = self.vision.grid(face, device=self.device)
        roi = self.vision.roi(face)
        return {'src': src, 'face': face, 'grid': grid, 'roi': roi,
                'arc': self.arc(src, grid), 'attr': self.attr(src, roi)}

    @torch.no_grad()
    def select_target(self, w0, ctx, seed):
        rng = torch.Generator(device=self.device).manual_seed(seed)
        records = []
        images = []
        for donor in range(2):
            raw = self.g.decoder.style(torch.randn((1, 512), device=self.device, generator=rng))
            avg = self.g.avg.reshape(1, -1, 512)[:, 0]
            donor_w = avg + .8 * (raw - avg)
            for alpha in [.4, .6, .8, 1.]:
                w = w0.clone()
                w[:, 3:9] = (1-alpha)*w0[:, 3:9] + alpha*donor_w[:, None]
                rgb = to_rgb(self.g.generate(w))
                index = len(records)
                # Exact round-two candidate metrics, without FaceNet/SFace/MediaPipe.
                # Their defaults use CUDA; this adapter keeps the optional CPU path local.
                if self.device == 'cpu':
                    m = self.cpu_candidate_metrics(rgb, ctx)
                else:
                    m = self.metrics(rgb, ctx, independent=False)
                m.update(index=index, donor=donor, alpha=alpha)
                m['score'] = -5*m['expression_mae']-m['pose_degrees']/15-2*m['arcface_cosine_fixed']+.1*m['detection_confidence']-.25*m['clipped_pixel_fraction']
                m['qualified'] = bool(m['face_detected'] and m['arcface_cosine_fixed'] <= .30 and m['expression_mae'] <= .15 and m['pose_degrees'] <= 5)
                records.append(m)
                images.append(rgb)
        qualified = [r for r in records if r['qualified']]
        selected = max(qualified or records, key=lambda r: r['score'])
        info = {'records': records, 'selected': selected['index'], 'qualified_count': len(qualified),
                'target_qualified': bool(qualified), 'selection_used_independent_models': False,
                'seed': seed, 'mode': 'generated_local_layer_mixing'}
        return images[selected['index']], info

    @torch.no_grad()
    def cpu_candidate_metrics(self, rgb, ctx):
        from runtime import pose_degrees
        x = tensor(rgb, device='cpu')
        attrs = self.attr(x, ctx['roi'])
        small = to_rgb(x)
        face = self.vision.detect(small)
        return {'arcface_cosine_fixed': float((self.arc(x, ctx['grid'])*ctx['arc']).sum()),
                'expression_mae': float((attrs['expression']-ctx['attr']['expression']).abs().mean()),
                'pose_degrees': pose_degrees(attrs['rotation'], ctx['attr']['rotation']),
                'face_detected': face is not None,
                'detection_confidence': float(face[-1]) if face is not None else 0.,
                'clipped_pixel_fraction': float(((small <= 1) | (small >= 254)).mean())}

    @staticmethod
    def seed_for(path):
        stem = Path(path).stem
        value = int(stem) if stem.isascii() and stem.isdigit() else int(sha(path)[:8], 16)
        return (20261020 + value) % (2**63 - 1)

    @staticmethod
    def nonid_seed_for(path):
        # The baseline's random identity is keyed by filename, not image pixels.
        stem = Path(path).stem
        value = int(stem) if stem.isascii() and stem.isdigit() else int(hashlib.sha256(stem.encode('utf-8')).hexdigest()[:8], 16)
        return (31000 + value) % (2**63 - 1)

    @torch.no_grad()
    def random_nonid_start(self, seed):
        rng = torch.Generator(device=self.device).manual_seed(seed)
        options = self.config['baseline']
        for attempt in range(options['max_random_attempts']):
            z = torch.randn((1, 512), device=self.device, generator=rng)
            prior = self.g.decoder.style(z)
            avg = self.g.avg.reshape(1, -1, 512)[:, 0]
            w = (avg + options['truncation']*(prior-avg))[:, None].repeat(1, 18, 1)
            face = self.vision.detect(to_rgb(self.g.generate(w), 256))
            if face is not None:
                return w, self.vision.roi(face), attempt
        raise ValueError('No face detected from the fixed random-prior attempts')

    def optimize_nonid(self, z0, seed):
        """Only source expression/rotation enter this independent baseline."""
        started = time.monotonic()
        z0 = {key: z0[key].detach().clone() for key in ['expression', 'rotation']}
        initial, roi, attempt = self.random_nonid_start(seed)
        w = initial.detach().clone().requires_grad_(True)
        options = self.config['baseline']
        steps = options['steps']
        optimizer = torch.optim.Adam([w], lr=options['lr_start'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=options['lr_end'])
        history = []
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            terms = self.nonid_losses(self.g.generate(w), w, initial, z0, roi)
            terms['total'].backward()
            if w.grad is None or not torch.isfinite(w.grad).all() or not all(torch.isfinite(value) for value in terms.values()):
                raise FloatingPointError(f'Nonfinite non-ID baseline loss/gradient at step {step+1}')
            self.assert_frozen()
            history.append({'step': step+1, 'lr': optimizer.param_groups[0]['lr'],
                            **{key: float(value.detach()) for key, value in terms.items()},
                            'latent_gradient_norm': float(w.grad.norm())})
            optimizer.step()
            scheduler.step()
            if step % 25 == 0 or step == steps-1:
                print(f'Non-ID baseline: {step+1}/{steps}, loss={history[-1]["total"]:.4f}', flush=True)
        with torch.no_grad():
            rgb = to_rgb(self.g.generate(w))
        info = {'kind': 'nonidentity_condition', 'steps': steps, 'seed': seed,
                'random_roi': roi, 'random_attempt': attempt, 'seconds': time.monotonic()-started,
                'source_information': ['expression10', 'rotation3x3'],
                'initialization': 'independent truncated random W',
                'network_weight_updates': 0, 'checkpoint': 'fixed_last_step'}
        return rgb, w.detach().cpu(), info, history

    def anonymize(self, input_path, output_dir, *, seed=None, teacher_path=None):
        """Run joint optimization and the non-ID baseline; output_dir must not exist.

        teacher_path is optional: by default the model builds its own teacher.
        It is useful for replaying an exact saved development experiment.
        """
        started = time.monotonic()
        input_path, folder = Path(input_path).resolve(), Path(output_dir).resolve()
        rgb = load_rgb(input_path)
        ctx = self.context(rgb)
        if folder.exists():
            raise FileExistsError(f'Output exists; choose a new output directory: {folder}')
        folder.mkdir(parents=True)
        try:
            Image.fromarray(rgb).save(folder / 'source.png')
            w0 = self.g.invert(rgb)
            torch.save(w0.detach().cpu(), folder / 'initial_latent.pt')
            if teacher_path is None:
                donor_seed = self.seed_for(input_path) if seed is None else int(seed)
                if not 0 <= donor_seed < 2**63:
                    raise ValueError('seed must be between 0 and 2**63-1')
                target_rgb, target_info = self.select_target(w0, ctx, donor_seed)
            else:
                teacher_path = Path(teacher_path).resolve()
                target_rgb = load_rgb(teacher_path)
                target_info = {'mode': 'provided_teacher', 'teacher_sha256': sha(teacher_path),
                               'selection_used_independent_models': False}
            Image.fromarray(target_rgb).save(folder / 'target.png')
            write_json(folder / 'candidates.json', target_info)
            target = tensor(target_rgb, device=self.device)
            target_face = self.vision.detect(to_rgb(target))
            if target_face is None:
                raise ValueError('Selected teacher has no detected face; refusing to fabricate eye/mouth locations.')
            mask = patch_mask(target, target_face)
            # Match the exact reference's final branch: same w0, Adam and schedule.
            w = w0.detach().clone().requires_grad_(True)
            steps = self.config['steps']
            optimizer = torch.optim.Adam([w], lr=self.config['lr_start'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=self.config['lr_end'])
            history = []
            if self.device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            for step in range(steps):
                optimizer.zero_grad(set_to_none=True)
                raw = self.g.generate(w)
                terms = self.new_losses(raw, w, w0, ctx, target, mask, True)
                terms['total'].backward()
                if w.grad is None or not torch.isfinite(w.grad).all() or not all(torch.isfinite(v) for v in terms.values()):
                    raise FloatingPointError(f'Nonfinite gradient/loss at step {step+1}')
                self.assert_frozen()
                record = {'step': step+1, 'lr': optimizer.param_groups[0]['lr'],
                          **{key: float(value.detach()) for key, value in terms.items()},
                          'latent_gradient_norm': float(w.grad.norm())}
                history.append(record)
                optimizer.step()
                scheduler.step()
                if step % 25 == 0 or step == steps-1:
                    write_json(folder / 'progress.json', record)
                    print(f'{input_path.name}: {step+1}/{steps}, loss={record["total"]:.4f}', flush=True)
            with torch.no_grad():
                result = to_rgb(self.g.generate(w))
            Image.fromarray(result).save(folder / 'final.png')
            torch.save(w.detach().cpu(), folder / 'latent.pt')
            write_json(folder / 'history.json', history)
            # This is a separate comparison branch. The joint result above starts at w0.
            z0 = {key: ctx['attr'][key].detach().clone() for key in ['expression', 'rotation']}
            baseline_rgb, baseline_w, baseline_info, baseline_history = self.optimize_nonid(z0, self.nonid_seed_for(input_path))
            Image.fromarray(baseline_rgb).save(folder / 'nonid_baseline.png')
            torch.save(baseline_w, folder / 'nonid_latent.pt')
            write_json(folder / 'nonid_history.json', baseline_history)
            self.assert_frozen()
            report = {'status': 'completed', 'method': self.config['version'], 'input_sha256': sha(input_path),
                      'initialization': 'e4e inversion w0', 'network_weight_updates': 0,
                      'steps': steps, 'checkpoint': 'fixed_last_step', 'device': self.device,
                      'seconds': time.monotonic()-started, 'teacher': target_info,
                      'output': 'final.png', 'latent': 'latent.pt',
                      'baseline_output': 'nonid_baseline.png', 'nonid_baseline': baseline_info,
                      'comparison_columns': ['source', 'nonid_baseline', 'target', 'final'],
                      'peak_allocated_gpu_gb': torch.cuda.max_memory_allocated()/1e9 if self.device == 'cuda' else None}
            write_json(folder / 'result.json', report)
            self.comparison(folder)
            return report
        except Exception as error:
            write_json(folder / 'failure.json', {'error': type(error).__name__, 'message': str(error)})
            raise

    @staticmethod
    def comparison(folder):
        paths = ['source.png', 'nonid_baseline.png', 'target.png', 'final.png']
        labels = ['Input', 'Non-ID baseline', 'Teacher', 'Joint optimized']
        canvas = Image.new('RGB', (4*256, 284), 'white')
        draw = ImageDraw.Draw(canvas)
        for column, (path, label) in enumerate(zip(paths, labels)):
            with Image.open(folder / path) as image:
                canvas.paste(image.resize((256, 256)), (column*256, 28))
            draw.text((column*256+8, 8), label, fill='black')
        canvas.save(folder / 'comparison.jpg', quality=95)

    @torch.no_grad()
    def render_latent(self, latent_path, output_path):
        w = torch.load(latent_path, map_location=self.device, weights_only=True)
        if not isinstance(w, torch.Tensor) or list(w.shape) != [1, 18, 512] or not torch.isfinite(w).all():
            raise ValueError('Expected a finite W+ tensor with shape [1,18,512]')
        target = Path(output_path)
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(to_rgb(self.g.generate(w))).save(target)
