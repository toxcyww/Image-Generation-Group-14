"""V3 source attributes, anonymous teacher identity, gaze and cheek chroma."""
import math
import torch
from torch.nn import functional as F
from runtime import to_rgb


class V3Losses:
    @staticmethod
    def small(raw):
        return F.interpolate(raw.clamp(-1, 1), (256, 256), mode='area')

    @staticmethod
    def gray(x):
        return (x*x.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1, keepdim=True).expand(-1, 3, -1, -1)

    def grids(self, face):
        return torch.cat([self.openings.grid(face, device=self.device, scale_factor=s)
                          for s in self.options['fan_crop_scales']])

    def cheek_chroma(self, image, face):
        crop = F.grid_sample((image+1)/2, self.gaze.grid(face, device=self.device).to(image),
                             align_corners=False, padding_mode='border')
        t = (torch.arange(224, device=image.device, dtype=image.dtype)+.5)/224
        yy, xx = torch.meshgrid(t, t, indexing='ij')
        mask = (((xx-.28)/.13)**2+((yy-.60)/.14)**2 < 1) | (((xx-.72)/.13)**2+((yy-.60)/.14)**2 < 1)
        rgb = crop[:, :, mask].mean(-1)
        matrix = image.new_tensor([[-.168736, -.331264, .5], [.5, -.418688, -.081312]])
        return rgb @ matrix.T

    @torch.no_grad()
    def targets(self, ctx, teacher, teacher_w):
        face = self.vision.detect(to_rgb(teacher))
        if face is None:
            raise ValueError('No face detected in teacher image')
        _, vector = self.gaze(ctx['src'], self.gaze.grid(ctx['face'], device=self.device))
        return {'openings': self.openings(ctx['src'].expand(2, -1, -1, -1), self.grids(ctx['face'])).detach(),
                'gray_rotation': self.attr(self.gray(ctx['src']), ctx['roi'])['rotation'].detach(),
                'teacher_identity': self.arc(teacher, self.vision.grid(face, device=self.device)).detach(),
                'teacher_w': teacher_w.detach(), 'teacher_face': face,
                'source_gaze_vector': vector.detach(),
                'source_cheek_chroma': self.cheek_chroma(ctx['src'], ctx['face']).detach()}

    def losses(self, raw, w, ctx, reference, face, branch):
        options = self.options
        x = self.small(raw)
        roi = self.vision.roi(face)
        attr = self.attr(x, roi)
        gray_attr = self.attr(self.gray(x), roi)
        expr = F.smooth_l1_loss((attr['expression']-ctx['attr']['expression'])/options['expression_scale'],
                                torch.zeros_like(attr['expression']), beta=1.)
        rotation_mse = (F.mse_loss(attr['rotation'], ctx['attr']['rotation'])+
                        F.mse_loss(gray_attr['rotation'], reference['gray_rotation']))/2
        unit = 4*(1-math.cos(math.radians(options['pose_scale_degrees'])))/9
        pose = rotation_mse/unit
        openings = self.openings(x.expand(2, -1, -1, -1), self.grids(face))
        difference = openings-reference['openings']
        eyes = F.smooth_l1_loss(difference[:, :2]/options['eye_scale'], torch.zeros_like(difference[:, :2]), beta=1.)
        mouth = F.smooth_l1_loss(difference[:, 2:]/options['mouth_scale'], torch.zeros_like(difference[:, 2:]), beta=1.)
        identity = self.arc(x, self.vision.grid(face, device=self.device))
        source_cos = (identity*ctx['arc']).sum()
        teacher_cos = (identity*reference['teacher_identity']).sum()
        terms = {'expression': expr, 'pose': pose, 'eyes': eyes, 'mouth': mouth,
                 'anonymity': F.relu(source_cos-options['identity_margin']),
                 'supervised_identity': 1-teacher_cos,
                 'latent': F.mse_loss(w, reference['teacher_w']),
                 'range': F.relu(raw.abs()-1).square().mean()}
        zeros = raw.new_zeros(())
        if branch == 'baseline':
            total = sum(options['weights'][name]*value for name, value in terms.items() if name != 'supervised_identity')
            additions = {'applied_supervision_weight': zeros, 'gaze_term': zeros, 'context_term': zeros, 'chroma_term': zeros}
        elif branch == 'optimized':
            total = sum(options['weights'][name]*value for name, value in terms.items())
            _, vector = self.gaze(x, self.gaze.grid(face, device=self.device))
            cosine = (vector*reference['source_gaze_vector']).sum(-1).clamp(-1, 1)
            tolerance = 1-math.cos(math.radians(options['gaze_tolerance_degrees']))
            unit = 1-math.cos(math.radians(options['gaze_normalization_degrees']))
            gaze_term = F.relu(1-cosine-tolerance).mean()/unit
            chroma = self.cheek_chroma(x, face)
            difference_chroma = (chroma-reference['source_cheek_chroma'])/options['chroma_scale']
            chroma_term = F.smooth_l1_loss(difference_chroma, torch.zeros_like(difference_chroma), beta=1.)
            total = total+options['gaze_weight']*gaze_term+options['chroma_weight']*chroma_term
            additions = {'applied_supervision_weight': raw.new_tensor(options['weights']['supervised_identity']),
                         'gaze_term': gaze_term, 'context_term': zeros, 'chroma_term': chroma_term,
                         'chroma_absolute_error': (chroma-reference['source_cheek_chroma']).abs().mean(),
                         'gaze_train_degrees': cosine.detach().acos().mean()*180/math.pi}
        else:
            raise ValueError('Expected baseline or optimized branch')
        return {'total': total, **terms, **additions, 'source_arcface_cosine_own': source_cos,
                'teacher_arcface_cosine_own': teacher_cos,
                'eye_ratio_mae': difference[:, :2].abs().mean(), 'mouth_ratio_error': difference[:, 2:].abs().mean()}
