"""ConvLSTM root tracker: walks a CT volume slice by slice along Z and segments root in each
slice, using the previous slice's predicted mask as extra input (closed loop -- it never sees
ground truth at inference time, only its own prior prediction).
"""
import random

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """(slice, prev_mask) stacked as 2 channels -> feature map, downsampled 4x by two stride-2 convs."""

    def __init__(self, base=24, latent=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, base, 3, stride=2, padding=1), nn.GroupNorm(4, base), nn.SiLU(),
            nn.Conv2d(base, latent, 3, stride=2, padding=1), nn.GroupNorm(4, latent), nn.SiLU(),
            nn.Conv2d(latent, latent, 3, padding=1), nn.GroupNorm(4, latent), nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """Feature map -> mask logits, upsampled back to full resolution."""

    def __init__(self, base=24, latent=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent, latent, 3, padding=1), nn.GroupNorm(4, latent), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(latent, base, 3, padding=1), nn.GroupNorm(4, base), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base, base, 3, padding=1), nn.GroupNorm(4, base), nn.SiLU(),
            nn.Conv2d(base, 1, 3, padding=1),
        )

    def forward(self, h):
        return self.net(h).squeeze(1)  # (B, P, P) logits


def _block(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1),
                         nn.GroupNorm(4, cout), nn.SiLU())


class SkipEncoder(nn.Module):
    """Like Encoder, but also returns the full-res and half-res feature maps so the decoder can
    recover thin structure that the 4x downsample would otherwise wash out."""

    def __init__(self, base=24, latent=48):
        super().__init__()
        self.c0 = max(base // 2, 8)
        self.s0 = _block(2, self.c0)
        self.s1 = _block(self.c0, base, stride=2)
        self.s2 = nn.Sequential(_block(base, latent, stride=2), _block(latent, latent))

    def forward(self, x):
        f0 = self.s0(x)
        f1 = self.s1(f0)
        return self.s2(f1), (f0, f1)


class SkipDecoder(nn.Module):
    """U-Net-style decoder: upsample the recurrent state, concatenating encoder skips."""

    def __init__(self, base=24, latent=48):
        super().__init__()
        c0 = max(base // 2, 8)
        self.d2 = _block(latent, latent)
        self.d1 = _block(latent + base, base)
        self.d0 = nn.Sequential(_block(base + c0, base), _block(base, base))
        self.out = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, h, skips):
        f0, f1 = skips
        x = self.d2(h)
        x = self.d1(torch.cat([nn.functional.interpolate(x, scale_factor=2, mode="nearest"), f1], 1))
        x = self.d0(torch.cat([nn.functional.interpolate(x, scale_factor=2, mode="nearest"), f0], 1))
        return self.out(x).squeeze(1)


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM update over a spatial feature map."""

    def __init__(self, latent):
        super().__init__()
        self.input_gate = nn.Conv2d(2 * latent, latent, 3, padding=1)
        self.forget_gate = nn.Conv2d(2 * latent, latent, 3, padding=1)
        self.output_gate = nn.Conv2d(2 * latent, latent, 3, padding=1)
        self.cell_gate = nn.Conv2d(2 * latent, latent, 3, padding=1)

    def forward(self, x, state):
        h, c = state
        xh = torch.cat([x, h], dim=1)
        i = torch.sigmoid(self.input_gate(xh))
        f = torch.sigmoid(self.forget_gate(xh))
        o = torch.sigmoid(self.output_gate(xh))
        g = torch.tanh(self.cell_gate(xh))
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class RootTracker(nn.Module):
    def __init__(self, base_channels=24, latent_channels=48, skips=False):
        super().__init__()
        self.latent = latent_channels
        self.skips = skips
        self.encoder = (SkipEncoder if skips else Encoder)(base_channels, latent_channels)
        self.lstm = ConvLSTMCell(latent_channels)
        self.predict_head = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels, 3, padding=1),
            nn.GroupNorm(4, latent_channels), nn.SiLU(),
            nn.Conv2d(latent_channels, latent_channels, 3, padding=1),
        )
        self.mask_decoder = (SkipDecoder if skips else Decoder)(base_channels, latent_channels)

    def init_state(self, batch, patch, device):
        h = torch.zeros(batch, self.latent, patch // 4, patch // 4, device=device)
        c = torch.zeros(batch, self.latent, patch // 4, patch // 4, device=device)
        mask_in = torch.zeros(batch, patch, patch, device=device)
        return h, c, mask_in

    def step(self, frame, mask_in, state):
        h, c = state
        x = torch.cat([frame, mask_in.unsqueeze(1)], dim=1)
        if self.skips:
            feat, sk = self.encoder(x)
            h, c = self.lstm(feat, (h, c))
            logits = self.mask_decoder(self.predict_head(h), sk)
        else:
            h, c = self.lstm(self.encoder(x), (h, c))
            logits = self.mask_decoder(self.predict_head(h))
        return logits, (h, c)

    def forward(self, frames, masks, tf_prob=0.0):
        """frames: (B, N, 1, P, P), masks: (B, N, P, P). Predicts mask[t+1] from frame[t] and
        the running state; tf_prob is the chance of feeding the true mask back in instead of
        the model's own prediction (0 during eval -- always closed loop).
        Returns mask_logits, target_masks, both (B, N-1, P, P)."""
        B, N, _, P, _ = frames.shape
        h, c, mask_in = self.init_state(B, P, frames.device)

        logits_list, targets = [], []
        for t in range(N - 1):
            logits, (h, c) = self.step(frames[:, t], mask_in, (h, c))
            logits_list.append(logits)
            targets.append(masks[:, t + 1])
            use_gt = random.random() < tf_prob
            mask_in = masks[:, t + 1] if use_gt else torch.sigmoid(logits).detach()

        return {
            "mask_logits": torch.stack(logits_list, dim=1),
            "target_masks": torch.stack(targets, dim=1),
        }
