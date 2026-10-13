"""Stage 2: SAR-conditioned diffusion-bridge denoiser for all 13 Sentinel-2 bands.

The network sees the bridge state x_t (13 bands), the time t, the optical condition (cloudy S2,
optionally with the masked pixels zeroed), the cloud mask, and S1. It predicts the clear image
x0 as a per-band Laplace distribution: a mean and (if the uncertainty head is on) a log-scale.

The mean is a residual on the condition (residual="cond"): clear pixels start out copied and the
network fills what is masked. A residual on the noisy state x_t ("xt") trained much worse in our
synthetic-data check and broke multi-step sampling.

Switches used by the ablations in NOVELTY.md:
  use_bga        band-group stems, attention and heads (C1) vs. plain all-band equivalents
  uncertainty    joint aleatoric uncertainty head (C3)
  (the NDVI-consistency loss, C2, lives in losses.py)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..bands import N_S1, N_S2
from .bga import BandGroupAttention, GroupHeads, GroupStem, SharedHead
from .blocks import ResBlock, SarAttention, SarEncoder, TimeEmbedding, norm

LOG_SCALE_RANGE = (-9.0, 1.0)  # Laplace scale b in roughly [1e-4, 2.7] reflectance units


class BridgeNet(nn.Module):
    def __init__(self, base=64, mults=(1, 2, 4, 4), blocks=2, attn_levels=(2, 3), sar_ch=64,
                 heads=4, dropout=0.0, use_bga=True, uncertainty=True, use_mask=True,
                 head_hidden=None, residual="cond"):
        super().__init__()
        self.use_bga, self.uncertainty, self.use_mask, self.residual = use_bga, uncertainty, use_mask, residual
        chs = [base * m for m in mults]
        self.n_levels = len(chs)
        temb = base * 4
        self.time = TimeEmbedding(base, temb)

        n_shared = N_S1 + (1 if use_mask else 0)
        if use_bga:
            self.stem = GroupStem(n_per_band=2, n_shared=n_shared, out=chs[0])
        else:
            self.stem = nn.Conv2d(2 * N_S2 + n_shared, chs[0], 3, padding=1)
        self.sar = SarEncoder(N_S1, sar_ch, self.n_levels)

        def attention(dim):
            return BandGroupAttention(dim, sar_ch, heads=heads) if use_bga else SarAttention(dim, sar_ch, heads=heads)

        self.attn_levels = set(attn_levels)
        self.down_blocks, self.down_attn, self.downsample = nn.ModuleList(), nn.ModuleDict(), nn.ModuleList()
        skips, prev = [chs[0]], chs[0]
        for lvl, c in enumerate(chs):
            for _ in range(blocks):
                self.down_blocks.append(ResBlock(prev, c, temb, dropout))
                prev = c
                skips.append(c)
            if lvl in self.attn_levels:
                self.down_attn[str(lvl)] = attention(c)
            if lvl < self.n_levels - 1:
                self.downsample.append(nn.Conv2d(c, c, 3, stride=2, padding=1))
                skips.append(c)
        self.mid1, self.mid_attn, self.mid2 = ResBlock(prev, prev, temb), attention(prev), ResBlock(prev, prev, temb)

        self.up_blocks, self.up_attn, self.upsample = nn.ModuleList(), nn.ModuleDict(), nn.ModuleList()
        for lvl in reversed(range(self.n_levels)):
            c = chs[lvl]
            for _ in range(blocks + 1):
                self.up_blocks.append(ResBlock(prev + skips.pop(), c, temb, dropout))
                prev = c
            if lvl in self.attn_levels:
                self.up_attn[str(lvl)] = attention(c)
            if lvl > 0:
                self.upsample.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                                   nn.Conv2d(c, c, 3, padding=1)))
        self.out_norm = norm(prev)
        k = 2 if uncertainty else 1
        hid = head_hidden or chs[0]
        self.head = GroupHeads(prev, hid, k) if use_bga else SharedHead(prev, hid, k)
        self.blocks = blocks

    def forward(self, x_t, t, cond, s1, mask=None):
        """x_t, cond: (B, 13, H, W); t: (B,); s1: (B, 2, H, W); mask: (B, 1, H, W) or None.

        Returns {"mean": (B, 13, H, W), "log_scale": (B, 13, H, W) or None}.
        H and W must be divisible by 2 ** (levels - 1).
        """
        temb = self.time(t)
        shared = [s1] + ([mask] if self.use_mask else [])
        shared = torch.cat(shared, 1)
        h = self.stem([x_t, cond], shared) if self.use_bga else self.stem(torch.cat([x_t, cond, shared], 1))
        sar = self.sar(s1)

        hs, bi = [h], 0
        for lvl in range(self.n_levels):
            for _ in range(self.blocks):
                h = self.down_blocks[bi](h, temb)
                bi += 1
                hs.append(h)
            if str(lvl) in self.down_attn:
                h = self.down_attn[str(lvl)](h, sar[lvl])
                hs[-1] = h
            if lvl < self.n_levels - 1:
                h = self.downsample[lvl](h)
                hs.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, temb), sar[-1]), temb)

        bi, ui = 0, 0
        for lvl in reversed(range(self.n_levels)):
            for _ in range(self.blocks + 1):
                h = self.up_blocks[bi](torch.cat([h, hs.pop()], 1), temb)
                bi += 1
            if str(lvl) in self.up_attn:
                h = self.up_attn[str(lvl)](h, sar[lvl])
            if lvl > 0:
                h = self.upsample[ui](h)
                ui += 1
        out = self.head(F.silu(self.out_norm(h)))
        delta = out[:, :N_S2]
        base = {"xt": x_t, "cond": cond, "none": torch.zeros_like(x_t)}[self.residual]
        mean = base + delta
        log_scale = None
        if self.uncertainty:
            # zero-initialised head, so b starts at e^-4 ~ 0.018 reflectance
            log_scale = (out[:, N_S2:] - 4.0).clamp(*LOG_SCALE_RANGE)
        return {"mean": mean, "log_scale": log_scale}
