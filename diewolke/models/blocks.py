"""Shared building blocks: time embedding, residual blocks, attention."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def norm(ch, max_groups=32):
    """GroupNorm with the largest group count <= max_groups that divides ch (>= 4 channels per group)."""
    groups = next(g for g in range(min(max_groups, max(1, ch // 4)), 0, -1) if ch % g == 0)
    return nn.GroupNorm(groups, ch)


def timestep_embedding(t, dim, max_period=10000.0):
    """Sinusoidal embedding of t in [0, 1] (scaled by 1000), shape (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = (t.float() * 1000)[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], -1)
    return F.pad(emb, (0, dim % 2))


class TimeEmbedding(nn.Module):
    def __init__(self, dim, out):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, out), nn.SiLU(), nn.Linear(out, out))

    def forward(self, t):
        return self.mlp(timestep_embedding(t, self.dim))


class ResBlock(nn.Module):
    """GroupNorm-SiLU-Conv x2 with FiLM (scale, shift) from the time embedding."""

    def __init__(self, cin, cout, temb=None, dropout=0.0):
        super().__init__()
        self.n1, self.c1 = norm(cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.n2, self.c2 = norm(cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.film = nn.Linear(temb, 2 * cout) if temb else None
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, x, temb=None):
        h = self.c1(F.silu(self.n1(x)))
        h = self.n2(h)
        if self.film is not None and temb is not None:
            scale, shift = self.film(F.silu(temb))[:, :, None, None].chunk(2, 1)
            h = h * (1 + scale) + shift
        h = self.c2(self.drop(F.silu(h)))
        return self.skip(x) + h


def to_tokens(x):
    return x.flatten(2).transpose(1, 2)  # (B, C, H, W) -> (B, HW, C)


def from_tokens(t, h, w):
    return t.transpose(1, 2).reshape(t.shape[0], -1, h, w)


class MHA(nn.Module):
    """Multi-head attention on token sequences, via scaled_dot_product_attention."""

    def __init__(self, dim, kv_dim=None, heads=4):
        super().__init__()
        assert dim % heads == 0, (dim, heads)
        self.heads = heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(kv_dim or dim, 2 * dim)
        self.out = nn.Linear(dim, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, ctx=None):
        ctx = x if ctx is None else ctx
        b, n, d = x.shape
        q = self.q(x).view(b, n, self.heads, -1).transpose(1, 2)
        k, v = self.kv(ctx).view(b, ctx.shape[1], 2, self.heads, -1).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.out(o.transpose(1, 2).reshape(b, n, d))


class FeedForward(nn.Module):
    def __init__(self, dim, mult=2):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * mult), nn.GELU(), nn.Linear(dim * mult, dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


class SarAttention(nn.Module):
    """Plain (all-band) attention block: self-attention, then cross-attention to SAR tokens.

    This is the 'no band-group attention' counterpart of BandGroupAttention.
    """

    def __init__(self, dim, sar_dim, heads=4):
        super().__init__()
        self.n1, self.attn = nn.LayerNorm(dim), MHA(dim, heads=heads)
        self.n2, self.cross = nn.LayerNorm(dim), MHA(dim, sar_dim, heads=heads)
        self.ff = FeedForward(dim)

    def forward(self, x, sar):
        h, w = x.shape[-2:]
        t, s = to_tokens(x), to_tokens(sar)
        t = t + self.attn(self.n1(t))
        t = t + self.cross(self.n2(t), s)
        return from_tokens(self.ff(t), h, w)


class SarEncoder(nn.Module):
    """Small conv pyramid over S1; returns features at strides 1, 2, 4, ... (one per level)."""

    def __init__(self, cin, ch, levels):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(cin, ch, 3, padding=1), norm(ch), nn.SiLU())
        self.downs = nn.ModuleList(
            nn.Sequential(nn.Conv2d(ch, ch, 3, stride=2, padding=1), norm(ch), nn.SiLU(),
                          nn.Conv2d(ch, ch, 3, padding=1), norm(ch), nn.SiLU())
            for _ in range(levels - 1))

    def forward(self, s1):
        feats = [self.stem(s1)]
        for d in self.downs:
            feats.append(d(feats[-1]))
        return feats
