"""Band-group attention (contribution C1).

The decoder features are projected into one stream per spectral group (visible, red-edge/NIR,
SWIR). Each stream attends over space and cross-attends to SAR features with its own weights,
then the streams exchange a pooled summary through a gate and are merged back. The output heads
are also group-specific, so each group's bands are predicted from that group's features.
"""
import math

import torch
import torch.nn as nn

from ..bands import BAND_GROUPS, N_S2
from .blocks import MHA, FeedForward, from_tokens, norm, to_tokens


class BandGroupAttention(nn.Module):
    def __init__(self, dim, sar_dim, n_groups=len(BAND_GROUPS), group_dim=None, heads=4):
        super().__init__()
        gd = group_dim or max(heads * 8, (dim // 2) // heads * heads)
        self.g, self.gd = n_groups, gd
        self.proj_in = nn.ModuleList(nn.Conv2d(dim, gd, 1) for _ in range(n_groups))
        self.n1 = nn.ModuleList(nn.LayerNorm(gd) for _ in range(n_groups))
        self.attn = nn.ModuleList(MHA(gd, heads=heads) for _ in range(n_groups))
        self.n2 = nn.ModuleList(nn.LayerNorm(gd) for _ in range(n_groups))
        self.cross = nn.ModuleList(MHA(gd, sar_dim, heads=heads) for _ in range(n_groups))
        self.ff = nn.ModuleList(FeedForward(gd) for _ in range(n_groups))
        # cross-group exchange: pooled summaries of all streams -> per-stream channel gates
        self.mix = nn.Sequential(nn.Linear(n_groups * gd, gd), nn.SiLU(), nn.Linear(gd, n_groups * gd))
        nn.init.zeros_(self.mix[-1].weight)
        nn.init.zeros_(self.mix[-1].bias)
        self.proj_out = nn.Conv2d(n_groups * gd, dim, 1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x, sar):
        b, _, h, w = x.shape
        s = to_tokens(sar)
        streams = []
        for g in range(self.g):
            t = to_tokens(self.proj_in[g](x))
            t = t + self.attn[g](self.n1[g](t))
            t = t + self.cross[g](self.n2[g](t), s)
            streams.append(t)
        summary = torch.cat([t.mean(1) for t in streams], -1)
        gates = 2 * torch.sigmoid(self.mix(summary)).view(b, self.g, 1, self.gd)  # 1 at init
        streams = [self.ff[g](streams[g] * gates[:, g]) for g in range(self.g)]
        out = torch.cat([from_tokens(t, h, w) for t in streams], 1)
        return x + self.proj_out(out)


class GroupStem(nn.Module):
    """Group-wise input encoding: each band group's channels get their own first convolution.

    shared: channels used by every group (mask, SAR); per_band: a list of (B, 13, H, W) tensors
    whose group slices are routed to that group's conv.
    """

    def __init__(self, n_per_band, n_shared, out, groups=BAND_GROUPS):
        super().__init__()
        self.groups = list(groups.values())
        sizes = [out // len(self.groups)] * len(self.groups)
        sizes[0] += out - sum(sizes)
        self.convs = nn.ModuleList(nn.Conv2d(n_per_band * len(idx) + n_shared, c, 3, padding=1)
                                   for idx, c in zip(self.groups, sizes))

    def forward(self, per_band, shared):
        outs = []
        for idx, conv in zip(self.groups, self.convs):
            parts = [t[:, idx] for t in per_band] + ([shared] if shared is not None else [])
            outs.append(conv(torch.cat(parts, 1)))
        return torch.cat(outs, 1)


def _head(cin, hid, cout):
    head = nn.Sequential(nn.Conv2d(cin, hid, 3, padding=1), norm(hid), nn.SiLU(),
                         nn.Conv2d(hid, hid, 3, padding=1), norm(hid), nn.SiLU(),
                         nn.Conv2d(hid, cout, 1))
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


class GroupHeads(nn.Module):
    """One output head per band group; returns (B, out_per_band * 13, H, W) in band order.

    Channel layout of the result: [values for 13 bands, then (if out_per_band == 2) 13 log-scales].
    """

    def __init__(self, cin, hid, out_per_band=2, groups=BAND_GROUPS):
        super().__init__()
        self.groups = list(groups.values())
        self.k = out_per_band
        self.heads = nn.ModuleList(_head(cin, hid, out_per_band * len(idx)) for idx in self.groups)

    def forward(self, x):
        b, _, h, w = x.shape
        out = x.new_zeros(b, self.k, N_S2, h, w)
        for idx, head in zip(self.groups, self.heads):
            out[:, :, idx] = head(x).view(b, self.k, len(idx), h, w)
        return out.view(b, self.k * N_S2, h, w)


class SharedHead(nn.Module):
    """All-band head with roughly the parameter count of GroupHeads (for the C1 ablation)."""

    def __init__(self, cin, hid, out_per_band=2, n_groups=len(BAND_GROUPS)):
        super().__init__()
        self.head = _head(cin, int(round(hid * math.sqrt(n_groups) / 8)) * 8, out_per_band * N_S2)

    def forward(self, x):
        return self.head(x)
