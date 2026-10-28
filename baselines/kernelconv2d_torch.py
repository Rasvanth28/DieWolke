"""Pure-PyTorch stand-in for GLF-CR's compiled KernelConv2D CUDA layer (inference and training).

The original layer (from STFAN, compiled with `setup.py` against PyTorch 1.4) computes a per-pixel
dynamic convolution: for every pixel, a k x k filter predicted by the network is applied to each
channel of the replication-padded input:
    out[b, c, y, x] = sum_{i, j} in_pad[b, c, y + i, x + j] * kernel[b, c*k*k + i*k + j, y, x]
`F.unfold` produces exactly this (channel-major, then row, then column) patch layout, so no
compilation is needed. `install()` registers it under the module names GLF-CR imports.
"""
import sys
import types

import torch.nn as nn
import torch.nn.functional as F


class KernelConv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        assert kernel_size % 2 == 1
        self.kernel_size = kernel_size
        p = (kernel_size - 1) // 2
        self.pad = nn.ReplicationPad2d([p, p, p, p])

    def forward(self, input, kernel):
        b, c, h, w = input.shape
        k = self.kernel_size
        patches = F.unfold(self.pad(input), k)  # (B, C*k*k, H*W)
        out = (patches * kernel.reshape(b, c * k * k, h * w)).view(b, c, k * k, h * w).sum(2)
        return out.view(b, c, h, w)


def install():
    """Make `from FAC.kernelconv2d import KernelConv2D` resolve to this module (call before importing GLF-CR)."""
    inner = types.ModuleType("FAC.kernelconv2d.KernelConv2D")
    inner.KernelConv2D = KernelConv2D
    pkg = types.ModuleType("FAC.kernelconv2d")
    pkg.KernelConv2D = inner
    fac = sys.modules.get("FAC") or types.ModuleType("FAC")
    fac.kernelconv2d = pkg
    sys.modules.update({"FAC": fac, "FAC.kernelconv2d": pkg, "FAC.kernelconv2d.KernelConv2D": inner})
