"""Stage 1 cloud/shadow masker: a plain U-Net on the 13-band cloudy image."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..bands import N_S2


def conv_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU())


class UNet(nn.Module):
    def __init__(self, cin=N_S2, cout=1, base=32, depth=4):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth)]
        self.downs, self.ups, self.decs = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        prev = cin
        for c in chs:
            self.downs.append(conv_block(prev, c))
            prev = c
        self.mid = conv_block(prev, prev * 2)
        prev *= 2
        for c in reversed(chs):
            self.ups.append(nn.ConvTranspose2d(prev, c, 2, stride=2))
            self.decs.append(conv_block(2 * c, c))
            prev = c
        self.head = nn.Conv2d(prev, cout, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.mid(x)
        for up, dec, skip in zip(self.ups, self.decs, reversed(skips)):
            x = dec(torch.cat([up(x), skip], 1))
        return self.head(x)


class Masker(UNet):
    """Predicts cloud+shadow logits (B, 1, H, W) from the cloudy image (B, 13, H, W)."""

    def __init__(self, base=32, depth=4):
        super().__init__(N_S2, 1, base, depth)

    @torch.no_grad()
    def predict(self, cloudy, threshold=0.5):
        return (torch.sigmoid(self(cloudy)) > threshold).float()
