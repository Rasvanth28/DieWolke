"""PyTorch datasets over a Cache (see sen12mscr.py)."""
import numpy as np
import torch

from ..bands import S2_MAX
from .clouds import CLOUD_PROB, donor_indices, real_mask, synthesize


class CloudPairs(torch.utils.data.Dataset):
    """Yields dicts of float32 tensors: s1 (2,H,W), cloudy (13,H,W), clear (13,H,W), mask (1,H,W).

    mode:
      "real"       SEN12MS-CR cloudy/clear pairs. The clear target is another date, and the mask
                   is s2cloudless (+ shadow heuristic), so it is an estimate. This is the
                   standard benchmark protocol.
      "synthetic"  real clouds (s2cloudless maps of other cloudy patches) blended onto the clear
                   patch, so ground truth and mask are exact.
      "mixed"      synthetic with probability p_synth, otherwise real.
    max_target_cloud: drop samples whose "clear" target has more than this cloud fraction.
    """

    def __init__(self, cache, mode="real", crop=None, train=True, p_synth=0.5, seed=0,
                 max_target_cloud=None, use_shadow=True):
        assert mode in ("real", "synthetic", "mixed"), mode
        self.c, self.mode, self.crop, self.train = cache, mode, crop, train
        self.p_synth, self.seed, self.use_shadow = p_synth, seed, use_shadow
        self.donors, self.cloud_frac = donor_indices(cache["prob"])
        self.index = np.arange(len(cache["clear"]))
        if max_target_cloud is not None:
            tgt = np.array([(np.asarray(cache["prob_clear"][i]) > CLOUD_PROB).mean() for i in self.index])
            self.index = self.index[tgt <= max_target_cloud]
        self.ids = getattr(cache, "ids", None)

    def __len__(self):
        return len(self.index)

    def _window(self, rng, size):
        s = self.crop or size
        if s >= size:
            return 0, 0, size
        y, x = rng.integers(0, size - s + 1, size=2)
        return int(y), int(x), s

    def __getitem__(self, k):
        i = int(self.index[k])
        rng = np.random.default_rng(None if self.train else self.seed * 100003 + i)
        size = self.c["clear"].shape[-1]
        y, x, s = self._window(rng, size)
        win = (slice(y, y + s), slice(x, x + s))
        s1 = np.asarray(self.c["s1"][i][(slice(None),) + win], dtype=np.float32)
        clear = np.asarray(self.c["clear"][i][(slice(None),) + win], dtype=np.float32) / S2_MAX
        synthetic = self.mode == "synthetic" or (self.mode == "mixed" and rng.random() < self.p_synth)
        if synthetic:
            choices = self.donors[self.donors != i] if len(self.donors) > 1 else self.donors
            j = int(rng.choice(choices))
            y2, x2, _ = self._window(rng, size)
            win2 = (slice(y2, y2 + s), slice(x2, x2 + s))
            donor = np.asarray(self.c["cloudy"][j][(slice(None),) + win2], dtype=np.float32) / S2_MAX
            prob = np.asarray(self.c["prob"][j][win2], dtype=np.float32)
            cloudy, mask, _ = synthesize(clear, donor, prob, rng)
        else:
            cloudy = np.asarray(self.c["cloudy"][i][(slice(None),) + win], dtype=np.float32) / S2_MAX
            prob = np.asarray(self.c["prob"][i][win], dtype=np.float32)
            if "mask" in self.c:  # exact mask shipped with the benchmark
                mask = np.asarray(self.c["mask"][i][win], dtype=np.float32)
            else:
                shadow = np.asarray(self.c["shadow"][i][win])
                mask = real_mask(prob, shadow, use_shadow=self.use_shadow)
        arrays = {"s1": s1, "cloudy": cloudy, "clear": clear, "mask": mask[None]}
        if self.train:
            rot, flip = rng.integers(4), rng.integers(2)
            for name, a in arrays.items():
                a = np.rot90(a, rot, axes=(-2, -1))
                arrays[name] = np.flip(a, -1) if flip else a
        out = {name: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)) for name, a in arrays.items()}
        out["synthetic"] = torch.tensor(bool(synthetic))
        out["index"] = torch.tensor(i)
        return out


def make_loader(cache, cfg_data, split, batch, workers=2, train=None, shuffle=None, mode=None):
    """DataLoader for one split using the `data` section of a config."""
    train = split == "train" if train is None else train
    ds = CloudPairs(cache, mode=mode or cfg_data.get(f"{split}_mode", cfg_data.get("mode", "real")),
                    crop=cfg_data.get("crop") if train else cfg_data.get("eval_crop"),
                    train=train, p_synth=cfg_data.get("p_synth", 0.5), seed=cfg_data.get("seed", 0),
                    max_target_cloud=cfg_data.get("max_target_cloud") if train else None,
                    use_shadow=cfg_data.get("use_shadow", True))
    return torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=train if shuffle is None else shuffle, drop_last=train,
        num_workers=workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0 and train)
