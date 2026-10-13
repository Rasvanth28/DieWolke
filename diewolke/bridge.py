"""Brownian-bridge diffusion between the clear image x0 (t = 0) and the cloudy image y (t = 1).

Forward marginal:  x_t = (1 - t) x0 + t y + sigma * sqrt(t (1 - t)) * eps
At t = 1 the state is exactly y, so sampling starts from the cloudy input with no noise.
Reverse step from t to s < t given a prediction x0_hat (the bridge from x0_hat at time 0 to x_t):
    mean = x0_hat + (s / t) (x_t - x0_hat),   var = sigma^2 * s (t - s) / t
With one step (t = 1 -> 0) the sampler returns the network's x0 prediction at t = 1, which is the
single-step inference regime DB-CR reports; more steps refine it.
"""
import torch


class BrownianBridge:
    def __init__(self, sigma=0.5, t_min=1e-3, p_t1=0.25):
        self.sigma, self.t_min, self.p_t1 = sigma, t_min, p_t1

    def sample_t(self, n, device):
        """Training times: uniform on [t_min, 1], with probability p_t1 exactly 1 (single-step regime)."""
        t = torch.rand(n, device=device) * (1 - self.t_min) + self.t_min
        return torch.where(torch.rand(n, device=device) < self.p_t1, torch.ones_like(t), t)

    def q_sample(self, x0, y, t, noise=None):
        tt = t.view(-1, 1, 1, 1)
        noise = torch.randn_like(x0) if noise is None else noise
        return (1 - tt) * x0 + tt * y + self.sigma * (tt * (1 - tt)).clamp(min=0).sqrt() * noise

    def step(self, x_t, x0_hat, t, s, eta=1.0, noise=None):
        tt, ss = t.view(-1, 1, 1, 1), s.view(-1, 1, 1, 1)
        mean = x0_hat + (ss / tt) * (x_t - x0_hat)
        std = eta * self.sigma * (ss * (tt - ss) / tt).clamp(min=0).sqrt()
        if noise is None:
            noise = torch.randn_like(x_t)
        return mean + std * noise

    @torch.no_grad()
    def sample(self, model_fn, y, steps=1, eta=0.0, generator=None):
        """Run the reverse bridge from y. model_fn(x_t, t) -> {"mean", "log_scale"}.

        Returns the final model output dict (its "mean" is the reconstruction).
        """
        n = y.shape[0]
        times = torch.linspace(1.0, 0.0, steps + 1, device=y.device)
        x = y
        out = None
        for k in range(steps):
            t = times[k].expand(n)
            out = model_fn(x, t)
            if k < steps - 1:
                s = times[k + 1].expand(n)
                noise = torch.randn(x.shape, device=x.device, generator=generator) if eta > 0 else torch.zeros_like(x)
                x = self.step(x, out["mean"], t, s, eta, noise)
        return out
