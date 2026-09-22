"""Velocity-field UNet for rectified-flow generation in the SD-2 VAE latent space.

Shared by `train_flow_matching.py` (fits v_theta) and `generate.py` (samples). The
network is deliberately small (4x14x14 latents) so it trains within the T4 budget.
`base` (channel width) is the main capacity knob.
"""
import math

import torch
import torch.nn as nn


class SinEmb(nn.Module):
    """Sinusoidal timestep embedding."""

    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, t):
        half = self.d // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t[:, None] * f[None]
        return torch.cat([a.sin(), a.cos()], -1)


class Block(nn.Module):
    """Two conv layers + GroupNorm/SiLU with a timestep-conditioned bias and a residual."""

    def __init__(self, ci, co, td):
        super().__init__()
        self.c1 = nn.Conv2d(ci, co, 3, padding=1)
        self.c2 = nn.Conv2d(co, co, 3, padding=1)
        self.n1 = nn.GroupNorm(8, co)
        self.n2 = nn.GroupNorm(8, co)
        self.tm = nn.Linear(td, co)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()

    def forward(self, x, t):
        h = self.act(self.n1(self.c1(x)))
        h = h + self.tm(t)[..., None, None]
        h = self.act(self.n2(self.c2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """Predicts the velocity v_theta(z, t) for a [B,4,14,14] latent and scalar time t."""

    def __init__(self, ch=4, base=128, td=256):
        super().__init__()
        self.temb = nn.Sequential(SinEmb(td), nn.Linear(td, td), nn.SiLU(), nn.Linear(td, td))
        self.inp = nn.Conv2d(ch, base, 3, padding=1)
        self.d1 = Block(base, base * 2, td)
        self.d2 = Block(base * 2, base * 2, td)
        self.mid = Block(base * 2, base * 2, td)
        self.u1 = Block(base * 2 + base * 2, base * 2, td)
        self.u2 = Block(base * 2 + base, base, td)
        self.out = nn.Conv2d(base, ch, 1)

    def forward(self, x, t):
        te = self.temb(t)
        x0 = self.inp(x)
        h1 = self.d1(x0, te)
        h2 = self.d2(h1, te)
        m = self.mid(h2, te)
        u = self.u1(torch.cat([m, h2], 1), te)
        u = self.u2(torch.cat([u, x0], 1), te)
        return self.out(u)
