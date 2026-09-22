#!/usr/bin/env python3
"""Challenge entry point: load the trained model and write N hot-dog PNGs.

CONTRACT (the challenge runner calls exactly this):
    python generate.py --out /output --n 256
It must write `--n` RGB PNG images into `--out`, with no network access at run time.

This reference loads the flow-matching checkpoint from `train_flow_matching.py` and
samples via a 2nd-order (Heun) ODE integration from noise (t=1) to data (t=0), then
decodes with the frozen VAE. Replace the sampler with your own method if you prefer;
only the contract above matters.
"""
import argparse
import os

import torch

from models import UNet
from vae import decode, load_vae


@torch.no_grad()
def sample(model, vae, n, steps, dev):
    z = torch.randn(n, 4, 14, 14, device=dev)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=dev)
    for i in range(steps):                                  # Heun (2nd-order) ODE step
        t = ts[i].expand(n)
        dt = ts[i] - ts[i + 1]
        v1 = model(z, t)
        v2 = model(z - v1 * dt, ts[i + 1].expand(n))
        z = z - 0.5 * (v1 + v2) * dt
    return decode(vae, z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--ckpt", default="checkpoint.pt")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=200, help="ODE integration steps")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location=dev)
    model = UNet(base=ck.get("base", 128)).to(dev).eval()
    model.load_state_dict(ck["ema_state_dict"])
    vae = load_vae(dev)

    from torchvision.utils import save_image
    written = 0
    while written < args.n:
        k = min(args.batch, args.n - written)
        imgs = sample(model, vae, k, args.steps, dev).cpu()
        for i in range(k):
            save_image(imgs[i], os.path.join(args.out, f"hotdog_{written + i:05d}.png"))
        written += k
        print(f"generated {written}/{args.n}", flush=True)


if __name__ == "__main__":
    main()
