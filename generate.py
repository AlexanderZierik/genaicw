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
from torchvision.utils import save_image

from models import UNet
from vae import decode, load_vae


@torch.no_grad()
def sample(model, count, steps, device):
    """Integrate the learned velocity field from Gaussian noise to data."""
    z = torch.randn(count, 4, 14, 14, device=device)
    times = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for index in range(steps):
        t = times[index].expand(count)
        next_t = times[index + 1].expand(count)
        dt = times[index] - times[index + 1]
        velocity_start = model(z, t)
        velocity_end = model(z - velocity_start * dt, next_t)
        z = z - 0.5 * (velocity_start + velocity_end) * dt
    return z


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Sample the Part 2 rectified-flow model")
    parser.add_argument("--out", required=True, help="directory for generated PNG files")
    parser.add_argument("--n", type=int, default=256, help="number of images to generate")
    parser.add_argument("--ckpt", default="checkpoint.pt", help="trained rectified-flow checkpoint")
    parser.add_argument("--batch", type=int, default=64, help="images decoded per batch")
    parser.add_argument("--steps", type=int, default=200, help="Heun ODE integration steps")
    parser.add_argument("--seed", type=int, help="optional seed for a reproducible sample set")
    args = parser.parse_args()

    if args.n < 1 or args.batch < 1 or args.steps < 1:
        raise ValueError("--n, --batch, and --steps must all be positive")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    if "ema_state_dict" not in checkpoint or "base" not in checkpoint:
        raise ValueError(f"{args.ckpt} is not a rectified-flow checkpoint")

    model = UNet(base=checkpoint["base"]).to(device).eval()
    model.load_state_dict(checkpoint["ema_state_dict"])
    vae = load_vae(device)

    os.makedirs(args.out, exist_ok=True)
    written = 0
    while written < args.n:
        count = min(args.batch, args.n - written)
        images = decode(vae, sample(model, count, args.steps, device)).cpu()
        for index in range(count):
            save_image(images[index], os.path.join(args.out, f"hotdog_{written + index:05d}.png"))
        written += count
        print(f"generated {written}/{args.n}", flush=True)


if __name__ == "__main__":
    main()
