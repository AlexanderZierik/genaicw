#!/usr/bin/env python3
"""Part 1 sampler: write N 112x112 RGB hot-dog PNGs from the trained VAE."""
import argparse
import os

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from vae_model import ConvVAE


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--ckpt", default="vae_hotdog.pt")
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()
    if args.n < 1 or args.batch < 1:
        raise ValueError("--n and --batch must be positive")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    if checkpoint["channels"] != 3 or checkpoint["output"] != "tanh":
        raise ValueError("checkpoint is not the Part 1 RGB hot-dog VAE")
    model = ConvVAE(checkpoint["channels"], checkpoint["latent_dim"], checkpoint["output"]).to(device).eval()
    model.load_state_dict(checkpoint["model"])

    os.makedirs(args.out, exist_ok=True)
    written = 0
    while written < args.n:
        count = min(args.batch, args.n - written)
        z = torch.randn(count, checkpoint["latent_dim"], device=device)
        images_28 = model.decode(z)
        images_112 = F.interpolate(images_28, size=(112, 112), mode="bilinear", align_corners=False)
        images_01 = ((images_112 + 1) / 2).clamp(0, 1).cpu()
        for index in range(count):
            save_image(images_01[index], os.path.join(args.out, f"hotdog_{written + index:05d}.png"))
        written += count
        print(f"generated {written}/{args.n}", flush=True)


if __name__ == "__main__":
    main()
