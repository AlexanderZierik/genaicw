#!/usr/bin/env python3
"""Self-evaluate a folder of generated images with the PUBLIC hot-dog classifier.

This is the exact detector used in the practice phase, so it lets you calibrate before
submitting. The FINAL phase uses a secret detector + a diversity guard, so a high score
here is necessary but not sufficient -- keep your samples diverse too.

Usage:  python selfeval.py <folder>
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from courseutils import get_classifier

HOTDOG_CLASS = 55


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "out"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    clf = get_classifier(dev)

    to_tensor = transforms.ToTensor()
    paths = sorted(p for p in Path(folder).iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not paths:
        print(f"no images in {folder}")
        return

    probs, embeds = [], []
    for i in range(0, len(paths), 64):
        imgs = [transforms.functional.resize(to_tensor(Image.open(p).convert("RGB")),
                                             [224, 224], antialias=True) for p in paths[i:i + 64]]
        batch = torch.stack(imgs).to(dev)
        _, p, _ = clf.predict(batch)
        probs.append(p[:, HOTDOG_CLASS].cpu())
        embeds.append(F.interpolate(batch, (16, 16)).reshape(batch.size(0), -1).cpu())

    p = torch.cat(probs)
    e = F.normalize(torch.cat(embeds), dim=1)
    sim = e @ e.t()
    sim.fill_diagonal_(-1)
    dup = (sim.max(1).values > 0.99).float().mean().item()
    print(f"images:          {len(paths)}")
    print(f"mean P(hot dog): {p.mean():.4f}")
    print(f"best P(hot dog): {p.max():.4f}")
    print(f"duplicate frac:  {dup:.4f}  -> diversity factor {1 - dup:.4f}")
    print(f"metric (mean x diversity): {p.mean().item() * (1 - dup):.4f}")


if __name__ == "__main__":
    main()
