#!/usr/bin/env python3
"""Reference rectified-flow (conditional flow matching) trainer for hot dogs.

Trains a velocity field v_theta(z,t) in the SD-2 VAE latent space and saves a
checkpoint that `generate.py` loads. Runs within the T4 / H200-slice budget.

Rectified flow:
  z_t = (1-t)*z_data + t*z_noise ,   target velocity = z_noise - z_data
  loss = || v_theta(z_t, t) - (z_noise - z_data) ||^2
Sampling (in generate.py): integrate the ODE from noise (t=1) to data (t=0).

Quality levers (all on by default here): weight EMA, mini-batch optimal-transport
coupling (`--ot`), and a wider network (`--base`). More steps generally helps.

Usage:
  python train_flow_matching.py --steps 20000 --base 128 --ot --out checkpoint.pt
"""
import argparse
import copy
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from courseutils import get_dataset
from models import UNet
from vae import decode, encode, load_vae  # noqa: F401 (decode used by generate.py)


def ot_pair(z0, z1):
    """Reorder the noise batch z1 to minimise total transport cost to z0 (minibatch
    optimal transport) -- straightens the flow paths, sharpening samples."""
    from scipy.optimize import linear_sum_assignment
    with torch.no_grad():
        cost = torch.cdist(z0.reshape(z0.size(0), -1), z1.reshape(z1.size(0), -1)).cpu().numpy()
        _, col = linear_sum_assignment(cost)
    return z1[torch.as_tensor(col, device=z1.device)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--base", type=int, default=128, help="UNet width (capacity)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--ot", action="store_true", help="minibatch OT coupling")
    ap.add_argument("--data-root", default="./_data")
    ap.add_argument("--out", default="checkpoint.pt")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"training on {dev}: steps={args.steps} base={args.base} ot={args.ot}", flush=True)

    # --- data -> precomputed latents (VAE is frozen) ---
    vae = load_vae(dev)
    tf = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor()])
    ds = get_dataset(args.data_root, transform=tf, split="train")
    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=2)
    lats = []
    print("encoding dataset to latents...", flush=True)
    for batch in dl:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        lats.append(encode(vae, x.to(dev)).cpu())
    lats = torch.cat(lats)
    print(f"latents: {tuple(lats.shape)}", flush=True)

    # --- train ---
    model = UNet(base=args.base).to(dev)
    ema_model = copy.deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    N = lats.shape[0]
    model.train()
    loss_ema = None
    for step in range(args.steps):
        idx = torch.randint(0, N, (args.batch,))
        z0 = lats[idx].to(dev)                      # data
        z1 = torch.randn_like(z0)                   # noise
        if args.ot:
            z1 = ot_pair(z0, z1)
        t = torch.rand(args.batch, device=dev)
        zt = (1 - t[:, None, None, None]) * z0 + t[:, None, None, None] * z1
        loss = F.mse_loss(model(zt, t), z1 - z0)    # target velocity = z1 - z0
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema_model.parameters(), model.parameters()):
                pe.mul_(args.ema).add_(pm, alpha=1 - args.ema)
            for be, bm in zip(ema_model.buffers(), model.buffers()):
                be.copy_(bm)
        loss_ema = loss.item() if loss_ema is None else 0.99 * loss_ema + 0.01 * loss.item()
        if step % 1000 == 0:
            print(f"step {step}/{args.steps} loss_ema={loss_ema:.4f}", flush=True)

    torch.save({"ema_state_dict": ema_model.state_dict(), "base": args.base}, args.out)
    print(f"saved {args.out} (final loss_ema={loss_ema:.4f})", flush=True)


if __name__ == "__main__":
    main()
