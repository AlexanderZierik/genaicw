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
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from courseutils import get_dataset
from models import UNet
from vae import VAE_ID, encode, load_vae


def set_seed(seed):
    """Seed every random source used by the trainer."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ot_pair(z0, z1):
    """Match noise to data by minibatch optimal transport."""
    from scipy.optimize import linear_sum_assignment

    with torch.no_grad():
        cost = torch.cdist(z0.flatten(1), z1.flatten(1)).cpu().numpy()
        _, columns = linear_sum_assignment(cost)
    return z1[torch.as_tensor(columns, device=z1.device)]


def cache_latents(args, device):
    cache_path = Path(args.latent_cache)
    if cache_path.exists() and not args.refresh_latents:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        latents = payload.get("latents") if isinstance(payload, dict) else None
        if (
            isinstance(latents, torch.Tensor)
            and payload.get("vae_id") == VAE_ID
            and tuple(latents.shape[1:]) == (4, 14, 14)
        ):
            print(f"loaded {len(latents)} cached latents from {cache_path}", flush=True)
            return latents
        print(f"ignoring incompatible latent cache at {cache_path}", flush=True)

    vae = load_vae(device)
    transform = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor()])
    dataset = get_dataset(args.data_root, transform=transform, split="train")
    loader = DataLoader(
        dataset,
        batch_size=args.encode_batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device == "cuda",
    )
    encoded = []
    print(f"encoding {len(dataset)} images to frozen-VAE latents...", flush=True)
    for batch in loader:
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        encoded.append(encode(vae, images.to(device, non_blocking=True)).cpu())
    latents = torch.cat(encoded)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"latents": latents, "vae_id": VAE_ID, "image_size": [112, 112]}, cache_path)
    print(f"saved {len(latents)} cached latents to {cache_path}", flush=True)
    return latents


def update_ema(ema_model, model, decay):
    with torch.no_grad():
        for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
            ema_parameter.mul_(decay).add_(parameter, alpha=1 - decay)
        for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


def save_loss_figure(history, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot([row["step"] for row in history], [row["loss"] for row in history], alpha=0.35, label="batch MSE")
    axis.plot([row["step"] for row in history], [row["loss_ema"] for row in history], linewidth=2, label="EMA loss")
    axis.set(xlabel="training step", ylabel="velocity MSE", title="Rectified-flow training")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Train a latent rectified-flow hot-dog generator")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--encode-batch", type=int, default=128)
    parser.add_argument("--base", type=int, default=128, help="U-Net channel width")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ema", type=float, default=0.999)
    parser.add_argument("--ot", action="store_true", help="use minibatch optimal-transport pairing")
    parser.add_argument("--seed", type=int, default=60038)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--data-root", default="./_data")
    parser.add_argument("--latent-cache", default="./_data/sd2_latents_112.pt")
    parser.add_argument("--refresh-latents", action="store_true")
    parser.add_argument("--out", default="checkpoint.pt")
    parser.add_argument("--summary", help="JSON run summary; defaults beside --out")
    parser.add_argument("--loss-figure", help="PNG loss curve; defaults beside --out")
    args = parser.parse_args()

    if args.steps < 1 or args.batch < 1 or args.encode_batch < 1 or args.log_every < 1:
        raise ValueError("step counts, batch sizes, and --log-every must be positive")
    if not 0 <= args.ema < 1:
        raise ValueError("--ema must be in [0, 1)")

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"training on {device}: steps={args.steps}, batch={args.batch}, "
        f"base={args.base}, ot={args.ot}, seed={args.seed}",
        flush=True,
    )
    latents = cache_latents(args, device)

    model = UNet(base=args.base).to(device)
    ema_model = copy.deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    started = time.perf_counter()
    loss_ema = None
    history = []
    for step in range(1, args.steps + 1):
        indices = torch.randint(0, len(latents), (args.batch,))
        z0 = latents[indices].to(device, non_blocking=True)
        z1 = torch.randn_like(z0)
        if args.ot:
            z1 = ot_pair(z0, z1)
        t = torch.rand(args.batch, device=device)
        zt = (1 - t[:, None, None, None]) * z0 + t[:, None, None, None] * z1
        loss = F.mse_loss(model(zt, t), z1 - z0)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        update_ema(ema_model, model, args.ema)

        value = loss.item()
        loss_ema = value if loss_ema is None else 0.99 * loss_ema + 0.01 * value
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            history.append({"step": step, "loss": value, "loss_ema": loss_ema})
            print(f"step {step}/{args.steps} loss={value:.5f} loss_ema={loss_ema:.5f}", flush=True)

    elapsed_seconds = time.perf_counter() - started
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "ema_state_dict": ema_model.state_dict(),
        "base": args.base,
        "method": "rectified_flow",
        "vae_id": VAE_ID,
        "latent_shape": [4, 14, 14],
        "training": {
            "steps": args.steps,
            "batch": args.batch,
            "lr": args.lr,
            "ema": args.ema,
            "ot": args.ot,
            "seed": args.seed,
        },
    }
    torch.save(checkpoint, output_path)

    summary_path = Path(args.summary) if args.summary else output_path.with_suffix(".summary.json")
    figure_path = Path(args.loss_figure) if args.loss_figure else output_path.with_suffix(".loss.png")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": "rectified_flow",
        "vae_id": VAE_ID,
        "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "parameter_count": parameter_count,
        "latent_count": len(latents),
        "elapsed_seconds": elapsed_seconds,
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else None,
        "config": checkpoint["training"],
        "history": history,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    save_loss_figure(history, figure_path)
    print(
        f"saved checkpoint={output_path}, summary={summary_path}, loss_figure={figure_path} "
        f"(training {elapsed_seconds / 60:.1f} min)",
        flush=True,
    )


if __name__ == "__main__":
    main()
