#!/usr/bin/env python3
"""Train and document the two Part 1 VAE stages.

``python train_vae.py --stage all`` trains MNIST then hot dogs and writes a
checkpoint, metrics, and the figures required by the Part 1 model card.
"""
import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from courseutils import get_dataset
from vae_model import ConvVAE, parameter_count, vae_loss


CONFIGS = {
    "mnist": {"channels": 1, "latent_dim": 32, "output": "sigmoid", "likelihood": "bernoulli", "epochs": 30, "checkpoint": "vae_mnist.pt"},
    "hotdog": {"channels": 3, "latent_dim": 10, "output": "tanh", "likelihood": "gaussian", "epochs": 60, "checkpoint": "vae_hotdog.pt"},
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def unwrap(batch):
    return batch[0] if isinstance(batch, (tuple, list)) else batch


def data_for(stage, data_root):
    if stage == "mnist":
        transform = transforms.ToTensor()
        return (
            datasets.MNIST(data_root, train=True, download=True, transform=transform),
            datasets.MNIST(data_root, train=False, download=True, transform=transform),
        )
    transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Lambda(lambda image: image * 2 - 1)])
    return get_dataset(data_root, transform=transform, split="train"), get_dataset(data_root, transform=transform, split="test")


def cache_images_on_gpu(dataset, device):
    """Decode/resize once, then keep the small 28x28 training set on GPU.

    The complete hot-dog set is only about 85 MB as float32 tensors, and MNIST
    about 188 MB. This avoids repeatedly decoding thousands of JPEGs over the
    WSL network filesystem every epoch.
    """
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    images = torch.cat([unwrap(batch) for batch in loader])
    return images.to(device, non_blocking=True)


def save_curves(history, destination, stage):
    epochs = [entry["epoch"] for entry in history]
    plt.figure(figsize=(8, 4.5))
    for key, label in (("total", "total ELBO"), ("reconstruction", "reconstruction"), ("kl", "KL")):
        plt.plot(epochs, [entry[key] for entry in history], label=label)
    plt.xlabel("epoch")
    plt.ylabel("loss per image")
    plt.title(f"{stage.upper()} VAE training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination, dpi=160)
    plt.close()


def save_pixel_histogram(images, destination, stage):
    """Record the observed pixel distribution used to justify the likelihood."""
    values = images.detach().flatten().float().cpu().numpy()
    if values.size > 1_000_000:
        values = values[:: values.size // 1_000_000]
    plt.figure(figsize=(7, 4))
    plt.hist(values, bins=64, density=True, color="#286090")
    plt.xlabel("normalised pixel value")
    plt.ylabel("density")
    plt.title(f"{stage.upper()} training-pixel distribution")
    plt.tight_layout()
    plt.savefig(destination, dpi=160)
    plt.close()


@torch.no_grad()
def save_figures(model, loader, config, stage, figures, device):
    model.eval()
    batch = unwrap(next(iter(loader))).to(device)[:16]
    reconstruction, _, _ = model(batch)
    shown_input = (batch + 1) / 2 if config["output"] == "tanh" else batch
    shown_recon = (reconstruction + 1) / 2 if config["output"] == "tanh" else reconstruction
    save_image(torch.cat((shown_input.cpu(), shown_recon.cpu())), figures / f"vae_{stage}_reconstructions.png", nrow=16)
    samples = model.decode(torch.randn(64, config["latent_dim"], device=device))
    shown_samples = (samples + 1) / 2 if config["output"] == "tanh" else samples
    save_image(shown_samples.cpu(), figures / f"vae_{stage}_samples.png", nrow=8)
    mu, _ = model.encode(batch[:2])
    alphas = torch.linspace(0, 1, 12, device=device).unsqueeze(1)
    interpolation = model.decode(mu[0].unsqueeze(0) * (1 - alphas) + mu[1].unsqueeze(0) * alphas)
    shown_interpolation = (interpolation + 1) / 2 if config["output"] == "tanh" else interpolation
    save_image(shown_interpolation.cpu(), figures / f"vae_{stage}_interpolation.png", nrow=12)


@torch.no_grad()
def save_mnist_tsne(model, dataset, figures, device):
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    latents, labels = [], []
    for images, target in loader:
        mu, _ = model.encode(images.to(device))
        latents.append(mu.cpu())
        labels.append(target)
        if sum(part.size(0) for part in latents) >= 2_000:
            break
    z = torch.cat(latents)[:2_000].numpy()
    y = torch.cat(labels)[:2_000].numpy()
    coords = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=0).fit_transform(z)
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=y, s=5, cmap="tab10")
    plt.colorbar(scatter, ticks=range(10), label="digit class")
    plt.title("MNIST VAE latent means (t-SNE)")
    plt.tight_layout()
    plt.savefig(figures / "vae_mnist_tsne.png", dpi=180)
    plt.close()


def train_stage(args, stage, device):
    config = CONFIGS[stage].copy()
    config["epochs"] = args.mnist_epochs if stage == "mnist" else args.hotdog_epochs
    train_set, evaluation_set = data_for(stage, args.data_root)
    loader = None
    cached_images = None
    if args.cache_gpu:
        print(f"{stage}: decoding and caching 28x28 training tensors on GPU", flush=True)
        cached_images = cache_images_on_gpu(train_set, device)
    else:
        loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=device == "cuda")
    model = ConvVAE(config["channels"], config["latent_dim"], config["output"]).to(device)
    count = parameter_count(model)
    if count >= 1_000_000:
        raise RuntimeError(f"VAE has {count:,} parameters; coursework limit is <1,000,000")
    print(f"{stage}: {len(train_set)} images, {count:,} parameters, device={device}", flush=True)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        beta = min(1.0, epoch / max(1, config["epochs"] // 3))
        model.train()
        totals = {"total": 0.0, "reconstruction": 0.0, "kl": 0.0, "images": 0}
        if cached_images is not None:
            batches = (cached_images[index] for index in torch.randperm(len(cached_images), device=device).split(args.batch))
        else:
            batches = (unwrap(raw_batch).to(device, non_blocking=True) for raw_batch in loader)
        for images in batches:
            reconstruction, mu, logvar = model(images)
            total, reconstruction_loss, kl = vae_loss(reconstruction, images, mu, logvar, beta, config["likelihood"])
            optimiser.zero_grad(set_to_none=True)
            total.backward()
            optimiser.step()
            count_in_batch = images.size(0)
            totals["total"] += total.item() * count_in_batch
            totals["reconstruction"] += reconstruction_loss.item() * count_in_batch
            totals["kl"] += kl.item() * count_in_batch
            totals["images"] += count_in_batch
        record = {key: totals[key] / totals["images"] for key in ("total", "reconstruction", "kl")}
        record.update(epoch=epoch, beta=beta)
        history.append(record)
        print(f"{stage} epoch {epoch:02d}/{config['epochs']}: total={record['total']:.3f} recon={record['reconstruction']:.3f} kl={record['kl']:.3f} beta={beta:.2f}", flush=True)
    elapsed = time.perf_counter() - started
    figures = Path(args.figures)
    figures.mkdir(parents=True, exist_ok=True)
    save_curves(history, figures / f"vae_{stage}_curves.png", stage)
    if cached_images is not None:
        save_pixel_histogram(cached_images, figures / f"vae_{stage}_pixel_histogram.png", stage)
    evaluation_loader = DataLoader(evaluation_set, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    save_figures(model, evaluation_loader, config, stage, figures, device)
    if stage == "mnist":
        save_mnist_tsne(model, evaluation_set, figures, device)
    payload = {"model": model.state_dict(), "channels": config["channels"], "latent_dim": config["latent_dim"], "output": config["output"], "likelihood": config["likelihood"], "parameters": count, "epochs": config["epochs"], "batch_size": args.batch, "lr": args.lr, "seed": args.seed}
    torch.save(payload, config["checkpoint"])
    summary = {**payload, "model": None, "wall_clock_seconds": elapsed, "device": torch.cuda.get_device_name(0) if device == "cuda" else "CPU", "history": history}
    Path(f"vae_{stage}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved {config['checkpoint']}; stage runtime {elapsed / 60:.1f} minutes", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("mnist", "hotdog", "all"), default="all")
    parser.add_argument("--mnist-epochs", type=int, default=30)
    parser.add_argument("--hotdog-epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=60038)
    parser.add_argument("--workers", type=int, default=0, help="Windows-safe DataLoader worker count")
    parser.add_argument("--cache-gpu", action=argparse.BooleanOptionalAction, default=True,
                        help="cache decoded 28x28 training tensors in GPU memory")
    parser.add_argument("--data-root", default="_data")
    parser.add_argument("--figures", default="figures")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA is required for this coursework training run")
    for stage in (("mnist", "hotdog") if args.stage == "all" else (args.stage,)):
        train_stage(args, stage, device)


if __name__ == "__main__":
    main()
