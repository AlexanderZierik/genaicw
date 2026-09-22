#!/usr/bin/env python3
"""Part 1 skeleton: your own variational autoencoder.

Fill in the parts marked TODO. Nothing here is pretrained -- that is the point of
Part 1. Expect blurry, blob-like samples; the specification explains why, and an
honest account of it earns more marks than trying to make it sharp.

    python vae_skeleton.py --epochs 60

Trains, saves `vae.pt`, and writes a contact sheet to `vae_samples.png` for your
model card. Peak memory for the staff reference is 1.35 GB, so this runs on a T4.
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from courseutils import get_dataset

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class VAE(nn.Module):
    """A convolutional VAE for 112x112 RGB images.

    The encoder must output TWO things -- a mean and a log-variance -- because
    it parameterises a distribution q(z|x), not a single code.
    """

    def __init__(self, latent_dim=128, base=64):
        super().__init__()
        self.latent_dim = latent_dim

        # 112 -> 56 -> 28 -> 14 -> 7
        self.encoder = nn.Sequential(
            nn.Conv2d(3, base, 4, 2, 1), nn.BatchNorm2d(base), nn.LeakyReLU(0.2),
            nn.Conv2d(base, base * 2, 4, 2, 1), nn.BatchNorm2d(base * 2), nn.LeakyReLU(0.2),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1), nn.BatchNorm2d(base * 4), nn.LeakyReLU(0.2),
            nn.Conv2d(base * 4, base * 8, 4, 2, 1), nn.BatchNorm2d(base * 8), nn.LeakyReLU(0.2),
        )
        self.flat_dim = base * 8 * 7 * 7

        # TODO (1): two linear heads mapping the flattened encoder output to
        #           mu and logvar, each of size latent_dim.
        self.fc_mu = None
        self.fc_logvar = None

        self.fc_dec = nn.Linear(latent_dim, self.flat_dim)
        self.base = base
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base * 8, base * 4, 4, 2, 1), nn.BatchNorm2d(base * 4), nn.ReLU(),
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1), nn.BatchNorm2d(base * 2), nn.ReLU(),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1), nn.BatchNorm2d(base), nn.ReLU(),
            nn.ConvTranspose2d(base, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(self, mu, logvar):
        """TODO (2): return a sample from N(mu, exp(logvar)).

        Sampling from the distribution directly would break the gradient. Write
        it as mu + sigma * eps with eps ~ N(0, I) so that mu and logvar stay
        differentiable -- this is the reparameterisation trick.
        """
        raise NotImplementedError

    def decode(self, z):
        h = self.fc_dec(z).view(-1, self.base * 8, 7, 7)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon, x, mu, logvar, beta):
    """TODO (3): the ELBO, as reconstruction + beta * KL.

    Reconstruction: squared error between recon and x, summed over pixels and
    averaged over the batch.

    KL: the closed form for a diagonal Gaussian against N(0, I),
        -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    again averaged over the batch.

    Return (total, reconstruction, kl) so you can plot the terms separately --
    watching them individually is how you diagnose posterior collapse.
    """
    raise NotImplementedError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--latent-dim", type=int, default=128)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--anneal", action="store_true", default=True,
                    help="ramp beta from 0 (strongly recommended)")
    ap.add_argument("--out", default="vae.pt")
    args = ap.parse_args()

    tf = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor()])
    ds = get_dataset("./_data", transform=tf)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)
    print("dataset: %d images on %s" % (len(ds), DEV))

    model = VAE(args.latent_dim).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for ep in range(args.epochs):
        # KL annealing: with the full KL term from step 0 the posterior collapses
        # to the prior, the decoder ignores z, and every sample looks the same.
        beta = args.beta * min(1.0, (ep + 1) / max(1, args.epochs // 3)) if args.anneal else args.beta
        model.train()
        tot = n = 0
        for batch in dl:
            x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(DEV)
            recon, mu, logvar = model(x)
            loss, rec, kl = vae_loss(recon, x, mu, logvar, beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item(); n += 1
        print("epoch %d/%d loss=%.1f beta=%.2f" % (ep, args.epochs, tot / n, beta), flush=True)

    torch.save({"model": model.state_dict(), "latent_dim": args.latent_dim}, args.out)
    print("saved %s" % args.out)

    # A contact sheet for your model card. Look at it.
    model.eval()
    with torch.no_grad():
        samples = model.decode(torch.randn(25, args.latent_dim, device=DEV)).cpu()
    save_image(samples, "vae_samples.png", nrow=5)
    print("wrote vae_samples.png -- put this in MODEL_CARD_VAE.md")


if __name__ == "__main__":
    main()
