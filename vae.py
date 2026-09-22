"""Frozen SD-2 VAE encode/decode helpers.

The VAE is pretrained and NOT trained by you: it maps a 112x112 RGB image in [0,1]
to a 4x14x14 latent and back. Your generator learns to produce these latents; the
decoder turns them into pictures.
"""
import torch

# NB: the pytorch/pytorch base image ships a conda `safetensors` that can shadow the
# pip one (breaks `safetensors.torch`). If VAE loading fails with a safetensors error,
# run: pip install --force-reinstall --no-cache-dir safetensors==0.4.3
from diffusers import AutoencoderKL

VAE_ID = "Manojb/stable-diffusion-2-base"


def load_vae(device):
    try:
        vae = AutoencoderKL.from_pretrained(VAE_ID, subfolder="vae")
    except Exception as e:  # safetensors env trouble -> fall back to .bin weights
        print(f"VAE safetensors load failed ({e}); retrying with use_safetensors=False")
        vae = AutoencoderKL.from_pretrained(VAE_ID, subfolder="vae", use_safetensors=False)
    vae = vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


@torch.no_grad()
def encode(vae, x01):
    """x01: [B,3,112,112] in [0,1] -> latent [B,4,14,14] (scaled)."""
    return vae.encode(x01 * 2 - 1).latent_dist.sample() * vae.config.scaling_factor


@torch.no_grad()
def decode(vae, z):
    """latent -> image [B,3,112,112] in [0,1]."""
    x = vae.decode(z / vae.config.scaling_factor, return_dict=False)[0]
    return ((x + 1) * 0.5).clamp(0, 1)
