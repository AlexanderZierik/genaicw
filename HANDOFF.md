# Part 1 handoff context

This document captures the completed work and the remaining manual steps for
another agent or collaborator. It is intentionally specific enough to resume
from the current working tree without reading the preceding chat.

## Goal and scope

This repository is for the Generative AI coursework hot-dog challenge. Work
completed here covers **Part 1 only**: a VAE implemented from scratch, trained
on MNIST and hot-dog images, with a sampler that meets the submission contract.

The coursework requires a small VAE with fewer than one million parameters.
It should work on MNIST but produce blurry/blob-like hot-dog prior samples; the
latter is expected and is analysed in the Part 1 model card. No pretrained
component, diffusion model, or flow-matching model is used for Part 1.

## What was implemented

- `vae_model.py`: the reusable 28x28 convolutional VAE, reparameterisation
  trick, negative ELBO, and parameter counter.
- `train_vae.py`: reproducible MNIST/hot-dog training, KL annealing, figures,
  JSON summaries, and an optional GPU tensor cache to avoid repeated JPEG
  decoding from the WSL filesystem.
- `vae_skeleton.py`: compatibility wrapper around `train_vae.py`.
- `generate.py`: Part 1 entry point. It loads `vae_hotdog.pt`, samples normal
  latent vectors, bilinearly upsamples 28x28 output to 112x112, and writes RGB
  PNG files using `--out` and `--n`.
- `courseutils.py`: changed to load the supplied helper archive without the
  symlink that Windows blocks when the repo is accessed through a WSL UNC path.
- `MODEL_CARD_VAE.md`: populated with implementation, training, figures,
  results, analysis, and the required AI-assistance disclosure.

## Trained artifacts and results

Training was done locally with native Windows Python 3.10, PyTorch 2.11.0 with
CUDA 12.8, and an NVIDIA GeForce RTX 4070 SUPER (12 GB).

| Stage | Configuration | Parameters | Result |
| --- | --- | ---: | --- |
| MNIST | 1 channel, latent 32, sigmoid + BCE, 30 epochs, batch 256 | 880,481 | Recognisable prior digits; 2.3-minute recorded loop time |
| Hot dogs | 3 channels, latent 10, tanh + MSE, 60 epochs, batch 1024 | 468,087 | Blurry but varied brown/beige prior samples; 16.8-second recorded optimization loop after one-time GPU tensor cache |

The first hot-dog attempt used JPEG loading each epoch and was stopped at epoch
35/60 because it was CPU/input-bound. The completed hot-dog run is the second,
GPU-cached 60-epoch run; only its checkpoint and summary are relevant.

`selfeval.py out` on 64 generated images reported:

```text
mean P(hot dog): 0.0054
best P(hot dog): 0.0093
duplicate frac:  0.0000
metric:          0.0054
```

This is the expected Part 1 near-noise-floor result. Visual inspection confirms
that MNIST prior samples are recognisable digits, hot-dog reconstructions retain
coarse structure, and hot-dog prior samples are blobs. The model card explains
why this is an expected limitation of the small Gaussian VAE.

## Local validation already completed

- CUDA availability and the RTX 4070 SUPER were verified.
- Both model variants are under the one-million-parameter cap.
- One-batch forward/loss checks passed for MNIST and hot dogs.
- `python generate.py --out out --n 64` produced 64 RGB 112x112 PNG files.
- Every image referenced by `MODEL_CARD_VAE.md` exists under `figures/`.
- `python -m py_compile vae_model.py train_vae.py vae_skeleton.py generate.py`
  passed.

## Important files to retain

For a complete Part 1 handoff, keep the source files, `vae_hotdog.pt`,
`vae_mnist.pt`, both JSON summaries, `MODEL_CARD_VAE.md`, and all
`figures/vae_*.png` files. `vae_hotdog.pt` is required by `generate.py` at
evaluation time. The MNIST weight and summaries are not needed by the sampler,
but substantiate the Part 1 report.

Do **not** add `_data/`, `_deps/`, or `out/`: they are local downloaded data,
helper cache, and test generations respectively, and are ignored by Git.

## Work still left to do

No commit, push, LabTS submission, automated test, or leaderboard action has
been performed by the agent.

To complete the actual coursework submission manually:

1. Review the changed files and generated images.
2. Commit/push the Part 1 source, weights, card, summaries, and figures to the
   course-issued GitLab repository.
3. Mark that commit in LabTS and run its automated test.
4. Once it has a leaderboard row, save a readable screenshot of the row,
   including commit hash and score, as exactly `figures/leaderboard.png`.
5. Commit that screenshot and select the intended final commit in LabTS.

The screenshot is intentionally absent now because it must be obtained after a
real submission; do not create a placeholder or fabricate it.

## Environment commands

The development virtual environment is local only and was created at
`C:\Users\alexz\.venvs\hotdog_vae`. From PowerShell in the repo:

```powershell
& 'C:\Users\alexz\.venvs\hotdog_vae\Scripts\python.exe' train_vae.py --stage all
& 'C:\Users\alexz\.venvs\hotdog_vae\Scripts\python.exe' generate.py --out out --n 64
& 'C:\Users\alexz\.venvs\hotdog_vae\Scripts\python.exe' selfeval.py out
```

The current `requirements.txt` includes `scikit-learn` for the MNIST t-SNE.
PyTorch and CUDA are intentionally not pinned there because the coursework
runner provides them.
