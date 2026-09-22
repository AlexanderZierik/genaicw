# Generative AI Coursework: starter

**Read `GenAI_Coursework.pdf` in this directory first.** It is the specification;
this README only tells you what the files are and how to get moving.

The coursework is worth 100 marks in three parts:

| | | marks |
|---|---|---:|
| **Part 1** | Build your own VAE from scratch, train it, submit it to the leaderboard | 15 |
| **Part 2** | Use a pretrained encoder and a modern generative model | 60 |
| **Part 3** | Automatic challenge score for your final submission | 25 |

---

## What you have to produce

One GitLab repository on `gitlab.doc.ic.ac.uk` containing, at its root:

| file | what it is |
|---|---|
| `generate.py` | **the contract**: `python generate.py --out DIR --n N` writes N RGB PNGs at 112×112 |
| `requirements.txt` | your dependencies (CUDA and PyTorch are already provided) |
| your weights | committed, or produced by code in the repository |
| `MODEL_CARD_VAE.md` | Part 1: your VAE, its samples, and your leaderboard submission |
| `figures/leaderboard.png` | Part 1: a screenshot of your leaderboard row. This exact path; it is printed on its own page of the marker's report |
| `MODEL_CARD.md` | Part 2: your final model, what you tried, and what it measured |

You submit by **marking a commit on LabTS** and running the automated test. The
commit hash *is* the submission, there is no upload.

Scores appear at **<https://gpudojo.doc.ic.ac.uk/leaderboard/>**, keyed by commit
hash only. Nobody's name or repository is shown, so find your row by the hash you
submitted.

---

## Files here

| file | for | what it does |
|---|---|---|
| `vae_skeleton.py` | Part 1 | scaffolding for your own VAE, with the parts you must write marked `TODO` |
| `courseutils.py` | both | fetches the dataset and the hot-dog classifier |
| `vae.py` | Part 2 | loads a **pretrained** autoencoder and encodes/decodes |
| `models.py` | Part 2 | a time-conditioned UNet you can build a diffusion or flow model on |
| `train_flow_matching.py` | Part 2 | a worked rectified-flow training loop |
| `selfeval.py` | both | scores your own images with the provided classifier |
| `generate.py` | both | the submission entry point — **edit this** |

---

## Getting started

```bash
pip install -r requirements.txt

# Part 1: train your own VAE (see the spec, section 4)
python vae_skeleton.py --epochs 60

# check what you produced, before spending a submission on it
python generate.py --out samples --n 64
python selfeval.py --dir samples
```

`selfeval.py` uses **the same classifier** as one half of the challenge score, so
your own numbers should match that half closely. The other half is
distributional and is not provided, the best proxy is whether your samples look
as *varied* as the training data, not just as convincing.

---

## Three things that will save you a day

**Do not train inside `generate.py`.** It is allowed, but every re-run then
retrains from scratch. Commit your weights and sample from them; the staff
reference generates 1024 images in about three minutes that way.

**Submit early, in Part 1, when nothing is at stake.** A Part 1 VAE scores
approximately zero and that is the expected result. What it buys you is a tested
route from your laptop to the leaderboard while there is still a month to fix
anything broken. Every commit is scored once and cached, so re-running an
unchanged commit is free and can never lower your score.

**Look at your own samples.** Both model cards ask for a contact sheet, and both
mark schemes reward an honest account of what is wrong with them. A card claiming
good results next to a grid of blobs scores badly; a card that says "these are
blobs, here is why" scores well.

---

## Where to run it

See the appendix of the spec for a proper how-to. You have four options, and all
of them work for both development and training:

- **GPU Dojo** (`gpudojo.doc.ic.ac.uk`), your own Azure VM with an NVIDIA T4.
  Set an SSH key, click start, and **click stop when you finish**: you have a
  GPU-hour budget and a running VM spends it whether you are using it or not.
- **Run:AI** (HX3 cluster), submit a job and walk away. The teaching project
  already caps you at 14 GB by default and 28 GB maximum, so you do not need to
  ask for memory.
- **Department lab workstations**, `ssh` in and check `nvidia-smi` first.
- **DoC GPU cluster**, `salloc --gres=gpu:1` on `gpucluster2`/`gpucluster3`.

The GPU Dojo dashboard shows live availability for the lab workstations and the
cluster too, so it is a useful page even if you do not use its VMs.

None of this should be a bottleneck. The Part 1 reference trains in about 20
minutes at **1.35 GB**; the Part 2 reference in about half an hour at **5 GB**.
