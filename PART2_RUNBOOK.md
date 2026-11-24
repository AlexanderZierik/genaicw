# Part 2 GPU Dojo runbook

Use the personal GPU Dojo T4 VM for all Part 2 training. Stop the VM whenever
you are not actively using it: its GPU-hour budget is consumed while it runs.

## Set up the VM

1. Add an SSH public key and start the VM from GPU Dojo.
2. Connect by SSH, clone the course GitLab repository, and check the GPU:

   ```bash
   nvidia-smi
   pip install -r requirements.txt
   ```

3. Run this once while the VM has network access. It downloads the supplied
   dataset and pretrained VAE and creates the local latent cache:

   ```bash
   python train_flow_matching.py --steps 1 --batch 2 --encode-batch 64 \
     --out _data/smoke.pt --loss-figure _data/smoke_loss.png
   ```

## Experiments

Keep checkpoints and generated images under `_data/` or `out/` until a final
model has been selected. Those directories are ignored and must not be pushed.

```bash
# Baseline
python train_flow_matching.py --steps 20000 --batch 256 --base 128 --ot \
  --out _data/baseline.pt --loss-figure _data/baseline_loss.png
python generate.py --ckpt _data/baseline.pt --out out/baseline --n 64 --seed 60038
python selfeval.py out/baseline

# Negative-control ablation: no optimal-transport pairing
python train_flow_matching.py --steps 10000 --batch 256 --base 128 \
  --out _data/no_ot.pt --loss-figure _data/no_ot_loss.png
python generate.py --ckpt _data/no_ot.pt --out out/no_ot --n 64 --seed 60038
python selfeval.py out/no_ot

# Final candidate
python train_flow_matching.py --steps 40000 --batch 256 --base 128 --ot \
  --out checkpoint.pt --loss-figure figures/flow_final_loss.png
python generate.py --ckpt checkpoint.pt --out out/final_64 --n 64 --seed 60038
python selfeval.py out/final_64
python generate.py --ckpt checkpoint.pt --out out/final_1024 --n 1024
```

Update `MODEL_CARD.md` from the `*.summary.json` files and save a contact sheet
of `out/final_64` as `figures/flow_final_samples.png`. Commit only the selected
`checkpoint.pt`, its source code, final figures, and the completed model card.
Before marking the final commit in LabTS, run the exact command below from a
fresh clone and verify that all requested images are RGB 112x112 PNG files:

```bash
python generate.py --out out/clean_clone --n 64
```
