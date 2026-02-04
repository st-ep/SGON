# SGON (Sheaf-Glue Operator Network)

This repo contains an SGON-style operator learning prototype for the Darcy 1D dataset.
It represents the solution using local patch coefficients and decodes with a partition of unity.

For fixed sensors on a fixed 1D grid, we found that adding a small 1D convolutional backbone over the input
(`--u_backbone`) substantially improves accuracy.

Quickstart:
1. Create a virtualenv and install deps.
2. Run the training script.

Install:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Recommended training config (fixed sensors):
```bash
python scripts/train_darcy1d_sgon.py \
  --multiscale --n_patches_coarse 8 --n_patches_mid 16 --n_patches 32 \
  --glue_mode poly --poly_k 3 \
  --attention_pool \
  --u_backbone
```

Noise/accuracy evaluation:
```bash
python scripts/eval_noise.py \
  --models sgon deeponet \
  --ckpts <sgon_best.pt> <deeponet_best.pt> \
  --labels SGON DeepONet \
  --noise_levels 0.0 0.02 0.04 0.06 0.08 0.1 \
  --n_eval 1024 \
  --batch_size 256 \
  --plot
```

Finding (no noise): on `Data/darcy_1d_data/darcy_1d_dataset_501` with `sensor_size=64` and `n_eval=1024`,
we observed:

| Model | Test rel L2 (mean) | Test MSE (mean) |
| --- | ---: | ---: |
| SGON (`--u_backbone`) | `1.55e-3` | `3.79e-8` |
| DeepONet | `4.19e-3` | `3.22e-7` |
