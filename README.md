# SGON (Sheaf-Glue Operator Network) sign-of-life

This repo contains a minimal SGON implementation for the Darcy 1D dataset.
It uses local patch coefficients, a sparse gluing solve, and partition of unity
for decoding.

Quickstart:
1. Create a virtualenv and install deps.
2. Run the training script.

Example:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/train_darcy1d_sgon.py \
  --data_path Data/darcy_1d_data/darcy_1d_dataset_501_sigma_0.008 \
  --device cuda \
  --steps 5000 \
  --batch_size 64 \
  --sensor_size 64 \
  --n_patches 64 \
  --degree 3 \
  --gluing_lam 5.0
```
