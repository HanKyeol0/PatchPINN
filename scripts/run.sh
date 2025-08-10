#!/usr/bin/env bash
set -e

# Change here per run
MODEL_NAME=mlp              # e.g., mlp, fourier_mlp
EXPERIMENT_NAME=burgers1d   # e.g., burgers1d, helmholtz2d, poisson2d

python -m pinnlab.train \
  --model_name $MODEL_NAME \
  --experiment_name $EXPERIMENT_NAME \
  --common_config configs/common_config.yaml \
  --model_config configs/model/${MODEL_NAME}.yaml \
  --exp_config configs/experiment/${EXPERIMENT_NAME}.yaml