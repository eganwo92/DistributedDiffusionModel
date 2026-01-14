#!/bin/bash
# Distributed sampling
# Usage: ./scripts/sample_ddp.sh <checkpoint_path>
CKPT=${1:-results/ckpt/ddp_e030.pt}
torchrun --standalone --nproc_per_node=2 src/sample_ddp.py --cfg configs/base.yaml --ckpt $CKPT --out results/samples_ddp.png
