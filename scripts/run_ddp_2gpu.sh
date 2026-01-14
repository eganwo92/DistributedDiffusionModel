#!/bin/bash
# 2 GPU training (if you have 2 GPUs)
torchrun --standalone --nproc_per_node=2 src/train_ddp.py --cfg configs/base.yaml
