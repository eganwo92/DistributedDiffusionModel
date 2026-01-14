#!/bin/bash
# Single GPU training (still uses torchrun with world_size=1)
torchrun --standalone --nproc_per_node=1 src/train_ddp.py --cfg configs/base.yaml
