@echo off
REM 2 GPU training (Windows)
torchrun --standalone --nproc_per_node=2 src/train_ddp.py --cfg configs/base.yaml
