@echo off
REM Single GPU training (Windows)
torchrun --standalone --nproc_per_node=1 src/train_ddp.py --cfg configs/base.yaml
