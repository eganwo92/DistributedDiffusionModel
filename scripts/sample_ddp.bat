@echo off
REM Distributed sampling (Windows)
REM Usage: sample_ddp.bat <checkpoint_path>
if "%1"=="" (
    set CKPT=results/ckpt/ddp_e030.pt
) else (
    set CKPT=%1
)
torchrun --standalone --nproc_per_node=2 src/sample_ddp.py --cfg configs/base.yaml --ckpt %CKPT% --out results/samples_ddp.png
