# Quick Start Guide

## Installation

```bash
pip install -r requirements.txt
```

## Data Preparation

**Note**: The dataset is not included in this repository. You need to download and organize your dataset.

The project uses the Describable Textures Dataset (DTD). Two dataset configurations are available:

1. **Full DTD** (`data_textures/dtd/images/`) - 47 classes, 5640 images
2. **Filtered DTD** (`data_textures/filtered/`) - 10 classes, 1200 images (for faster testing)

**Download DTD**: [https://www.robots.ox.ac.uk/~vgg/data/dtd/](https://www.robots.ox.ac.uk/~vgg/data/dtd/)

**Verify your dataset:**
```bash
python src/check_data.py --data_dir data_textures/dtd/images
```

The default config (`configs/base.yaml`) uses the full DTD dataset. To use the filtered dataset, use:
```bash
torchrun --standalone --nproc_per_node=1 src/train_ddp.py --cfg configs/dtd_filtered.yaml
```

## Training

### Single GPU (Windows)
```bash
scripts\run_single_gpu.bat
```

### Single GPU (Linux/Mac)
```bash
bash scripts/run_single_gpu.sh
```

### Multi-GPU (2 GPUs, Windows)
```bash
scripts\run_ddp_2gpu.bat
```

### Multi-GPU (2 GPUs, Linux/Mac)
```bash
bash scripts/run_ddp_2gpu.sh
```

### Custom Configuration
```bash
torchrun --standalone --nproc_per_node=2 src/train_ddp.py --cfg configs/base.yaml
```

## Sampling

### Distributed Sampling (Windows)
```bash
scripts\sample_ddp.bat results\ckpt\ddp_e030.pt
```

### Distributed Sampling (Linux/Mac)
```bash
bash scripts/sample_ddp.sh results/ckpt/ddp_e030.pt
```

### Custom Sampling
```bash
torchrun --standalone --nproc_per_node=2 src/sample_ddp.py \
    --cfg configs/base.yaml \
    --ckpt results/ckpt/ddp_e030.pt \
    --out results/samples_ddp.png
```

## Evaluation

```bash
python src/eval_fid.py --cfg configs/base.yaml --ckpt results/ckpt/ddp_e030.pt --n_samples 1000
```

## Troubleshooting

1. **CUDA Out of Memory**: Reduce `batch_size` in `configs/base.yaml`
2. **Data Not Found**: Update `data_dir` in `configs/base.yaml`
3. **Import Errors**: Make sure you're running from the project root directory
