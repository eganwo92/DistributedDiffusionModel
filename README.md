# Distributed Conditional Diffusion for Scalable Image Generation

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A distributed training and sampling pipeline for class-conditional diffusion models (DDPM), featuring data sharding, gradient synchronization, EMA synchronization, and distributed sampling with aggregation.

**Repository**: [https://github.com/eganwo92/DistributedDiffusionModel](https://github.com/eganwo92/DistributedDiffusionModel)

## Features

- **Distributed Data Parallel (DDP) Training**: Multi-GPU training with synchronized gradients
- **Data Sharding**: Automatic data distribution across workers via `DistributedSampler`
- **EMA Synchronization**: Periodic broadcasting of EMA weights from rank0 to prevent drift
- **Distributed Sampling**: Parallel generation with `all_gather` aggregation
- **Flexible Deployment**: Works on 1 GPU, multiple GPUs, or CPU-only (debug mode)

## Project Structure

```
DistributedDiffusionModel/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── test_quick.py          # Quick setup test
├── configs/               # Configuration files
│   ├── base.yaml          # Main configuration (full DTD)
│   ├── dtd_full.yaml      # Full DTD dataset config
│   └── dtd_filtered.yaml  # Filtered DTD dataset config
├── src/                   # Source code
│   ├── data.py            # Data loading utilities
│   ├── unet.py            # U-Net architecture
│   ├── ddpm.py            # DDPM diffusion process
│   ├── ema.py             # Exponential Moving Average
│   ├── dist_utils.py      # Distributed utilities
│   ├── train_ddp.py       # DDP training script
│   ├── sample_ddp.py      # Distributed sampling
│   ├── eval_fid.py        # Evaluation utilities
│   ├── check_data.py      # Dataset validation script
│   └── test_setup.py      # Comprehensive test script
├── scripts/               # Execution scripts
│   ├── run_single_gpu.sh  # Single GPU training (Linux/Mac)
│   ├── run_single_gpu.bat # Single GPU training (Windows)
│   ├── run_ddp_2gpu.sh    # Multi-GPU training (Linux/Mac)
│   ├── run_ddp_2gpu.bat   # Multi-GPU training (Windows)
│   ├── sample_ddp.sh      # Distributed sampling (Linux/Mac)
│   └── sample_ddp.bat     # Distributed sampling (Windows)
├── data_textures/         # Dataset directory (not included in repo)
│   ├── dtd/images/        # Full DTD dataset
│   └── filtered/          # Filtered DTD dataset
└── results/               # Output directory (generated)
    ├── samples/           # Generated samples
    ├── logs/              # Training logs
    └── ckpt/              # Model checkpoints
```

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0+ with CUDA support (recommended) or CPU-only
- CUDA-capable GPU (optional but recommended)
- Local machine or server with PyTorch installed (not designed for cloud notebooks)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/eganwo92/DistributedDiffusionModel.git
cd DistributedDiffusionModel
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python test_quick.py
```

## Quick Start

### 1. Prepare Data

**Note**: The `data_textures/` directory is not included in this repository. You need to download and organize your dataset.

The project is configured to use the Describable Textures Dataset (DTD) structure:
```
data_textures/
├── dtd/
│   └── images/
│       ├── banded/
│       ├── blotchy/
│       ├── braided/
│       └── ... (47 texture classes)
└── filtered/
    ├── cracked/
    ├── crystalline/
    └── ... (10 filtered classes)
```

**Verify your dataset:**
```bash
python src/check_data.py --data_dir data_textures/dtd/images
```

**Dataset Download**: 
- Download the Describable Textures Dataset (DTD) from [https://www.robots.ox.ac.uk/~vgg/data/dtd/](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
- Extract and organize it in the `data_textures/` directory following the structure shown above
- Or use your own dataset organized in the same folder structure (each class in its own subdirectory)

**Available configurations:**
- `configs/base.yaml` - Full DTD dataset (47 classes, 5640 images)
- `configs/dtd_full.yaml` - Same as base (full DTD)
- `configs/dtd_filtered.yaml` - Filtered subset (10 classes, 1200 images) for faster testing

### 2. Single GPU Training

```bash
torchrun --standalone --nproc_per_node=1 src/train_ddp.py --cfg configs/base.yaml
```

### 3. Multi-GPU Training (2 GPUs)

```bash
torchrun --standalone --nproc_per_node=2 src/train_ddp.py --cfg configs/base.yaml
```

### 4. Distributed Sampling

```bash
torchrun --standalone --nproc_per_node=2 src/sample_ddp.py \
    --cfg configs/base.yaml \
    --ckpt results/ckpt/ddp_e030.pt \
    --out results/samples_ddp.png
```

Or use the script:
```bash
bash scripts/sample_ddp.sh results/ckpt/ddp_e030.pt
```

## Architecture

### Method: Conditional DDPM

- **U-Net**: Predicts noise at each timestep
- **Class Conditioning**: Class embeddings injected via residual blocks
- **Time Embedding**: Sinusoidal positional embeddings for timesteps

### Distributed Design

#### 1. Data Parallel Training
- Each worker sees a shard of data via `DistributedSampler`
- `DistributedDataParallel` synchronizes gradients each step
- Effective global batch size = `local_batch_size × world_size`

#### 2. EMA Synchronization
- **Problem**: In distributed training, EMA weights can drift across ranks
- **Solution**: Rank0 maintains canonical EMA snapshot
- **Implementation**: Every `ema_sync_every` steps, broadcast EMA weights from rank0 to all ranks
- This ensures consistent sampling quality across all workers

#### 3. Deterministic Sampling
- Each rank generates disjoint samples using `seed = base_seed + rank`
- `all_gather` collects samples from all ranks on rank0
- Rank0 saves aggregated results

## Dataset Information

The project includes:
- **Full DTD Dataset**: 47 texture classes, 120 images per class (5640 total) - located at `data_textures/dtd/images/`
- **Filtered DTD Dataset**: 10 texture classes, 120 images per class (1200 total) - located at `data_textures/filtered/`

Use `src/check_data.py` to verify your dataset structure:
```bash
python src/check_data.py --data_dir data_textures/dtd/images
```

## Configuration

Edit `configs/base.yaml` (or `configs/dtd_full.yaml` / `configs/dtd_filtered.yaml`) to customize:

- `data_dir`: Path to your dataset (default: `./data_textures/dtd/images`)
- `img_size`: Image resolution (default: 64)
- `batch_size`: Per-GPU batch size
- `timesteps`: Number of diffusion timesteps (default: 1000)
- `epochs`: Number of training epochs
- `lr`: Learning rate
- `ema_decay`: EMA decay factor
- `ema_sync_every`: Frequency of EMA synchronization (0 to disable)

## Experiments & Analysis

### Throughput Analysis

Measure images/sec per GPU:
- Single GPU baseline
- Multi-GPU scaling efficiency
- Communication overhead

### Sample Quality

- Qualitative: Visual inspection of generated samples
- Quantitative: FID score (requires additional setup)

### Failure Modes & Fixes

1. **Stale EMA Drift**
   - Symptom: Degraded sample quality over time
   - Fix: Periodic EMA synchronization (`ema_sync_every`)

2. **Class Imbalance**
   - Symptom: Uneven class distribution in batches
   - Fix: Use `DistributedSampler` with proper shuffling

3. **Communication Overhead**
   - Symptom: Slow training with many GPUs
   - Fix: Adjust `ema_sync_every` to balance sync frequency

## Key Implementation Details

### Distributed Setup
- Uses `torchrun` for process management
- Automatically detects CUDA availability
- Falls back to `gloo` backend for CPU-only mode

### Gradient Synchronization
- Automatic via `DistributedDataParallel`
- Gradient clipping for stability
- Mixed precision training (AMP) support

### EMA Broadcast
```python
# Every ema_sync_every steps:
if global_step % ema_sync_every == 0:
    broadcast_model(ema.ema_model, src=0)
```

### Sample Aggregation
```python
# Each rank generates n_local samples
samples = ddpm.sample(model, n=n_local, ...)

# Gather from all ranks
gathered = all_gather_tensor(samples)  # Shape: (n_total, C, H, W)
```

## Performance Tips

1. **Batch Size**: Increase `batch_size` per GPU for better GPU utilization
2. **Num Workers**: Adjust `num_workers` based on your CPU cores
3. **AMP**: Enable automatic mixed precision for faster training
4. **EMA Sync**: Balance `ema_sync_every` between sync frequency and overhead

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config
- Reduce `img_size`
- Disable AMP if enabled

### Slow Training
- Check `num_workers` (too high can cause overhead)
- Verify data loading isn't the bottleneck
- Monitor GPU utilization

### Poor Sample Quality
- Increase training epochs
- Adjust learning rate
- Verify EMA synchronization is working
- Check data quality and preprocessing

## Testing

Run the quick test to verify everything works:

```bash
python test_quick.py
```

Or check your dataset structure:

```bash
python src/check_data.py --data_dir data_textures/dtd/images
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Built with [PyTorch](https://pytorch.org/)
- Uses the [Describable Textures Dataset (DTD)](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
- Inspired by DDPM (Denoising Diffusion Probabilistic Models)
