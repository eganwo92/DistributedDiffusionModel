# Dataset Information

This document describes the datasets used in the Distributed Diffusion Model project.

## Describable Textures Dataset (DTD)

This project uses the Describable Textures Dataset (DTD) for training conditional diffusion models.

### Dataset Structure

```
data_textures/
├── dtd/
│   └── images/          # Full DTD dataset
│       ├── banded/
│       ├── blotchy/
│       ├── braided/
│       └── ... (47 classes total)
└── filtered/            # Filtered subset
    ├── cracked/
    ├── crystalline/
    └── ... (10 classes total)
```

### Full DTD Dataset

- **Location**: `data_textures/dtd/images/`
- **Classes**: 47 texture categories
- **Images per class**: 120
- **Total images**: 5,640
- **Config file**: `configs/base.yaml` or `configs/dtd_full.yaml`

**Classes included:**
banded, blotchy, braided, bubbly, bumpy, chequered, cobwebbed, cracked, crosshatched, crystalline, dotted, fibrous, flecked, freckled, frilly, gauzy, grid, grooved, honeycombed, interlaced, knitted, lacelike, lined, marbled, matted, meshed, paisley, perforated, pitted, pleated, polka-dotted, porous, potholed, scaly, smeared, spiralled, sprinkled, stained, stratified, striped, studded, swirly, veined, waffled, woven, wrinkled, zigzagged

### Filtered DTD Dataset

- **Location**: `data_textures/filtered/`
- **Classes**: 10 texture categories
- **Images per class**: 120
- **Total images**: 1,200
- **Config file**: `configs/dtd_filtered.yaml`

**Classes included:**
cracked, crystalline, fibrous, grooved, honeycombed, marbled, scaly, studded, veined, woven

### Usage

**Check dataset structure:**
```bash
# Check full dataset
python src/check_data.py --data_dir data_textures/dtd/images

# Check filtered dataset
python src/check_data.py --data_dir data_textures/filtered
```

**Train with full dataset:**
```bash
torchrun --standalone --nproc_per_node=1 src/train_ddp.py --cfg configs/base.yaml
```

**Train with filtered dataset (faster for testing):**
```bash
torchrun --standalone --nproc_per_node=1 src/train_ddp.py --cfg configs/dtd_filtered.yaml
```

### Dataset Format

The dataset follows the PyTorch `ImageFolder` structure:
- Each class is a subdirectory
- Images can be `.jpg`, `.png`, or `.jpeg`
- Images are automatically resized and normalized during training

### Image Preprocessing

Images are preprocessed with:
- Resize to `img_size` (default: 64x64)
- Random resized crop (scale: 0.6-1.0)
- Random horizontal flip
- Normalize to [-1, 1] range

These transformations are defined in `src/train_ddp.py` and can be customized in the config file.
