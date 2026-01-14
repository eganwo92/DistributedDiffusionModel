# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of Distributed Conditional Diffusion Model
- DDP training with gradient synchronization
- EMA synchronization across distributed ranks
- Distributed sampling with all_gather aggregation
- Support for Describable Textures Dataset (DTD)
- Configuration files for full and filtered datasets
- Data validation script (`check_data.py`)
- Quick test script (`test_quick.py`)
- Comprehensive test suite (`test_setup.py`)
- Windows and Linux/Mac execution scripts
- Documentation (README, QUICKSTART, DATASET_INFO)

### Features
- Class-conditional DDPM with U-Net architecture
- Automatic mixed precision (AMP) training support
- Flexible deployment (1 GPU, multi-GPU, CPU-only)
- Deterministic sampling across distributed ranks
- Checkpoint saving and loading
- Sample generation and visualization

## [1.0.0] - 2024-12-XX

### Initial Release
- Core distributed diffusion training pipeline
- Support for DTD dataset
- Basic evaluation utilities
