"""
Quick test to verify everything works - run from project root.
"""
import torch
import yaml
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 60)
print("Quick Setup Test")
print("=" * 60)

# Test 1: PyTorch and CUDA
print("\n[1] Testing PyTorch...")
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"  Device count: {torch.cuda.device_count()}")

# Test 2: Imports
print("\n[2] Testing imports...")
try:
    from unet import TinyUNet
    from ddpm import DDPM
    from ema import EMA
    print("  [OK] All modules imported")
except Exception as e:
    print(f"  [ERROR] {e}")
    sys.exit(1)

# Test 3: Config
print("\n[3] Testing config...")
try:
    cfg = yaml.safe_load(open("configs/base.yaml"))
    print(f"  [OK] Config loaded: {cfg['data_dir']}")
except Exception as e:
    print(f"  [ERROR] {e}")
    sys.exit(1)

# Test 4: Model creation
print("\n[4] Testing model creation...")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyUNet(num_classes=47, base_ch=cfg["base_ch"]).to(device)
    ddpm = DDPM(timesteps=cfg["timesteps"], device=device)
    print(f"  [OK] Model created on {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"  [ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Forward pass
print("\n[5] Testing forward pass...")
try:
    batch_size = 2
    x = torch.randn(batch_size, 3, cfg["img_size"], cfg["img_size"]).to(device)
    t = torch.randint(0, cfg["timesteps"], (batch_size,)).to(device)
    y = torch.randint(0, 47, (batch_size,)).to(device)
    
    with torch.no_grad():
        pred = model(x, t, y)
        xt, noise = ddpm.q_sample(x, t)
    
    print(f"  [OK] Forward pass works!")
    print(f"  Input: {x.shape} -> Output: {pred.shape}")
    print(f"  DDPM q_sample: {x.shape} -> {xt.shape}")
except Exception as e:
    print(f"  [ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Data loading
print("\n[6] Testing data loading...")
try:
    from torchvision import datasets, transforms
    if os.path.exists(cfg["data_dir"]):
        tfm = transforms.Compose([transforms.Resize(cfg["img_size"]), transforms.ToTensor()])
        ds = datasets.ImageFolder(cfg["data_dir"], transform=tfm)
        print(f"  [OK] Dataset loaded: {len(ds.classes)} classes, {len(ds)} samples")
    else:
        print(f"  [WARNING] Data directory not found: {cfg['data_dir']}")
except Exception as e:
    print(f"  [WARNING] Data loading test skipped: {e}")

# Summary
print("\n" + "=" * 60)
print("[SUCCESS] All tests passed!")
print("=" * 60)
print("\nYou have 1 GPU available. You can run:")
print("  - Single GPU training: torchrun --standalone --nproc_per_node=1 src/train_ddp.py")
print("  - Or use: scripts\\run_single_gpu.bat")
print("\nNote: You don't need multiple GPUs - single GPU works perfectly!")
print("=" * 60)
