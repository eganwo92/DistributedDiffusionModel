"""
Quick test script to verify the project setup works without full training.
"""
import torch
import yaml
import sys
import os

def test_imports():
    """Test that all modules can be imported."""
    print("=" * 60)
    print("Testing Imports")
    print("=" * 60)
    try:
        import sys
        import os
        # Add src to path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        # Change to src directory for relative imports
        os.chdir(os.path.join(os.path.dirname(__file__), '..'))
        from src.unet import TinyUNet
        from src.ddpm import DDPM
        from src.ema import EMA
        from src.dist_utils import setup_distributed, is_main
        from src.data import get_dataset
        print("[OK] All modules imported successfully")
        return True
    except Exception as e:
        print(f"[ERROR] Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config():
    """Test config file loading."""
    print("\n" + "=" * 60)
    print("Testing Configuration")
    print("=" * 60)
    try:
        cfg = yaml.safe_load(open("configs/base.yaml"))
        print(f"[OK] Config loaded")
        print(f"  Data dir: {cfg['data_dir']}")
        print(f"  Image size: {cfg['img_size']}")
        print(f"  Batch size: {cfg['batch_size']}")
        print(f"  Timesteps: {cfg['timesteps']}")
        return True, cfg
    except Exception as e:
        print(f"[ERROR] Config load failed: {e}")
        return False, None

def test_model_creation(cfg):
    """Test model creation."""
    print("\n" + "=" * 60)
    print("Testing Model Creation")
    print("=" * 60)
    try:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        os.chdir(os.path.join(os.path.dirname(__file__), '..'))
        from src.unet import TinyUNet
        from src.ddpm import DDPM
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        num_classes = 47  # DTD has 47 classes
        
        model = TinyUNet(num_classes=num_classes, base_ch=cfg["base_ch"]).to(device)
        ddpm = DDPM(timesteps=cfg["timesteps"], device=device)
        
        print(f"[OK] Model created on {device}")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  DDPM timesteps: {ddpm.timesteps}")
        
        # Test forward pass
        batch_size = 2
        x = torch.randn(batch_size, 3, cfg["img_size"], cfg["img_size"]).to(device)
        t = torch.randint(0, cfg["timesteps"], (batch_size,)).to(device)
        y = torch.randint(0, num_classes, (batch_size,)).to(device)
        
        with torch.no_grad():
            pred = model(x, t, y)
        
        print(f"[OK] Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {pred.shape}")
        
        return True
    except Exception as e:
        print(f"[ERROR] Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading(cfg):
    """Test data loading."""
    print("\n" + "=" * 60)
    print("Testing Data Loading")
    print("=" * 60)
    try:
        from torchvision import datasets, transforms
        
        if not os.path.exists(cfg["data_dir"]):
            print(f"[WARNING] Data directory not found: {cfg['data_dir']}")
            print("  Skipping data loading test")
            return True
        
        tfm = transforms.Compose([
            transforms.Resize(cfg["img_size"]),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        
        ds = datasets.ImageFolder(cfg["data_dir"], transform=tfm)
        print(f"[OK] Dataset loaded")
        print(f"  Classes: {len(ds.classes)}")
        print(f"  Total samples: {len(ds)}")
        print(f"  First 5 classes: {ds.classes[:5]}")
        
        # Test loading one sample
        sample, label = ds[0]
        print(f"[OK] Sample loaded")
        print(f"  Sample shape: {sample.shape}")
        print(f"  Label: {ds.classes[label]}")
        
        return True
    except Exception as e:
        print(f"[ERROR] Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_distributed_setup():
    """Test distributed setup (without actually initializing)."""
    print("\n" + "=" * 60)
    print("Testing Distributed Setup")
    print("=" * 60)
    try:
        import torch.distributed as dist
        print(f"[OK] torch.distributed available: {dist.is_available()}")
        print(f"[OK] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device count: {torch.cuda.device_count()}")
            print(f"  Current device: {torch.cuda.current_device()}")
            print(f"  Device name: {torch.cuda.get_device_name(0)}")
        print("\n[INFO] Distributed training works with:")
        print("  - Single GPU: torchrun --nproc_per_node=1")
        print("  - Multiple GPUs: torchrun --nproc_per_node=N (where N = number of GPUs)")
        print("  - CPU only: torchrun --nproc_per_node=1 (uses gloo backend)")
        return True
    except Exception as e:
        print(f"[ERROR] Distributed setup check failed: {e}")
        return False

def main():
    print("\n" + "=" * 60)
    print("Distributed Diffusion - Setup Test")
    print("=" * 60)
    print()
    
    results = []
    
    # Test imports
    results.append(("Imports", test_imports()))
    
    # Test config
    cfg_ok, cfg = test_config()
    results.append(("Configuration", cfg_ok))
    
    if cfg_ok and cfg:
        # Test model creation
        results.append(("Model Creation", test_model_creation(cfg)))
        
        # Test data loading
        results.append(("Data Loading", test_data_loading(cfg)))
    
    # Test distributed setup
    results.append(("Distributed Setup", test_distributed_setup()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "[OK]" if passed else "[FAIL]"
        print(f"{status} {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("[SUCCESS] All tests passed! Project is ready to use.")
        print("\nYou can start training with:")
        print("  Single GPU: scripts\\run_single_gpu.bat")
        print("  Or: torchrun --standalone --nproc_per_node=1 src/train_ddp.py --cfg configs/base.yaml")
    else:
        print("[WARNING] Some tests failed. Please check the errors above.")
    print("=" * 60)

if __name__ == "__main__":
    main()
