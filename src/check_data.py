"""
Script to verify dataset structure and count samples per class.
"""
import os
import argparse
from pathlib import Path
from torchvision import datasets

def check_dataset(data_dir):
    """Check dataset structure and print statistics."""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"[ERROR] Data directory '{data_dir}' does not exist!")
        return False
    
    print(f"[OK] Data directory found: {data_path.absolute()}")
    
    # Check if it's a valid ImageFolder structure
    try:
        dataset = datasets.ImageFolder(str(data_path))
        num_classes = len(dataset.classes)
        print(f"[OK] Found {num_classes} classes")
        print(f"\nClasses:")
        for i, class_name in enumerate(dataset.classes):
            class_path = data_path / class_name
            if class_path.exists():
                num_images = len(list(class_path.glob("*.jpg")) + 
                               list(class_path.glob("*.png")) +
                               list(class_path.glob("*.jpeg")))
                print(f"  {i:2d}. {class_name:20s} - {num_images:4d} images")
            else:
                print(f"  {i:2d}. {class_name:20s} - [WARNING] directory not found")
        
        print(f"\n[OK] Dataset structure is valid!")
        print(f"  Total classes: {num_classes}")
        print(f"  Total samples: {len(dataset)}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error loading dataset: {e}")
        print(f"\nExpected structure:")
        print(f"  {data_dir}/")
        print(f"    class0/")
        print(f"      img1.jpg")
        print(f"      img2.jpg")
        print(f"      ...")
        print(f"    class1/")
        print(f"      ...")
        return False

def main():
    parser = argparse.ArgumentParser(description="Check dataset structure")
    parser.add_argument("--data_dir", type=str, default="data_textures/dtd/images",
                       help="Path to dataset directory")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Dataset Structure Checker")
    print("=" * 60)
    print()
    
    success = check_dataset(args.data_dir)
    
    print()
    if success:
        print("=" * 60)
        print("[OK] Dataset is ready for training!")
        print("=" * 60)
    else:
        print("=" * 60)
        print("[ERROR] Please fix the dataset structure before training")
        print("=" * 60)

if __name__ == "__main__":
    main()
