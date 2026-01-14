"""
Data loading utilities for distributed diffusion training.
"""
import torch
from torchvision import datasets, transforms

def get_dataset(data_dir, img_size=64, train=True):
    """Load dataset from directory."""
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    return dataset
