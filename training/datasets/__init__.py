import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import logging

logger = logging.getLogger("evolvelab.datasets")

def get_dataset_loaders(dataset_name: str = "mnist", batch_size: int = 64, val_split: float = 0.1):
    """
    Returns train and validation data loaders for the specified dataset.
    Supported: 'mnist', 'cifar10', 'fashion_mnist'
    """
    dataset_name = dataset_name.lower()
    data_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(data_dir, exist_ok=True)

    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        full_train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    
    elif dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        full_train = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
    
    elif dataset_name == "fashion_mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        full_train = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Split train into train/val
    num_train = len(full_train)
    indices = list(range(num_train))
    split = int(num_train * val_split)
    
    # Simple deterministic split for consistency during NAS
    val_idx, train_idx = indices[:split], indices[split:]
    
    train_subset = Subset(full_train, train_idx)
    val_subset = Subset(full_train, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info(f"Loaded {dataset_name}: {len(train_subset)} train, {len(val_subset)} val, {len(test_dataset)} test")
    
    return train_loader, val_loader, test_loader
