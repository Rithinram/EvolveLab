import pytest
import torch
from unittest.mock import patch, MagicMock
from training.datasets import get_dataset_loaders

@patch('torchvision.datasets.MNIST')
def test_mnist_loader(mock_mnist):
    # Mock dataset
    mock_data = MagicMock()
    mock_data.__len__.return_value = 100
    mock_mnist.return_value = mock_data
    
    train, val, test = get_dataset_loaders('mnist', batch_size=10, val_split=0.2)
    
    assert train is not None
    assert val is not None
    assert test is not None
    # 100 total, 0.2 val split -> 80 train, 20 val
    assert len(train.dataset) == 80
    assert len(val.dataset) == 20

def test_unsupported_dataset():
    with pytest.raises(ValueError):
        get_dataset_loaders('invalid_dataset')

def test_cifar_shapes():
    # This would require real CIFAR10 or a very complex mock
    # For now, let's just test that the logic branches correctly
    with patch('torchvision.datasets.CIFAR10') as mock_cifar:
        mock_data = MagicMock()
        mock_data.__len__.return_value = 100
        mock_cifar.return_value = mock_data
        
        train, _, _ = get_dataset_loaders('cifar10', batch_size=5)
        assert train.batch_size == 5
