import pytest
import torch
import torch.nn as nn
from training.code_generator import CodeGenerator
from evolution.genome import Genome

def test_conv_after_dense_robustness():
    """Test that CodeGenerator handles Conv2d after a Dense layer by reshaping."""
    genome = Genome()
    genome.architecture["layers"] = [
        {"type": "dense", "params": {"units": 64, "activation": "relu"}},
        {"type": "conv2d", "params": {"filters": 32, "kernel": 3, "activation": "relu"}}
    ]
    
    codegen = CodeGenerator()
    code = codegen.generate_module_code(genome)
    
    # Check for the view/reshape logic
    assert "x.view" in code
    
    # Verify it compiles and runs
    local_scope = {}
    exec_globals = {"torch": torch, "nn": nn, "torch.nn": nn, "F": torch.nn.functional}
    exec(code, exec_globals, local_scope)
    
    ModelClass = local_scope['EvolvedModel']
    model = ModelClass(input_shape=[1, 28, 28], num_classes=10)
    
    # Forward pass
    x = torch.randn(2, 1, 28, 28)
    output = model(x)
    
    assert output.shape == (2, 10)

def test_pure_cnn_final_flatten():
    """Test that a pure CNN model still gets flattened before the classifier."""
    genome = Genome()
    genome.architecture["layers"] = [
        {"type": "conv2d", "params": {"filters": 16, "kernel": 3, "activation": "relu"}}
    ]
    
    codegen = CodeGenerator()
    code = codegen.generate_module_code(genome)
    
    assert "AdaptiveAvgPool2d" in code
    assert "Flatten" in code
    
    local_scope = {}
    exec_globals = {"torch": torch, "nn": nn, "torch.nn": nn, "F": torch.nn.functional}
    exec(code, exec_globals, local_scope)
    
    ModelClass = local_scope['EvolvedModel']
    model = ModelClass(input_shape=[1, 28, 28], num_classes=10)
    
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    assert output.shape == (1, 10)

def test_dense_after_pooling():
    """Test standard Pooling -> Dense transition."""
    genome = Genome()
    genome.architecture["layers"] = [
        {"type": "conv2d", "params": {"filters": 8, "kernel": 3}},
        {"type": "pooling", "params": {"type": "max", "size": 2}},
        {"type": "dense", "params": {"units": 32}}
    ]
    
    codegen = CodeGenerator()
    code = codegen.generate_module_code(genome)
    
    local_scope = {}
    exec_globals = {"torch": torch, "nn": nn, "torch.nn": nn, "F": torch.nn.functional}
    exec(code, exec_globals, local_scope)
    
    ModelClass = local_scope['EvolvedModel']
    model = ModelClass(input_shape=[1, 28, 28], num_classes=10)
    
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    assert output.shape == (1, 10)
