import torch
import torch.nn as nn
from training.code_generator import CodeGenerator
from evolution.genome import Genome

def test_residual_add_code_gen():
    # Define a genome with a residual skip
    genome = Genome(genome_id="res_test")
    genome.architecture = {
        "input_shape": [3, 32, 32],
        "layers": [
            {"type": "conv2d", "params": {"filters": 16, "kernel": 3}}, # Index 0
            {"type": "conv2d", "params": {"filters": 32, "kernel": 3}}, # Index 1
            {"type": "residual_add", "params": {"skip_back": 2}},      # Index 2: skips back to layer 0 output
            {"type": "pooling", "params": {"type": "max", "size": 2}}
        ]
    }
    
    gen = CodeGenerator()
    code = gen.generate_module_code(genome)
    
    # Check if code contains projection (since layer 0 has 16 filters, layer 2 has 32)
    assert "proj_2" in code
    assert "nn.Conv2d(16, 32, kernel_size=1)" in code
    assert "x = x + res_2" in code
    
    # Try to execute the code
    namespace = {}
    exec(code, {"torch": torch, "nn": nn, "F": torch.nn.functional}, namespace)
    model_class = namespace['EvolvedModel']
    model = model_class()
    
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)
    assert output.shape == (1, 10) # 10 classes default
    print("Residual Add execution successful!")

def test_residual_add_same_channels():
    # Define a genome where channels match (no projection needed)
    genome = Genome(genome_id="res_same")
    genome.architecture = {
        "input_shape": [3, 32, 32],
        "layers": [
            {"type": "conv2d", "params": {"filters": 32, "kernel": 3}}, # Index 0
            {"type": "conv2d", "params": {"filters": 32, "kernel": 3}}, # Index 1
            {"type": "residual_add", "params": {"skip_back": 2}}       # Index 2: back to layer 0
        ]
    }
    
    gen = CodeGenerator()
    code = gen.generate_module_code(genome)
    
    assert "proj_2" not in code
    assert "res_2 = tensors[0]" in code
    
    # Try to execute
    namespace = {}
    exec(code, {"torch": torch, "nn": nn, "F": torch.nn.functional}, namespace)
    model_class = namespace['EvolvedModel']
    model = model_class()
    
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)
    assert output.shape == (1, 10)
