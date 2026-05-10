"""
Verification script for EvolveLab Code Generator
"""

import sys
import os
import torch

# Add the root directory to path
sys.path.append(os.getcwd())

from evolution.genome import Genome
from training.code_generator import CodeGenerator

def test_generation():
    # 1. Create a dummy genome with a mix of layers
    genome = Genome()
    genome.architecture["layers"] = [
        {"type": "conv2d", "params": {"filters": 16, "kernel": 3, "activation": "relu"}},
        {"type": "pooling", "params": {"type": "max", "size": 2}},
        {"type": "projection", "target_dim": 32}, # Test our adapter logic
        {"type": "conv2d", "params": {"filters": 32, "kernel": 3, "activation": "relu"}},
        {"type": "dropout", "params": {"rate": 0.2}},
        {"type": "dense", "params": {"units": 64, "activation": "relu"}},
        {"type": "batch_norm", "params": {}}
    ]
    
    # 2. Generate code
    gen = CodeGenerator()
    code = gen.generate_module_code(genome)
    
    print("--- Generated Code Preview ---")
    print(code)
    print("------------------------------")
    
    # 3. Verify it's valid Python by saving and importing
    temp_file = "temp_model.py"
    try:
        with open(temp_file, "w") as f:
            f.write(code)
        
        import importlib.util
        spec = importlib.util.spec_from_file_location("temp_model", temp_file)
        temp_model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(temp_model)
        
        ModelClass = temp_model.EvolvedModel
        model = ModelClass(input_shape=[1, 28, 28], num_classes=10)
        model.eval() # Avoid BatchNorm error with batch size 1
        
        # 4. Test a forward pass
        dummy_input = torch.randn(1, 1, 28, 28)
        output = model(dummy_input)
        
        print(f"\n[OK] Model instantiated and forward pass successful!")
        print(f"Output shape: {output.shape}")
        
    except Exception as e:
        print(f"\n[FAIL] Code generation test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == "__main__":
    test_generation()
