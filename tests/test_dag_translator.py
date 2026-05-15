import torch
import torch.nn as nn
from training.code_generator import CodeGenerator
from evolution.genome import Genome

def test_full_dag_translation():
    # Define a complex DAG genome (Inception-like/Residual cell)
    # Node 0: Input
    # Node 1: Conv 3x3 (Input -> 1)
    # Node 2: Conv 3x3 (Input -> 2)
    # Node 3: Add (1 + 2)
    # Node 4: Pooling (3 -> 4)
    # Node 5: Add (Input + 4) -> Skip connection across the whole cell
    
    genome = Genome(genome_id="dag_complex")
    genome.architecture = {
        "input_shape": [3, 32, 32],
        "nodes": [
            {"id": 0, "op": "input"},
            {"id": 1, "op": "conv2d", "params": {"filters": 32, "kernel": 3}},
            {"id": 2, "op": "conv2d", "params": {"filters": 32, "kernel": 3}},
            {"id": 3, "op": "add"}, # Conceptual op
            {"id": 4, "op": "pooling", "params": {"type": "max", "size": 2}},
            {"id": 5, "op": "conv2d", "params": {"filters": 32, "kernel": 1}} # Final projection
        ],
        "adjacency": {
            "0": ["1", "2", "5"],
            "1": ["3"],
            "2": ["3"],
            "3": ["4"],
            "4": ["5"]
        }
    }
    
    gen = CodeGenerator()
    code = gen.generate_module_code(genome)
    
    # print(code) # For debugging
    
    # Verify topological order is present
    assert "self.topo_order" in code
    assert "'0'" in code # Node IDs are strings in JSON usually
    assert "torch.stack(inputs_3).sum(dim=0)" in code # Node 3 has 2 inputs
    assert "torch.stack(inputs_5).sum(dim=0)" in code # Node 5 has 2 inputs (0 and 4)
    
    # Try to execute
    namespace = {}
    exec(code, {"torch": torch, "nn": nn, "F": torch.nn.functional}, namespace)
    model_class = namespace['EvolvedModel']
    model = model_class()
    
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)
    assert output.shape == (1, 10)
    print("Full DAG execution successful!")

def test_leaf_concatenation():
    # Test that multiple sink nodes are concatenated
    genome = Genome(genome_id="dag_multi_sink")
    genome.architecture = {
        "input_shape": [3, 32, 32],
        "nodes": [
            {"id": 0, "op": "input"},
            {"id": 1, "op": "conv2d", "params": {"filters": 16, "kernel": 3}},
            {"id": 2, "op": "conv2d", "params": {"filters": 16, "kernel": 3}}
        ],
        "adjacency": {
            "0": ["1", "2"]
        }
    }
    
    gen = CodeGenerator()
    code = gen.generate_module_code(genome)
    
    # Node 1 and 2 are leaves
    assert "torch.cat" in code
    assert "'1'" in code
    assert "'2'" in code
    
    namespace = {}
    exec(code, {"torch": torch, "nn": nn, "F": torch.nn.functional}, namespace)
    model_class = namespace['EvolvedModel']
    model = model_class()
    
    # Combined output channels = 16 + 16 = 32
    # The LazyLinear will handle this 32 -> 10 mapping
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)
    assert output.shape == (1, 10)
