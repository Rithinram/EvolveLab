import pytest
import torch
from unittest.mock import MagicMock, patch
from training.pytorch_evaluator import PyTorchEvaluator
from evolution.genome import Genome

@pytest.fixture
def mock_config():
    return {
        "evaluation": {
            "mode": "pytorch",
            "dataset": "synthetic",
            "batch_size": 4
        },
        "fitness": {
            "accuracy_weight": 0.7,
            "cost_weight": 0.3
        }
    }

def test_evaluator_synthetic(mock_config):
    evaluator = PyTorchEvaluator(mock_config)
    genome = Genome()
    genome.architecture["layers"] = [{"type": "dense", "params": {"units": 10}}]
    
    result = evaluator.evaluate(genome, epochs=1, dataset="synthetic")
    
    assert result["status"] == "success"
    assert "accuracy" in result
    assert "param_count" in result
    assert result["param_count"] > 0

def test_evaluator_oom_handling(mock_config):
    evaluator = PyTorchEvaluator(mock_config)
    genome = Genome()
    
    # Mock CodeGenerator to return a valid class that accepts arguments
    mock_code = """
import torch
import torch.nn as nn
class EvolvedModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.p = nn.Parameter(torch.randn(1))
    def forward(self, x):
        return torch.randn(x.size(0), 10)
"""
    with patch('training.code_generator.CodeGenerator.generate_module_code', return_value=mock_code):
        # Mock torch.randn to raise the error
        with patch('torch.randn', side_effect=RuntimeError("CUDA out of memory")):
            result = evaluator.evaluate(genome, epochs=1, dataset="synthetic")
            assert result["status"] == "error"
            assert result["error"] == "OOM"

def test_evaluate_population(mock_config):
    evaluator = PyTorchEvaluator(mock_config)
    genomes = [Genome() for _ in range(2)]
    for g in genomes:
        g.architecture["layers"] = [{"type": "dense", "params": {"units": 4}}]
        
    evaluated = evaluator.evaluate_population(genomes, fidelity=1)
    
    assert len(evaluated) == 2
    for g in evaluated:
        assert g.metrics["fitness_score"] >= 0
        assert "accuracy" in g.metrics
