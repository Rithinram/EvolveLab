import pytest
import torch
from training.proxy_evaluator import ProxyEvaluator
from evolution.genome import Genome

def test_synflow_basic():
    # Simple linear model
    genome = Genome(genome_id="proxy_test")
    genome.architecture = {
        "input_shape": [1, 28, 28],
        "layers": [
            {"type": "conv2d", "params": {"filters": 16, "kernel": 3}},
            {"type": "dense", "params": {"units": 10}}
        ]
    }
    
    evaluator = ProxyEvaluator(device="cpu")
    res = evaluator.score_genome(genome, proxy_type="synflow")
    
    assert res["status"] == "success"
    assert res["proxy_score"] > 0
    print(f"Synflow score: {res['proxy_score']}")

def test_grad_norm_basic():
    genome = Genome(genome_id="grad_test")
    genome.architecture = {
        "input_shape": [1, 28, 28],
        "layers": [
            {"type": "conv2d", "params": {"filters": 8, "kernel": 3}}
        ]
    }
    
    evaluator = ProxyEvaluator(device="cpu")
    res = evaluator.score_genome(genome, proxy_type="grad_norm")
    
    assert res["status"] == "success"
    assert res["proxy_score"] > 0
    print(f"Grad Norm score: {res['proxy_score']}")

def test_synflow_bottleneck():
    # A model that should have a lower synflow score (disconnected or extreme bottleneck)
    genome_good = Genome(genome_id="good")
    genome_good.architecture = {
        "input_shape": [1, 28, 28],
        "layers": [{"type": "conv2d", "params": {"filters": 32, "kernel": 3}}]
    }
    
    genome_bad = Genome(genome_id="bad")
    genome_bad.architecture = {
        "input_shape": [1, 28, 28],
        "layers": [{"type": "conv2d", "params": {"filters": 1, "kernel": 1}}] # Tiny bottleneck
    }
    
    evaluator = ProxyEvaluator(device="cpu")
    score_good = evaluator.score_genome(genome_good, proxy_type="synflow")["proxy_score"]
    score_bad = evaluator.score_genome(genome_bad, proxy_type="synflow")["proxy_score"]
    
    # Larger models with more parameters typically have higher synflow
    assert score_good > score_bad
