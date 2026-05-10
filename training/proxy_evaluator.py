"""
EvolveLab — Zero-Cost Proxy Evaluator
Calculates architecture potential without training (Synflow, Grad Norm).
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, List
from training.code_generator import CodeGenerator

logger = logging.getLogger("evolvelab.proxy_evaluator")

class ProxyEvaluator:
    """Calculates zero-cost proxies to estimate genome potential."""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.code_gen = CodeGenerator()

    def compute_synflow(self, model: nn.Module, input_shape: List[int]) -> float:
        """
        Synflow: Measures total path connectivity.
        Uses unit weights to focus purely on structural potential.
        """
        model.eval()
        
        # 1. Set all parameters to 1.0 (Unit Weights)
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(1.0)

        # 2. Forward pass with all-ones input
        input_data = torch.ones(1, *input_shape).to(self.device)
        
        # 3. Compute Synflow score
        output = model(input_data)
        loss = output.sum()
        
        model.zero_grad()
        loss.backward()

        score = 0
        for param in model.parameters():
            if param.grad is not None:
                # Since param is 1.0, score is just the sum of gradients
                score += param.grad.sum().item()
                
        return score

    def compute_grad_norm(self, model: nn.Module, input_shape: List[int], num_classes: int) -> float:
        """Grad Norm: Euclidean norm of gradients after a single forward/backward."""
        model.train()
        input_data = torch.randn(1, *input_shape).to(self.device)
        targets = torch.randint(0, num_classes, (1,)).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        output = model(input_data)
        loss = criterion(output, targets)
        
        model.zero_grad()
        loss.backward()
        
        grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm(2).item() ** 2
        
        return grad_norm ** 0.5

    def score_genome(self, genome, proxy_type: str = "synflow") -> Dict[str, float]:
        """Instantiates a model and computes the requested proxy score."""
        try:
            code = self.code_gen.generate_module_code(genome)
            local_scope = {}
            exec(code, {"torch": torch, "nn": nn, "F": torch.nn.functional}, local_scope)
            
            ModelClass = local_scope['EvolvedModel']
            input_shape = genome.architecture.get("input_shape", [1, 28, 28])
            num_classes = genome.architecture.get("output_classes", 10)
            
            model = ModelClass(input_shape=input_shape, num_classes=num_classes).to(self.device)
            
            # Initialize Lazy layers with a dummy pass
            with torch.no_grad():
                dummy_input = torch.zeros(1, *input_shape).to(self.device)
                model(dummy_input)

            if proxy_type == "synflow":
                score = self.compute_synflow(model, input_shape)
            else:
                score = self.compute_grad_norm(model, input_shape, num_classes)
                
            # Cleanup
            del model
            return {"proxy_score": score, "status": "success"}
        except Exception as e:
            logger.error(f"Proxy scoring failed for {genome.id}: {e}")
            return {"proxy_score": 0.0, "status": "error"}
