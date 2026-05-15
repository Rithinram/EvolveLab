"""
EvolveLab — Baselines
Implementation of Random Search and other baselines for comparison.
"""

import time
import logging
from typing import List, Dict
from evolution.genome import Genome
from agents.builder import BuilderAgent
from training.pytorch_evaluator import PyTorchEvaluator
from database.crud import DatabaseManager

logger = logging.getLogger("evolvelab.baselines")

class RandomSearch:
    """Standard Random Search baseline for NAS."""

    def __init__(self, config: dict, db: DatabaseManager):
        self.config = config
        self.db = db
        self.builder = BuilderAgent(species="random_baseline")
        self.evaluator = PyTorchEvaluator(config)
        logger.info("Random Search baseline initialized")

    def run(self, total_samples: int = 50, fidelity: int = 3):
        """Runs random search for a fixed number of samples."""
        logger.info(f"Starting Random Search baseline with {total_samples} samples")
        
        results = []
        best_fitness = -1.0
        
        for i in range(total_samples):
            # 1. Generate random genome
            genome = self.builder.generate_gen_0(1)[0]
            genome.id = f"random_{i}"
            
            # 2. Evaluate
            metrics = self.evaluator.evaluate(genome, epochs=fidelity)
            genome.metrics.update(metrics)
            
            # Calculate fitness (accuracy / cost)
            acc = metrics.get("accuracy", 0)
            cost = metrics.get("compute_cost", 1)
            fitness = acc / (1 + 0.1 * cost)
            genome.metrics["fitness_score"] = fitness
            
            if fitness > best_fitness:
                best_fitness = fitness
                logger.info(f"New best random architecture found: {fitness:.4f}")
            
            results.append({
                "sample": i,
                "fitness": fitness,
                "accuracy": acc,
                "params": metrics.get("param_count", 0)
            })
            
        logger.info(f"Random Search complete. Best fitness: {best_fitness:.4f}")
        return results
