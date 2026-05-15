"""
EvolveLab — Evaluator Agent
Orchestrates genome evaluation using heuristic scoring.
"""

import logging
from typing import List, Dict, Optional, Any, Callable
from evolution.genome import Genome
from training.heuristic import HeuristicEvaluator
from training.pytorch_evaluator import PyTorchEvaluator

logger = logging.getLogger("evolvelab.evaluator_agent")


class EvaluatorAgent:
    """Evaluates genomes using either heuristic scoring or real PyTorch training."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.eval_cfg = self.config.get("evaluation", {})
        self.mode = self.eval_cfg.get("mode", "heuristic").lower()
        
        self.heuristic = HeuristicEvaluator(config)
        self.pytorch = PyTorchEvaluator(config)
        self.total_evaluated = 0
        self.on_batch_callback = None
        
        logger.info(f"Evaluator agent initialized in {self.mode} mode")

    def set_on_batch_callback(self, callback: Callable):
        self.on_batch_callback = callback

    def evaluate_population(self, genomes: List[Genome],
                            fidelity: int = None,
                            fitness_weights: dict = None) -> List[Genome]:
        """Evaluate all genomes in a population."""
        # Use provided weights or defaults from config
        weights = fitness_weights or self.config.get("fitness", {"accuracy_weight": 0.7, "cost_weight": 0.3})
        
        if self.mode == "pytorch":
            # Real training mode
            eval_fidelity = fidelity or self.eval_cfg.get("fidelity", 3)
            self.pytorch.evaluate_population(genomes, fidelity=eval_fidelity, on_batch=self.on_batch_callback)
        else:
            # Simulation / Heuristic mode
            acc_w = weights.get("accuracy_weight", 0.7)
            cost_w = weights.get("cost_weight", 0.3)

            for genome in genomes:
                metrics = self.heuristic.evaluate(genome)
                accuracy = metrics.get("accuracy", 0)
                cost = metrics.get("compute_cost", 0)
                
                fitness = (acc_w * accuracy) - (cost_w * cost)
                fitness = max(0.0, round(fitness, 4))
                
                genome.metrics.update(metrics)
                genome.metrics["fitness_score"] = fitness
                self.total_evaluated += 1

        logger.info(
            "Evaluated %d genomes. Best: %.4f, Avg: %.4f",
            len(genomes),
            max(g.metrics.get("fitness_score", 0) for g in genomes) if genomes else 0,
            sum(g.metrics.get("fitness_score", 0) for g in genomes) / len(genomes) if genomes else 0,
        )

        return genomes
