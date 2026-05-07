"""
EvolveLab — Evaluator Agent
Orchestrates genome evaluation using heuristic scoring.
"""

import logging
from typing import List, Dict
from evolution.genome import Genome
from evaluator.heuristic import HeuristicEvaluator

logger = logging.getLogger("evolvelab.evaluator_agent")


class EvaluatorAgent:
    """Evaluates genomes using heuristic scoring model."""

    def __init__(self, config: dict = None):
        self.heuristic = HeuristicEvaluator(config)
        self.total_evaluated = 0
        logger.info("Evaluator agent initialized (heuristic mode)")

    def evaluate_population(self, genomes: List[Genome],
                            fitness_weights: dict = None) -> List[Genome]:
        """Evaluate all genomes in a population."""
        weights = fitness_weights or {"accuracy_weight": 0.7, "cost_weight": 0.3}
        acc_w = weights.get("accuracy_weight", 0.7)
        cost_w = weights.get("cost_weight", 0.3)

        for genome in genomes:
            metrics = self.heuristic.evaluate(genome)

            # Calculate composite fitness
            accuracy = metrics.get("accuracy", 0)
            cost = metrics.get("compute_cost", 0)
            fitness = (acc_w * accuracy) - (cost_w * cost)
            fitness = max(0.0, round(fitness, 4))

            genome.metrics["fitness_score"] = fitness
            self.total_evaluated += 1

        logger.info(
            "Evaluated %d genomes. Best: %.4f, Avg: %.4f",
            len(genomes),
            max(g.metrics.get("fitness_score", 0) for g in genomes) if genomes else 0,
            sum(g.metrics.get("fitness_score", 0) for g in genomes) / len(genomes) if genomes else 0,
        )

        return genomes
