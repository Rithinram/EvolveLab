"""
EvolveLab — Heuristic Evaluation Engine
Estimates ML model performance from architecture structure without training.
Uses architecture analysis, known patterns, and noise to produce realistic scores.
"""

import math
import random
import logging
from typing import Dict, Any
from evolution.genome import Genome

logger = logging.getLogger("evolvelab.heuristic")


class HeuristicEvaluator:
    """Evaluates genomes using architecture analysis heuristics."""

    def __init__(self, config: dict = None):
        cfg = (config or {}).get("heuristic", {})
        self.base_accuracy_range = cfg.get("base_accuracy_range", [0.45, 0.98])
        self.attention_bonus = cfg.get("attention_bonus", 0.08)
        self.depth_bonus = cfg.get("depth_bonus_per_layer", 0.015)
        self.width_bonus = cfg.get("width_bonus_per_unit", 0.0001)
        self.dropout_penalty = cfg.get("dropout_penalty", 0.02)
        self.complexity_penalty = cfg.get("complexity_penalty", 0.005)
        self.noise_std = cfg.get("noise_std", 0.02)

    def evaluate(self, genome: Genome) -> Dict[str, float]:
        """Evaluate a genome and return performance metrics."""
        layers = genome.architecture.get("layers", [])
        training = genome.training_strategy

        # Start with base accuracy based on architecture type
        arch_type = genome.architecture.get("type", "cnn")
        base_accuracy = self._base_accuracy_for_type(arch_type)

        # Depth bonus (diminishing returns)
        depth = len(layers)
        depth_bonus = self.depth_bonus * min(depth, 10) * (1 - 0.05 * max(0, depth - 6))

        # Layer composition analysis
        attention_layers = sum(1 for l in layers if l.get("type") == "attention")
        conv_layers = sum(1 for l in layers if l.get("type") in ("conv2d", "depthwise_conv"))
        dense_layers = sum(1 for l in layers if l.get("type") == "dense")
        norm_layers = sum(1 for l in layers if l.get("type") in ("batch_norm", "layer_norm"))
        dropout_layers = sum(1 for l in layers if l.get("type") == "dropout")

        # Attention bonus
        attn_bonus = self.attention_bonus * min(attention_layers, 3)

        # Width bonus
        total_units = sum(
            l.get("units", 0) + l.get("filters", 0) + l.get("dim", 0)
            for l in layers
        )
        width_bonus = self.width_bonus * min(total_units, 2000)

        # Normalization bonus
        norm_bonus = 0.02 * min(norm_layers, 3)

        # Dropout penalty (too much hurts)
        d_penalty = self.dropout_penalty * max(0, dropout_layers - 2)

        # Training strategy bonus
        lr = training.get("lr", 0.001)
        lr_bonus = 0.0
        if 0.0003 <= lr <= 0.003:
            lr_bonus = 0.02  # Sweet spot
        elif lr > 0.01:
            lr_bonus = -0.03  # Too high

        optimizer = training.get("optimizer", "adam")
        opt_bonus = {"adam": 0.02, "adamw": 0.025, "sgd": 0.0, "rmsprop": 0.01, "adagrad": -0.01}.get(optimizer, 0)

        scheduler = training.get("scheduler", "none")
        sched_bonus = {"cosine": 0.015, "step": 0.01, "exponential": 0.005, "plateau": 0.01, "none": 0}.get(scheduler, 0)

        # Complexity penalty
        param_count = genome.estimate_param_count()
        complexity = math.log10(max(param_count, 1)) / 7  # Normalize to ~0-1
        comp_penalty = self.complexity_penalty * max(0, complexity - 0.5)

        # Calculate accuracy
        accuracy = (
            base_accuracy + depth_bonus + attn_bonus + width_bonus +
            norm_bonus - d_penalty + lr_bonus + opt_bonus + sched_bonus - comp_penalty
        )

        # Add realistic noise
        noise = random.gauss(0, self.noise_std)
        accuracy = max(0.1, min(0.99, accuracy + noise))

        # Compute cost (normalized 0-1)
        compute_cost = self._estimate_compute_cost(genome, param_count)

        # Complexity score
        complexity_score = min(1.0, depth / 15 + complexity * 0.5)

        # Inference speed (inverse of compute cost)
        inference_speed = max(0.01, 1.0 - compute_cost * 0.8)

        metrics = {
            "accuracy": round(accuracy, 4),
            "compute_cost": round(compute_cost, 4),
            "complexity": round(complexity_score, 4),
            "inference_speed": round(inference_speed, 4),
            "param_count": param_count,
        }

        # Update genome metrics
        genome.metrics.update(metrics)

        logger.debug(
            "Evaluated %s: acc=%.3f cost=%.3f params=%d",
            genome.id[:8], accuracy, compute_cost, param_count
        )

        return metrics

    def _base_accuracy_for_type(self, arch_type: str) -> float:
        """Return base accuracy for architecture type."""
        bases = {
            "cnn": 0.72,
            "transformer": 0.68,
            "hybrid_transformer_cnn": 0.75,
            "mlp": 0.60,
            "attention_net": 0.70,
            "residual_cnn": 0.74,
            "efficient_net_style": 0.73,
        }
        return bases.get(arch_type, 0.65)

    def _estimate_compute_cost(self, genome: Genome, param_count: int) -> float:
        """Estimate normalized compute cost."""
        # Base cost from parameters
        param_cost = min(1.0, param_count / 1_000_000)

        # Epochs multiplier
        epochs = genome.training_strategy.get("epochs", 5)
        epoch_cost = epochs / 20.0

        # Batch size efficiency (larger batches = faster)
        batch_size = genome.training_strategy.get("batch_size", 64)
        batch_eff = 1.0 - (batch_size / 256.0) * 0.3

        # Attention layers are expensive
        attention_count = sum(
            1 for l in genome.architecture.get("layers", [])
            if l.get("type") == "attention"
        )
        attn_cost = attention_count * 0.08

        cost = (param_cost * 0.4 + epoch_cost * 0.3 + attn_cost * 0.2) * batch_eff
        return max(0.01, min(1.0, cost))
