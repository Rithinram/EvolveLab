"""
EvolveLab — Crossover Agent
Combines two parent genomes to produce offspring through genetic crossover.
"""

import uuid
import random
import logging
from typing import List, Tuple
from evolution.genome import Genome

logger = logging.getLogger("evolvelab.crossover")


class CrossoverAgent:
    """Performs crossover between parent genomes."""

    def __init__(self, config: dict = None):
        cx_cfg = (config or {}).get("crossover", {})
        self.rate = cx_cfg.get("rate", 0.7)
        self.blend_ratio = cx_cfg.get("blend_ratio", 0.5)
        self.total_crossovers = 0
        logger.info("Crossover agent: rate=%.2f, blend=%.2f", self.rate, self.blend_ratio)

    def crossover(self, parent_a: Genome, parent_b: Genome,
                  generation: int = 0) -> Genome:
        """Produce a child genome from two parents."""
        child = Genome(
            species=self._inherit_species(parent_a, parent_b),
            generation=generation,
        )
        child.lineage = {
            "parent_a": parent_a.id,
            "parent_b": parent_b.id,
            "creation_method": "crossover",
        }

        # Layer crossover (single-point)
        child.architecture = self._crossover_architecture(parent_a, parent_b)

        # Training strategy blending
        child.training_strategy = self._blend_training(parent_a, parent_b)

        # Prompt strategy crossover
        child.prompt_strategy = self._crossover_prompts(parent_a, parent_b)

        child.estimate_param_count()
        self.total_crossovers += 1

        logger.debug(
            "Crossover: %s x %s -> %s (species: %s)",
            parent_a.id[:8], parent_b.id[:8], child.id[:8], child.species
        )

        return child

    def crossover_pairs(self, parents: List[Genome], num_children: int,
                        generation: int = 0) -> List[Genome]:
        """Generate children from random parent pairs.

        Always produces exactly num_children offspring. When the crossover
        probability roll fails, the fitter parent is cloned instead of
        producing nothing — this prevents the engine from filling gaps with
        random builder genomes that dilute evolutionary pressure.
        """
        children = []
        for _ in range(num_children):
            if len(parents) < 2:
                break
            p1, p2 = random.sample(parents, 2)
            if random.random() < self.rate:
                child = self.crossover(p1, p2, generation)
                children.append(child)
            else:
                # Clone fitter parent instead of producing nothing
                fitter = p1 if (p1.metrics.get("fitness_score") or 0) >= (p2.metrics.get("fitness_score") or 0) else p2
                clone = fitter.clone(new_id=True)
                clone.generation = generation
                clone.lineage["parent_a"] = fitter.id
                clone.lineage["creation_method"] = "clone"
                children.append(clone)

        logger.info("Produced %d children from %d parents", len(children), len(parents))
        return children

    def _crossover_architecture(self, pa: Genome, pb: Genome) -> dict:
        """Single-point crossover of layer lists."""
        layers_a = pa.architecture.get("layers", [])
        layers_b = pb.architecture.get("layers", [])

        if not layers_a or not layers_b:
            return dict(pa.architecture) if layers_a else dict(pb.architecture)

        # Single point crossover
        point_a = random.randint(0, len(layers_a))
        point_b = random.randint(0, len(layers_b))

        child_layers = layers_a[:point_a] + layers_b[point_b:]

        # Limit layer count
        if len(child_layers) > 12:
            child_layers = child_layers[:12]
        if len(child_layers) < 1:
            # Guarantee at least one compute layer (not batch_norm/dropout alone)
            compute_types = {"conv2d", "dense"}
            fallback = None
            for src in (layers_a, layers_b):
                for l in src:
                    if l.get("type") in compute_types:
                        fallback = l
                        break
                if fallback:
                    break
            child_layers = [fallback] if fallback else [{"type": "conv2d", "params": {"filters": 32, "kernel": 3, "activation": "relu"}}]

        # Inherit architecture type from fitter parent
        fit_a = pa.metrics.get("fitness_score") or 0
        fit_b = pb.metrics.get("fitness_score") or 0
        arch_type = pa.architecture.get("type") if fit_a >= fit_b else pb.architecture.get("type")

        # Fix dimension mismatches at the splice boundary
        child_layers = self._validate_and_fix_dimensions(child_layers)

        return {
            "type": arch_type or "cnn",
            "layers": child_layers,
            "input_shape": pa.architecture.get("input_shape", [1, 28, 28]),
            "output_classes": pa.architecture.get("output_classes", 10),
        }

    def _blend_training(self, pa: Genome, pb: Genome) -> dict:
        """Blend training strategies from both parents."""
        ta = pa.training_strategy
        tb = pb.training_strategy
        r = self.blend_ratio

        # Blend numerical values
        lr = ta.get("lr", 0.001) * r + tb.get("lr", 0.001) * (1 - r)
        wd = ta.get("weight_decay", 0) * r + tb.get("weight_decay", 0) * (1 - r)
        batch = int(ta.get("batch_size", 64) * r + tb.get("batch_size", 64) * (1 - r))
        epochs = int(ta.get("epochs", 5) * r + tb.get("epochs", 5) * (1 - r))

        # Discrete choices from random parent
        optimizer = random.choice([ta.get("optimizer", "adam"), tb.get("optimizer", "adam")])
        scheduler = random.choice([ta.get("scheduler", "none"), tb.get("scheduler", "none")])

        return {
            "optimizer": optimizer,
            "lr": round(lr, 6),
            "scheduler": scheduler,
            "batch_size": max(16, batch),
            "epochs": max(1, epochs),
            "weight_decay": round(wd, 6),
        }

    def _crossover_prompts(self, pa: Genome, pb: Genome) -> dict:
        """Crossover prompt strategies."""
        psa = pa.prompt_strategy or {}
        psb = pb.prompt_strategy or {}
        r = self.blend_ratio

        return {
            "style": random.choice([psa.get("style", "balanced"), psb.get("style", "balanced")]),
            "focus": random.choice([psa.get("focus", "accuracy"), psb.get("focus", "accuracy")]),
            "creativity": round(psa.get("creativity", 0.5) * r + psb.get("creativity", 0.5) * (1 - r), 2),
        }

    def _inherit_species(self, pa: Genome, pb: Genome) -> str:
        """Inherit species from the fitter parent."""
        fit_a = pa.metrics.get("fitness_score") or 0
        fit_b = pb.metrics.get("fitness_score") or 0
        if fit_a >= fit_b:
            return pa.species
        return pb.species

    # ── Dimension Validation ─────────────────────────────────────────

    @staticmethod
    def _get_layer_output_dim(layer: dict) -> int:
        """Infer the output dimensionality of a layer (0 = passthrough/unknown)."""
        lt = layer.get("type", "")
        if lt == "conv2d":
            return layer.get("filters", 32)
        if lt == "depthwise_conv":
            return 0  # preserves input channels
        if lt == "dense":
            return layer.get("units", 128)
        if lt == "attention":
            return layer.get("dim", 64)
        # pooling, dropout, norm, flatten, residual — passthrough
        return 0

    @staticmethod
    def _get_layer_input_expectation(layer: dict) -> int:
        """Infer what input dim a layer expects (0 = any/flexible)."""
        lt = layer.get("type", "")
        if lt == "dense":
            return 0  # dense layers accept any flattened input
        if lt == "attention":
            return layer.get("dim", 64)  # expects matching embed dim
        if lt == "conv2d":
            return 0  # conv accepts any channel count
        return 0

    def _validate_and_fix_dimensions(self, layers: list) -> list:
        """Scan layer list for dimension mismatches and insert adapters.

        Detects transitions like conv2d(128 out) → attention(dim=64) where
        the output of one layer doesn't match the expected input of the next.
        Inserts a minimal adapter (dense projection) to bridge the gap.
        """
        if len(layers) < 2:
            return layers

        fixed = [layers[0]]
        for i in range(1, len(layers)):
            prev = fixed[-1]
            curr = layers[i]

            prev_out = self._get_layer_output_dim(prev)
            curr_in = self._get_layer_input_expectation(curr)

            # Only insert adapter when both dims are known AND they mismatch
            if prev_out > 0 and curr_in > 0 and prev_out != curr_in:
                adapter = {"type": "dense", "units": curr_in, "activation": "relu"}
                fixed.append(adapter)
                logger.debug(
                    "Inserted adapter dense(%d) between %s(%d) and %s(%d)",
                    curr_in, prev.get("type"), prev_out,
                    curr.get("type"), curr_in,
                )

            fixed.append(curr)

        return fixed
