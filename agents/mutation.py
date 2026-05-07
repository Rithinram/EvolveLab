"""
EvolveLab — Mutation Agent
Applies adaptive mutations to genomes with historical success tracking.
Mutation rates evolve dynamically based on what has worked.
"""

import uuid
import copy
import random
import logging
from typing import List, Dict, Optional, Tuple
from evolution.genome import Genome, generate_random_layer, OPTIMIZERS, SCHEDULERS
from memory.memory_store import MemoryStore

logger = logging.getLogger("evolvelab.mutation")

MUTATION_TYPES = [
    "layer_addition", "layer_removal", "layer_modification",
    "optimizer_change", "lr_perturbation", "dropout_change",
    "architecture_resize", "training_strategy",
]


class MutationAgent:
    """Applies adaptive mutations guided by historical success."""

    def __init__(self, config: dict = None):
        mut_cfg = (config or {}).get("mutation", {})
        self.base_rate = mut_cfg.get("base_rate", 0.3)
        self.min_rate = mut_cfg.get("min_rate", 0.05)
        self.max_rate = mut_cfg.get("max_rate", 0.8)
        self.adaptive = mut_cfg.get("adaptive", True)
        self.decay_factor = mut_cfg.get("decay_factor", 0.95)

        # Default type weights
        self.type_weights = mut_cfg.get("types", {
            "layer_addition": 0.15, "layer_removal": 0.10,
            "layer_modification": 0.20, "optimizer_change": 0.10,
            "lr_perturbation": 0.20, "dropout_change": 0.10,
            "architecture_resize": 0.10, "training_strategy": 0.05,
        })

        self.current_rate = self.base_rate
        self.total_mutations = 0
        self.mutations_log: List[dict] = []

        logger.info("Mutation agent: rate=%.2f, adaptive=%s", self.base_rate, self.adaptive)

    def mutate(self, genome: Genome, memory: MemoryStore = None,
               generation: int = 0) -> Tuple[Genome, List[dict]]:
        """Apply mutations to a genome. Returns mutated genome and mutation records."""
        mutant = genome.clone(new_id=True)
        mutant.generation = generation
        mutant.lineage["parent_a"] = genome.id
        mutant.lineage["creation_method"] = "mutation"

        records = []

        # Determine mutation rate (adaptive)
        rate = self._get_adaptive_rate(memory)

        # Decide how many mutations (at least 1)
        num_mutations = 1
        while random.random() < rate and num_mutations < 4:
            num_mutations += 1

        # Select mutation types using adaptive weights
        weights = self._get_adaptive_weights(memory)
        mutation_types = self._select_mutation_types(num_mutations, weights)

        for mt in mutation_types:
            record = self._apply_mutation(mutant, mt)
            if record:
                record["genome_id"] = mutant.id
                record["generation_number"] = generation
                record["fitness_before"] = genome.metrics.get("fitness_score")
                records.append(record)
                mutant.mutation_metadata["mutations_applied"].append(mt)

        mutant.mutation_metadata["mutation_rate"] = rate
        mutant.estimate_param_count()
        self.total_mutations += len(records)

        return mutant, records

    def _get_adaptive_rate(self, memory: MemoryStore = None) -> float:
        """Calculate adaptive mutation rate based on history."""
        rate = self.current_rate

        if self.adaptive and memory:
            # If mutations are succeeding, keep rate; if failing, increase rate
            all_attempts = sum(memory.mutation_attempts.values())
            all_successes = sum(memory.mutation_successes.values())

            if all_attempts > 10:
                success_rate = all_successes / all_attempts
                if success_rate > 0.4:
                    rate *= self.decay_factor  # Converging, reduce mutations
                elif success_rate < 0.2:
                    rate *= 1.1  # Struggling, increase mutations

        rate = max(self.min_rate, min(self.max_rate, rate))
        self.current_rate = rate
        return rate

    def _get_adaptive_weights(self, memory: MemoryStore = None) -> Dict[str, float]:
        """Get adaptive mutation type weights from memory."""
        if not self.adaptive or not memory:
            return dict(self.type_weights)

        learned = memory.get_mutation_weights()
        if not learned:
            return dict(self.type_weights)

        # Blend default weights with learned weights
        blended = {}
        for mt in MUTATION_TYPES:
            default_w = self.type_weights.get(mt, 0.1)
            learned_w = learned.get(mt, default_w)
            blended[mt] = 0.5 * default_w + 0.5 * learned_w

        # Normalize
        total = sum(blended.values())
        if total > 0:
            blended = {k: v / total for k, v in blended.items()}

        return blended

    def _select_mutation_types(self, count: int, weights: Dict[str, float]) -> List[str]:
        """Select mutation types using weighted random selection."""
        types = list(weights.keys())
        w = [weights.get(t, 0.1) for t in types]
        selected = random.choices(types, weights=w, k=count)
        return selected

    def _apply_mutation(self, genome: Genome, mutation_type: str) -> Optional[dict]:
        """Apply a specific mutation type to a genome."""
        layers = genome.architecture.get("layers", [])
        training = genome.training_strategy
        record = {"mutation_type": mutation_type}

        if mutation_type == "layer_addition":
            new_layer = generate_random_layer()
            pos = random.randint(0, len(layers))
            layers.insert(pos, new_layer)
            record["field_changed"] = f"layers[{pos}]"
            record["old_value"] = "none"
            record["new_value"] = str(new_layer)

        elif mutation_type == "layer_removal" and len(layers) > 2:
            idx = random.randint(0, len(layers) - 1)
            removed = layers.pop(idx)
            record["field_changed"] = f"layers[{idx}]"
            record["old_value"] = str(removed)
            record["new_value"] = "removed"

        elif mutation_type == "layer_modification" and layers:
            idx = random.randint(0, len(layers) - 1)
            layer = layers[idx]
            old = str(layer)
            self._modify_layer(layer)
            record["field_changed"] = f"layers[{idx}]"
            record["old_value"] = old
            record["new_value"] = str(layer)

        elif mutation_type == "optimizer_change":
            old_opt = training.get("optimizer", "adam")
            new_opt = random.choice([o for o in OPTIMIZERS if o != old_opt])
            training["optimizer"] = new_opt
            record["field_changed"] = "optimizer"
            record["old_value"] = old_opt
            record["new_value"] = new_opt

        elif mutation_type == "lr_perturbation":
            old_lr = training.get("lr", 0.001)
            factor = random.choice([0.5, 0.7, 1.5, 2.0, 3.0])
            new_lr = round(max(0.00001, min(0.1, old_lr * factor)), 6)
            training["lr"] = new_lr
            record["field_changed"] = "lr"
            record["old_value"] = str(old_lr)
            record["new_value"] = str(new_lr)

        elif mutation_type == "dropout_change" and layers:
            dropout_layers = [i for i, l in enumerate(layers) if l.get("type") == "dropout"]
            if dropout_layers:
                idx = random.choice(dropout_layers)
                old_rate = layers[idx].get("rate", 0.3)
                new_rate = round(max(0.05, min(0.7, old_rate + random.gauss(0, 0.1))), 2)
                layers[idx]["rate"] = new_rate
                record["field_changed"] = f"layers[{idx}].rate"
                record["old_value"] = str(old_rate)
                record["new_value"] = str(new_rate)
            else:
                return None

        elif mutation_type == "architecture_resize":
            old_type = genome.architecture.get("type", "cnn")
            from evolution.genome import ARCHITECTURE_TYPES
            new_type = random.choice([t for t in ARCHITECTURE_TYPES if t != old_type])
            genome.architecture["type"] = new_type
            record["field_changed"] = "architecture.type"
            record["old_value"] = old_type
            record["new_value"] = new_type

        elif mutation_type == "training_strategy":
            old_sched = training.get("scheduler", "none")
            new_sched = random.choice([s for s in SCHEDULERS if s != old_sched])
            training["scheduler"] = new_sched
            record["field_changed"] = "scheduler"
            record["old_value"] = old_sched
            record["new_value"] = new_sched

        else:
            return None

        genome.architecture["layers"] = layers
        record["success"] = None  # Determined after evaluation
        return record

    def _modify_layer(self, layer: dict):
        """Modify parameters of an existing layer."""
        lt = layer.get("type", "")
        if lt == "conv2d":
            if random.random() < 0.5 and "filters" in layer:
                layer["filters"] = random.choice([16, 32, 64, 128, 256])
            elif "kernel" in layer:
                layer["kernel"] = random.choice([3, 5, 7])
        elif lt == "dense":
            if "units" in layer:
                layer["units"] = random.choice([64, 128, 256, 512, 1024])
        elif lt == "attention":
            if random.random() < 0.5 and "heads" in layer:
                layer["heads"] = random.choice([2, 4, 8])
            elif "dim" in layer:
                layer["dim"] = random.choice([32, 64, 128, 256])
