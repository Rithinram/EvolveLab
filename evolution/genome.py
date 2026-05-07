"""
EvolveLab — Genome Representation
JSON-based genome system for ML architecture representation.
"""

import uuid
import copy
import random
import logging
from typing import List, Optional, Dict, Any

logger = logging.getLogger("evolvelab.genome")

# ── Layer Templates ─────────────────────────────────────────────────

LAYER_TYPES = {
    "conv2d": {"filters": [16, 32, 64, 128, 256], "kernel": [3, 5, 7], "activation": ["relu", "gelu", "silu"]},
    "dense": {"units": [64, 128, 256, 512, 1024], "activation": ["relu", "gelu", "silu"]},
    "attention": {"heads": [2, 4, 8], "dim": [32, 64, 128, 256]},
    "pooling": {"type": ["max", "avg"], "size": [2, 3]},
    "dropout": {"rate": [0.1, 0.2, 0.3, 0.4, 0.5]},
    "batch_norm": {},
    "layer_norm": {},
    "residual": {"inner_type": ["conv2d", "dense"]},
    "depthwise_conv": {"kernel": [3, 5], "activation": ["relu", "gelu"]},
    "flatten": {},
}

OPTIMIZERS = ["adam", "sgd", "adamw", "rmsprop", "adagrad"]
SCHEDULERS = ["cosine", "step", "exponential", "plateau", "none"]
ARCHITECTURE_TYPES = [
    "cnn", "transformer", "hybrid_transformer_cnn", "mlp",
    "attention_net", "residual_cnn", "efficient_net_style",
]


class Genome:
    """Represents a complete ML architecture genome."""

    def __init__(self, genome_id: str = None, species: str = "unknown",
                 generation: int = 0):
        self.id = genome_id or str(uuid.uuid4())
        self.species = species
        self.generation = generation
        self.architecture = {
            "type": "cnn",
            "layers": [],
            "input_shape": [1, 28, 28],
            "output_classes": 10,
        }
        self.training_strategy = {
            "optimizer": "adam",
            "lr": 0.001,
            "scheduler": "cosine",
            "batch_size": 64,
            "epochs": 5,
            "weight_decay": 0.0001,
        }
        self.prompt_strategy = {
            "style": "balanced",
            "focus": "accuracy",
            "creativity": 0.5,
        }
        self.lineage = {
            "parent_a": None,
            "parent_b": None,
            "creation_method": "generated",
        }
        self.mutation_metadata = {
            "mutations_applied": [],
            "mutation_rate": 0.3,
        }
        self.metrics = {
            "fitness_score": None,
            "accuracy": None,
            "compute_cost": None,
            "complexity": None,
            "inference_speed": None,
            "param_count": 0,
        }
        self.is_elite = False
        self.survived = False
        self.rank = None

    def to_dict(self) -> dict:
        """Serialize genome to dictionary for DB storage."""
        return {
            "id": self.id,
            "species": self.species,
            "generation_number": self.generation,
            "architecture": self.architecture,
            "training_strategy": self.training_strategy,
            "prompt_strategy": self.prompt_strategy,
            "fitness_score": self.metrics.get("fitness_score"),
            "accuracy": self.metrics.get("accuracy"),
            "compute_cost": self.metrics.get("compute_cost"),
            "complexity": self.metrics.get("complexity"),
            "inference_speed": self.metrics.get("inference_speed"),
            "param_count": self.metrics.get("param_count", 0),
            "parent_a_id": self.lineage.get("parent_a"),
            "parent_b_id": self.lineage.get("parent_b"),
            "creation_method": self.lineage.get("creation_method", "generated"),
            "mutation_history": self.mutation_metadata.get("mutations_applied", []),
            "is_elite": self.is_elite,
            "survived": self.survived,
            "rank": self.rank,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Genome":
        """Reconstruct genome from dictionary."""
        g = cls(
            genome_id=data.get("id"),
            species=data.get("species", "unknown"),
            generation=data.get("generation_number", 0),
        )
        g.architecture = data.get("architecture", g.architecture)
        g.training_strategy = data.get("training_strategy", g.training_strategy)
        g.prompt_strategy = data.get("prompt_strategy", g.prompt_strategy)
        g.lineage = {
            "parent_a": data.get("parent_a_id"),
            "parent_b": data.get("parent_b_id"),
            "creation_method": data.get("creation_method", "generated"),
        }
        g.mutation_metadata = {
            "mutations_applied": data.get("mutation_history", []),
            "mutation_rate": 0.3,
        }
        g.metrics = {
            "fitness_score": data.get("fitness_score"),
            "accuracy": data.get("accuracy"),
            "compute_cost": data.get("compute_cost"),
            "complexity": data.get("complexity"),
            "inference_speed": data.get("inference_speed"),
            "param_count": data.get("param_count", 0),
        }
        g.is_elite = data.get("is_elite", False)
        g.survived = data.get("survived", False)
        g.rank = data.get("rank")
        return g

    def clone(self, new_id: bool = True) -> "Genome":
        """Deep copy the genome with optional new ID."""
        g = copy.deepcopy(self)
        if new_id:
            g.id = str(uuid.uuid4())
        return g

    def estimate_param_count(self) -> int:
        """Estimate total parameter count from architecture."""
        total = 0
        prev_channels = self.architecture.get("input_shape", [1, 28, 28])[0]
        prev_units = 0

        for layer in self.architecture.get("layers", []):
            lt = layer.get("type", "")
            if lt == "conv2d":
                f = layer.get("filters", 32)
                k = layer.get("kernel", 3)
                total += prev_channels * f * k * k + f
                prev_channels = f
            elif lt == "dense":
                u = layer.get("units", 128)
                inp = prev_units if prev_units > 0 else prev_channels * 7 * 7
                total += inp * u + u
                prev_units = u
            elif lt == "attention":
                d = layer.get("dim", 64)
                h = layer.get("heads", 4)
                total += 4 * d * d * h
                prev_units = d
            elif lt == "depthwise_conv":
                k = layer.get("kernel", 3)
                total += prev_channels * k * k + prev_channels
            elif lt in ("batch_norm", "layer_norm"):
                total += prev_channels * 2 if prev_units == 0 else prev_units * 2

        # Output layer
        out_classes = self.architecture.get("output_classes", 10)
        if prev_units > 0:
            total += prev_units * out_classes + out_classes
        else:
            total += prev_channels * 7 * 7 * out_classes + out_classes

        self.metrics["param_count"] = total
        return total

    def get_depth(self) -> int:
        return len(self.architecture.get("layers", []))

    def get_layer_types(self) -> List[str]:
        return [l.get("type", "unknown") for l in self.architecture.get("layers", [])]


def generate_random_layer(layer_type: str = None) -> dict:
    """Generate a random layer configuration."""
    if layer_type is None:
        layer_type = random.choice(list(LAYER_TYPES.keys()))

    template = LAYER_TYPES.get(layer_type, {})
    layer = {"type": layer_type}

    for key, choices in template.items():
        if isinstance(choices, list):
            layer[key] = random.choice(choices)
        else:
            layer[key] = choices

    return layer


def generate_random_architecture(species: str = "unknown",
                                  min_layers: int = 2,
                                  max_layers: int = 8) -> dict:
    """Generate a random architecture based on species tendencies."""
    arch_type = random.choice(ARCHITECTURE_TYPES)
    num_layers = random.randint(min_layers, max_layers)
    layers = []

    # Species-specific layer preferences
    species_preferences = {
        "transformer_specialist": ["attention", "layer_norm", "dense", "dropout"],
        "efficient_architect": ["depthwise_conv", "pooling", "batch_norm", "dense"],
        "hybrid_innovator": ["conv2d", "attention", "dense", "pooling", "dropout"],
        "accuracy_maximizer": ["conv2d", "conv2d", "dense", "dense", "batch_norm", "dropout"],
        "cost_minimizer": ["conv2d", "pooling", "dense"],
    }

    preferred = species_preferences.get(species, list(LAYER_TYPES.keys()))

    for i in range(num_layers):
        lt = random.choice(preferred)
        layers.append(generate_random_layer(lt))

    # Ensure at least one dense layer at the end
    if not any(l["type"] == "dense" for l in layers):
        layers.append(generate_random_layer("dense"))

    return {
        "type": arch_type,
        "layers": layers,
        "input_shape": [1, 28, 28],
        "output_classes": 10,
    }


def generate_random_training_strategy() -> dict:
    """Generate a random training strategy."""
    return {
        "optimizer": random.choice(OPTIMIZERS),
        "lr": random.choice([0.0001, 0.0005, 0.001, 0.005, 0.01]),
        "scheduler": random.choice(SCHEDULERS),
        "batch_size": random.choice([16, 32, 64, 128]),
        "epochs": random.choice([3, 5, 8, 10]),
        "weight_decay": random.choice([0, 0.0001, 0.001, 0.01]),
    }
