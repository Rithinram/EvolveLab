"""
EvolveLab — Builder Agents
Species-specialized agents that generate ML architecture genomes.
Each agent has a unique personality, strategy, and evolving prompt system.
"""

import uuid
import random
import logging
from typing import List, Dict, Optional, Any
from evolution.genome import (
    Genome, generate_random_architecture, generate_random_training_strategy,
    ARCHITECTURE_TYPES, OPTIMIZERS, SCHEDULERS
)

logger = logging.getLogger("evolvelab.builder")

# ── Curated Seed Architectures ──────────────────────────────────────
# Proven mini-architectures that give evolution a reliable starting point
# instead of fully random layer combos that may be structurally nonsensical.

SEED_ARCHITECTURES = {
    "transformer_specialist": [
        {   # Mini-ViT style
            "type": "transformer",
            "layers": [
                {"type": "conv2d", "filters": 64, "kernel": 3, "activation": "gelu"},
                {"type": "layer_norm"},
                {"type": "attention", "heads": 4, "dim": 64},
                {"type": "dropout", "rate": 0.1},
                {"type": "attention", "heads": 4, "dim": 64},
                {"type": "layer_norm"},
                {"type": "dense", "units": 128, "activation": "gelu"},
            ],
            "input_shape": [1, 28, 28], "output_classes": 10,
        },
    ],
    "efficient_architect": [
        {   # MobileNet-style depthwise separable
            "type": "efficient_net_style",
            "layers": [
                {"type": "conv2d", "filters": 32, "kernel": 3, "activation": "relu"},
                {"type": "batch_norm"},
                {"type": "depthwise_conv", "kernel": 3, "activation": "relu"},
                {"type": "batch_norm"},
                {"type": "pooling", "type_": "avg", "size": 2},
                {"type": "dense", "units": 64, "activation": "relu"},
            ],
            "input_shape": [1, 28, 28], "output_classes": 10,
        },
    ],
    "hybrid_innovator": [
        {   # CNN + Attention hybrid
            "type": "hybrid_transformer_cnn",
            "layers": [
                {"type": "conv2d", "filters": 64, "kernel": 3, "activation": "relu"},
                {"type": "batch_norm"},
                {"type": "conv2d", "filters": 128, "kernel": 3, "activation": "relu"},
                {"type": "attention", "heads": 4, "dim": 128},
                {"type": "dropout", "rate": 0.2},
                {"type": "dense", "units": 256, "activation": "gelu"},
                {"type": "dense", "units": 128, "activation": "relu"},
            ],
            "input_shape": [1, 28, 28], "output_classes": 10,
        },
    ],
    "accuracy_maximizer": [
        {   # Mini-ResNet style (deep CNN)
            "type": "residual_cnn",
            "layers": [
                {"type": "conv2d", "filters": 64, "kernel": 3, "activation": "relu"},
                {"type": "batch_norm"},
                {"type": "conv2d", "filters": 64, "kernel": 3, "activation": "relu"},
                {"type": "batch_norm"},
                {"type": "conv2d", "filters": 128, "kernel": 3, "activation": "relu"},
                {"type": "batch_norm"},
                {"type": "dropout", "rate": 0.3},
                {"type": "dense", "units": 512, "activation": "relu"},
                {"type": "dropout", "rate": 0.3},
                {"type": "dense", "units": 256, "activation": "relu"},
            ],
            "input_shape": [1, 28, 28], "output_classes": 10,
        },
    ],
    "cost_minimizer": [
        {   # Minimal MLP baseline
            "type": "mlp",
            "layers": [
                {"type": "dense", "units": 128, "activation": "relu"},
                {"type": "dropout", "rate": 0.2},
                {"type": "dense", "units": 64, "activation": "relu"},
            ],
            "input_shape": [1, 28, 28], "output_classes": 10,
        },
        {   # Tiny CNN
            "type": "cnn",
            "layers": [
                {"type": "conv2d", "filters": 16, "kernel": 3, "activation": "relu"},
                {"type": "pooling", "type_": "max", "size": 2},
                {"type": "dense", "units": 64, "activation": "relu"},
            ],
            "input_shape": [1, 28, 28], "output_classes": 10,
        },
    ],
}


class BuilderAgent:
    """A species-specialized builder agent with memory and prompt evolution."""

    def __init__(self, species_config: dict):
        self.id = str(uuid.uuid4())
        self.name = species_config.get("name", "unknown")
        self.species = self.name
        self.personality = species_config.get("personality", "")
        self.focus = species_config.get("focus", "balanced")
        self.creativity = species_config.get("creativity", 0.5)

        # Performance tracking
        self.generations_active = 0
        self.total_genomes_created = 0
        self.fitness_history: List[float] = []
        self.best_fitness = 0.0
        self.survival_count = 0
        self.total_produced = 0

        # Prompt strategy (evolves over time)
        self.prompt_strategy = {
            "style": self._initial_style(),
            "focus": self.focus,
            "creativity": self.creativity,
            "depth_preference": random.uniform(0.3, 0.8),
            "width_preference": random.uniform(0.3, 0.8),
            "exploration_rate": 0.5,
        }

        # Memory of what worked
        self.memory: Dict[str, Any] = {
            "best_architectures": [],
            "preferred_layers": [],
            "preferred_optimizers": [],
            "successful_patterns": [],
        }

        logger.info("Builder agent initialized: %s (%s)", self.name, self.personality[:50])

    def _initial_style(self) -> str:
        styles = {
            "transformer_specialist": "deep_attention",
            "efficient_architect": "minimal",
            "hybrid_innovator": "experimental",
            "accuracy_maximizer": "aggressive",
            "cost_minimizer": "conservative",
        }
        return styles.get(self.name, "balanced")

    def generate(self, generation: int, count: int = 1,
                 memory_store=None) -> List[Genome]:
        """Generate genomes using species-specific strategy and memory."""
        genomes = []
        for _ in range(count):
            genome = self._create_genome(generation, memory_store)
            genomes.append(genome)
            self.total_genomes_created += 1

        self.generations_active += 1
        logger.info(
            "Agent %s generated %d genomes for gen %d",
            self.name, len(genomes), generation
        )
        return genomes

    def _create_genome(self, generation: int, memory_store=None) -> Genome:
        """Create a single genome guided by personality and memory."""
        genome = Genome(species=self.species, generation=generation)

        # Gen 0: 40% chance to use a curated seed architecture for a
        # reliable fitness floor; later generations always randomize so
        # evolution has room to explore.
        use_seed = (
            generation == 0
            and random.random() < 0.4
            and self.species in SEED_ARCHITECTURES
        )

        if use_seed:
            import copy
            arch = copy.deepcopy(random.choice(SEED_ARCHITECTURES[self.species]))
            logger.debug("Agent %s using seed architecture (%s)", self.name, arch["type"])
        else:
            # Generate architecture based on species preferences
            arch = generate_random_architecture(
                species=self.species,
                min_layers=self._preferred_min_layers(),
                max_layers=self._preferred_max_layers(),
            )

        # Apply memory-guided adjustments
        if memory_store and random.random() < 0.4:
            recommended = memory_store.get_recommended_architecture()
            if recommended:
                arch = self._blend_with_recommendation(arch, recommended)

        genome.architecture = arch
        genome.training_strategy = self._create_training_strategy()
        genome.prompt_strategy = dict(self.prompt_strategy)

        # Estimate params
        genome.estimate_param_count()

        return genome

    def _preferred_min_layers(self) -> int:
        prefs = {
            "transformer_specialist": 3,
            "efficient_architect": 2,
            "hybrid_innovator": 3,
            "accuracy_maximizer": 4,
            "cost_minimizer": 1,
        }
        return prefs.get(self.name, 2)

    def _preferred_max_layers(self) -> int:
        prefs = {
            "transformer_specialist": 8,
            "efficient_architect": 5,
            "hybrid_innovator": 10,
            "accuracy_maximizer": 12,
            "cost_minimizer": 4,
        }
        return prefs.get(self.name, 8)

    def _create_training_strategy(self) -> dict:
        """Species-specific training strategy."""
        strategy = generate_random_training_strategy()

        # Apply species preferences
        if self.name == "accuracy_maximizer":
            strategy["optimizer"] = random.choice(["adam", "adamw"])
            strategy["lr"] = random.choice([0.0005, 0.001])
            strategy["epochs"] = random.choice([8, 10])
        elif self.name == "cost_minimizer":
            strategy["batch_size"] = random.choice([128, 256])
            strategy["epochs"] = random.choice([3, 5])
        elif self.name == "efficient_architect":
            strategy["optimizer"] = "adamw"
            strategy["weight_decay"] = random.choice([0.001, 0.01])
        elif self.name == "transformer_specialist":
            strategy["scheduler"] = "cosine"
            strategy["lr"] = random.choice([0.0001, 0.0005])

        return strategy

    def _blend_with_recommendation(self, arch: dict, recommended: dict) -> dict:
        """Blend current architecture with a recommended pattern."""
        if recommended.get("optimizer"):
            # 30% chance to adopt recommended optimizer
            pass  # Handled in training strategy
        if recommended.get("layer_types") and random.random() < 0.3:
            # Insert a recommended layer type
            from evolution.genome import generate_random_layer
            rec_type = random.choice(recommended["layer_types"])
            arch["layers"].insert(
                random.randint(0, len(arch["layers"])),
                generate_random_layer(rec_type)
            )
        return arch

    def update_from_results(self, generation_fitness: List[float],
                            best_genome_dict: dict = None):
        """Update agent memory based on generation results."""
        if generation_fitness:
            avg = sum(generation_fitness) / len(generation_fitness)
            self.fitness_history.append(avg)
            if max(generation_fitness) > self.best_fitness:
                self.best_fitness = max(generation_fitness)

        if best_genome_dict:
            self.memory["best_architectures"].append({
                "type": best_genome_dict.get("architecture", {}).get("type"),
                "fitness": best_genome_dict.get("fitness_score", 0),
            })
            # Keep last 10
            self.memory["best_architectures"] = self.memory["best_architectures"][-10:]

    def to_dict(self) -> dict:
        """Serialize agent state for DB storage."""
        return {
            "id": self.id,
            "name": self.name,
            "species": self.species,
            "personality": self.personality,
            "focus": self.focus,
            "creativity": self.creativity,
            "generations_active": self.generations_active,
            "total_genomes_created": self.total_genomes_created,
            "best_fitness_achieved": self.best_fitness,
            "avg_fitness": sum(self.fitness_history) / len(self.fitness_history) if self.fitness_history else 0,
            "survival_rate": self.survival_count / max(self.total_produced, 1),
            "memory": self.memory,
            "prompt_strategy": self.prompt_strategy,
            "performance_history": [{"gen": i, "fitness": f} for i, f in enumerate(self.fitness_history)],
        }


def create_builder_agents(config: dict) -> List[BuilderAgent]:
    """Create all builder agents from config."""
    species_list = config.get("builder_species", [])
    agents = []
    for sp in species_list:
        agents.append(BuilderAgent(sp))
    logger.info("Created %d builder agents", len(agents))
    return agents
