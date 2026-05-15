"""
EvolveLab — Memory Store
Historical memory system that agents use to improve future decisions.
Tracks mutation success rates, best patterns, and species statistics.
"""

import logging
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque

logger = logging.getLogger("evolvelab.memory")


class MemoryStore:
    """Persistent evolutionary memory for the agent system."""

    # Rolling window caps to prevent unbounded memory growth
    MAX_HISTORY = 50
    MAX_SURVIVAL_HISTORY = 100

    def __init__(self):
        # Mutation type success tracking
        self.mutation_successes: Dict[str, int] = defaultdict(int)
        self.mutation_attempts: Dict[str, int] = defaultdict(int)
        self.mutation_deltas: Dict[str, deque] = defaultdict(lambda: deque(maxlen=MemoryStore.MAX_HISTORY))

        # Best architecture patterns
        self.best_patterns: List[dict] = []
        self.max_patterns = 20

        # Species performance tracking
        self.species_fitness: Dict[str, deque] = defaultdict(lambda: deque(maxlen=MemoryStore.MAX_HISTORY))
        self.species_survival: Dict[str, deque] = defaultdict(lambda: deque(maxlen=MemoryStore.MAX_SURVIVAL_HISTORY))

        # Prompt success tracking
        self.prompt_fitness_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=MemoryStore.MAX_HISTORY))
        self.best_prompts: Dict[str, dict] = {}

        # Generation-level statistics
        self.generation_stats: deque = deque(maxlen=MemoryStore.MAX_HISTORY)

        logger.info("Memory store initialized")

    def record_mutation(self, mutation_type: str, fitness_before: float,
                        fitness_after: float):
        """Record a mutation outcome for adaptive rates."""
        self.mutation_attempts[mutation_type] += 1
        delta = fitness_after - fitness_before
        self.mutation_deltas[mutation_type].append(delta)

        if delta > 0:
            self.mutation_successes[mutation_type] += 1

    def get_mutation_success_rate(self, mutation_type: str) -> float:
        """Get historical success rate for a mutation type."""
        attempts = self.mutation_attempts.get(mutation_type, 0)
        if attempts == 0:
            return 0.5  # Prior
        return self.mutation_successes.get(mutation_type, 0) / attempts

    def get_mutation_weights(self) -> Dict[str, float]:
        """Get adaptive weights for mutation type selection."""
        weights = {}
        all_types = set(self.mutation_attempts.keys())
        if not all_types:
            return {}

        for mt in all_types:
            rate = self.get_mutation_success_rate(mt)
            avg_delta = 0.0
            deltas = self.mutation_deltas.get(mt, [])
            if deltas:
                avg_delta = sum(deltas[-20:]) / len(deltas[-20:])

            # Weight = success_rate * (1 + avg_delta)
            weights[mt] = max(0.05, rate * (1.0 + max(0, avg_delta * 5)))

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def record_best_pattern(self, genome_dict: dict):
        """Store a high-performing architecture pattern."""
        pattern = {
            "architecture_type": genome_dict.get("architecture", {}).get("type"),
            "layer_types": [l.get("type") for l in genome_dict.get("architecture", {}).get("layers", [])],
            "optimizer": genome_dict.get("training_strategy", {}).get("optimizer"),
            "lr": genome_dict.get("training_strategy", {}).get("lr"),
            "fitness": genome_dict.get("fitness_score", 0),
            "species": genome_dict.get("species"),
        }
        self.best_patterns.append(pattern)
        # Keep only top patterns
        self.best_patterns.sort(key=lambda p: p.get("fitness", 0), reverse=True)
        self.best_patterns = self.best_patterns[:self.max_patterns]

    def record_species_performance(self, species: str, fitness: float, survived: bool):
        """Track species-level performance."""
        self.species_fitness[species].append(fitness)
        self.species_survival[species].append(survived)

    def get_species_stats(self) -> Dict[str, dict]:
        """Get aggregated species statistics."""
        stats = {}
        for species in self.species_fitness:
            fitnesses = self.species_fitness[species]
            survivals = self.species_survival.get(species, [])
            stats[species] = {
                "avg_fitness": sum(fitnesses) / len(fitnesses) if fitnesses else 0,
                "max_fitness": max(fitnesses) if fitnesses else 0,
                "survival_rate": sum(survivals) / len(survivals) if survivals else 0,
                "total_genomes": len(fitnesses),
            }
        return stats

    def record_prompt_fitness(self, agent_id: str, fitness: float, prompt_data: dict):
        """Track prompt performance for an agent."""
        self.prompt_fitness_history[agent_id].append(fitness)
        current_best = self.best_prompts.get(agent_id, {}).get("fitness", -999)
        if fitness > current_best:
            self.best_prompts[agent_id] = {**prompt_data, "fitness": fitness}

    def get_best_prompt(self, agent_id: str) -> Optional[dict]:
        """Get the best-performing prompt for an agent."""
        return self.best_prompts.get(agent_id)

    def record_generation_stats(self, stats: dict):
        """Store generation-level statistics."""
        self.generation_stats.append(stats)

    def get_recommended_architecture(self) -> Optional[dict]:
        """Recommend architecture patterns based on best performers."""
        if not self.best_patterns:
            return None
        return self.best_patterns[0]

    def to_dict(self) -> dict:
        """Serialize memory state."""
        return {
            "mutation_successes": dict(self.mutation_successes),
            "mutation_attempts": dict(self.mutation_attempts),
            "best_patterns_count": len(self.best_patterns),
            "species_stats": self.get_species_stats(),
            "generation_count": len(self.generation_stats),
        }
