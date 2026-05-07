"""
EvolveLab — Population Management
Manages the collection of genomes in the current generation.
"""

import logging
from typing import List, Optional, Dict
from collections import Counter
from evolution.genome import Genome

logger = logging.getLogger("evolvelab.population")


class Population:
    """Manages a population of genomes for a generation."""

    def __init__(self, generation: int = 0):
        self.generation = generation
        self.genomes: List[Genome] = []

    def add(self, genome: Genome):
        self.genomes.append(genome)

    def add_many(self, genomes: List[Genome]):
        self.genomes.extend(genomes)

    @property
    def size(self) -> int:
        return len(self.genomes)

    def ranked(self) -> List[Genome]:
        """Return genomes sorted by fitness (descending)."""
        return sorted(
            self.genomes,
            key=lambda g: g.metrics.get("fitness_score") or -999,
            reverse=True,
        )

    def best(self) -> Optional[Genome]:
        ranked = self.ranked()
        return ranked[0] if ranked else None

    def worst(self) -> Optional[Genome]:
        ranked = self.ranked()
        return ranked[-1] if ranked else None

    def avg_fitness(self) -> float:
        scores = [g.metrics.get("fitness_score", 0) for g in self.genomes
                   if g.metrics.get("fitness_score") is not None]
        return sum(scores) / len(scores) if scores else 0.0

    def avg_accuracy(self) -> float:
        scores = [g.metrics.get("accuracy", 0) for g in self.genomes
                   if g.metrics.get("accuracy") is not None]
        return sum(scores) / len(scores) if scores else 0.0

    def species_distribution(self) -> Dict[str, int]:
        return dict(Counter(g.species for g in self.genomes))

    def diversity_score(self) -> float:
        """Measure architectural diversity in the population."""
        if len(self.genomes) <= 1:
            return 0.0

        # Count unique architecture types, layer counts, optimizers
        arch_types = set()
        layer_counts = set()
        optimizers = set()
        for g in self.genomes:
            arch_types.add(g.architecture.get("type", ""))
            layer_counts.add(len(g.architecture.get("layers", [])))
            optimizers.add(g.training_strategy.get("optimizer", ""))

        max_diversity = 3.0  # Normalize across 3 dimensions
        diversity = (
            len(arch_types) / max(len(self.genomes), 1) +
            len(layer_counts) / max(len(self.genomes), 1) +
            len(optimizers) / max(len(self.genomes), 1)
        ) / max_diversity

        return min(diversity, 1.0)

    def mark_survivors(self, survivor_ids: set):
        """Mark genomes as survived for analytics."""
        for g in self.genomes:
            g.survived = g.id in survivor_ids

    def assign_ranks(self):
        """Assign rank based on fitness."""
        ranked = self.ranked()
        for i, g in enumerate(ranked):
            g.rank = i + 1

    def to_dicts(self, generation_id: int) -> List[dict]:
        """Convert all genomes to dicts for DB storage."""
        return [{**g.to_dict(), "generation_id": generation_id} for g in self.genomes]
