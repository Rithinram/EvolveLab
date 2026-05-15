"""
EvolveLab — Selection Agent
Tournament selection with elitism and diversity preservation.
"""

import random
import logging
from typing import List, Set
from evolution.genome import Genome

logger = logging.getLogger("evolvelab.selection")


class SelectionAgent:
    """Selects survivors using tournament selection + elitism."""

    def __init__(self, config: dict = None):
        evo_cfg = (config or {}).get("evolution", {})
        self.elite_count = evo_cfg.get("elite_count", 2)
        self.tournament_size = evo_cfg.get("tournament_size", 3)
        self.diversity_threshold = evo_cfg.get("diversity_threshold", 0.4)
        logger.info(
            "Selection agent: elites=%d, tournament=%d",
            self.elite_count, self.tournament_size
        )

    def select(self, genomes: List[Genome], num_parents: int) -> List[Genome]:
        """Select parents for reproduction."""
        if not genomes:
            return []

        # Sort by fitness
        ranked = sorted(
            genomes,
            key=lambda g: g.metrics.get("fitness_score") or -999,
            reverse=True
        )

        selected = []
        selected_ids: Set[str] = set()

        # Elitism: top N always survive
        for elite in ranked[:self.elite_count]:
            elite.is_elite = True
            elite.survived = True
            selected.append(elite)
            selected_ids.add(elite.id)

        # Tournament selection for remaining slots
        remaining = num_parents - len(selected)
        attempts = 0
        max_attempts = remaining * 5

        while len(selected) < num_parents and attempts < max_attempts:
            attempts += 1
            winner = self._tournament(genomes)
            if winner and winner.id not in selected_ids:
                winner.survived = True
                selected.append(winner)
                selected_ids.add(winner.id)

        # Diversity injection: if selection is too homogeneous, add a random genome
        if self._needs_diversity(selected) and len(genomes) > len(selected):
            non_selected = [g for g in genomes if g.id not in selected_ids]
            if non_selected:
                # Ensure we don't replace an elite
                if len(selected) > self.elite_count:
                    diverse = random.choice(non_selected)
                    diverse.survived = True
                    selected[-1] = diverse  # Replace weakest NON-ELITE selection
                else:
                    # If we only have elites, just add the diverse one as an extra parent 
                    # if there's room, or skip
                    pass

        logger.info(
            "Selected %d parents from %d genomes. Elites: %d",
            len(selected), len(genomes), self.elite_count
        )

        return selected

    def _tournament(self, genomes: List[Genome]) -> Genome:
        """Run a tournament and return the winner."""
        size = min(self.tournament_size, len(genomes))
        contestants = random.sample(genomes, size)
        return max(
            contestants,
            key=lambda g: g.metrics.get("fitness_score") or -999
        )

    def _needs_diversity(self, selected: List[Genome]) -> bool:
        """Check if selected population lacks diversity."""
        if len(selected) <= 2:
            return False
        species = set(g.species for g in selected)
        return len(species) / len(selected) < self.diversity_threshold
