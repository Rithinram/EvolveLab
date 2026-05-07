"""
EvolveLab — Meta-Prompt Evolution Agent
Evolves the prompting/reasoning strategies used by Builder Agents.
Tracks prompt fitness, performs prompt mutation and crossover.
This is a KEY NOVELTY FEATURE of the system.
"""

import random
import logging
from typing import List, Dict, Optional
from memory.memory_store import MemoryStore

logger = logging.getLogger("evolvelab.meta_prompt")

# Prompt style vocabulary
STYLES = ["balanced", "aggressive", "conservative", "experimental", "deep_attention", "minimal"]
FOCUSES = ["accuracy", "efficiency", "novelty", "cost", "attention_depth", "balanced"]


class MetaPromptAgent:
    """Evolves prompts used by builder agents over generations."""

    def __init__(self, config: dict = None):
        self.prompt_history: Dict[str, List[dict]] = {}  # agent_id -> prompt history
        self.prompt_fitness: Dict[str, List[float]] = {}  # agent_id -> fitness values
        self.generation_count = 0
        logger.info("Meta-prompt evolution agent initialized")

    def evolve_prompts(self, agents, generation: int,
                       generation_fitness: Dict[str, float],
                       memory: MemoryStore = None) -> List[dict]:
        """Evolve prompt strategies for all builder agents.
        
        Args:
            agents: List of BuilderAgent instances
            generation: Current generation number
            generation_fitness: Dict mapping agent species -> avg fitness of their genomes
            memory: Memory store for tracking
            
        Returns:
            List of prompt records for DB storage
        """
        self.generation_count = generation
        records = []

        for agent in agents:
            agent_id = agent.id
            species = agent.species

            # Get this agent's genome fitness for this generation
            agent_fitness = generation_fitness.get(species, 0.0)

            # Initialize history
            if agent_id not in self.prompt_history:
                self.prompt_history[agent_id] = []
                self.prompt_fitness[agent_id] = []

            # Record current prompt fitness
            self.prompt_fitness[agent_id].append(agent_fitness)

            # Decide mutation strategy
            old_strategy = dict(agent.prompt_strategy)
            mutation_type = self._select_prompt_mutation(agent_id)

            # Apply prompt mutation
            new_strategy = self._mutate_prompt(old_strategy, mutation_type, agent_fitness)

            # Update agent's prompt strategy
            agent.prompt_strategy = new_strategy

            # Track in memory
            if memory:
                memory.record_prompt_fitness(agent_id, agent_fitness, new_strategy)

            # Create record
            parent_id = None
            if self.prompt_history[agent_id]:
                parent_id = self.prompt_history[agent_id][-1].get("id")

            record = {
                "agent_id": agent_id,
                "generation_number": generation,
                "prompt_text": self._strategy_to_text(new_strategy, species),
                "prompt_fitness": agent_fitness,
                "parent_prompt_id": parent_id,
                "mutation_type": mutation_type,
                "strategy_params": new_strategy,
            }
            records.append(record)
            self.prompt_history[agent_id].append(record)

            logger.debug(
                "Agent %s prompt evolved: %s (fitness=%.3f, mutation=%s)",
                species, new_strategy.get("style"), agent_fitness, mutation_type
            )

        logger.info("Evolved prompts for %d agents in generation %d", len(agents), generation)
        return records

    def _select_prompt_mutation(self, agent_id: str) -> str:
        """Select prompt mutation type based on fitness trajectory."""
        history = self.prompt_fitness.get(agent_id, [])

        if len(history) < 2:
            return "tweak"

        # Check if fitness is improving
        recent = history[-3:] if len(history) >= 3 else history
        improving = all(recent[i] <= recent[i+1] for i in range(len(recent)-1))
        declining = all(recent[i] >= recent[i+1] for i in range(len(recent)-1))

        if improving:
            # Keep momentum, small tweaks
            return random.choice(["tweak", "tweak", "refine"])
        elif declining:
            # Need change, bigger mutations
            return random.choice(["radical", "crossover", "reset"])
        else:
            # Oscillating, try something new
            return random.choice(["tweak", "radical", "crossover"])

    def _mutate_prompt(self, strategy: dict, mutation_type: str,
                       fitness: float) -> dict:
        """Apply a prompt mutation."""
        new = dict(strategy)

        if mutation_type == "tweak":
            # Small adjustments
            if "creativity" in new:
                delta = random.gauss(0, 0.05)
                new["creativity"] = round(max(0.1, min(0.95, new["creativity"] + delta)), 2)
            if "depth_preference" in new:
                delta = random.gauss(0, 0.05)
                new["depth_preference"] = round(max(0.1, min(0.9, new.get("depth_preference", 0.5) + delta)), 2)
            if "exploration_rate" in new:
                # Decay exploration over time
                new["exploration_rate"] = round(max(0.1, new.get("exploration_rate", 0.5) * 0.95), 2)

        elif mutation_type == "radical":
            # Major changes
            new["style"] = random.choice(STYLES)
            new["focus"] = random.choice(FOCUSES)
            new["creativity"] = round(random.uniform(0.2, 0.9), 2)

        elif mutation_type == "crossover":
            # Cross with a random style
            new["style"] = random.choice(STYLES)
            new["creativity"] = round((new.get("creativity", 0.5) + random.uniform(0.3, 0.7)) / 2, 2)

        elif mutation_type == "refine":
            # Intensify current focus
            if fitness > 0.5:
                new["creativity"] = round(max(0.1, new.get("creativity", 0.5) - 0.05), 2)
            else:
                new["creativity"] = round(min(0.9, new.get("creativity", 0.5) + 0.05), 2)

        elif mutation_type == "reset":
            # Reset to defaults
            new = {
                "style": "balanced",
                "focus": "accuracy",
                "creativity": 0.5,
                "depth_preference": 0.5,
                "width_preference": 0.5,
                "exploration_rate": 0.5,
            }

        return new

    def _strategy_to_text(self, strategy: dict, species: str) -> str:
        """Convert strategy params to a human-readable prompt description."""
        style = strategy.get("style", "balanced")
        focus = strategy.get("focus", "accuracy")
        creativity = strategy.get("creativity", 0.5)

        creative_desc = "cautious" if creativity < 0.3 else "balanced" if creativity < 0.6 else "creative" if creativity < 0.8 else "highly experimental"

        return (
            f"As a {species} agent with {style} style, "
            f"focus on {focus} with {creative_desc} architecture generation. "
            f"Creativity: {creativity:.2f}, "
            f"Depth preference: {strategy.get('depth_preference', 0.5):.2f}, "
            f"Exploration: {strategy.get('exploration_rate', 0.5):.2f}."
        )

    def get_prompt_lineage(self, agent_id: str) -> List[dict]:
        """Get the full prompt evolution history for an agent."""
        return self.prompt_history.get(agent_id, [])
