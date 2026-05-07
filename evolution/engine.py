"""
EvolveLab — Evolution Engine
Orchestrates the full evolutionary loop across all agents.
"""

import time
import logging
import random
from typing import Optional, Callable, List, Dict, Any
from collections import defaultdict

from utils.helpers import load_config
from evolution.genome import Genome
from evolution.population import Population
from agents.builder import BuilderAgent, create_builder_agents
from agents.evaluator import EvaluatorAgent
from agents.selection import SelectionAgent
from agents.mutation import MutationAgent
from agents.crossover import CrossoverAgent
from agents.meta_prompt import MetaPromptAgent
from memory.memory_store import MemoryStore
from database.crud import DatabaseManager

logger = logging.getLogger("evolvelab.engine")


class EvolutionEngine:
    """Orchestrates the complete evolutionary loop."""

    def __init__(self, config: dict = None, db: DatabaseManager = None):
        self.config = config or load_config()
        evo_cfg = self.config.get("evolution", {})
        self.population_size = evo_cfg.get("population_size", 8)
        self.max_generations = evo_cfg.get("max_generations", 20)
        self.elite_count = evo_cfg.get("elite_count", 2)
        self.checkpoint_interval = evo_cfg.get("checkpoint_interval", 5)

        self.fitness_config = self.config.get("fitness", {})

        # Database
        self.db = db or DatabaseManager(self.config.get("database", {}).get("url", "sqlite:///evolvelab.db"))

        # Agents
        self.builders = create_builder_agents(self.config)
        self.evaluator = EvaluatorAgent(self.config)
        self.selector = SelectionAgent(self.config)
        self.mutator = MutationAgent(self.config)
        self.crossover = CrossoverAgent(self.config)
        self.meta_prompt = MetaPromptAgent(self.config)

        # Memory
        self.memory = MemoryStore()

        # State
        self.current_generation = 0
        self.running = False
        self.paused = False
        self.best_genome: Optional[Genome] = None
        self.best_fitness = 0.0
        self.best_accuracy = 0.0
        self.history: List[dict] = []

        # Callback for WebSocket updates
        self._event_callback: Optional[Callable] = None

        logger.info(
            "Evolution engine initialized: pop=%d, gens=%d, elites=%d",
            self.population_size, self.max_generations, self.elite_count
        )

    def set_event_callback(self, callback: Callable):
        """Set callback for real-time event broadcasting."""
        self._event_callback = callback

    def _emit(self, event_type: str, data: dict):
        """Emit an event to the callback and DB."""
        event = {
            "generation_number": self.current_generation,
            "event_type": event_type,
            "description": data.get("description", ""),
            "metadata_json": data,
        }
        self.db.save_event(event)

        if self._event_callback:
            try:
                self._event_callback({"type": event_type, "generation": self.current_generation, **data})
            except Exception:
                pass

    def run(self, max_generations: int = None, on_generation: Callable = None):
        """Run the full evolutionary loop."""
        if max_generations:
            self.max_generations = max_generations

        self.running = True
        self.paused = False

        logger.info("=== Evolution started: %d generations ===", self.max_generations)
        self._emit("evolution_start", {"max_generations": self.max_generations, "population_size": self.population_size})

        # Save initial agent states
        for agent in self.builders:
            self.db.save_agent(agent.to_dict())

        try:
            for gen in range(self.current_generation, self.max_generations):
                if not self.running:
                    logger.info("Evolution stopped at generation %d", gen)
                    break
                while self.paused:
                    time.sleep(0.5)
                    if not self.running:
                        break

                self.current_generation = gen
                gen_result = self._run_generation(gen)
                self.history.append(gen_result)

                if on_generation:
                    on_generation(gen, gen_result)

                # Checkpoint
                if gen > 0 and gen % self.checkpoint_interval == 0:
                    self._save_checkpoint(gen)

        except Exception as e:
            logger.error("Evolution error at gen %d: %s", self.current_generation, e, exc_info=True)
            self._emit("evolution_error", {"error": str(e)})
        finally:
            self.running = False

        self._emit("evolution_complete", {
            "total_generations": len(self.history),
            "best_fitness": self.best_fitness,
            "best_accuracy": self.best_accuracy,
            "description": f"Evolution complete: {len(self.history)} generations, best fitness {self.best_fitness:.4f}",
        })

        logger.info(
            "=== Evolution complete: %d generations, best fitness %.4f ===",
            len(self.history), self.best_fitness
        )

        return self.history

    def _run_generation(self, gen: int) -> dict:
        """Execute a single generation of evolution."""
        start_time = time.time()
        logger.info("--- Generation %d ---", gen)
        self._emit("generation_start", {"description": f"Generation {gen} started"})

        population = Population(generation=gen)

        if gen == 0:
            # Initial population from builder agents
            genomes = self._generate_initial_population(gen)
        else:
            # Evolve from previous generation
            genomes = self._evolve_population(gen)

        population.add_many(genomes)

        # 2. Evaluate
        self._emit("evaluation_start", {
            "description": f"Evaluating {population.size} genomes",
            "count": population.size,
        })
        self.evaluator.evaluate_population(population.genomes, self.fitness_config)

        # 3. Rank
        population.assign_ranks()

        # 4. Update best
        best = population.best()
        if best and (best.metrics.get("fitness_score") or 0) > self.best_fitness:
            self.best_fitness = best.metrics.get("fitness_score", 0)
            self.best_accuracy = best.metrics.get("accuracy", 0)
            self.best_genome = best

        # 5. Compute stats
        elapsed = time.time() - start_time
        gen_stats = {
            "generation": gen,
            "best_fitness": best.metrics.get("fitness_score", 0) if best else 0,
            "avg_fitness": population.avg_fitness(),
            "worst_fitness": population.worst().metrics.get("fitness_score", 0) if population.worst() else 0,
            "best_accuracy": best.metrics.get("accuracy", 0) if best else 0,
            "avg_accuracy": population.avg_accuracy(),
            "diversity": population.diversity_score(),
            "species_distribution": population.species_distribution(),
            "population_size": population.size,
            "elapsed": elapsed,
        }

        # 6. Save generation to DB
        gen_db = {
            "number": gen,
            "population_size": population.size,
            "best_fitness": gen_stats["best_fitness"],
            "avg_fitness": gen_stats["avg_fitness"],
            "worst_fitness": gen_stats["worst_fitness"],
            "best_accuracy": gen_stats["best_accuracy"],
            "avg_accuracy": gen_stats["avg_accuracy"],
            "diversity_score": gen_stats["diversity"],
            "species_distribution": gen_stats["species_distribution"],
            "elapsed_seconds": elapsed,
        }
        gen_id = self.db.save_generation(gen_db)

        # 7. Save genomes
        genome_dicts = population.to_dicts(gen_id)
        self.db.save_genomes_batch(genome_dicts)

        # 8. Record to memory
        self.memory.record_generation_stats(gen_stats)
        if best:
            self.memory.record_best_pattern(best.to_dict())

        for g in population.genomes:
            self.memory.record_species_performance(
                g.species,
                g.metrics.get("fitness_score", 0),
                g.survived,
            )

        # 9. Evolve prompts
        agent_fitness = self._compute_agent_fitness(population)
        prompt_records = self.meta_prompt.evolve_prompts(
            self.builders, gen, agent_fitness, self.memory
        )
        for pr in prompt_records:
            self.db.save_prompt(pr)

        # 10. Update agent states
        for agent in self.builders:
            agent_genomes = [g for g in population.genomes if g.species == agent.species]
            fitnesses = [g.metrics.get("fitness_score", 0) for g in agent_genomes]
            best_dict = best.to_dict() if best else None
            agent.update_from_results(fitnesses, best_dict)
            self.db.save_agent(agent.to_dict())

        self._emit("generation_end", {
            "description": f"Gen {gen}: best={gen_stats['best_fitness']:.4f}, avg={gen_stats['avg_fitness']:.4f}",
            **gen_stats,
        })

        logger.info(
            "Gen %d complete: best=%.4f avg=%.4f diversity=%.2f (%.1fs)",
            gen, gen_stats["best_fitness"], gen_stats["avg_fitness"],
            gen_stats["diversity"], elapsed,
        )

        return gen_stats

    def _generate_initial_population(self, generation: int) -> List[Genome]:
        """Generate initial population from builder agents."""
        genomes = []
        per_agent = max(1, self.population_size // len(self.builders))
        remainder = self.population_size - per_agent * len(self.builders)

        for i, agent in enumerate(self.builders):
            count = per_agent + (1 if i < remainder else 0)
            agent_genomes = agent.generate(generation, count, self.memory)
            genomes.extend(agent_genomes)

            self._emit("genome_generated", {
                "agent_id": agent.id,
                "agent_species": agent.species,
                "count": count,
                "description": f"Agent {agent.species} generated {count} genomes",
            })

        return genomes[:self.population_size]

    def _evolve_population(self, generation: int) -> List[Genome]:
        """Evolve new population from previous generation's survivors."""
        # Get previous generation genomes from DB
        prev_genomes_data = self.db.get_genomes_by_generation(generation - 1)
        if not prev_genomes_data:
            logger.warning("No previous genomes found, generating fresh population")
            return self._generate_initial_population(generation)

        prev_genomes = [Genome.from_dict(d) for d in prev_genomes_data]

        # Select parents
        parents = self.selector.select(prev_genomes, max(4, self.population_size // 2))

        self._emit("selection_complete", {
            "description": f"Selected {len(parents)} parents",
            "parent_count": len(parents),
            "elite_count": sum(1 for p in parents if p.is_elite),
        })

        new_genomes = []

        # Elites carry over with new IDs
        elites = [p for p in parents if p.is_elite]
        for elite in elites[:self.elite_count]:
            e = elite.clone(new_id=True)
            e.generation = generation
            e.lineage["parent_a"] = elite.id
            e.lineage["creation_method"] = "elite"
            e.is_elite = True
            new_genomes.append(e)

        # Crossover
        num_children = self.population_size - len(new_genomes)
        children = self.crossover.crossover_pairs(parents, num_children, generation)
        new_genomes.extend(children)

        self._emit("crossover_complete", {
            "description": f"Crossover produced {len(children)} children",
            "children_count": len(children),
        })

        # Mutation
        mutation_records = []
        for i, genome in enumerate(new_genomes):
            if not genome.is_elite and random.random() < self.mutator.current_rate:
                mutated, records = self.mutator.mutate(genome, self.memory, generation)
                new_genomes[i] = mutated
                mutation_records.extend(records)

        # Save mutation records
        for mr in mutation_records:
            self.db.save_mutation(mr)

        self._emit("mutation_complete", {
            "description": f"Applied {len(mutation_records)} mutations",
            "mutation_count": len(mutation_records),
            "mutation_rate": self.mutator.current_rate,
        })

        # Fill remaining slots with new genomes from builders
        while len(new_genomes) < self.population_size:
            agent = random.choice(self.builders)
            new = agent.generate(generation, 1, self.memory)
            new_genomes.extend(new)

        return new_genomes[:self.population_size]

    def _compute_agent_fitness(self, population: Population) -> Dict[str, float]:
        """Compute average fitness per agent species."""
        species_fitness = defaultdict(list)
        for g in population.genomes:
            f = g.metrics.get("fitness_score", 0)
            species_fitness[g.species].append(f)

        return {
            species: sum(fits) / len(fits)
            for species, fits in species_fitness.items()
            if fits
        }

    def _save_checkpoint(self, generation: int):
        """Save an evolution checkpoint."""
        cp = {
            "generation_number": generation,
            "engine_state": {
                "best_fitness": self.best_fitness,
                "best_accuracy": self.best_accuracy,
                "current_generation": self.current_generation,
                "mutation_rate": self.mutator.current_rate,
                "memory_summary": self.memory.to_dict(),
            },
            "population_snapshot": [],
            "agent_states": {a.id: a.to_dict() for a in self.builders},
        }
        self.db.save_checkpoint(cp)
        logger.info("Checkpoint saved at generation %d", generation)

    def pause(self):
        self.paused = True
        logger.info("Evolution paused")

    def resume(self):
        self.paused = False
        logger.info("Evolution resumed")

    def stop(self):
        self.running = False
        logger.info("Evolution stopping")

    def get_status(self) -> dict:
        """Get current evolution status."""
        return {
            "running": self.running,
            "paused": self.paused,
            "current_generation": self.current_generation,
            "max_generations": self.max_generations,
            "population_size": self.population_size,
            "best_fitness": self.best_fitness,
            "best_accuracy": self.best_accuracy,
            "total_generations_completed": len(self.history),
            "mutation_rate": self.mutator.current_rate,
            "agents": [{"name": a.name, "species": a.species, "genomes_created": a.total_genomes_created} for a in self.builders],
        }
