"""
EvolveLab — Evolution Engine
Orchestrates the full evolutionary loop across all agents.
"""

import time
import logging
import random
import torch
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
from training.proxy_evaluator import ProxyEvaluator
from training.pytorch_evaluator import PyTorchEvaluator
from utils.hardware import get_compute_capability

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
        self.eval_cfg = self.config.get("evaluation", {})

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
        
        # Adaptive Hardware
        self.hw_profile = get_compute_capability()
        self.proxy_evaluator = ProxyEvaluator(device=self.hw_profile["device"])

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
        
        # Propagate batch telemetry to evaluator
        def on_batch_telemetry(data):
            self._emit("training_batch", data)
            
        self.evaluator.set_on_batch_callback(on_batch_telemetry)

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
            mutation_map = {}  # No mutations in gen 0
        else:
            # Evolve from previous generation
            genomes, mutation_map = self._evolve_population(gen)

        population.add_many(genomes)

        # 2. Evaluate
        fidelity = self._get_fidelity(gen)
        self._emit("evaluation_start", {
            "description": f"Evaluating {population.size} genomes (fidelity: {fidelity} epochs)",
            "count": population.size,
            "fidelity": fidelity,
        })
        self.evaluator.evaluate_population(population.genomes, fidelity=fidelity)

        # 3. Rank
        population.assign_ranks()

        # 4. Close the mutation feedback loop
        # Now that genomes are evaluated, backfill fitness_after on every
        # mutation record and feed the results into the memory store so
        # adaptive mutation rates/weights are based on real data.
        if mutation_map:
            self._close_mutation_feedback(population, mutation_map, gen)

        # 5. Update best
        best = population.best()
        if best and (best.metrics.get("fitness_score") or 0) > self.best_fitness:
            self.best_fitness = best.metrics.get("fitness_score", 0)
            self.best_accuracy = best.metrics.get("accuracy", 0)
            self.best_genome = best

        # 6. Compute stats
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

        # 7. Dynamic fitness weight shifting
        self._maybe_shift_fitness_weights(gen_stats)

        # 8. Save generation to DB
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

        # 9. Save genomes
        genome_dicts = population.to_dicts(gen_id)
        self.db.save_genomes_batch(genome_dicts)

        # 10. Record to memory
        self.memory.record_generation_stats(gen_stats)
        if best:
            self.memory.record_best_pattern(best.to_dict())

        for g in population.genomes:
            self.memory.record_species_performance(
                g.species,
                g.metrics.get("fitness_score", 0),
                g.survived,
            )

        # 11. Evolve prompts
        agent_fitness = self._compute_agent_fitness(population)
        prompt_records = self.meta_prompt.evolve_prompts(
            self.builders, gen, agent_fitness, self.memory
        )
        for pr in prompt_records:
            self.db.save_prompt(pr)

        # 12. Update agent states
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
            "Gen %d complete: best=%.4f avg=%.4f diversity=%.2f mut_rate=%.3f (%.1fs)",
            gen, gen_stats["best_fitness"], gen_stats["avg_fitness"],
            gen_stats["diversity"], self.mutator.current_rate, elapsed,
        )

        return gen_stats

    def _generate_initial_population(self, generation: int) -> List[Genome]:
        """
        Generate initial population from builder agents.
        Uses Proxy Warmup: Generates N-x candidates and picks the best by Synflow.
        """
        ratio = self.hw_profile.get("proxy_ratio", 3)
        candidate_size = self.population_size * ratio
        candidates = []
        
        per_agent = max(1, candidate_size // len(self.builders))
        remainder = candidate_size - per_agent * len(self.builders)

        logger.info(f"Warmup: Generating {candidate_size} candidates for screening...")

        for i, agent in enumerate(self.builders):
            count = per_agent + (1 if i < remainder else 0)
            agent_genomes = agent.generate(generation, count, self.memory)
            candidates.extend(agent_genomes)

        # Proxy Screening
        self._emit("warmup_start", {"candidate_count": len(candidates)})
        
        scored_candidates = []
        for g in candidates:
            res = self.proxy_evaluator.score_genome(g, proxy_type="synflow")
            g.metrics["proxy_score"] = res["proxy_score"]
            scored_candidates.append(g)
            
        # Sort by proxy score and take top N
        scored_candidates.sort(key=lambda x: x.metrics.get("proxy_score", 0), reverse=True)
        final_population = scored_candidates[:self.population_size]
        
        if not final_population:
            logger.error("All proxy candidates failed. Falling back to unsorted candidates.")
            final_population = candidates[:self.population_size]

        top_score = final_population[0].metrics.get("proxy_score", 0) if final_population else 0
        self._emit("warmup_complete", {
            "top_proxy_score": top_score,
            "description": f"Warmup complete. Selected top {len(final_population)} from {len(candidates)} candidates."
        })

        return final_population

    def _evolve_population(self, generation: int):
        """Evolve new population from previous generation's survivors.

        Returns:
            Tuple of (genomes list, mutation_map) where mutation_map maps
            genome_id -> list of mutation record dicts with fitness_before.
        """
        # Get previous generation genomes from DB
        prev_genomes_data = self.db.get_genomes_by_generation(generation - 1)
        if not prev_genomes_data:
            logger.warning("No previous genomes found, generating fresh population")
            return self._generate_initial_population(generation), {}

        prev_genomes = [Genome.from_dict(d) for d in prev_genomes_data]

        # Select parents
        parents = self.selector.select(prev_genomes, max(4, self.population_size // 2))

        # Fix #4: Mark survivors from previous generation in DB
        survivor_ids = {p.id for p in parents}
        self.db.update_survived_flags(generation - 1, survivor_ids)

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

        # Mutation — track which genomes were mutated and their pre-mutation fitness
        mutation_map = {}  # genome_id -> [mutation records with fitness_before]
        for i, genome in enumerate(new_genomes):
            if not genome.is_elite and random.random() < self.mutator.current_rate:
                mutated, records = self.mutator.mutate(genome, self.memory, generation)
                new_genomes[i] = mutated
                if records:
                    mutation_map[mutated.id] = records

        # Save mutation records to DB (fitness_after still NULL at this point)
        for records in mutation_map.values():
            for mr in records:
                self.db.save_mutation(mr)

        self._emit("mutation_complete", {
            "description": f"Applied {sum(len(r) for r in mutation_map.values())} mutations",
            "mutation_count": sum(len(r) for r in mutation_map.values()),
            "mutation_rate": self.mutator.current_rate,
        })

        # Fill remaining slots with new genomes from builders
        while len(new_genomes) < self.population_size:
            agent = random.choice(self.builders)
            new = agent.generate(generation, 1, self.memory)
            new_genomes.extend(new)

        return new_genomes[:self.population_size], mutation_map

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

    def _close_mutation_feedback(self, population: Population,
                                 mutation_map: dict, generation: int):
        """Backfill fitness_after on mutation records and feed into memory.

        This is the critical step that makes adaptive mutation rates work:
        without it, the memory store has no data and adaptive weights/rates
        fall through to static defaults.
        """
        genome_fitness = {g.id: g.metrics.get("fitness_score", 0)
                          for g in population.genomes}

        # When fitness_before is None (mutations on unevaluated children),
        # use the previous generation's avg fitness as a meaningful baseline.
        prev_avg = 0.0
        if self.history:
            prev_avg = self.history[-1].get("avg_fitness", 0)

        for genome_id, records in mutation_map.items():
            fitness_after = genome_fitness.get(genome_id, 0)
            for rec in records:
                fitness_before = rec.get("fitness_before")
                if fitness_before is None:
                    fitness_before = prev_avg
                delta = round(fitness_after - fitness_before, 6)
                success = delta > 0

                # Update DB record
                self.db.update_mutation_result(
                    genome_id=genome_id,
                    generation_number=generation,
                    mutation_type=rec["mutation_type"],
                    fitness_after=fitness_after,
                    fitness_delta=delta,
                    success=success,
                )

                # Feed into memory store — this is what drives adaptive rates
                self.memory.record_mutation(
                    rec["mutation_type"], fitness_before, fitness_after
                )

        logger.debug(
            "Mutation feedback closed: %d genomes, %d records",
            len(mutation_map),
            sum(len(r) for r in mutation_map.values()),
        )

    def _maybe_shift_fitness_weights(self, gen_stats: dict):
        """Dynamically shift accuracy/cost weights based on convergence.

        If the population is converging (low diversity), shift toward cost
        optimization. If still exploring (high diversity), favor accuracy.
        """
        if not self.fitness_config.get("dynamic_weights", False):
            return

        shift_rate = self.fitness_config.get("weight_shift_rate", 0.02)
        diversity = gen_stats.get("diversity", 0.5)
        acc_w = self.fitness_config.get("accuracy_weight", 0.7)
        cost_w = self.fitness_config.get("cost_weight", 0.3)

        if diversity < 0.3:
            # Converging — push toward cost efficiency to differentiate
            acc_w = max(0.4, acc_w - shift_rate)
            cost_w = min(0.6, cost_w + shift_rate)
        elif diversity > 0.7:
            # High diversity — push toward accuracy to converge
            acc_w = min(0.9, acc_w + shift_rate)
            cost_w = max(0.1, cost_w - shift_rate)

        self.fitness_config["accuracy_weight"] = round(acc_w, 4)
        self.fitness_config["cost_weight"] = round(cost_w, 4)

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

    def restore_from_checkpoint(self, checkpoint: dict):
        """Restore engine state from a saved checkpoint."""
        state = checkpoint.get("engine_state", {})
        self.best_fitness = state.get("best_fitness", 0.0)
        self.best_accuracy = state.get("best_accuracy", 0.0)
        self.current_generation = state.get("current_generation", 0)
        self.mutator.current_rate = state.get("mutation_rate", self.mutator.base_rate)
        logger.info(
            "Restored from checkpoint: gen=%d, best_fitness=%.4f",
            self.current_generation, self.best_fitness,
        )

    def pause(self):
        self.paused = True
        logger.info("Evolution paused")

    def resume(self):
        self.paused = False
        logger.info("Evolution resumed")

    def stop(self):
        self.running = False
        logger.info("Evolution stopping")

    def _get_fidelity(self, gen: int) -> int:
        """Determine the number of training epochs based on generation."""
        # Simple adaptive schedule:
        # Gen 0-2: 1 epoch
        # Gen 3-5: 3 epochs
        # Gen 6+: 5 epochs
        if gen <= 2:
            return 1
        elif gen <= 5:
            return 3
        else:
            return self.eval_cfg.get("fidelity", 5)

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
