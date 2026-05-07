"""
EvolveLab — Database CRUD Operations
All database read/write operations for the evolution system.
"""

import json
import logging
from typing import Optional, List, Dict, Any
from sqlalchemy import desc, func, Integer
from sqlalchemy.orm import Session

from database.models import (
    Generation, Genome, Agent, PromptRecord,
    MutationRecord, Checkpoint, EvolutionEvent,
    create_db_engine, create_session_factory
)

logger = logging.getLogger("evolvelab.db")


class DatabaseManager:
    """Manages all database operations for EvolveLab."""

    def __init__(self, db_url: str = "sqlite:///evolvelab.db"):
        self.engine = create_db_engine(db_url)
        self.SessionFactory = create_session_factory(self.engine)
        logger.info("Database initialized: %s", db_url)

    def get_session(self) -> Session:
        return self.SessionFactory()

    # ── Generations ─────────────────────────────────────────────

    def save_generation(self, gen_data: dict) -> int:
        session = self.get_session()
        try:
            gen = Generation(**gen_data)
            session.add(gen)
            session.commit()
            gen_id = gen.id
            return gen_id
        except Exception as e:
            session.rollback()
            logger.error("Failed to save generation: %s", e)
            raise
        finally:
            session.close()

    def get_generation(self, number: int) -> Optional[dict]:
        session = self.get_session()
        try:
            gen = session.query(Generation).filter_by(number=number).first()
            if not gen:
                return None
            return {
                "id": gen.id, "number": gen.number,
                "population_size": gen.population_size,
                "best_fitness": gen.best_fitness, "avg_fitness": gen.avg_fitness,
                "worst_fitness": gen.worst_fitness,
                "best_accuracy": gen.best_accuracy, "avg_accuracy": gen.avg_accuracy,
                "diversity_score": gen.diversity_score,
                "species_distribution": gen.species_distribution or {},
                "elapsed_seconds": gen.elapsed_seconds,
                "timestamp": gen.timestamp.isoformat() if gen.timestamp else None,
            }
        finally:
            session.close()

    def get_all_generations(self) -> List[dict]:
        session = self.get_session()
        try:
            gens = session.query(Generation).order_by(Generation.number).all()
            return [{
                "id": g.id, "number": g.number,
                "population_size": g.population_size,
                "best_fitness": g.best_fitness, "avg_fitness": g.avg_fitness,
                "best_accuracy": g.best_accuracy, "avg_accuracy": g.avg_accuracy,
                "diversity_score": g.diversity_score,
                "species_distribution": g.species_distribution or {},
                "elapsed_seconds": g.elapsed_seconds,
                "timestamp": g.timestamp.isoformat() if g.timestamp else None,
            } for g in gens]
        finally:
            session.close()

    # ── Genomes ─────────────────────────────────────────────────

    def save_genome(self, genome_data: dict) -> str:
        session = self.get_session()
        try:
            genome = Genome(**genome_data)
            session.merge(genome)
            session.commit()
            return genome.id
        except Exception as e:
            session.rollback()
            logger.error("Failed to save genome %s: %s", genome_data.get("id"), e)
            raise
        finally:
            session.close()

    def save_genomes_batch(self, genomes: List[dict]):
        session = self.get_session()
        try:
            for g in genomes:
                session.merge(Genome(**g))
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error("Batch genome save failed: %s", e)
            raise
        finally:
            session.close()

    def get_genome(self, genome_id: str) -> Optional[dict]:
        session = self.get_session()
        try:
            g = session.query(Genome).filter_by(id=genome_id).first()
            if not g:
                return None
            return self._genome_to_dict(g)
        finally:
            session.close()

    def get_genomes_by_generation(self, gen_number: int) -> List[dict]:
        session = self.get_session()
        try:
            genomes = session.query(Genome).filter_by(
                generation_number=gen_number
            ).order_by(desc(Genome.fitness_score)).all()
            return [self._genome_to_dict(g) for g in genomes]
        finally:
            session.close()

    def get_best_genome(self) -> Optional[dict]:
        session = self.get_session()
        try:
            g = session.query(Genome).filter(
                Genome.fitness_score.isnot(None)
            ).order_by(desc(Genome.fitness_score)).first()
            if not g:
                return None
            return self._genome_to_dict(g)
        finally:
            session.close()

    def get_genome_lineage(self, genome_id: str, max_depth: int = 10) -> List[dict]:
        """Trace ancestry chain for a genome."""
        lineage = []
        current_id = genome_id
        visited = set()
        for _ in range(max_depth):
            if not current_id or current_id in visited:
                break
            visited.add(current_id)
            genome = self.get_genome(current_id)
            if not genome:
                break
            lineage.append(genome)
            current_id = genome.get("parent_a_id")
        return lineage

    def get_all_genomes(self, limit: int = 200) -> List[dict]:
        session = self.get_session()
        try:
            genomes = session.query(Genome).order_by(
                desc(Genome.generation_number), desc(Genome.fitness_score)
            ).limit(limit).all()
            return [self._genome_to_dict(g) for g in genomes]
        finally:
            session.close()

    def _genome_to_dict(self, g: Genome) -> dict:
        return {
            "id": g.id, "generation_id": g.generation_id,
            "generation_number": g.generation_number,
            "species": g.species,
            "architecture": g.architecture or {},
            "training_strategy": g.training_strategy or {},
            "prompt_strategy": g.prompt_strategy or {},
            "fitness_score": g.fitness_score, "accuracy": g.accuracy,
            "compute_cost": g.compute_cost, "complexity": g.complexity,
            "inference_speed": g.inference_speed,
            "param_count": g.param_count,
            "parent_a_id": g.parent_a_id, "parent_b_id": g.parent_b_id,
            "creation_method": g.creation_method,
            "mutation_history": g.mutation_history or [],
            "is_elite": g.is_elite, "survived": g.survived,
            "rank": g.rank,
        }

    # ── Agents ──────────────────────────────────────────────────

    def save_agent(self, agent_data: dict):
        session = self.get_session()
        try:
            session.merge(Agent(**agent_data))
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error("Failed to save agent: %s", e)
            raise
        finally:
            session.close()

    def get_all_agents(self) -> List[dict]:
        session = self.get_session()
        try:
            agents = session.query(Agent).all()
            return [{
                "id": a.id, "name": a.name, "species": a.species,
                "personality": a.personality, "focus": a.focus,
                "creativity": a.creativity,
                "generations_active": a.generations_active,
                "total_genomes_created": a.total_genomes_created,
                "best_fitness_achieved": a.best_fitness_achieved,
                "avg_fitness": a.avg_fitness,
                "survival_rate": a.survival_rate,
                "memory": a.memory or {},
                "prompt_strategy": a.prompt_strategy or {},
                "performance_history": a.performance_history or [],
            } for a in agents]
        finally:
            session.close()

    def get_agent(self, agent_id: str) -> Optional[dict]:
        session = self.get_session()
        try:
            a = session.query(Agent).filter_by(id=agent_id).first()
            if not a:
                return None
            return {
                "id": a.id, "name": a.name, "species": a.species,
                "personality": a.personality, "focus": a.focus,
                "creativity": a.creativity,
                "generations_active": a.generations_active,
                "total_genomes_created": a.total_genomes_created,
                "best_fitness_achieved": a.best_fitness_achieved,
                "avg_fitness": a.avg_fitness, "survival_rate": a.survival_rate,
                "memory": a.memory or {},
                "prompt_strategy": a.prompt_strategy or {},
                "performance_history": a.performance_history or [],
            }
        finally:
            session.close()

    # ── Prompts ─────────────────────────────────────────────────

    def save_prompt(self, prompt_data: dict):
        session = self.get_session()
        try:
            session.add(PromptRecord(**prompt_data))
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error("Failed to save prompt: %s", e)
        finally:
            session.close()

    def get_prompts(self, agent_id: str = None, limit: int = 100) -> List[dict]:
        session = self.get_session()
        try:
            q = session.query(PromptRecord)
            if agent_id:
                q = q.filter_by(agent_id=agent_id)
            prompts = q.order_by(desc(PromptRecord.generation_number)).limit(limit).all()
            return [{
                "id": p.id, "agent_id": p.agent_id,
                "generation_number": p.generation_number,
                "prompt_text": p.prompt_text,
                "prompt_fitness": p.prompt_fitness,
                "parent_prompt_id": p.parent_prompt_id,
                "mutation_type": p.mutation_type,
                "strategy_params": p.strategy_params or {},
            } for p in prompts]
        finally:
            session.close()

    # ── Mutations ───────────────────────────────────────────────

    def save_mutation(self, mutation_data: dict):
        session = self.get_session()
        try:
            session.add(MutationRecord(**mutation_data))
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error("Failed to save mutation: %s", e)
        finally:
            session.close()

    def get_mutations(self, generation: int = None, limit: int = 200) -> List[dict]:
        session = self.get_session()
        try:
            q = session.query(MutationRecord)
            if generation is not None:
                q = q.filter_by(generation_number=generation)
            muts = q.order_by(desc(MutationRecord.timestamp)).limit(limit).all()
            return [{
                "id": m.id, "genome_id": m.genome_id,
                "generation_number": m.generation_number,
                "mutation_type": m.mutation_type,
                "field_changed": m.field_changed,
                "old_value": m.old_value, "new_value": m.new_value,
                "fitness_before": m.fitness_before,
                "fitness_after": m.fitness_after,
                "fitness_delta": m.fitness_delta,
                "success": m.success,
            } for m in muts]
        finally:
            session.close()

    def get_mutation_analytics(self) -> dict:
        session = self.get_session()
        try:
            results = session.query(
                MutationRecord.mutation_type,
                func.count(MutationRecord.id).label("total"),
                func.sum(func.cast(MutationRecord.success, Integer)).label("successes"),
                func.avg(MutationRecord.fitness_delta).label("avg_delta"),
            ).group_by(MutationRecord.mutation_type).all()

            analytics = {}
            for r in results:
                total = r.total or 0
                successes = r.successes or 0
                analytics[r.mutation_type] = {
                    "total": total,
                    "successes": successes,
                    "success_rate": successes / total if total > 0 else 0,
                    "avg_fitness_delta": float(r.avg_delta) if r.avg_delta else 0,
                }
            return analytics
        finally:
            session.close()

    # ── Events ──────────────────────────────────────────────────

    def save_event(self, event_data: dict):
        session = self.get_session()
        try:
            session.add(EvolutionEvent(**event_data))
            session.commit()
        except Exception as e:
            session.rollback()
        finally:
            session.close()

    def get_events(self, generation: int = None, limit: int = 500) -> List[dict]:
        session = self.get_session()
        try:
            q = session.query(EvolutionEvent)
            if generation is not None:
                q = q.filter_by(generation_number=generation)
            events = q.order_by(desc(EvolutionEvent.timestamp)).limit(limit).all()
            return [{
                "id": e.id, "generation_number": e.generation_number,
                "event_type": e.event_type, "agent_id": e.agent_id,
                "genome_id": e.genome_id, "description": e.description,
                "metadata": e.metadata_json or {},
                "timestamp": e.timestamp.isoformat() if e.timestamp else None,
            } for e in events]
        finally:
            session.close()

    # ── Checkpoints ─────────────────────────────────────────────

    def save_checkpoint(self, cp_data: dict):
        session = self.get_session()
        try:
            session.add(Checkpoint(**cp_data))
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error("Failed to save checkpoint: %s", e)
        finally:
            session.close()

    def get_checkpoints(self) -> List[dict]:
        session = self.get_session()
        try:
            cps = session.query(Checkpoint).order_by(desc(Checkpoint.generation_number)).all()
            return [{
                "id": c.id, "generation_number": c.generation_number,
                "timestamp": c.timestamp.isoformat() if c.timestamp else None,
            } for c in cps]
        finally:
            session.close()

    def get_checkpoint(self, cp_id: int) -> Optional[dict]:
        session = self.get_session()
        try:
            c = session.query(Checkpoint).filter_by(id=cp_id).first()
            if not c:
                return None
            return {
                "id": c.id, "generation_number": c.generation_number,
                "engine_state": c.engine_state,
                "population_snapshot": c.population_snapshot,
                "agent_states": c.agent_states,
            }
        finally:
            session.close()

    # ── Analytics Queries ───────────────────────────────────────

    def get_fitness_trends(self) -> List[dict]:
        return self.get_all_generations()

    def get_species_distribution(self) -> dict:
        session = self.get_session()
        try:
            results = session.query(
                Genome.species, Genome.generation_number,
                func.count(Genome.id).label("count")
            ).group_by(Genome.species, Genome.generation_number).all()

            dist = {}
            for r in results:
                gen = r.generation_number
                if gen not in dist:
                    dist[gen] = {}
                dist[gen][r.species] = r.count
            return dist
        finally:
            session.close()

    def get_survival_rates(self) -> List[dict]:
        session = self.get_session()
        try:
            results = session.query(
                Genome.generation_number,
                func.count(Genome.id).label("total"),
                func.sum(func.cast(Genome.survived, Integer)).label("survived_count"),
            ).group_by(Genome.generation_number).order_by(Genome.generation_number).all()

            return [{
                "generation": r.generation_number,
                "total": r.total,
                "survived": r.survived_count or 0,
                "rate": (r.survived_count or 0) / r.total if r.total > 0 else 0,
            } for r in results]
        finally:
            session.close()

    def reset(self):
        """Drop and recreate all tables."""
        from database.models import Base
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)
        logger.info("Database reset complete")
