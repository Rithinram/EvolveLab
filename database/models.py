"""
EvolveLab — Database ORM Models
SQLAlchemy models for persisting all evolution state.
"""

import datetime
from sqlalchemy import (
    Column, Integer, Float, String, Text, Boolean,
    DateTime, ForeignKey, create_engine, JSON
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()


class Generation(Base):
    __tablename__ = "generations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    number = Column(Integer, nullable=False, unique=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    population_size = Column(Integer, nullable=False)
    best_fitness = Column(Float, default=0.0)
    avg_fitness = Column(Float, default=0.0)
    worst_fitness = Column(Float, default=0.0)
    best_accuracy = Column(Float, default=0.0)
    avg_accuracy = Column(Float, default=0.0)
    diversity_score = Column(Float, default=0.0)
    species_distribution = Column(JSON, default=dict)
    elapsed_seconds = Column(Float, default=0.0)
    genomes = relationship("Genome", back_populates="generation", cascade="all, delete-orphan")


class Genome(Base):
    __tablename__ = "genomes"
    id = Column(String, primary_key=True)
    generation_id = Column(Integer, ForeignKey("generations.id"), nullable=False, index=True)
    generation_number = Column(Integer, nullable=False, index=True)
    species = Column(String, nullable=False, index=True)
    architecture = Column(JSON, nullable=False)
    training_strategy = Column(JSON, nullable=False)
    prompt_strategy = Column(JSON, default=dict)
    fitness_score = Column(Float, default=None)
    accuracy = Column(Float, default=None)
    compute_cost = Column(Float, default=None)
    complexity = Column(Float, default=None)
    inference_speed = Column(Float, default=None)
    param_count = Column(Integer, default=0)
    parent_a_id = Column(String, default=None)
    parent_b_id = Column(String, default=None)
    creation_method = Column(String, default="generated")
    mutation_history = Column(JSON, default=list)
    is_elite = Column(Boolean, default=False)
    survived = Column(Boolean, default=False)
    rank = Column(Integer, default=None)
    generation = relationship("Generation", back_populates="genomes")


class Agent(Base):
    __tablename__ = "agents"
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    species = Column(String, nullable=False)
    personality = Column(Text, default="")
    focus = Column(String, default="")
    creativity = Column(Float, default=0.5)
    generations_active = Column(Integer, default=0)
    total_genomes_created = Column(Integer, default=0)
    best_fitness_achieved = Column(Float, default=0.0)
    avg_fitness = Column(Float, default=0.0)
    survival_rate = Column(Float, default=0.0)
    memory = Column(JSON, default=dict)
    prompt_strategy = Column(JSON, default=dict)
    performance_history = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)


class PromptRecord(Base):
    __tablename__ = "prompt_records"
    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(String, ForeignKey("agents.id"), nullable=False, index=True)
    generation_number = Column(Integer, nullable=False, index=True)
    prompt_text = Column(Text, nullable=False)
    prompt_fitness = Column(Float, default=0.0)
    parent_prompt_id = Column(Integer, default=None)
    mutation_type = Column(String, default=None)
    strategy_params = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class MutationRecord(Base):
    __tablename__ = "mutation_records"
    id = Column(Integer, primary_key=True, autoincrement=True)
    genome_id = Column(String, nullable=False, index=True)
    generation_number = Column(Integer, nullable=False, index=True)
    mutation_type = Column(String, nullable=False)
    field_changed = Column(String, default="")
    old_value = Column(Text, default="")
    new_value = Column(Text, default="")
    fitness_before = Column(Float, default=None)
    fitness_after = Column(Float, default=None)
    fitness_delta = Column(Float, default=None)
    success = Column(Boolean, default=None)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)


class Checkpoint(Base):
    __tablename__ = "checkpoints"
    id = Column(Integer, primary_key=True, autoincrement=True)
    generation_number = Column(Integer, nullable=False)
    engine_state = Column(JSON, nullable=False)
    population_snapshot = Column(JSON, default=list)
    agent_states = Column(JSON, default=dict)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)


class EvolutionEvent(Base):
    __tablename__ = "evolution_events"
    id = Column(Integer, primary_key=True, autoincrement=True)
    generation_number = Column(Integer, nullable=False, index=True)
    event_type = Column(String, nullable=False)
    agent_id = Column(String, default=None)
    genome_id = Column(String, default=None)
    description = Column(Text, default="")
    metadata_json = Column(JSON, default=dict)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)


def create_db_engine(db_url="sqlite:///evolvelab.db", echo=False):
    engine = create_engine(db_url, echo=echo)
    Base.metadata.create_all(engine)
    return engine


def create_session_factory(engine):
    return sessionmaker(bind=engine)
