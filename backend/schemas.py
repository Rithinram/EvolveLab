"""
EvolveLab — Pydantic Response Schemas
API response models for type safety and documentation.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"


class EvolutionStartRequest(BaseModel):
    population_size: Optional[int] = None
    max_generations: Optional[int] = None
    accuracy_weight: Optional[float] = None
    cost_weight: Optional[float] = None


class StatusResponse(BaseModel):
    running: bool
    paused: bool
    current_generation: int
    max_generations: int
    population_size: int
    best_fitness: float
    best_accuracy: float
    total_generations_completed: int
    mutation_rate: float
    agents: List[dict]


class GenerationResponse(BaseModel):
    id: int
    number: int
    population_size: int
    best_fitness: float
    avg_fitness: float
    best_accuracy: float
    avg_accuracy: float
    diversity_score: float
    species_distribution: Dict[str, Any]
    elapsed_seconds: float
    timestamp: Optional[str]


class GenomeResponse(BaseModel):
    id: str
    generation_number: int
    species: str
    architecture: Dict[str, Any]
    training_strategy: Dict[str, Any]
    prompt_strategy: Dict[str, Any]
    fitness_score: Optional[float]
    accuracy: Optional[float]
    compute_cost: Optional[float]
    complexity: Optional[float]
    inference_speed: Optional[float]
    param_count: int
    parent_a_id: Optional[str]
    parent_b_id: Optional[str]
    creation_method: str
    mutation_history: List[Any]
    is_elite: bool
    survived: bool
    rank: Optional[int]
