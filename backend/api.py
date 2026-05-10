"""
EvolveLab — FastAPI Backend
REST API and WebSocket endpoints for the evolution system.
"""

import json
import threading
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware

from utils.helpers import load_config, setup_logging
from database.crud import DatabaseManager
from evolution.engine import EvolutionEngine
from backend.websocket import WebSocketManager
from backend.schemas import EvolutionStartRequest, HealthResponse

logger = logging.getLogger("evolvelab.api")

# ── Global State ────────────────────────────────────────────────

config = load_config()
db = DatabaseManager(config.get("database", {}).get("url", "sqlite:///evolvelab.db"))
engine: Optional[EvolutionEngine] = None
ws_manager = WebSocketManager()
evolution_thread: Optional[threading.Thread] = None
engine_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("EvolveLab API starting")
    yield
    logger.info("EvolveLab API shutting down")
    if engine and engine.running:
        engine.stop()


app = FastAPI(
    title="EvolveLab API",
    description="Multi-agent evolutionary AI research platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health & Status ─────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/api/status")
def get_status():
    if engine:
        return engine.get_status()
    return {
        "running": False, "paused": False, "current_generation": 0,
        "max_generations": 0, "population_size": 0,
        "best_fitness": 0, "best_accuracy": 0,
        "total_generations_completed": 0, "mutation_rate": 0.3,
        "agents": [],
    }


# ── Evolution Control ──────────────────────────────────────────

@app.post("/api/evolution/start")
def start_evolution(req: EvolutionStartRequest = None):
    global engine, evolution_thread

    with engine_lock:
        if engine and engine.running:
            return {"error": "Evolution already running"}

    cfg = load_config()
    if req:
        if req.population_size:
            cfg["evolution"]["population_size"] = req.population_size
        if req.max_generations:
            cfg["evolution"]["max_generations"] = req.max_generations
        if req.accuracy_weight is not None:
            cfg["fitness"]["accuracy_weight"] = req.accuracy_weight
        if req.cost_weight is not None:
            cfg["fitness"]["cost_weight"] = req.cost_weight

    engine = EvolutionEngine(cfg, db)
    engine.set_event_callback(ws_manager.sync_broadcast)

    def run_evolution():
        try:
            engine.run()
        except Exception as e:
            logger.error("Evolution thread error: %s", e, exc_info=True)

    evolution_thread = threading.Thread(target=run_evolution, daemon=True)
    evolution_thread.start()

    return {"status": "started", "population_size": engine.population_size, "max_generations": engine.max_generations}


@app.post("/api/evolution/pause")
def pause_evolution():
    if engine and engine.running:
        engine.pause()
        return {"status": "paused"}
    return {"error": "No evolution running"}


@app.post("/api/evolution/resume")
def resume_evolution():
    if engine and engine.paused:
        engine.resume()
        return {"status": "resumed"}
    return {"error": "Evolution not paused"}


@app.post("/api/evolution/stop")
def stop_evolution():
    if engine and engine.running:
        engine.stop()
        return {"status": "stopping"}
    return {"error": "No evolution running"}


# ── Generations ────────────────────────────────────────────────

@app.get("/api/generations")
def get_generations():
    return db.get_all_generations()


@app.get("/api/generations/{number}")
def get_generation(number: int):
    gen = db.get_generation(number)
    if not gen:
        return {"error": "Generation not found"}
    return gen


# ── Genomes ────────────────────────────────────────────────────

@app.get("/api/genomes")
def get_genomes(generation: Optional[int] = Query(None), limit: int = Query(200)):
    if generation is not None:
        return db.get_genomes_by_generation(generation)
    return db.get_all_genomes(limit=limit)


@app.get("/api/genomes/best")
def get_best_genome():
    best = db.get_best_genome()
    if not best:
        return {"error": "No genomes found"}
    return best


@app.get("/api/genomes/{genome_id}")
def get_genome(genome_id: str):
    genome = db.get_genome(genome_id)
    if not genome:
        return {"error": "Genome not found"}
    return genome


@app.get("/api/genomes/{genome_id}/lineage")
def get_genome_lineage(genome_id: str, depth: int = Query(10)):
    return db.get_genome_lineage(genome_id, max_depth=depth)


# ── Agents ─────────────────────────────────────────────────────

@app.get("/api/agents")
def get_agents():
    return db.get_all_agents()


@app.get("/api/agents/{agent_id}")
def get_agent(agent_id: str):
    agent = db.get_agent(agent_id)
    if not agent:
        return {"error": "Agent not found"}
    return agent


# ── Mutations ──────────────────────────────────────────────────

@app.get("/api/mutations")
def get_mutations(generation: Optional[int] = Query(None), limit: int = Query(200)):
    return db.get_mutations(generation=generation, limit=limit)


@app.get("/api/mutations/analytics")
def get_mutation_analytics():
    return db.get_mutation_analytics()


# ── Prompts ────────────────────────────────────────────────────

@app.get("/api/prompts")
def get_prompts(agent_id: Optional[str] = Query(None), limit: int = Query(100)):
    return db.get_prompts(agent_id=agent_id, limit=limit)


# ── Analytics ──────────────────────────────────────────────────

@app.get("/api/analytics/fitness")
def get_fitness_trends():
    return db.get_fitness_trends()


@app.get("/api/analytics/species")
def get_species_distribution():
    return db.get_species_distribution()


@app.get("/api/analytics/survival")
def get_survival_rates():
    return db.get_survival_rates()


# ── Checkpoints ────────────────────────────────────────────────

@app.get("/api/checkpoints")
def get_checkpoints():
    return db.get_checkpoints()


@app.post("/api/checkpoints/{cp_id}/restore")
def restore_checkpoint(cp_id: int):
    global engine

    cp = db.get_checkpoint(cp_id)
    if not cp:
        return {"error": "Checkpoint not found"}

    if engine and engine.running:
        return {"error": "Cannot restore while evolution is running"}

    # Create a fresh engine and restore state from checkpoint
    cfg = load_config()
    engine = EvolutionEngine(cfg, db)
    engine.restore_from_checkpoint(cp)

    return {
        "status": "restored",
        "generation": cp.get("generation_number"),
        "best_fitness": cp.get("engine_state", {}).get("best_fitness", 0),
    }


# ── Events ─────────────────────────────────────────────────────

@app.get("/api/events")
def get_events(generation: Optional[int] = Query(None), limit: int = Query(500)):
    return db.get_events(generation=generation, limit=limit)


# ── WebSocket ──────────────────────────────────────────────────

@app.websocket("/ws/evolution")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Clients can send ping/status requests
            if data == "ping":
                await websocket.send_text('{"type": "pong"}')
            elif data == "status":
                status = get_status()
                await websocket.send_text(json.dumps(status))
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
