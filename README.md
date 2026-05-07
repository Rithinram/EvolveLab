# EvolveLab

**Agents That Evolve Their Own Intelligence**

An enterprise-grade multi-agent evolutionary AI research platform where autonomous agents collaboratively generate, evaluate, mutate, evolve, and optimize ML architectures and training strategies using genuine evolutionary algorithms, historical memory, and meta-prompt evolution.

---

## Architecture

```
Multi-Agent Layer (6 agents)
  |-- Builder Agents (5 species)
  |-- Evaluator Agent (heuristic scoring)
  |-- Selection Agent (tournament + elitism)
  |-- Mutation Agent (adaptive rates)
  |-- Crossover Agent (architecture + training)
  |-- Meta-Prompt Evolution Agent
  |
Evolution Engine (orchestrator)
  |
Memory System (historical success tracking)
  |
FastAPI Backend (24+ endpoints + WebSocket)
  |
React Frontend (7 pages, dark enterprise UI)
  |
SQLite Database (full persistence)
```

## Quick Start

### 1. Install Backend Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Frontend Dependencies

```bash
cd frontend
npm install
```

### 3. Run Evolution (CLI)

```bash
python main.py run --gens 10 --pop 8
```

### 4. Run Full System (API + Frontend)

```bash
# Terminal 1: Start API
python main.py api

# Terminal 2: Start Frontend
cd frontend
npm run dev
```

Open **http://localhost:5173** for the dashboard.

## Features

### Multi-Agent System
- **5 Builder Agent Species**: transformer_specialist, efficient_architect, hybrid_innovator, accuracy_maximizer, cost_minimizer
- Each agent has a unique personality, generation strategy, and evolving prompt system
- Agents use memory of past generations to improve future decisions

### Genuine Evolutionary Computation
- **Tournament selection** with elitism and diversity preservation
- **Single-point crossover** of architecture layers + training strategy blending
- **8 mutation types**: layer addition/removal/modification, optimizer change, LR perturbation, dropout change, architecture resize, training strategy
- **Adaptive mutation rates** that evolve based on historical success (key novelty)
- **Lineage tracking** with full parent-child ancestry chains

### Meta-Prompt Evolution (Key Novelty)
- Agents evolve their own prompting/reasoning strategies over generations
- Prompt mutations: tweak, radical, crossover, refine, reset
- Prompt fitness tracking and lineage
- Self-improving generation pipeline

### Heuristic Evaluation (No GPU Required)
- Estimates model performance from architecture structure
- Analyzes depth, width, attention layers, normalization, dropout
- Accounts for training strategy (optimizer, LR, scheduler)
- Produces realistic accuracy/cost estimates with noise

### Persistence
- Full SQLite database with 7 tables
- Generations, genomes, agents, prompts, mutations, checkpoints, events
- Checkpoint save/restore for pause/resume

### Dashboard (React)
- **Dashboard**: KPI cards, fitness progression chart, best genome display
- **Evolution Monitor**: Start/pause/stop controls, live stats, event timeline, mutation log
- **Genome Explorer**: Searchable genome table with expandable JSON config viewer
- **Lineage Tree**: Ancestry chain visualization and generation-based genome grid
- **Agent Intelligence**: Agent personalities, stats, prompt evolution charts
- **Analytics**: Survival rates, mutation distribution, fitness/diversity trends
- **Settings**: Evolution parameters, fitness weights, mutation configuration

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | /api/health | Health check |
| GET | /api/status | Evolution status |
| POST | /api/evolution/start | Start evolution |
| POST | /api/evolution/pause | Pause evolution |
| POST | /api/evolution/resume | Resume evolution |
| POST | /api/evolution/stop | Stop evolution |
| GET | /api/generations | All generations |
| GET | /api/genomes | All genomes |
| GET | /api/genomes/best | Best genome |
| GET | /api/genomes/{id}/lineage | Genome ancestry |
| GET | /api/agents | All agents |
| GET | /api/mutations | Mutation history |
| GET | /api/mutations/analytics | Mutation success rates |
| GET | /api/prompts | Prompt evolution history |
| GET | /api/analytics/fitness | Fitness trends |
| GET | /api/analytics/species | Species distribution |
| GET | /api/analytics/survival | Survival rates |
| WS | /ws/evolution | Real-time updates |

## Genome Schema

```json
{
  "species": "accuracy_maximizer",
  "architecture": {
    "type": "hybrid_transformer_cnn",
    "layers": [
      {"type": "conv2d", "filters": 64, "kernel": 3, "activation": "relu"},
      {"type": "attention", "heads": 4, "dim": 128},
      {"type": "dense", "units": 256, "dropout": 0.3}
    ]
  },
  "training_strategy": {
    "optimizer": "adamw",
    "lr": 0.001,
    "scheduler": "cosine"
  },
  "lineage": {
    "parent_a": "uuid",
    "parent_b": "uuid",
    "creation_method": "crossover"
  }
}
```

## Fitness Function

```
fitness = (accuracy_weight * accuracy) - (cost_weight * compute_cost)
```

Default: `(0.7 * accuracy) - (0.3 * cost)`

## Project Structure

```
evolvelab/
├── agents/           # 6 specialized agents
├── evolution/        # Engine, genome, population
├── evaluator/        # Heuristic scoring
├── memory/           # Historical memory
├── database/         # SQLAlchemy models + CRUD
├── backend/          # FastAPI application
├── frontend/         # React + Vite dashboard
├── configs/          # Default parameters
├── utils/            # Helpers
├── main.py           # CLI entry point
└── requirements.txt
```

## Tech Stack

- **Backend**: Python, FastAPI, SQLAlchemy, SQLite
- **Frontend**: React, Vite, Recharts, React Router
- **No GPU required** — runs on any laptop
- **No external API dependency** — fully self-contained
