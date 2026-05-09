EvolveLab v2 - Revised Implementation Plan
A self-evolving multi-agent system that automatically discovers and improves ML model
architectures. This document corrects fundamental flaws in the original specification, replacing
naive assumptions with robust engineering practices.

1. Project Scaffolding & Configuration
   [MODIFIED] requirements.txt Removed sqlite , added psycopg2-binary for PostgreSQL
   support to enable parallel writes.
   torch>=2.0.0
   torchvision>=0.15.0
   fastapi>=0.103.0
   uvicorn>=0.23.0
   streamlit>=1.28.0
   sqlalchemy>=2.0.0
   psycopg2-binary>=2.9.0
   openai>=1.0.0
   pydantic>=2.0.0
   numpy>=1.21.0
   python-dotenv>=1.0.0
   [NEW] .env Requires OPENAI_API_KEY and a valid DATABASE_URL pointing to a PostgreSQL
   instance.
   [NEW] config.py Global configuration: population size, generation limit, database connection
   strings, and multi-fidelity training parameters.
2. Config Schema & Pipeline System
   [MODIFIED] core/config_schema.py Pydantic models define the universal pipeline config.
   Added strict schema validation for input/output dimensions to flag mismatch errors before
   training.
3. Agent Layer Updates
   The Agent Layer requires strict constraints to prevent the system from generating
   mathematically invalid models or exhausting API budgets.
   [MODIFIED] agents/builder.py
   LLM Mode Protocol: API calls to OpenAI are strictly restricted to Generation 0 to seed
   the initial population. Subsequent generations must rely purely on evolutionary
   mathematics.
   Fallback/Offline Mode: The previous "random config generation" is removed. Random
   architectures fail to converge. The fallback must pull from a static dictionary of proven
   seed architectures (e.g., miniature ResNets, standard MLPs).
   [MODIFIED] agents/evaluator.py
   Multi-Fidelity Training: The "Max 3 epochs" constraint is removed, as it penalizes deep
   architectures.
   Stage 1: Train full population for 3 epochs.
   Stage 2: Retain top 50%; train for 10 epochs.
   Stage 3: Retain top 10%; train to convergence (up to 50 epochs).
   [MODIFIED] agents/crossover.py & agents/mutation.py
   When combining layers from different parent models or mutating existing layers, these
   agents must tag the transition points for the Code Generator to resolve dimension
   mismatches.
4. Evolution Engine
   [MODIFIED] core/evolution_engine.py Orchestrates the evolutionary loop with cost controls:
5. Gen 0 Seeding: Builder Agent (LLM + Hardcoded Baselines).
6. Multi-Fidelity Evaluation: Evaluator trains and scores the current generation.
7. Selection: Tournament and Elitism select the top architectures based on Stage 3 fidelity
   scores.
8. Crossover & Mutation: Generates the next generation.
9. Iteration: Loop without LLM calls until max generations.
10. Code Generation & Training
    [MODIFIED] training/code_generator.py
    Projection Layer Module: Randomly crossing over Layer A (e.g., 128 output channels) to
    Layer B (e.g., 64 input channels) results in matrix multiplication crashes. The Code
    Generator must dynamically inject a 1x1 Conv2d or a nn.Linear projection layer to force
    output sizes to match input sizes at transition points.
    [MODIFIED] training/trainer.py
    Executes the Multi-Fidelity Training loop.
    Implements strict Out-Of-Memory (OOM) handling; models that exceed VRAM limits are
    immediately assigned a fitness score of 0.
11. Database Layer
    [MODIFIED] db/database.py & db/models.py
    PostgreSQL Migration: SQLite is incapable of handling concurrent writes from parallel
    Evaluator Agents without throwing database is locked errors. The backend must use
    PostgreSQL via SQLAlchemy.
    Tables: generations , pipelines , prompts , agent_logs .
12. Backend API
    [NEW] backend/api.py FastAPI application to manage the background tasks and expose data
    to the dashboard.
    POST /evolution/start
    GET /evolution/status
    GET /generations
    GET /pipelines/{id}
    WS /ws/evolution (Real-time updates)
13. Frontend Dashboard
    [NEW] frontend/app.py Streamlit dashboard for real-time monitoring.
    Overview & Status Controls
    Evolution Graph (Plotly)
    Population View (Table of current generation)
    Lineage Tree (Visual parent-child mapping)
    Log Viewer
