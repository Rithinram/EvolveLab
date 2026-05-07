"""
EvolveLab — Main Entry Point
CLI interface for running evolution, starting the API, or launching the frontend.

Usage:
    python main.py run              # Run evolution loop directly
    python main.py run --gens 10    # Custom generation count
    python main.py api              # Start FastAPI server
    python main.py all              # Start API + run evolution
"""

import sys
import os
import argparse
import logging
import threading

# Fix Windows encoding
os.environ.setdefault("PYTHONUTF8", "1")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.helpers import setup_logging, load_config


def cmd_run(args):
    """Run evolution loop directly."""
    setup_logging("INFO")
    from evolution.engine import EvolutionEngine

    config = load_config()
    if args.gens:
        config["evolution"]["max_generations"] = args.gens
    if args.pop:
        config["evolution"]["population_size"] = args.pop

    engine = EvolutionEngine(config)

    print("=" * 60)
    print("  EvolveLab — Evolution Engine")
    print(f"  Population: {engine.population_size}")
    print(f"  Generations: {engine.max_generations}")
    print("=" * 60)

    def on_generation(gen, stats):
        print(
            f"  Gen {gen:3d} | "
            f"Best: {stats['best_fitness']:.4f} | "
            f"Avg: {stats['avg_fitness']:.4f} | "
            f"Acc: {stats['best_accuracy']:.4f} | "
            f"Diversity: {stats['diversity']:.2f} | "
            f"{stats['elapsed']:.1f}s"
        )

    history = engine.run(on_generation=on_generation)

    print("=" * 60)
    print(f"  Evolution complete: {len(history)} generations")
    print(f"  Best fitness: {engine.best_fitness:.4f}")
    print(f"  Best accuracy: {engine.best_accuracy:.4f}")
    print("=" * 60)


def cmd_api(args):
    """Start the FastAPI server."""
    setup_logging("INFO")
    import uvicorn

    port = args.port or 8000
    print("=" * 60)
    print("  EvolveLab — API Server")
    print(f"  http://localhost:{port}")
    print(f"  API docs: http://localhost:{port}/docs")
    print("=" * 60)

    uvicorn.run(
        "backend.api:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )


def cmd_all(args):
    """Start API server and optionally run evolution."""
    setup_logging("INFO")
    import uvicorn

    port = args.port or 8000
    print("=" * 60)
    print("  EvolveLab — Full System")
    print(f"  API: http://localhost:{port}")
    print(f"  Docs: http://localhost:{port}/docs")
    print("  Start evolution via POST /api/evolution/start")
    print("=" * 60)

    uvicorn.run(
        "backend.api:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )


def main():
    parser = argparse.ArgumentParser(
        description="EvolveLab — Agents That Evolve Their Own Intelligence"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run evolution loop")
    run_parser.add_argument("--gens", type=int, help="Number of generations")
    run_parser.add_argument("--pop", type=int, help="Population size")

    # API command
    api_parser = subparsers.add_parser("api", help="Start API server")
    api_parser.add_argument("--port", type=int, default=8000, help="Port number")

    # All command
    all_parser = subparsers.add_parser("all", help="Start full system")
    all_parser.add_argument("--port", type=int, default=8000, help="Port number")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "api":
        cmd_api(args)
    elif args.command == "all":
        cmd_all(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
