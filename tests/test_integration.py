import os
import pytest
import json
from evolution.engine import EvolutionEngine
from database.crud import DatabaseManager
from utils.helpers import load_config

def test_full_evolution_data_path():
    """Integration test to verify the flow from evolution to database."""
    # 1. Setup in-memory or temp DB
    db_path = "test_integration.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    db_url = f"sqlite:///{db_path}"
    db = DatabaseManager(db_url)
    
    # 2. Configure for a very fast run
    config = {
        "evolution": {
            "population_size": 2,
            "max_generations": 2,
            "elite_count": 1
        },
        "evaluation": {
            "mode": "pytorch",
            "dataset": "synthetic",
            "batch_size": 4,
            "fidelity": 1
        },
        "fitness": {
            "accuracy_weight": 0.7,
            "cost_weight": 0.3
        },
        "database": {"url": db_url},
        "builder_species": [
            {"name": "accuracy_maximizer", "personality": "Aggressive accuracy-focused agent", "focus": "accuracy"},
            {"name": "cost_minimizer", "personality": "Conservative cost-focused agent", "focus": "cost"}
        ]
    }
    
    engine = EvolutionEngine(config, db)
    
    # 3. Run Evolution
    engine.run()
    
    # 4. Verify Data Paths
    # Check Generations
    gens = db.get_all_generations()
    assert len(gens) == 2, "Should have 2 generations in DB"
    
    # Check Genomes
    genomes = db.get_genomes_by_generation(0)
    assert len(genomes) == 2, "Should have 2 genomes in Gen 0"
    
    for g in genomes:
        assert g["accuracy"] is not None
        assert g["param_count"] > 0
        assert g["fitness_score"] >= 0
    
    # Check Events
    events = db.get_events()
    assert len(events) > 0, "Should have generated events"
    
    # Check Mutations (if any happened)
    # With 2 generations and pop size 2, mutations are likely
    mutations = db.get_mutations()
    # Even if 0, the table should exist
    assert isinstance(mutations, list)
    
    # 5. Cleanup
    engine.db.engine.dispose()
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except PermissionError:
            pass # On Windows, file might still be locked briefly

if __name__ == "__main__":
    test_full_evolution_data_path()
