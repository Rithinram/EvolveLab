import os
import sys
import logging
import json
import torch
from evolution.engine import EvolutionEngine
from database.crud import DatabaseManager
from utils.helpers import load_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stress_test")

def run_stress_test():
    # 1. Load config and override for stress test
    config = load_config()
    config["evaluation"]["mode"] = "pytorch"
    config["evaluation"]["dataset"] = "mnist"
    config["evaluation"]["batch_size"] = 16 # Small for speed
    config["evolution"]["population_size"] = 4
    config["evolution"]["max_generations"] = 2
    
    db_url = "sqlite:///evolvelab_stress.db"
    if os.path.exists("evolvelab_stress.db"):
        os.remove("evolvelab_stress.db")
    
    db = DatabaseManager(db_url)
    
    logger.info("Starting stress test: PyTorch mode on MNIST")
    engine = EvolutionEngine(config, db)
    
    # 2. Run evolution
    try:
        engine.run()
    except Exception as e:
        logger.error(f"Stress test failed during run: {e}", exc_info=True)
        return False
    
    # 3. Verify Data Paths
    logger.info("Verifying data paths...")
    
    # Check generations
    gens = db.get_all_generations()
    if len(gens) < 2:
        logger.error(f"Expected 2 generations, found {len(gens)}")
        return False
    
    # Check genomes
    genomes = db.get_genomes_by_generation(0)
    if not genomes:
        logger.error("No genomes found for Gen 0")
        return False
    
    for g in genomes:
        logger.info(f"Genome {g['id']} - Accuracy: {g.get('accuracy')}, Params: {g.get('param_count')}")
        
        if g.get("accuracy") is None or g.get("param_count") is None:
            logger.error(f"Missing critical metrics in genome {g['id']}")
            return False
            
    logger.info("Stress test PASSED: All data paths verified.")
    return True

if __name__ == "__main__":
    success = run_stress_test()
    sys.exit(0 if success else 1)
