import pytest
from evolution.engine import EvolutionEngine
from utils.helpers import load_config
from unittest.mock import MagicMock

def test_engine_fidelity_schedule():
    config = {
        "evolution": {"population_size": 2, "max_generations": 10},
        "evaluation": {"fidelity": 10},
        "database": {"url": "sqlite:///:memory:"}
    }
    engine = EvolutionEngine(config)
    
    # Gen 0-2: 1 epoch
    assert engine._get_fidelity(0) == 1
    assert engine._get_fidelity(2) == 1
    
    # Gen 3-5: 3 epochs
    assert engine._get_fidelity(3) == 3
    assert engine._get_fidelity(5) == 3
    
    # Gen 6+: Use config (10)
    assert engine._get_fidelity(6) == 10
    assert engine._get_fidelity(10) == 10
