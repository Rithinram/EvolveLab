"""
EvolveLab — Export Utility
Exports evolved architectures to standalone PyTorch files.
"""

import os
import logging
from evolution.genome import Genome
from training.code_generator import CodeGenerator

logger = logging.getLogger("evolvelab.export")

def export_genome(genome: Genome, directory: str = "exported_models"):
    """Exports a genome to a standalone .py file."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    filename = f"model_{genome.id[:8]}.py"
    filepath = os.path.join(directory, filename)
    
    gen = CodeGenerator()
    code = gen.generate_module_code(genome)
    
    # Add metadata as comments
    header = [
        f"# EvolveLab Exported Model",
        f"# Genome ID: {genome.id}",
        f"# Species: {genome.species}",
        f"# Generation: {genome.generation}",
        f"# Fitness: {genome.metrics.get('fitness_score', 'N/A')}",
        f"# Accuracy: {genome.metrics.get('accuracy', 'N/A')}",
        f"# Params: {genome.metrics.get('param_count', 'N/A')}",
        "#" + "-"*40,
        ""
    ]
    
    full_code = "\n".join(header) + code
    
    with open(filepath, "w") as f:
        f.write(full_code)
        
    logger.info(f"Successfully exported genome to {filepath}")
    return filepath
