"""
EvolveLab — Hardware & Resource Analyzer
Detects available compute resources to adaptively scale evolution fidelity.
"""

import torch
import multiprocessing
import logging

logger = logging.getLogger("evolvelab.hardware")

def get_compute_capability():
    """
    Returns a 'compute_profile' based on available hardware.
    Profiles: 'low' (CPU/Laptop), 'medium' (Entry GPU), 'high' (Server GPU).
    """
    if not torch.cuda.is_available():
        return {
            "profile": "low",
            "device": "cpu",
            "cores": multiprocessing.cpu_count(),
            "proxy_ratio": 10,  # Screen 10x candidates on CPU
            "max_batches": 5,    # Very fast evaluation
            "fidelity_scale": 0.5
        }
    
    # Check GPU VRAM
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    if vram_gb < 6:
        return {
            "profile": "medium",
            "device": "cuda",
            "vram_gb": round(vram_gb, 1),
            "proxy_ratio": 5,
            "max_batches": 20,
            "fidelity_scale": 1.0
        }
    else:
        return {
            "profile": "high",
            "device": "cuda",
            "vram_gb": round(vram_gb, 1),
            "proxy_ratio": 3,
            "max_batches": 50,
            "fidelity_scale": 2.0
        }

def log_hardware_report():
    cap = get_compute_capability()
    logger.info(f"Hardware Profile Detected: {cap['profile'].upper()} ({cap['device']})")
    if cap['device'] == 'cuda':
        logger.info(f"VRAM: {cap['vram_gb']}GB | Recommended Batches: {cap['max_batches']}")
    else:
        logger.info(f"CPU Cores: {cap['cores']} | Recommended Batches: {cap['max_batches']}")
    return cap
