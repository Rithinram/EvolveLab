"""
EvolveLab — Optional Lightweight Benchmark
Placeholder for tiny real-training benchmarks (optional).
"""

import logging

logger = logging.getLogger("evolvelab.benchmark")


class LightweightBenchmark:
    """Optional real-training benchmark using tiny datasets.
    Not used by default — system uses heuristic evaluation."""

    def __init__(self):
        self.enabled = False
        logger.info("Benchmark module initialized (disabled by default)")

    def run(self, genome_dict: dict) -> dict:
        """Placeholder — returns None to signal heuristic should be used."""
        return None
