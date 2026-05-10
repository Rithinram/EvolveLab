"""
EvolveLab — PyTorch Evaluation Engine
Executes real training loops for genomes using generated code.
"""

import time
import logging
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional, Callable
from evolution.genome import Genome
from training.code_generator import CodeGenerator

from training.datasets import get_dataset_loaders
from utils.hardware import get_compute_capability

logger = logging.getLogger("evolvelab.pytorch_evaluator")

class PyTorchEvaluator:
    """Evaluates genomes by actually training them in PyTorch."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.eval_cfg = self.config.get("evaluation", {})
        self.code_gen = CodeGenerator()
        
        # Adaptive Hardware Setup
        self.hw_profile = get_compute_capability()
        self.device = torch.device(self.hw_profile["device"])
        
        self._loaders_cache = {}
        logger.info(f"PyTorch Evaluator initialized on {self.device} (Profile: {self.hw_profile['profile']})")

    def _get_loaders(self, dataset_name):
        if dataset_name not in self._loaders_cache:
            batch_size = self.eval_cfg.get("batch_size", 64)
            val_split = self.eval_cfg.get("val_split", 0.1)
            self._loaders_cache[dataset_name] = get_dataset_loaders(dataset_name, batch_size, val_split)
        return self._loaders_cache[dataset_name]

    def evaluate(self, genome: Genome, epochs: int = 1, 
                 dataset: str = None, 
                 on_batch: Optional[Callable] = None) -> Dict[str, float]:
        """
        Trains the genome for specified epochs and returns real metrics.
        """
        start_time = time.time()
        dataset = dataset or self.eval_cfg.get("dataset", "synthetic")
        
        model = None
        try:
            # 1. Generate Model
            code = self.code_gen.generate_module_code(genome)
            
            # Use dynamic execution to get the class
            local_scope = {}
            exec_globals = {"torch": torch, "nn": nn, "torch.nn": nn, "F": torch.nn.functional}
            exec(code, exec_globals, local_scope)
            
            ModelClass = local_scope['EvolvedModel']
            input_shape = genome.architecture.get("input_shape", [1, 28, 28])
            num_classes = genome.architecture.get("output_classes", 10)
            
            model = ModelClass(input_shape=input_shape, num_classes=num_classes).to(self.device)
            
            # Optional: Performance Boost with torch.compile
            if self.eval_cfg.get("use_compile", False) and hasattr(torch, "compile"):
                try:
                    model = torch.compile(model)
                    logger.debug("Model compiled successfully")
                except Exception as e:
                    logger.warning(f"torch.compile failed: {e}")
            
            # 2. Setup Training
            strategy = genome.training_strategy
            optimizer_name = strategy.get("optimizer", "adam").lower()
            lr = strategy.get("lr", 0.001)
            
            if optimizer_name == "adam":
                optimizer = optim.Adam(model.parameters(), lr=lr)
            elif optimizer_name == "sgd":
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            else:
                optimizer = optim.AdamW(model.parameters(), lr=lr)
                
            criterion = nn.CrossEntropyLoss()
            
            # 3. Training Loop
            model.train()
            
            if dataset == "synthetic":
                batch_size = strategy.get("batch_size", 32)
                num_batches = 5
                for epoch in range(epochs):
                    epoch_loss = 0
                    for _ in range(num_batches):
                        inputs = torch.randn(batch_size, *input_shape).to(self.device)
                        targets = torch.randint(0, num_classes, (batch_size,)).to(self.device)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                        
                        if on_batch:
                            on_batch({
                                "genome_id": genome.id,
                                "epoch": epoch,
                                "batch": _,
                                "loss": loss.item(),
                                "mode": "synthetic"
                            })
                final_loss = epoch_loss / num_batches if epochs > 0 else 1.0
            else:
                train_loader, val_loader, _ = self._get_loaders(dataset)
                for epoch in range(epochs):
                    epoch_loss = 0
                    for batch_idx, (inputs, targets) in enumerate(train_loader):
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                        
                        if on_batch:
                            on_batch({
                                "genome_id": genome.id,
                                "epoch": epoch,
                                "batch": batch_idx,
                                "loss": loss.item(),
                                "mode": "real"
                            })
                            
                        # Adaptive early break for search speed
                        max_b = self.eval_cfg.get("max_batches", self.hw_profile.get("max_batches", 10))
                        if batch_idx >= max_b:
                            break
                final_loss = epoch_loss / (batch_idx + 1) if epochs > 0 else 1.0
            
            # 4. Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                if dataset == "synthetic":
                    val_inputs = torch.randn(100, *input_shape).to(self.device)
                    val_targets = torch.randint(0, num_classes, (100,)).to(self.device)
                    val_outputs = model(val_inputs)
                    _, predicted = torch.max(val_outputs.data, 1)
                    total = val_targets.size(0)
                    correct = (predicted == val_targets).sum().item()
                else:
                    _, val_loader, _ = self._get_loaders(dataset)
                    for val_inputs, val_targets in val_loader:
                        val_inputs, val_targets = val_inputs.to(self.device), val_targets.to(self.device)
                        val_outputs = model(val_inputs)
                        _, predicted = torch.max(val_outputs.data, 1)
                        total += val_targets.size(0)
                        correct += (predicted == val_targets).sum().item()
                        if total > 500: # Fast validation
                            break
            
            accuracy = correct / total if total > 0 else 0
            elapsed = time.time() - start_time
            
            # 5. Model Analysis (FLOPs / Params)
            analysis = self.code_gen.analyze_model(model, input_shape)
            param_count = analysis.get("total_params", 0)
            flops = analysis.get("total_mult_adds", 0)
            
            return {
                "accuracy": accuracy,
                "loss": final_loss,
                "compute_cost": elapsed,
                "param_count": param_count,
                "complexity": flops, # FLOPs proxy
                "model_size_mb": analysis.get("model_size_mb", 0),
                "status": "success"
            }
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM for genome {genome.id}. Model too large.")
                return {
                    "accuracy": 0.0, "loss": 10.0, "compute_cost": time.time() - start_time,
                    "param_count": 0, "status": "error", "error": "OOM"
                }
            raise e
        except Exception as e:
            logger.error(f"PyTorch Evaluation failed for genome {genome.id}: {e}")
            return {
                "accuracy": 0.0, "loss": 10.0, "compute_cost": time.time() - start_time,
                "param_count": 0, "status": "error", "error": str(e)
            }
        finally:
            if model is not None:
                del model
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def evaluate_population(self, genomes: List[Genome], 
                            fidelity: int = 3,
                            on_batch: Optional[Callable] = None) -> List[Genome]:
        """Evaluate a list of genomes with a specific fidelity level (epochs)."""
        logger.info(f"Evaluating population of {len(genomes)} with fidelity {fidelity} epochs")
        
        # Use config weights if available
        fit_cfg = self.config.get("fitness", {})
        acc_w = fit_cfg.get("accuracy_weight", 0.7)
        cost_w = fit_cfg.get("cost_weight", 0.3)
        
        for genome in genomes:
            metrics = self.evaluate(genome, epochs=fidelity, on_batch=on_batch)
            genome.metrics.update(metrics)
            
            if metrics["status"] == "success":
                acc = metrics["accuracy"]
                params = metrics["param_count"]
                
                # Multi-Objective Fitness: Fitness = alpha * Acc - beta * log10(Params)
                # This encourages small models without being too aggressive.
                import math
                param_penalty = math.log10(params + 1) / 7.0 # Normalize log(params) to ~0-1 range
                
                fitness = (acc_w * acc) - (cost_w * param_penalty)
                genome.metrics["fitness_score"] = max(0.01, round(fitness, 4))
            else:
                genome.metrics["fitness_score"] = 0.0
            
        return genomes
