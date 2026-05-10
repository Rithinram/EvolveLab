"""
EvolveLab — Code Generator
Translates Genome JSON into runnable PyTorch nn.Module code.
Supports both Sequential and DAG-based architectures.
"""

import logging
import re
from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
from torchinfo import summary
from evolution.genome import Genome
from utils.graph import get_topological_sort, get_leaf_nodes

logger = logging.getLogger("evolvelab.code_generator")

class CodeGenerator:
    """Orchestrates PyTorch code generation from an EvolveLab Genome."""

    def __init__(self):
        self.indent = "    "

    def analyze_model(self, model: nn.Module, input_shape: List[int]) -> Dict[str, Any]:
        """Analyzes model efficiency (Params, FLOPs) using torchinfo."""
        try:
            batch_shape = (1, *input_shape)
            stats = summary(model, input_size=batch_shape, verbose=0)
            return {
                "total_params": stats.total_params,
                "trainable_params": stats.trainable_params,
                "total_mult_adds": stats.total_mult_adds,
                "input_size": str(batch_shape),
                "model_size_mb": stats.total_param_bytes / (1024 ** 2)
            }
        except Exception as e:
            logger.error(f"Model analysis failed: {e}")
            return {"total_params": 0, "total_mult_adds": 0, "error": str(e)}

    def validate_genome(self, genome: Genome):
        """
        Security check: ensures the genome contains no malicious strings 
        before passing it to exec().
        """
        # 1. Check for dangerous keywords anywhere in the architecture JSON
        dangerous_patterns = [
            "import", "os.", "sys.", "subprocess", "eval", "exec", 
            "builtins", "getattr", "setattr", "__", "pickle", "requests",
            "shutil", "threading", "multiprocessing", "socket"
        ]
        
        arch_str = str(genome.architecture).lower()
        for pattern in dangerous_patterns:
            if pattern in arch_str:
                raise SecurityError(f"Security violation: dangerous pattern '{pattern}' detected in genome.")

        # 2. Validate Layer Names and Types
        # They must be alphanumeric or underscores only
        for layer in genome.architecture.get("layers", []):
            l_type = layer.get("type", "")
            if not re.match(r"^[a-zA-Z0-9_]+$", l_type):
                raise SecurityError(f"Security violation: invalid layer type '{l_type}'")

    def sanitize_string(self, s: str) -> str:
        """Removes any characters that aren't alphanumeric or underscores."""
        return re.sub(r"[^a-zA-Z0-9_]", "", s)

    def generate_module_code(self, genome: Genome) -> str:
        """Determines the architecture type and uses the appropriate translator."""
        # Security first
        self.validate_genome(genome)
        
        arch = genome.architecture
        
        # Check if it's a DAG-based genome (has 'nodes' or 'normal_cell')
        if "nodes" in arch or "normal_cell" in arch:
            return self._generate_dag_code(genome)
        else:
            return self._generate_sequential_code(genome)

    def _generate_sequential_code(self, genome: Genome) -> str:
        """Original sequential logic, refactored for modularity."""
        arch = genome.architecture
        layers = arch.get("layers", [])
        input_shape = arch.get("input_shape", [3, 32, 32])
        
        code = [
            "import torch",
            "import torch.nn as nn",
            "import torch.nn.functional as F",
            "",
            "class EvolvedModel(nn.Module):",
            f"{self.indent}def __init__(self, input_shape={input_shape}, num_classes={arch.get('output_classes', 10)}):",
            f"{self.indent}{self.indent}super().__init__()",
            f"{self.indent}{self.indent}self.input_shape = input_shape",
            ""
        ]

        module_defs = []
        forward_steps = []
        # Track history: {index: {"channels": int, "is_flattened": bool}}
        history = {-1: {"channels": input_shape[0], "is_flattened": False}}
        
        current_channels = input_shape[0]
        is_flattened = False
        
        for i, layer in enumerate(layers):
            l_id = f"layer_{i}"
            l_type = layer.get("type")
            
            if l_type == "residual_add":
                # Skip back to a previous layer
                skip_index = i - layer.get("params", {}).get("skip_back", 2)
                if skip_index < -1: skip_index = -1
                
                prev_info = history[skip_index]
                prev_channels = prev_info["channels"]
                prev_flattened = prev_info["is_flattened"]
                
                # 1. Align Channels if mismatch
                if prev_channels != current_channels:
                    proj_id = f"proj_{i}"
                    if not is_flattened:
                        module_defs.append(f"{self.indent}{self.indent}self.{proj_id} = nn.Conv2d({prev_channels}, {current_channels}, kernel_size=1)")
                        forward_steps.append(f"res_{i} = self.{proj_id}(tensors[{skip_index}])")
                    else:
                        module_defs.append(f"{self.indent}{self.indent}self.{proj_id} = nn.Linear({prev_channels}, {current_channels})")
                        forward_steps.append(f"res_{i} = self.{proj_id}(tensors[{skip_index}])")
                else:
                    forward_steps.append(f"res_{i} = tensors[{skip_index}]")
                
                # 2. Align Spatial if needed (Simplified: Adaptive Pooling)
                if not is_flattened and not prev_flattened:
                    # In a real DAG, we'd calculate H/W, here we use Lazy pooling or interpolation
                    # For prototype, we'll use F.interpolate to match current x size
                    forward_steps.append(f"if res_{i}.shape[-2:] != x.shape[-2:]:")
                    forward_steps.append(f"{self.indent}res_{i} = F.interpolate(res_{i}, size=x.shape[-2:], mode='bilinear', align_corners=False)")
                
                forward_steps.append(f"x = x + res_{i}")
                forward_steps.append(f"tensors[{i}] = x")
                history[i] = {"channels": current_channels, "is_flattened": is_flattened}
            else:
                l_code, f_code, current_channels, is_flattened = self._translate_layer(
                    layer, l_id, current_channels, is_flattened
                )
                module_defs.extend(l_code)
                forward_steps.extend(f_code)
                forward_steps.append(f"tensors[{i}] = x")
                history[i] = {"channels": current_channels, "is_flattened": is_flattened}

        # Final Classifier Head
        if not is_flattened:
            module_defs.append(f"{self.indent}{self.indent}self.global_pool = nn.AdaptiveAvgPool2d((1, 1))")
            forward_steps.append("x = self.global_pool(x)")
            module_defs.append(f"{self.indent}{self.indent}self.final_flatten = nn.Flatten()")
            forward_steps.append("x = self.final_flatten(x)")
        
        module_defs.append(f"{self.indent}{self.indent}self.classifier = nn.LazyLinear(num_classes)")
        forward_steps.append("x = self.classifier(x)")
        forward_steps.append("return x")

        code.extend(module_defs)
        # Assemble forward
        code.append(f"{self.indent}def forward(self, x):")
        code.append(f"{self.indent}{self.indent}tensors = {{-1: x}}")
        for step in forward_steps:
            code.append(f"{self.indent}{self.indent}{step}")
            
        return "\n".join(code)

    def _translate_layer(self, layer: Dict, l_id: str, current_channels: int, is_flattened: bool):
        """Translates a single layer dict into PyTorch init and forward code."""
        l_type = layer.get("type") or layer.get("op")
        module_defs = []
        forward_steps = []
        
        if l_type == "conv2d":
            if is_flattened:
                forward_steps.append(f"x = x.view(x.size(0), {current_channels}, 1, 1)")
                is_flattened = False
            out_channels = layer.get("params", {}).get("filters", 32)
            kernel = layer.get("params", {}).get("kernel", 3)
            
            if current_channels == 0:
                module_defs.append(f"{self.indent}{self.indent}self.{l_id} = nn.LazyConv2d({out_channels}, kernel_size={kernel}, padding={kernel//2})")
            else:
                module_defs.append(f"{self.indent}{self.indent}self.{l_id} = nn.Conv2d({current_channels}, {out_channels}, kernel_size={kernel}, padding={kernel//2})")
            
            forward_steps.append(f"x = F.{layer.get('params', {}).get('activation', 'relu')}(self.{l_id}(x))")
            current_channels = out_channels
            
        elif l_type == "dense":
            if not is_flattened:
                module_defs.append(f"{self.indent}{self.indent}self.flatten_{l_id} = nn.Flatten()")
                forward_steps.append(f"x = self.flatten_{l_id}(x)")
                is_flattened = True
                module_defs.append(f"{self.indent}{self.indent}self.{l_id} = nn.LazyLinear({layer.get('params', {}).get('units', 128)})")
            else:
                module_defs.append(f"{self.indent}{self.indent}self.{l_id} = nn.Linear({current_channels}, {layer.get('params', {}).get('units', 128)})")
            forward_steps.append(f"x = F.{layer.get('params', {}).get('activation', 'relu')}(self.{l_id}(x))")
            current_channels = layer.get("params", {}).get("units", 128)

        elif l_type == "pooling":
            if is_flattened:
                forward_steps.append(f"x = x.view(x.size(0), {current_channels}, 1, 1)")
                is_flattened = False
            p_type = layer.get("params", {}).get("type", "max")
            size = layer.get("params", {}).get("size", 2)
            p_class = "MaxPool2d" if p_type == "max" else "AvgPool2d"
            module_defs.append(f"{self.indent}{self.indent}self.{l_id} = nn.{p_class}(kernel_size={size})")
            forward_steps.append(f"x = self.{l_id}(x)")

        elif l_type == "dropout":
            rate = layer.get("params", {}).get("rate", 0.2)
            module_defs.append(f"{self.indent}{self.indent}self.{l_id} = nn.Dropout({rate})")
            forward_steps.append(f"x = self.{l_id}(x)")

        elif l_type == "batch_norm":
            if not is_flattened:
                module_defs.append(f"{self.indent}{self.indent}self.{l_id} = nn.BatchNorm2d({current_channels})")
            else:
                module_defs.append(f"{self.indent}{self.indent}self.{l_id} = nn.BatchNorm1d({current_channels})")
            forward_steps.append(f"x = self.{l_id}(x)")

        return module_defs, forward_steps, current_channels, is_flattened

    def _generate_dag_code(self, genome: Genome) -> str:
        """
        Translates a DAG-based cell genome into a PyTorch Module.
        Handles topological sorting, multiple inputs, and shape alignment.
        """
        arch = genome.architecture
        # For simplicity in this step, we assume 'nodes' and 'adjacency' are in arch
        # This corresponds to a single-cell macro or a global DAG
        nodes = arch.get("nodes", [])
        adj = arch.get("adjacency", {})
        input_shape = arch.get("input_shape", [3, 32, 32])
        
        if not nodes:
            return self._generate_sequential_code(genome)

        try:
            topo_order = get_topological_sort(nodes, adj)
            leaf_nodes = get_leaf_nodes(nodes, adj)
        except ValueError as e:
            logger.error(f"DAG Validation failed: {e}")
            return self._generate_sequential_code(genome)

        code = [
            "import torch",
            "import torch.nn as nn",
            "import torch.nn.functional as F",
            "",
            "class EvolvedModel(nn.Module):",
            f"{self.indent}def __init__(self, input_shape={input_shape}, num_classes={arch.get('output_classes', 10)}):",
            f"{self.indent}{self.indent}super().__init__()",
            f"{self.indent}{self.indent}self.input_shape = input_shape",
            f"{self.indent}{self.indent}self.topo_order = {topo_order}",
            f"{self.indent}{self.indent}self.leaf_nodes = {leaf_nodes}",
            ""
        ]

        # 1. Inverse Adjacency for forward pass (to know which tensors to grab)
        parents = {str(n['id']): [] for n in nodes}
        for src, targets in adj.items():
            for target in targets:
                if target in parents:
                    parents[target].append(src)

        module_defs = []
        # Pre-registration of modules
        for node in nodes:
            n_id = str(node['id'])
            op = node.get('op')
            if op == 'input': continue
            
            # Build the operation module
            l_code, _, _, _ = self._translate_layer(node, f"op_{n_id}", 0, False) # Channels resolved by Lazy/Shape Inference later
            module_defs.extend(l_code)
            
            # Handle Alignment Modules for multiple inputs
            node_parents = parents.get(n_id, [])
            if len(node_parents) > 1:
                # Determine target channels for alignment
                target_channels = 32 # Default
                if op == 'conv2d':
                    target_channels = node.get('params', {}).get('filters', 32)
                
                for p_id in node_parents:
                    module_defs.append(f"{self.indent}{self.indent}self.align_{p_id}_to_{n_id} = nn.LazyConv2d({target_channels}, kernel_size=1)")

        # 2. Forward Pass Construction
        forward_steps = [f"tensors = {{-1: x}}"] # Note: genome IDs might differ, usually node 0 is input
        
        for n_id in topo_order:
            node_data = next(n for n in nodes if str(n['id']) == n_id)
            if node_data['op'] == 'input':
                forward_steps.append(f"tensors['{n_id}'] = x")
                continue
                
            node_parents = parents.get(n_id, [])
            
            # Gather and align inputs
            if len(node_parents) == 0:
                forward_steps.append(f"tensors['{n_id}'] = x # Orphan fallback")
            elif len(node_parents) == 1:
                p_id = node_parents[0]
                forward_steps.append(f"in_{n_id} = tensors['{p_id}']")
                forward_steps.append(f"tensors['{n_id}'] = self.op_{n_id}(in_{n_id})")
            else:
                # Multiple inputs -> Fusion (Add by default)
                forward_steps.append(f"inputs_{n_id} = []")
                for p_id in node_parents:
                    # Apply alignment logic from NAS.md
                    forward_steps.append(f"t_{p_id} = self.align_{p_id}_to_{n_id}(tensors['{p_id}'])")
                    # Spatial matching
                    forward_steps.append(f"if t_{p_id}.shape[-2:] != tensors['{node_parents[0]}'].shape[-2:]:")
                    forward_steps.append(f"{self.indent}t_{p_id} = F.interpolate(t_{p_id}, size=tensors['{node_parents[0]}'].shape[-2:], mode='bilinear', align_corners=False)")
                    forward_steps.append(f"inputs_{n_id}.append(t_{p_id})")
                
                # Fusion (Addition)
                fusion_out = f"torch.stack(inputs_{n_id}).sum(dim=0)"
                if node_data['op'] == 'concat':
                    fusion_out = f"torch.cat(inputs_{n_id}, dim=1)"
                
                if node_data['op'] in ['add', 'concat']:
                    forward_steps.append(f"tensors['{n_id}'] = {fusion_out}")
                else:
                    forward_steps.append(f"tensors['{n_id}'] = self.op_{n_id}({fusion_out})")

        # 3. Output Fusion (Leaf concatenation)
        if len(leaf_nodes) > 1:
            forward_steps.append(f"out = torch.cat([tensors[nid] for nid in {leaf_nodes}], dim=1)")
        else:
            forward_steps.append(f"out = tensors['{leaf_nodes[0]}']")
            
        forward_steps.append("return self.classifier(self.global_pool(out).flatten(1))")
        
        # Add boilerplate
        module_defs.append(f"{self.indent}{self.indent}self.global_pool = nn.AdaptiveAvgPool2d((1, 1))")
        module_defs.append(f"{self.indent}{self.indent}self.classifier = nn.LazyLinear(num_classes)")

        code.extend(module_defs)
        code.append("")
        code.append(f"{self.indent}def forward(self, x):")
        for step in forward_steps:
            code.append(f"{self.indent}{self.indent}{step}")
            
        return "\n".join(code)

    def save_to_file(self, genome: Genome, filepath: str):
        """Generates code and saves it to a file."""
        code = self.generate_module_code(genome)
        with open(filepath, "w") as f:
            f.write(code)
        logger.info(f"Generated code saved to {filepath}")

class SecurityError(Exception):
    """Raised when a genome violates security policies."""
    pass
