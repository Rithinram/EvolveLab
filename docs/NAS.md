## EvolveLab Genome Representation for DAG Architectures

This specification defines a JSON-based genome, a mutation-safe DAG structure, and a robust PyTorch translation engine that together support NAS with ResNet‑style skips, DenseNet concats, and arbitrary directed acyclic graphs.

---

### 1. JSON Data Schema

The genome describes an entire network as a **macro‑architecture** that stacks **cell templates**, where each cell is a DAG of primitive operations. This hierarchical decomposition follows the best practice of separating cell search (micro) from network depth/width (macro), keeping the search space manageable.

#### 1.1 Root Level

```json
{
  "version": "1.0",
  "macro": { … },
  "normal_cell": { … },
  "reduction_cell": { … }
}
```

- `macro` – defines how cells are stacked, channel expansion, and number of repeats.
- `normal_cell` / `reduction_cell` – reusable DAG blueprints for feature extraction and down‑sampling stages.

#### 1.2 Macro Structure

```json
"macro": {
  "input_shape": [3, 32, 32],
  "stem_channels": 16,
  "stem_ops": [
    {"op": "conv3x3", "out_channels": 16, "stride": 1}
  ],
  "stages": [
    {"normal_cells": 5, "reduction": false, "base_channels": 16},
    {"normal_cells": 5, "reduction": true,  "base_channels": 32},
    {"normal_cells": 5, "reduction": true,  "base_channels": 64}
  ]
}
```

- `stem_ops` – initial convolution before cells (optional).
- `stages` – each stage applies a sequence of cells. When `reduction: true`, the first cell uses the `reduction_cell` (or a reduction cell is prepended); all subsequent cells use `normal_cell`. Channel counts double after each reduction.

#### 1.3 Cell DAG – Node‑Based Representation

A cell is a DAG of **nodes**, each representing an operation that takes inputs from earlier nodes. The output of the cell is formed by concatenating all “leaf” nodes (nodes not fed into any other node) along the channel dimension.

**Node object:**

| Field       | Type             | Description                                                                                                                                     |
| ----------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`      | `int`          | Unique within the cell, corresponding to the positional index in the `nodes` list.                                                            |
| `op`      | `string`       | Operation type (see §1.4).                                                                                                                     |
| `params`  | `object`       | Hyper‑parameters for the operation (e.g.,`out_channels`, `kernel_size`, `stride`).                                                       |
| `inputs`  | `array[int]`   | List of node IDs providing input to this node.**Must be monotonically decreasing (all IDs < current `id`)** to guarantee acyclic graph. |
| `comment` | `string` (opt) | Human‑readable label, ignored by the engine.                                                                                                   |

**Special nodes:**

- The cell’s input is represented by a virtual node with `id = -1` (or `0` if we shift all IDs). To simplify, we assign `id = 0` to the *cell input placeholder*, which always has `op = "input"` and no `inputs`.
- There is no separate output node; all nodes with no children (i.e., not referenced by any other node’s `inputs`) are concatenated to form the cell output.

**Cell JSON structure:**

```json
"normal_cell": {
  "nodes": [
    {
      "id": 0,
      "op": "input",
      "params": {},
      "inputs": []
    },
    {
      "id": 1,
      "op": "conv3x3",
      "params": {"out_channels": 64, "stride": 1, "padding": 1},
      "inputs": [0]
    },
    {
      "id": 2,
      "op": "conv3x3",
      "params": {"out_channels": 64, "stride": 1, "padding": 1},
      "inputs": [0]
    },
    {
      "id": 3,
      "op": "add",
      "params": {},
      "inputs": [1, 2]
    }
  ]
}
```

**Key design decisions:**

- **Topological order enforced by construction:** nodes are added in strict increasing order, and `inputs` only reference earlier IDs. This makes cycle detection trivial (impossible) and mutations safe.
- **Edges are encoded implicitly as references in `inputs`.** There is no separate adjacency list; the DAG is fully described by the node list.
- **Skip connection = adding an edge**, i.e., inserting a source ID into the `inputs` of a later node.
- **Multiple input nodes (`add`, `concat`) are first‑class operations.** Shape mismatches are resolved automatically by the translation engine (see §3).

---

### 2. Mutation Operations and DAG Integrity

All mutation operators preserve the DAG property because nodes remain topologically ordered (`new_id > max(inputs)`). The genome manipulation library must enforce this invariant after every mutation (e.g., by keeping the node list sorted or by generating new IDs that are strictly larger).

#### Core mutation primitives

| Operation                        | How to Apply                                                                                                                                                                   | DAG Guarantee                                        |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------- |
| **Add edge** (insert skip) | Append `src_id` to `dest.inputs` if `src_id < dest_id` and edge does not already exist.                                                                                  | `src_id < dest_id` ⇒ no cycle                     |
| **Remove edge**            | Remove `src_id` from `dest.inputs`.                                                                                                                                        | always safe                                          |
| **Insert node on edge**    | Replace edge `A → B` with `A → C → B`: create new node `C` with `inputs = [A]` and `id = max_id+1`; remove `A` from `B.inputs`; add `C.id` to `B.inputs`. | `A < C.id`, `C.id < B.id` must hold → see below |
| **Delete node**            | Remove node `X`; for all nodes that had `X` in `inputs`, replace with `X.inputs` (or simply remove the edge). Advanced: reconnect to preserve connectivity.            | always safe                                          |
| **Change operation**       | Modify `op` or `params` of a node.                                                                                                                                         | DAG untouched                                        |

**Insert‑node‑on‑edge without breaking order:**
If we always assign new IDs greater than all existing IDs, inserting a node on edge `A → B` would place `C` at the *end* of the list, but then `C.id > B.id`, which violates the ordering constraint. To avoid re‑sorting the whole node list, we can simply **append** the new node with a higher ID and adjust the graph to maintain topological order: let `C` be the new node, `id = max_id+1`. To keep `B` still receiving from an earlier node, we must ensure `C.id < B.id`, which is false.

**Solution:** when inserting a node on edge `A→B`, we actually **do not require strict monotonicity for all references**; we only require that the graph remains acyclic. A simpler and more common approach is to **store nodes in topological order after each mutation** by performing a topological sort. This adds minimal overhead (cell graphs have < 20 nodes) and frees us from artificial ID ordering. The genome JSON does not rely on IDs being sorted – the translation engine will topological sort anyway before building the model.

Thus the **recommended mutation‑safety protocol:**

1. Perform mutation (add node, connect edges).
2. Verify that the graph is acyclic (DFS cycle detection).
3. Re‑index nodes in topological order and rewrite `inputs` accordingly.
4. Optionally, prune orphaned nodes.

This guarantees a valid DAG after every mutation and keeps the JSON representation clean.

**Example mutation code (pseudocode):**

```python
def insert_skip_connection(cell, src_id, dest_id):
    # validate acyclic
    if not would_create_cycle(cell.nodes, src_id, dest_id):
        node = find_node(cell.nodes, dest_id)
        if src_id not in node.inputs:
            node.inputs.append(src_id)
```

---

### 3. PyTorch Translation Pattern

The `CodeGenerator` produces an `nn.Module` whose `forward()` method evaluates the DAG according to the topological order of nodes. The generator must also resolve spatial and channel mismatches when combining tensors (e.g., for `add` and `concat` operations).

#### 3.1 Building the Module

Parse steps:

1. **Topological sort** nodes (ignoring the list order) using a standard Kahn’s algorithm. This gives execution order.
2. For each node, create a PyTorch module that implements its operation. For simple ops (`conv3x3`, `batch_norm`, `relu`, etc.) we create a `nn.Sequential` block. For `add`/`concat`, no trainable parameters – they are pure tensor operations.
3. The `forward` method:
   - Maintain a dictionary `tensor_map` mapping `node_id -> output_tensor`.
   - Iterate over nodes in topological order.
   - Gather inputs: for each `input_id` in `node.inputs`, fetch `tensor_map[input_id]`.
   - If the operation is `add` or `concat`, automatically apply **shape alignment** to all inputs (see below).
   - Apply the node’s operation and store the result.
   - After processing all nodes, collect tensors from all *leaf nodes* (nodes not used as input to any other node) and concatenate them along channel dimension for the cell output.

#### 3.2 Automatic Shape Resolution for `add` and `concat`

When an `add` (element‑wise addition) or `concat` (channel concatenation) node receives tensors with different spatial sizes or channels, the engine must equalise them. The **target shape** is defined as the **maximum spatial resolution and channel count** among all inputs (or, for `add`, all inputs must have exactly the same spatial size and channel count after alignment). For `concat`, spatial sizes must match, but channel sizes may differ.

**Alignment rules (executed on the fly in `forward`):**

1. **For `add`:**

   - Determine maximum spatial dimensions (H,W) and channels (C) across all input tensors.
   - For each input tensor:
     - If spatial dims mismatch → apply adaptive pooling (e.g., `torch.nn.functional.interpolate` with `mode='nearest'` if needed) or a stride‑2/2x2 pooling to reach target size. **Better:** use a learnable `AdaptiveAvgPool2d(target_size)` or a 1×1 conv with stride.
     - If channel count mismatch → apply a 1×1 convolution with `out_channels = target_channels` (bias=False). An `nn.Conv2d(in_ch, target_ch, 1)` will match channels without affecting spatial size if stride=1.

   After alignment, all tensors are summed element‑wise.
2. **For `concat`:**

   - Spatial sizes must be equal; if they differ, upsample/downsample each to the **largest** spatial size among inputs using interpolation or `AdaptiveAvgPool2d`.
   - Then concatenate along the channel dimension (no channel alignment needed).

**Implementation:** The `CodeGenerator` injects a helper function `_align_shapes(tensors, mode='add')` that is called inside the generated `forward()`:

```python
import torch
import torch.nn.functional as F

class CellModule(nn.Module):
    def __init__(self, genome_cell, in_channels, out_channels):
        super().__init__()
        # … build node modules …
        self._node_modules = nn.ModuleList([...])

    def forward(self, x):
        # special handling for input node
        tensors = {0: x}   # node 0 is input
        # … store intermediate tensors …
        for node in self._topo_order[1:]:
            inputs = [tensors[in_id] for in_id in node.inputs]
            if node.op == 'add':
                aligned = self._align_for_add(inputs)
                out = sum(aligned)
            elif node.op == 'concat':
                aligned = self._align_for_concat(inputs)
                out = torch.cat(aligned, dim=1)
            else:
                # regular operation
                out = self._node_modules[node.id](inputs[0])
            tensors[node.id] = out
        # gather leaf outputs
        leaf_ids = self._leaf_nodes
        return torch.cat([tensors[lid] for lid in leaf_ids], dim=1)

    @staticmethod
    def _align_for_add(tensor_list):
        # find target spatial size (max H, W) and max channels
        shapes = [t.shape[-2:] for t in tensor_list]  # (H, W)
        max_h, max_w = max(s[0] for s in shapes), max(s[1] for s in shapes)
        max_c = max(t.shape[1] for t in tensor_list)
        aligned = []
        for t in tensor_list:
            if t.shape[1] != max_c:
                # 1x1 conv projection (global for cell) – need to store module
                # In practice, we create a nn.Conv2d per edge, but here we use a temp conv
                t = F.conv2d(t, _get_proj_weight(t.shape[1], max_c), stride=1)
            if t.shape[-2:] != (max_h, max_w):
                t = F.interpolate(t, size=(max_h, max_w), mode='bilinear', align_corners=False)
            aligned.append(t)
        return aligned
```

**Better practice:** Instead of creating temporary convolutions every forward pass, the code generator should **materialise learnable adapter modules** (e.g., a `nn.Conv2d` for each edge that leads to an `add` node) during initialisation. This can be done by analysing mismatches at graph construction time and inserting hidden adapter nodes in the internal module list, but that would complicate the genome. For a clean separation, the genome can optionally include explicit `proj_1x1` nodes, and the mutation engine can insert them when adding a skip connection across different resolutions/channels. The choice depends on how much automatic handling you want.

**Recommended hybrid approach:**

- The translation engine **first builds the graph with explicit adapters**. For every edge entering an `add` or `concat`, it checks if the source node’s output shape will differ from the expected target shape. If so, it **wraps the edge** with a small adapter (1×1 conv for channels, pooling/interpolation for spatial) *before* the merge, effectively treating the adapter as part of the edge. This is transparent to the genome – the mutation agent just adds connections, and the builder fixes dimensions.
- The builder knows the shape evolution because it simulates the forward pass on a sample input to record output shapes per node (shape inference). With that, it can pre‑compute which edges need projection and insert appropriate modules. This is how AutoGluon and many NAS frameworks handle wild‑card skips.

For the specification, we define:

> The PyTorch translator must perform **shape inference** on the DAG by tracing a dummy input through the operations. It then inserts **edge adapters** (1×1 convolutions, bilinear upsampling, or adaptive pooling) where dimensions of two inputs to an `add`/`concat` node differ. These adapters are ordinary `nn.Module`s stored in a dictionary keyed by `(source_id, dest_id)` and applied during forward.

---

### 4. Expert Best Practices from Modern NAS

Leading frameworks (DARTS, NAS‑Bench‑201, AutoGluon, ENAS) consistently use **cell‑based micro‑search** with a fixed macro‑architecture for the following reasons:

#### 4.1 Cell vs. Macro‑Architecture Separation

- **Cell (Micro‑structure):** A small DAG (typically 4–7 nodes) that defines connectivity and operation choices. It is repeated many times in the network with different channel widths and spatial resolutions (normal cell / reduction cell).*NAS‑Bench‑201* defines each cell as a fully connected DAG of 4 intermediate nodes, searching only which edge operation (conv1×1, conv3×3, skip, etc.) is assigned.
- **Macro‑architecture:** Specifies the number of stacked cells, the number of channels at each stage, and where down‑sampling occurs. In most search spaces, the macro skeleton is fixed (e.g., 3 stages, 5 cells each) so that the search complexity is bounded by the cell DAG alone.

**Why this works:** Searching a complete network DAG of 50+ layers leads to an explosion of topology choices and makes it nearly impossible to train and compare architectures fairly. By confining the DAG to a small, repeatable cell, the search space stays exponential in the cell size but linear in network depth – a sweet spot that has produced state‑of‑the‑art results.

#### 4.2 Genome Design Implications

For EvolveLab:

- Define **two cell genomes** (`normal_cell`, `reduction_cell`) as separate DAGs.
- The **macro genome** stores the stage layout and channel multipliers but does *not* describe connectivity beyond cell placement.
- Mutation agents operate **primarily on the cell DAG** (adding/removing edges, swapping operations, inserting nodes). Macro mutations (e.g., adding a whole stage) are possible but less frequent to keep the search stable.

#### 4.3 Manageable Search Space Size

- Keep the cell node count between **4 and 8** (excluding input/output) to prevent unbounded growth.
- Use a predefined operation set (e.g., `none`, `skip`, `conv1x1`, `conv3x3`, `sep_conv3x3`, `dil_conv3x3`, `avg_pool3x3`, `max_pool3x3`).
  This limits the combinatorial explosion while covering building blocks of known high‑performance models.

---

### 5. Full Example

**Compact JSON Genome (2‑node cell with a skip connection):**

```json
{
  "version": "1.0",
  "macro": {
    "input_shape": [3, 32, 32],
    "stem_channels": 16,
    "stages": [
      {"normal_cells": 3, "reduction": false, "base_channels": 16},
      {"normal_cells": 3, "reduction": true,  "base_channels": 32},
      {"normal_cells": 3, "reduction": true,  "base_channels": 64}
    ]
  },
  "normal_cell": {
    "nodes": [
      {"id": 0, "op": "input",  "inputs": []},
      {"id": 1, "op": "conv3x3", "params": {"out_channels": 32}, "inputs": [0]},
      {"id": 2, "op": "conv3x3", "params": {"out_channels": 32}, "inputs": [1]},
      {"id": 3, "op": "add",     "inputs": [1, 2]},
      {"id": 4, "op": "output",  "inputs": [3]}   // optional, to mark leaf explicitly
    ]
  },
  "reduction_cell": {
    "nodes": [
      {"id": 0, "op": "input",  "inputs": []},
      {"id": 1, "op": "conv3x3", "params": {"out_channels": 32, "stride": 2}, "inputs": [0]},
      {"id": 2, "op": "conv3x3", "params": {"out_channels": 32, "stride": 2}, "inputs": [0]},
      {"id": 3, "op": "add",     "inputs": [1, 2]},
      {"id": 4, "op": "output",  "inputs": [3]}
    ]
  }
}
```

This clean schema, together with the automatic shape‑resolving translator and mutation‑friendly topological indexing, forms the foundation of EvolveLab’s powerful and flexible neural architecture search engine.
