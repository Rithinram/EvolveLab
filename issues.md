
Here's what remains broken or newly introduced after the fixes:

---

## Still Broken

**`database/crud.py` — `SessionFactory` doesn't support context manager protocol**

```python
with self.SessionFactory() as session:
```

`sessionmaker()` returns a factory whose instances are `Session` objects. `Session` supports `__enter__`/`__exit__` only in SQLAlchemy 1.4+ with `Session(bind=engine)` directly, not via a factory-produced session in all configurations. The safe pattern is:

```python
session = self.SessionFactory()
try:
    ...
    session.commit()
    return result
except Exception as e:
    session.rollback()
    raise
finally:
    session.close()
```

Or configure the factory with `autoclose=True` and use `with Session(engine) as session:` instead of the factory pattern entirely.

**`database/crud.py` — `get_genome_lineage` opens N sessions in a loop**

```python
def get_genome_lineage(self, genome_id: str, max_depth: int = 10) -> List[dict]:
    for _ in range(max_depth):
        genome = self.get_genome(current_id)  # Opens/closes session each iteration
```

Each call to `get_genome` opens and closes a separate session. For a depth-10 lineage that's 10 round-trips. Load the full lineage in a single recursive CTE query instead.

**`backend/api.py` — `pause_evolution`, `resume_evolution`, `stop_evolution` still read `engine` without the lock**

```python
@app.post("/api/evolution/pause")
def pause_evolution():
    if engine and engine.running:  # ← race: engine could be None or swapped here
        engine.pause()
```

The fix applied to `get_status` was not applied to these three endpoints. All reads of the global `engine` need `with engine_lock:`.

**`backend/api.py` — `restore_checkpoint` assigns to global `engine` without the lock**

```python
@app.post("/api/checkpoints/{cp_id}/restore")
def restore_checkpoint(cp_id: int):
    global engine
    ...
    engine = EvolutionEngine(cfg, db)  # ← unsynchronized write
```

---

## New Issues Introduced

**`training/pytorch_evaluator.py` — `gc.collect()` runs even on success path, every evaluation**

```python
finally:
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

`gc.collect()` is an O(heap) operation. Running it after every single genome evaluation (potentially hundreds per run) adds measurable overhead, especially on CPU where there's no CUDA cache to justify it. Restrict to OOM recovery or call it every N evaluations.

**`training/pytorch_evaluator.py` — `import math` inside a hot loop**

```python
def evaluate_population(self, genomes, fidelity, on_batch):
    for genome in genomes:
        ...
        import math  # ← inside the loop body
        param_penalty = math.log10(params + 1) / 7.0
```

Module imports inside loops are cached by Python after the first call so this isn't a correctness bug, but it's misleading style and a lint error. Move to top-level import.

---

## Issues From Before That Were Not Fixed

**`training/code_generator.py` — Activation function injection still unsanitized**

```python
forward_steps.append(f"x = F.{layer.get('params', {}).get('activation', 'relu')}(self.{l_id}(x))")
```

The activation string is still interpolated directly into `exec`'d code. `validate_genome` checks `layer.get("type")` with a regex, but not `params` values. A genome with `{"activation": "dropout + os.system('id') #"}` still passes validation and produces injectable code. Add a whitelist:

```python
VALID_ACTIVATIONS = {"relu", "gelu", "silu", "sigmoid", "tanh", "leaky_relu", "elu"}
activation = layer.get("params", {}).get("activation", "relu")
if activation not in VALID_ACTIVATIONS:
    activation = "relu"
```

**`evolution/engine.py` — `_generate_initial_population` crashes if all candidates fail proxy scoring**

```python
final_population = scored_candidates[:self.population_size]
self._emit("warmup_complete", {
    "top_proxy_score": final_population[0].metrics["proxy_score"],  # IndexError if empty
```

If `ProxyEvaluator` returns `status: error` for all candidates (e.g., PyTorch not installed), `scored_candidates` is populated but all have `proxy_score=0`. The slice is fine, but if `population_size > len(scored_candidates)` (e.g., proxy errors reduced the pool), `final_population` could be shorter than expected. No guard exists.

**`agents/crossover.py` — Minimum layer count fallback selects structurally invalid layers**

```python
if len(child_layers) < 1:
    child_layers = [layers_a[0]] if layers_a else [layers_b[0]]
```

`layers_a[0]` could be `{"type": "batch_norm"}` or `{"type": "dropout", "rate": 0.3}` — layers that have no meaningful output on their own and will crash the code generator (BatchNorm2d with unknown channels, no Conv2d preceding it). The fallback should guarantee at least one conv or dense layer.

**`agents/selection.py` — Diversity threshold logic is inverted**

```python
def _needs_diversity(self, selected: List[Genome]) -> bool:
    species = set(g.species for g in selected)
    return len(species) / len(selected) < self.diversity_threshold
```

`diversity_threshold` defaults to `0.15`. With 5 species across 8 selected genomes, `5/8 = 0.625`, which is `> 0.15`, so diversity injection never triggers. With 1 species, `1/8 = 0.125 < 0.15`, so it triggers. The threshold is calibrated for nearly homogeneous populations only. With the 5 distinct species in the default config, this will essentially never fire. The threshold should be `> 0.5` or the metric should be defined differently (e.g., Gini coefficient over species counts).

**`memory/memory_store.py` — `mutation_deltas` still uses list slice instead of deque**

The cap is applied via slice after-the-fact on every append, allocating `MAX_HISTORY + 1` elements each time instead of using `deque(maxlen=50)`. Same for `species_fitness`, `species_survival`, `prompt_fitness_history`, and `generation_stats` — all have the same pattern. Not a correctness bug but unnecessary allocation at scale.

**`database/models.py` — `datetime.datetime.utcnow` is deprecated in Python 3.12+**

```python
timestamp = Column(DateTime, default=datetime.datetime.utcnow)
```

`datetime.utcnow()` is deprecated since Python 3.12 and removed in future versions. Replace with `datetime.datetime.now(datetime.timezone.utc)` and store as timezone-aware, or use `func.now()` in SQLAlchemy for DB-side timestamps.

**`.github/workflows/ci.yml` — Tests require torch/torchvision but neither is in `requirements.txt`**

The CI installs `requirements.txt` and then runs `pytest tests/`. Every test in `tests/` imports from `training.*` which imports `torch`. The CI will fail with `ModuleNotFoundError: No module named 'torch'` on a fresh Ubuntu runner. Add a CPU-only torch install step or add `torch` and `torchvision` to requirements with the `--index-url https://download.pytorch.org/whl/cpu` index.
