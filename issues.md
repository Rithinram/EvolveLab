Here's a dense audit across the codebase, organized by severity and category.

---

## Critical / Correctness Bugs

**`backend/websocket.py` — `sync_broadcast` is broken**

```python
loop = asyncio.get_event_loop()
if loop.is_running():
    asyncio.ensure_future(self.broadcast(message))
```

`asyncio.ensure_future` called from a non-async thread without a running loop in that thread's context. `ensure_future` requires the coroutine be scheduled on the *event loop's* thread. Use `asyncio.run_coroutine_threadsafe(self.broadcast(message), loop)` instead, caching the loop reference at startup (not at call time via `get_event_loop`).

**`database/crud.py` — Session leak on exception paths**

Every method does `session = self.get_session()` then `finally: session.close()`, but `get_session()` returns a raw `Session` without being used as a context manager. If the session factory itself throws, `session` is unbound and the `finally` block will `NameError`. Use `with self.SessionFactory() as session:` (SQLAlchemy 1.4+ supports this).

**`evolution/engine.py` — `engine_lock` never guards the engine assignment**

```python
with engine_lock:
    if engine and engine.running:
        return {"error": "Evolution already running"}
# ← engine assigned OUTSIDE the lock
engine = EvolutionEngine(cfg, db)
```

Classic TOCTOU. Two concurrent requests both pass the guard, then both create engines and start threads. The lock needs to wrap the check+assign atomically.

**`training/code_generator.py` — `exec()` with user-controlled `genome` data**

The `validate_genome` regex check is insufficient. It lowercases the arch string and checks for `"import"`, but:

* `{"type": "IMPORT"}` passes (it converts to lowercase but only for pattern matching, and the check is on `arch_str = str(genome.architecture).lower()` — actually this is fine)
* More critically, `layer.get("params", {})` values are never validated. A param value like `"__import__('os').system('rm -rf /')"` would pass the layer name check (name is fine) but the value goes directly into `exec`'d code via f-strings in `_translate_layer`. The activation function is interpolated directly: `F.{layer.get('params', {}).get('activation', 'relu')}` — this must be whitelisted, not passed raw.

**`agents/crossover.py` — `_validate_and_fix_dimensions` is dead on genomes without DAG nodes**

The method only handles `attention.dim` as a "known input expectation". In practice, dense→attention transitions and conv→dense transitions (which go through flatten) aren't modeled, so the adapter insertion misses the most common dimension mismatches in the actual genome schema.

---

## Race Conditions / Thread Safety

**`backend/api.py` — Global mutable state shared across threads**

```python
engine: Optional[EvolutionEngine] = None
evolution_thread: Optional[threading.Thread] = None
```

`engine` is read in `get_status`, `pause_evolution`, `resume_evolution`, `stop_evolution` without holding `engine_lock`. Any of these can race with `start_evolution` or the evolution thread completing and setting `engine.running = False`.

**`evolution/engine.py` — `self.running` / `self.paused` accessed from two threads**

The evolution runs in a daemon thread, while `pause()` / `resume()` / `stop()` are called from FastAPI's thread pool. These boolean writes are not protected by a lock. CPython's GIL makes this *usually* safe for simple booleans, but it's undefined behavior in the language spec. Use `threading.Event` instead.

---

## Memory / Resource Issues

**`training/pytorch_evaluator.py` — GPU memory not cleared on OOM path**

```python
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        return {..., "error": "OOM"}
```

The `finally` block does `torch.cuda.empty_cache()`, but `del model` only runs `if model is not None`. On OOM during `ModelClass(...)` instantiation, `model` stays `None`, but partial CUDA allocations from the failed instantiation are not freed by `empty_cache()` alone — you need to also call the garbage collector explicitly: `import gc; gc.collect(); torch.cuda.empty_cache()`.

**`training/pytorch_evaluator.py` — `_loaders_cache` holds DataLoader references forever**

DataLoaders pin memory workers (even with `num_workers=0` they hold dataset references in memory). The cache grows unbounded across generations. The dataset name set is small, so this isn't catastrophic, but the underlying dataset objects (60K MNIST samples in RAM) are never released even if the evaluator switches modes.

**`memory/memory_store.py` — `mutation_deltas` cap is applied after-the-fact**

```python
self.mutation_deltas[mutation_type].append(delta)
if len(self.mutation_deltas[mutation_type]) > self.MAX_HISTORY:
    self.mutation_deltas[mutation_type] = self.mutation_deltas[mutation_type][-self.MAX_HISTORY:]
```

This allocates `MAX_HISTORY + 1` entries before trimming every call. Use `collections.deque(maxlen=MAX_HISTORY)` to avoid the repeated slice allocation.

---

## Logic / Algorithmic Issues

**`training/heuristic.py` — `estimate_param_count` is called inside `evaluate` but `genome.metrics` has a side effect**

`evaluate` calls `genome.estimate_param_count()` which writes `self.metrics["param_count"]`. But `EvaluatorAgent.evaluate_population` then calls `genome.metrics.update(metrics)` which would overwrite it with a possibly different value from heuristic. The write order is fragile.

**`agents/crossover.py` — Crossover can produce 0-layer genomes**

```python
child_layers = layers_a[:point_a] + layers_b[point_b:]
if len(child_layers) < 1:
    child_layers = [layers_a[0]] if layers_a else [layers_b[0]]
```

`point_a = 0` and `point_b = len(layers_b)` is a valid random outcome — produces an empty list, then falls through to the 1-layer fallback. But the single-layer fallback might be a `batch_norm` or `dropout` layer as the entire architecture, which will produce a degenerate (and possibly broken) model.

**`evolution/engine.py` — `_close_mutation_feedback` uses `prev_avg` as baseline when `fitness_before` is None**

The comment is correct that children of crossover have no pre-mutation fitness. But the fix (using `prev_avg`) means a mutation that produces a genome with fitness *above the previous generation's average* is marked successful, even if it's below the elite's fitness. The adaptive weight calculation will skew toward mutations that simply beat an already-outdated baseline.

**`agents/selection.py` — Diversity injection can swap out an elite**

```python
selected[-1] = diverse  # Replace weakest selection
```

If `elite_count >= num_parents - 1`, `selected[-1]` could be an elite. There's no guard ensuring elites are preserved when diversity injection fires.

**`training/code_generator.py` — `_generate_dag_code` uses `LazyConv2d` for all alignment ops**

The alignment modules `self.align_{p}_to_{n}` are always registered as `LazyConv2d`, but input tensors might be already-flattened 1D tensors at that point in the graph (e.g., after a Dense node). Applying Conv2d to a 2D tensor (batch × features) will crash.

---

## Security

**`training/code_generator.py` — `validate_genome` doesn't cover `params` values**

Already noted above. Specifically:

```python
forward_steps.append(f"x = F.{layer.get('params', {}).get('activation', 'relu')}(self.{l_id}(x))")
```

A genome with `{"activation": "dropout"}` would call `F.dropout(...)` which is valid but unexpected. More critically, `{"activation": "__import__('os').system('id')"}` would produce executable injection. Whitelist activations: `VALID_ACTIVATIONS = {"relu", "gelu", "silu", "sigmoid", "tanh", "leaky_relu"}`.

**`backend/api.py` — No rate limiting or auth on evolution control endpoints**

`POST /api/evolution/start` is publicly accessible. Any client can trigger unbounded PyTorch training. If deployed anywhere beyond localhost, this is a remote compute abuse vector.

---

## Design / Architecture Issues

**`training/datasets/__init__.py` — `get_dataset_loaders` with `num_workers=0` hardcoded**

Removes multiprocessing entirely. On CPU machines, data loading becomes the bottleneck during real training. This should come from hardware profile: `num_workers = min(4, hw_profile["cores"] // 2)`.

**`evolution/engine.py` — `_generate_initial_population` ratio logic doesn't distribute evenly**

```python
per_agent = max(1, candidate_size // len(self.builders))
remainder = candidate_size - per_agent * len(self.builders)
```

The remainder is distributed only to the first `remainder` agents (indices 0 to `remainder-1`). With 5 agents and `candidate_size=24`, `per_agent=4`, `remainder=4` — so 4 agents get 5 and 1 gets 4. This is cosmetically fine, but the first agent (transformer_specialist) always gets the extra, systematically overrepresenting it in the warmup pool.

**`agents/mutation.py` — `_modify_layer` mutates in-place on a shallow copy**

`mutant = genome.clone(new_id=True)` does `copy.deepcopy(self)`, so mutations are safe. But `layers = genome.architecture.get("layers", [])` in `_apply_mutation` grabs the reference from `mutant`, and then direct `layer["units"] = ...` mutations work correctly. This is actually fine, but the comment about `genome` vs `mutant` is confusing since `_apply_mutation` receives `mutant` named `genome`.

**`backend/websocket.py` — `_event_queue: asyncio.Queue = None` declared but never used**

Dead code that misleads readers into thinking there's an async queue-based broadcast pattern.

**`configs/default.json` — `"evaluation.mode": "heuristic"` but `requirements.txt` doesn't include `torch`**

The requirements file has `torchinfo` but no `torch` or `torchvision`. The system silently works in heuristic mode but will fail at import time in `training/pytorch_evaluator.py` and `training/proxy_evaluator.py` if PyTorch isn't installed separately (e.g., fresh CI environment). Torch should be in requirements with an appropriate index URL, or conditionally imported.

---

## Test Coverage Gaps

* `tests/test_integration.py` — `evaluation.dataset = "synthetic"` bypasses real data loading; doesn't test the `_get_loaders` cache or actual MNIST path.
* `tests/test_graph_utils.py` — No test for `get_node_depths`. No test for `get_leaf_nodes` when *all* nodes have outgoing edges (should return empty list).
* `agents/meta_prompt.py` — Zero test coverage on the prompt mutation strategy selection logic (the `improving`/`declining` branch).
* `agents/selection.py` — No test for the diversity injection path; the elite-overwrite bug above would be caught here.
* `training/heuristic.py` — No tests at all; given it's the default evaluation mode, this is the highest-ROI missing coverage.
