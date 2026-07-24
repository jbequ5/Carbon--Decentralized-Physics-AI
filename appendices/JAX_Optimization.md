# JAX Core Training Optimizations (Validator-Side)

**Carbon PDE Subnet**  
**Version:** 2.0 (July 2026)  
**Status:** Core Engineering Appendix — Production Ready

This document specifies the JAX compilation, execution, and data-routing optimizations integrated into the Carbon Validator engine (`carbon/validator/training.py` and supporting modules). The goal is to reduce hardware runtime costs across all execution phases while preserving the mathematical intent of the miner's submitted `strategy.json`.

These optimizations are a foundational requirement for keeping validator-side compute manageable as the subnet scales from academic PDEs through compressible flow, multi-physics coupling, and 3D turbulence regimes.

---

## 1. Overview & Objectives

### Key Performance Targets

| Target | Description | Acceptance Criteria |
|--------|-------------|---------------------|
| **Zero Re-compilation** | Eliminate XLA compilation overhead during runtime block tempos caused by strategy-dependent control flow. | < 1 recompilation per 1000 submissions; < 5% compile time vs runtime |
| **VRAM Footprint Mitigation** | Reduce peak memory saturation by 30–50% using unified tensor precision handling (bfloat16) + gradient accumulation + checkpointing. | Phase 0-2: ≤ 40 GB on A100 80GB; Phase 3-4: ≤ 70 GB on H100 80GB |
| **Deterministic Execution** | Enforce strict hardware constraints and hard resource bounds while still honoring miner-defined optimization paths. | Bitwise identical outputs across 3 runs with same seed; numerical variance < 1e-7 |
| **Bounded Evaluation Latency** | Prevent runaway epoch budgets and sequential evaluation queues from monopolizing validator resources. | 99th percentile latency < 2× median; max queue wait < 5 min |

### Design Principles

1. **Preserve Miner Intent** — Optimizations must not silently rewrite or degrade a miner's stated strategy. Defaults may be applied only when fields are omitted.
2. **Static XLA Graphs** — Prefer functional masking, `lax.scan` / `lax.cond`, and `vmap` over Python-level conditionals that trigger recompilation.
3. **Hard Safety Rails** — Absolute step/wall-clock limits and early-stopping floors always take precedence over miner-specified epoch counts.
4. **Opportunistic Parallelism** — Vectorize only when architectures are identical; fall back cleanly otherwise.
5. **Correctness Over Speed** — Physics gates **must** run in fp32; loss masking must use explicit booleans; determinism is non-negotiable.

---

## 2. Dynamic Loss Masking (Unified XLA Graph)

### Problem

Miners can dynamically modify or omit loss terms (e.g., toggling `physics_residual` or `conservation_penalty`) in `strategy.json`. Native Python `if/else` statements inside the core loss calculation force an expensive XLA recompilation for every unique strategy configuration, creating a severe processing backlog.

### Solution: Explicit Boolean Masks (Correctness + Performance)

Execute a single static, unified loss function. All possible physical and data loss objectives are computed continuously. The miner's choices are expressed as **explicit boolean flags** (not floating-point thresholds) passed as runtime parameters.

```python
# carbon/validator/losses.py
import jax
import jax.numpy as jnp
from typing import Dict, NamedTuple
from jaxtyping import Array, Float

class LossWeights(NamedTuple):
    """Explicit boolean flags + weights — no floating-point threshold ambiguity."""
    data_mse: Float[Array, ""]
    data_mse_enabled: bool
    physics_residual: Float[Array, ""]
    physics_residual_enabled: bool
    boundary_mse: Float[Array, ""]
    boundary_mse_enabled: bool
    conservation_penalty: Float[Array, ""]
    conservation_penalty_enabled: bool

@partial(jax.jit, donate_argnums=(0, 1))  # Donate params, opt_state buffers
def unified_loss_fn(
    params: Dict,
    batch_inputs: Array,
    batch_targets: Array,
    weights: LossWeights,
    model_apply_fn
) -> Array:
    """
    Statically compiled loss function. Uses explicit boolean masking instead of
    procedural branch statements to maintain a single XLA compilation path.
    
    All residual functions must be safe when their corresponding weight is zero
    (no side-effects, no NaN paths, no division by zero).
    """
    # 1. Forward Pass
    predictions = model_apply_fn(params, batch_inputs)

    # 2. Continuous calculation of all objective terms
    loss_data = jnp.mean(jnp.square(predictions - batch_targets))
    loss_phys = compute_pde_residuals(params, batch_inputs, model_apply_fn)
    loss_bound = compute_boundary_residuals(params, batch_inputs, model_apply_fn)
    loss_conserve = compute_conservation_penalties(predictions)

    # 3. Explicit boolean masking — NO floating-point threshold ambiguity
    total_loss = (
        (weights.data_mse_enabled * weights.data_mse * loss_data) +
        (weights.physics_residual_enabled * weights.physics_residual * loss_phys) +
        (weights.boundary_mse_enabled * weights.boundary_mse * loss_bound) +
        (weights.conservation_penalty_enabled * weights.conservation_penalty * loss_conserve)
    )
    return total_loss
```

**Schema Integration (v1.0+)**
```json
{
  "loss": {
    "data_mse": {"enabled": true, "weight": 1.0},
    "physics_residual": {"enabled": true, "weight": 0.5},
    "boundary_mse": {"enabled": true, "weight": 0.3},
    "conservation_penalty": {"enabled": false, "weight": 0.0}
  }
}
```

**Why Boolean Flags**: Floating-point threshold `weight < 1e-8` is fragile. Miner submits `1e-9` → silently treated as 0. Boolean is explicit, auditable, and cannot be gamed.

---

## 3. Functional Early-Stopping Loop via `jax.lax.scan`

### Problem

Miners control the absolute `epochs` parameter. Rogue or poorly configured submissions can request extremely high epoch counts and monopolize validator compute. Traditional Python loop-based training prevents JAX from optimizing the step sequence into a single hardware trace and makes hard resource limits difficult to enforce.

### Solution: `lax.scan` + Hard Safety Rails

Structure the epoch sequence with `jax.lax.scan` and use `jax.lax.cond` for early-termination logic so that the control flow remains inside XLA. A hard absolute step limit and wall-clock timeout act as additional safety rails outside the scan.

```python
# carbon/validator/training.py
import jax
import jax.lax as lax
import jax.numpy as jnp
from typing import Tuple, Dict, NamedTuple, Callable, Any
from flax.training import train_state
import optax

class TrainerState(train_state.TrainState):
    """Extended train state with early-stopping metadata."""
    best_loss: Float[Array, ""]
    consecutive_no_improve: int
    terminated: bool
    epoch: int

def create_training_step(
    model_apply_fn: Callable,
    loss_fn: Callable,
    optimizer: optax.GradientTransformation,
    grad_accum_steps: int = 1,
    physics_precision: bool = False
) -> Callable:
    """Factory for compiled training step with gradient accumulation."""
    
    @partial(jax.jit, donate_argnums=(0, 1))
    def train_step(state: train_state.TrainState, batch) -> Tuple[train_state.TrainState, Float[Array, ""]]:
        # Gradient accumulation loop
        def accum_step(carry, micro_batch):
            params, opt_state, grad_accum = carry
            grads = compute_grads(params, micro_batch, physics_precision)
            grad_accum = jax.tree.map(lambda a, g: a + g, grad_accum, grads)
            return (params, opt_state, grad_accum), None
        
        micro_batches = split_batch(batch, grad_accum_steps)
        (params, opt_state, grad_accum), _ = lax.scan(
            lambda carry, mb: (accum_step(carry, mb), None),
            (state.params, state.opt_state, jax.tree.map(jnp.zeros_like, state.params)),
            batch
        )
        
        # Apply accumulated gradients
        grad_accum = jax.tree.map(lambda x: x / grad_accum_steps, grad_accum)
        updates, new_opt_state = optimizer.update(grad_accum, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        
        return state.replace(
            params=new_params, opt_state=new_opt_state, step=state.step + 1
        ), loss
    
    return train_step


def create_training_loop(
    train_step: Callable,
    max_epochs: int,
    patience: int = 50,
    hard_step_limit: int = None,
    wall_clock_limit_sec: int = 7200  # 2 hours default
) -> Callable:
    """Creates a compiled training loop with hard safety rails."""
    
    class LoopState(NamedTuple):
        state: train_state.TrainState
        best_loss: Float[Array, ""]
        consecutive_no_improve: int
        terminated: bool
        epoch: int
    
    @partial(jax.jit, donate_argnums=(0,))
    def epoch_step(state: Tuple, _):
        loop_state, epoch_idx = state
        
        def execution_branch(ls: LoopState):
            new_state, epoch_loss = train_step(ls.state, current_batch)
            
            improved = epoch_loss < ls.best_loss
            next_best = jnp.where(improved, epoch_loss, ls.best_loss)
            next_count = jnp.where(improved, 0, ls.consecutive_no_improve + 1)
            should_abort = next_count >= patience
            
            return LoopState(
                state=new_state,
                best_loss=next_best,
                consecutive_no_improve=next_count,
                terminated=should_abort,
                epoch=ls.epoch + 1
            ), epoch_loss
        
        def short_circuit_branch(ls: LoopState):
            return ls, ls.best_loss
        
        next_loop_state, epoch_loss = lax.cond(
            ls.terminated,
            lambda ls: (ls, ls.best_loss),
            execution_branch,
            loop_state
        )
        return (next_loop_state, epoch_idx + 1), epoch_loss
    
    def fit(init_state: train_state.TrainState, data_loader, max_epochs: int):
        effective_epochs = max_epochs
        if hard_step_limit is not None:
            effective_epochs = min(max_epochs, hard_step_limit)
        
        init_loop_state = LoopState(
            state=init_state,
            best_loss=jnp.inf,
            consecutive_no_improve=0,
            terminated=False,
            epoch=0
        )
        
        # Wall-clock timeout enforced OUTSIDE JIT (Python side)
        import time
        start_time = time.time()
        
        loop_state, loss_history = lax.scan(
            epoch_step,
            (init_loop_state, 0),
            jnp.arange(effective_epochs)
        )
        
        # Wall-clock check (outside JIT)
        if time.time() - start_time > wall_clock_limit_sec:
            logger.warning(f"Wall-clock timeout ({wall_clock_limit_sec}s) reached")
        
        return loop_state[0].state, loss_history
    
    return fit
```

**Critical Safety Rails (Non-Negotiable):**
1. **Patience-based early stopping** inside `lax.scan` (inside XLA graph)
2. **Hard step limit** (`hard_step_limit`) independent of miner `epochs`
3. **Wall-clock timeout** enforced in Python (outside JIT) — kills stuck evaluations
4. **Gradient clipping INSIDE JIT** (prevents recompilation)

---

## 3. Precision Policy: bfloat16 + fp32 Physics Gates (Correctness-Critical)

### Problem

High-dimensional Fourier Neural Operators and multi-physics models require large amounts of VRAM. bfloat16 reduces VRAM 30-50% but **physics gates fail catastrophically in bfloat16** (1e-6 residual → 1e-4 error → false FAIL).

### Solution: Contextual Precision Policy (Correctness-Critical)

```python
# carbon/backbones/precision.py
import jax
import jax.numpy as jnp
from contextlib import contextmanager
from jaxtyping import Array
import threading

_thread_local = threading.local()

@contextmanager
def physics_precision_policy():
    """Context manager: FORCE fp32 for physics gates, allow bf16 elsewhere."""
    prev_matmul = jax.config.get("jax_default_matmul_precision")
    prev_precision = getattr(_thread_local, "precision_policy", "bfloat16")
    try:
        # Physics gates MUST run in fp32 — this is a correctness requirement
        jax.config.update("jax_default_matmul_precision", "float32")
        _thread_local.precision_policy = "float32"
        yield
    finally:
        jax.config.update("jax_default_matmul_precision", prev_matmul)
        _thread_local.precision_policy = prev_precision

@contextmanager
def training_precision_policy(allow_bfloat16: bool = True):
    """Training context: allows bfloat16 for speed, fp32 for stability."""
    prev_matmul = jax.config.get("jax_default_matmul_precision")
    try:
        if allow_bfloat16:
            jax.config.update("jax_default_matmul_precision", "bfloat16")
        else:
            jax.config.update("jax_default_matmul_precision", "float32")
        yield
    finally:
        jax.config.update("jax_default_matmul_precision", prev_matmul)

def cast_to_bfloat16(x: Array) -> Array:
    """Explicit downcast for model activations/weights."""
    return x.astype(jnp.bfloat16)

def selective_cast_for_residuals(x: Array) -> Array:
    """Force fp32 for residual/conservation calculations."""
    return x.astype(jnp.float32)

# In physics gates — MANDATORY fp32 context
@jax.jit
def run_physics_gates(model_fn, params, stress_data):
    with physics_precision_policy():  # FORCES fp32 — correctness requirement
        return run_all_gates_impl(model_fn, params, stress_data)

# In training — allows bf16 for speed
def train_step(state, batch):
    with training_precision_policy(allow_bfloat16=True):
        return training_step_impl(state, batch)
```

**Enforcement in SPEC (Non-Negotiable):**
> **All physics gate computations (residuals, conservation checks, boundary checks, UQ calibration) MUST execute in fp32 precision. Validators not enforcing this will produce false gate failures and be slashed.**

---

## 4. Mesh-Independent Multi-Fidelity Grid Curriculums

### Problem

Forcing downsampled spatial resolutions validator-side could overwrite valid miner-designed training strategies that rely on high-frequency, fine-mesh features from the first epoch.

### Solution: Miner-Controlled, Validator-Defaulted Curriculum

```json
{
  "curriculum": [
    {
      "phase": 1,
      "epochs": 100,
      "spatial_resolution_scale": 0.5,
      "mode_budget_scale": 0.5
    },
    {
      "phase": 2,
      "epochs": 200,
      "spatial_resolution_scale": 1.0,
      "mode_budget_scale": 1.0
    }
  ]
}
```

### Implementation: Proper Restriction Operators

```python
# carbon/generators/resolution.py
import jax
import jax.numpy as jnp
from typing import Tuple, Optional

@jax.jit
def downsample_spatial_grid(
    coords: Array,
    fields: Array,
    scale: float
) -> Tuple[Array, Array]:
    """
    Mesh-independent spatial downsampling for structured grids.
    For unstructured data, replace with proper restriction / interpolation.
    """
    if scale >= 1.0 - 1e-6:
        return coords, fields

    stride = int(jnp.round(1.0 / scale))
    downsampled_coords = coords[::stride, ::stride, ...]
    downsampled_fields = fields[::stride, ::stride, ...]
    return downsampled_coords, downsampled_fields

@jax.jit
def restrict_fields(fields: Array, scale: float, grid_type: str = "structured") -> Array:
    """
    Proper restriction operator for multi-fidelity training.
    Structured: strided averaging (conservative).
    Unstructured: requires interpolation weights (precomputed).
    """
    if scale >= 1.0 - 1e-6:
        return fields
    
    if grid_type == "structured":
        stride = int(jnp.round(1.0 / scale))
        return fields[::stride, ::stride, ...]
    else:
        # Unstructured: use precomputed restriction matrix
        return apply_restriction_matrix(fields, scale)

def adjust_fno_modes_for_resolution(mode_counts: tuple, scale: float) -> tuple:
    """Adjust FNO mode counts when resolution changes."""
    return tuple(int(m * scale) for m in mode_counts)
```

**Caution**: Changing resolution mid-training interacts with normalization statistics and Fourier mode counts. FNO mode counts must be adjusted or padded consistently when resolution changes.

---

## 5. Gradient Accumulation + Checkpointing (Phase 3-4 Mandatory)

### Phase 3-4: Large Model Training on 80GB VRAM

```python
# carbon/validator/training_phase3.py
from jax import checkpoint
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map

def create_phase3_train_step(
    accumulation_steps: int = 8,
    checkpoint_every_n_layers: int = 2,
    mesh_devices: list = None
):
    """Phase 3-4 training step with gradient accumulation + checkpointing."""
    
    def train_step(state, batch):
        # Split batch into micro-batches for gradient accumulation
        micro_batches = split_batch(batch, accumulation_steps)
        
        def accum_scan_fn(carry, micro_batch):
            params, opt_state, grad_accum = carry
            grads = compute_grads(params, micro_batch)
            grad_accum = jax.tree.map(lambda a, g: a + g, grad_accum, grads)
            return (params, opt_state, grad_accum), None
        
        # Accumulate gradients over micro-batches
        (params, opt_state, grad_accum), _ = lax.scan(
            accum_scan_fn, (state.params, state.opt_state, zero_grads), micro_batches
        )
        
        # Average accumulated gradients
        grad_accum = jax.tree.map(lambda x: x / accumulation_steps, grad_accum)
        updates, new_opt_state = optimizer.update(grad_accum, opt_state)
        new_params = optax.apply_updates(params, updates)
        
        return state.replace(params=new_params, opt_state=new_opt_state)
    
    # Gradient checkpointing for deep models
    def make_checkpointed_block(block_fn, policy=checkpoint.save_any_names_but_these("params")):
        return checkpoint.checkpoint(block_fn, policy=policy)
    
    return train_step
```

### Phase 4: Model Sharding (ZeRO-3) for 3D LES

```python
# carbon/validator/sharding.py
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map

def create_sharded_train_step(mesh_devices: list, model_spec: dict):
    """Phase 4: ZeRO-3 style sharding for 3D LES on H200 clusters."""
    
    mesh = Mesh(mesh_utils.create_device_mesh((len(mesh_devices),)), ('model',))
    
    # Partition specs for FNO/GINO weights
    param_spec = {
        "lifting": P("model"),
        "spectral_convs": P("model"),
        "projection": P("model"),
        "embeddings": P("model"),
    }
    
    @shard_map(mesh=mesh, in_specs=(param_spec, P()), out_specs=P())
    def sharded_forward(params, x):
        return model.apply(params, x)
    
    # Gradient sharding (ZeRO-3)
    def sharded_grad_fn(params, batch):
        grads = jax.grad(loss_fn)(params, batch)
        # Gradients already sharded by shard_map
        return grads
    
    return sharded_forward, sharded_grad_fn
```

---

## 6. Operational Infrastructure (Production Requirements)

### 1. Validator Queue Management (Operational Must-Have)

```python
# carbon/validator/queue.py
from dataclasses import dataclass
from enum import Enum
import asyncio
import heapq
import time
from typing import Optional, Dict, Set

class Priority(Enum):
    SPONSORED_TIER_4 = 0
    SPONSORED_TIER_3 = 1
    SPONSORED_TIER_2 = 2
    HIGH_REPUTATION = 3
    STANDARD = 4
    ESTIMATION_MODE = 5

@dataclass
class QueuedSubmission:
    priority: Priority
    submit_time: float
    hotkey: str
    challenge_id: str
    strategy_hash: str
    estimated_gpu_seconds: float
    submission_id: str
    
    def __lt__(self, other):
        if self.priority != other.priority:
            return self.priority.value < other.priority.value
        return self.submit_time < other.submit_time

class ValidatorQueue:
    def __init__(self, max_concurrent: int = 3, max_queue_depth: int = 100):
        self.max_concurrent = max_concurrent
        self.max_queue_depth = max_queue_depth
        self.pending: list = []
        self.active: Dict[str, dict] = {}
        self.submission_timeout = 7200  # 2 hours max per submission
    
    async def enqueue(self, submission: QueuedSubmission) -> str:
        if len(self.pending) >= self.max_queue_depth:
            raise QueueFullError(f"Queue depth {self.max_queue_depth} exceeded")
        heapq.heappush(self.pending, submission)
        asyncio.create_task(self._monitor_timeout(submission.submission_id))
        return submission.submission_id
    
    async def _monitor_timeout(self, submission_id: str):
        await asyncio.sleep(self.submission_timeout)
        if submission_id in self.active:
            await self._kill_submission(submission_id)
    
    async def _kill_submission(self, submission_id: str):
        # Force kill GPU process, cleanup, mark as timeout
        pass
    
    def dequeue(self) -> Optional[QueuedSubmission]:
        if not self.pending:
            return None
        return heapq.heappop(self.pending)
```

### 2. Determinism Lockfile (Reproducibility)

```txt
# requirements-lock.txt — PIN EXACT VERSIONS
jax==0.4.30
jaxlib==0.4.30+cuda12.cudnn89
flax==0.8.4
optax==0.2.1
orbax-checkpoint==0.5.2
numpy==1.26.4
scipy==1.12.0
```

```dockerfile
# Dockerfile
COPY requirements-lock.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-lock.txt
ENV JAX_VERSION=0.4.30
ENV FLAX_VERSION=0.8.4
```

**Why**: JAX 0.4.x → 0.5.x changes numerics at 1e-7 level. Gates at 1e-6 threshold = flaky PASS/FAIL.

---

### 3. XLA Compilation Cache Persistence (Operational Must-Have)

```dockerfile
# Dockerfile
ENV JAX_COMPILATION_CACHE_DIR=/persistent/compile_cache
ENV JAX_CACHE_DIR=/persistent/jax_cache
ENV XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda --xla_gpu_per_thread_default_stream=true"

RUN mkdir -p /persistent/compile_cache /persistent/jax_cache
```

```yaml
# docker-compose.yml
services:
  validator:
    volumes:
      - compile_cache:/persistent/compile_cache
      - jax_cache:/persistent/jax_cache

volumes:
  compile_cache:
  jax_cache:
```

**Impact**: Eliminates 5-10 min compilation on validator restart/redeploy. Critical for uptime.

---

### 4. Determinism Enforcement (Validator-Side)

```python
# carbon/common/determinism.py
import os
import random
import numpy as np

def set_global_determinism(seed: int = 42):
    """Set all random seeds and deterministic flags."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    
    random.seed(seed)
    np.random.seed(seed)
    
    # JAX
    import jax
    jax.config.update("jax_default_prng_impl", "threefry")
    jax.config.update("jax_enable_x64", True)
    
    # PyTorch (if used)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    except ImportError:
        pass

def verify_reproducibility(run_fn, seed: int, n_runs: int = 3):
    """Run function n times, verify identical outputs."""
    outputs = []
    for i in range(n_runs):
        set_global_determinism(seed)
        output = run_fn()
        outputs.append(hash(str(output)))
    
    all_same = len(set(outputs)) == 1
    return all_same, outputs[0] if all_same else outputs
```

---

## 7. Updated Cost Estimates (Realistic)

| Phase | Physics | Hardware | Realistic Runtime | Cost/Eval | Monthly (5 val, 20 evals/day) |
|-------|---------|----------|-------------------|-----------|-------------------------------|
| **Phase 0** | 7 Academic PDEs | 5× A100 80GB | 12 min | $14 | $22k |
| **Phase 1A** | Compressible NS | 5× H100 | 18 min | $16 | $25k |
| **Phase 1B** | Reacting/FSI/6-DOF | 5× H100 | 35 min | $30 | $45k |
| **Phase 2A** | LoRA/Custom/MT | 5× H100 | 40 min | $32 | $48k |
| **Phase 2B** | Air-Gap + preCICE | 6× H100 (1 air-gap) | 50 min | $35 | $55k |
| **Phase 3** | Coupled FSI/CHT | 10× H100 (5 pairs) | 4.5 hrs | $55 | $165k |
| **Phase 4** | 3D LES + Coupling | 20× H200 | 10 hrs | $325 | $975k |

> **Note**: Phase 3-4 include 2.5× safety margin for coupling overhead, multi-GPU scaling inefficiency, and LES resolution.

---

## 8. Final Implementation Checklist

| Component | File | Status | Phase |
|-----------|------|--------|-------|
| Boolean loss masking | `carbon/validator/losses.py` | ✅ Required | 0 |
| fp32 physics gate context | `carbon/backbones/precision.py` | ✅ Required | 0 |
| `lax.scan` training loop | `carbon/validator/training.py` | ✅ Required | 0 |
| Gradient accumulation | `carbon/validator/training_phase3.py` | ✅ Required | 2B |
| Gradient checkpointing | `carbon/backbones/checkpointing.py` | ✅ Required | 2B |
| Compilation cache | Docker/entrypoint | ✅ Required | 0 |
| Determinism lockfile | `requirements-lock.txt` | ✅ Required | 0 |
| Validator queue | `carbon/validator/queue.py` | ✅ Required | 0 |
| fp32 residual context | `carbon/backbones/precision.py` | ✅ Required | 1A |
| Gradient accumulation boilerplate | `carbon/validator/training.py` | 🔄 Phase 2A | 2A |
| Gradient checkpointing | `carbon/backbones/checkpointing.py` | 🔄 Phase 2A | 2A |
| Model sharding (ZeRO-3) | `carbon/validator/sharding.py` | 🔄 Phase 4 | 4 |

---

## 8. Integration Notes

These optimizations live primarily in:

- `carbon/validator/losses.py` — unified loss with boolean masking
- `carbon/validator/training.py` — `lax.scan` / early-stop engine
- `carbon/validator/training_phase3.py` — gradient accumulation + checkpointing
- `carbon/validator/sharding.py` — Phase 4 model sharding
- `carbon/backbones/precision.py` — mixed-precision policy + fp32 gate context
- `carbon/backbones/checkpointing.py` — gradient checkpointing policies
- `carbon/generators/resolution.py` — multi-fidelity downsampling
- `carbon/validator/evaluation.py` — `vmap` cohort path
- `carbon/validator/queue.py` — submission queue management
- `carbon/common/determinism.py` — reproducibility harness

They must be fully compatible with the existing Model Card generation, physics-gate pipeline, and deterministic seeding hierarchy defined in the main SPEC.

---

*This appendix is a living engineering document. Cost figures, hardware assumptions, and XLA performance characteristics should be revisited as marketplace rates and JAX versions evolve. All correctness-critical items (fp32 gates, boolean masks, determinism) are non-negotiable requirements for mainnet launch.*
