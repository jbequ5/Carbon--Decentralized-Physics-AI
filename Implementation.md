# IMPLEMENTATION.md — Carbon Physics Intelligence Subnet

**Version:** 2.0 (July 2026)  
**Status:** Core Engineering Implementation Guide  
**Audience:** Harshdeep Sharma (Tech Lead) + Engineering Team  
**Purpose:** Build-level implementation details for every component in `SPEC.md`

---

# TABLE OF CONTENTS

1. [JAX Core Training Optimizations](#1-jax-core-training-optimizations)
2. [Physics Gates Implementation](#2-physics-gates-implementation)
3. [Procedural Generators](#3-procedural-generators)
4. [Validator Training Pipeline](#4-validator-training-pipeline)
5. [Landscape Agent Pipeline](#5-landscape-agent-pipeline)
6. [Miner Toolkit & SDK](#6-miner-toolkit--sdk)
7. [Genesis Contracts](#7-genesis-contracts)
8. [MCP Protocol](#8-mcp-protocol)
9. [Reproducibility & Determinism](#9-reproducibility--determinism)
10. [Operational Infrastructure](#10-operational-infrastructure)

---

# 1. JAX CORE TRAINING OPTIMIZATIONS

## 1.1 Unified Loss Masking (Zero Recompilation)

**Problem**: Miners dynamically enable/disable loss terms via `strategy.json`. Python `if/else` in loss computation triggers XLA recompilation per unique strategy.

**Solution**: Single static JIT-compiled loss function with explicit boolean masks.

```python
# carbon/validator/losses.py
import jax
import jax.numpy as jnp
from typing import Dict, NamedTuple
from jaxtyping import Array, Float, Bool
from flax.core import FrozenDict

class LossWeights(NamedTuple):
    """Explicit boolean flags + weights — NO floating-point thresholds."""
    data_mse: Float[Array, ""]
    data_mse_enabled: Bool[Array, ""]
    physics_residual: Float[Array, ""]
    physics_residual_enabled: Bool[Array, ""]
    boundary_mse: Float[Array, ""]
    boundary_mse_enabled: Bool[Array, ""]
    conservation_penalty: Float[Array, ""]
    conservation_penalty_enabled: Bool[Array, ""]

@partial(jax.jit, donate_argnums=(0, 1))  # Donate params, opt_state buffers
def unified_loss_fn(
    params: FrozenDict,
    batch_inputs: Array,
    batch_targets: Array,
    weights: LossWeights,
    model_apply_fn: Callable
) -> Array:
    """
    Statically compiled loss function. Uses EXPLICIT boolean masking — NO fp thresholds.
    All residual functions must be safe when weight=0 (no side-effects, no NaN paths).
    """
    # 1. Forward Pass
    predictions = model_apply_fn(params, batch_inputs)

    # 2. Continuous calculation of ALL objective terms (always computed)
    loss_data = jnp.mean(jnp.square(predictions - batch_targets))
    loss_phys = compute_pde_residuals(params, batch_inputs, model_apply_fn)
    loss_bound = compute_boundary_residuals(params, batch_inputs, model_apply_fn)
    loss_conserve = compute_conservation_penalties(predictions)

    # 3. Explicit boolean masking — NO fp threshold ambiguity
    total_loss = (
        (weights.data_mse_enabled * weights.data_mse * loss_data) +
        (weights.physics_residual_enabled * weights.physics_residual * loss_phys) +
        (weights.boundary_mse_enabled * weights.boundary_mse * loss_bound) +
        (weights.conservation_penalty_enabled * weights.conservation_penalty * loss_conserve)
    )
    return total_loss
```

**Schema Integration (v1.0+)**:
```json
{
  "loss": {
    "data_mse": {"enabled": true, "weight": 1.0},
    "physics_residual": {"enabled": true, "weight": 0.5},
    "boundary_mse": {"enabled": true, "weight": 0.3},
    "conservation_penalty": {"enabled": false, "weight": 0.2}
  }
}
```

**Why Boolean Flags**: Floating-point threshold `weight < 1e-8` is fragile. Miner submits `1e-9` → silently treated as 0. Boolean is explicit, auditable, and cannot be gamed.

---

## 1.2 Functional Training Loop with `lax.scan` + Hard Safety Rails

### Problem
Miners control `epochs` parameter. Rogue submissions can request extreme epoch counts. Python loops prevent JAX optimization and make hard limits unenforceable.

### Solution: `lax.scan` + Hard Safety Rails (Inside + Outside JIT)

```python
# carbon/validator/training.py
import jax
import jax.lax as lax
import jax.numpy as jnp
from typing import Tuple, Dict, NamedTuple, Callable, Any, Optional
from flax.training import train_state
import optax
from flax.core import FrozenDict
from jaxtyping import Array, Float, Int
import time

class TrainerState(train_state.TrainState):
    """Extended train state with early-stopping metadata."""
    best_loss: Float[Array, ""]
    consecutive_no_improve: Int[Array, ""]
    terminated: bool
    epoch: Int[Array, ""]

def create_training_step(
    model_apply_fn: Callable,
    loss_fn: Callable,
    optimizer: optax.GradientTransformation,
    grad_accum_steps: int = 1,
    physics_precision: bool = False
) -> Callable:
    """Factory for compiled training step with gradient accumulation."""
    
    @partial(jax.jit, donate_argnums=(0, 1))
    def train_step(state: TrainState, batch) -> Tuple[train_state.TrainState, Float[Array, ""]]:
        # Gradient accumulation over micro-batches
        def accum_step(carry, micro_batch):
            params, opt_state, grad_accum = carry
            grads = compute_grads(params, micro_batch, physics_precision)
            grad_accum = jax.tree.map(lambda a, g: a + g, grad_accum, grads)
            return (params, opt_state, grad_accum), None
        
        micro_batches = split_batch(batch, grad_accum_steps)
        (params, opt_state, grad_accum), _ = lax.scan(
            lambda carry, mb: (accum_step(carry, mb), None),
            (state.params, state.opt_state, jax.tree.map(jnp.zeros_like, state.params)),
            micro_batches
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
    """Creates a compiled training loop with HARD safety rails."""
    
    class LoopState(NamedTuple):
        state: TrainState
        best_loss: Array
        consecutive_no_improve: int
        terminated: bool
        epoch: int
    
    @partial(jax.jit, donate_argnums=(0,))
    def epoch_step(state: Tuple, _):
        loop_state, epoch_idx = state
        
        def execution_branch(ls):
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
        
        def short_circuit_branch(ls):
            return ls, ls.best_loss
        
        next_loop_state, epoch_loss = lax.cond(
            ls.terminated,
            short_circuit_branch,
            execution_branch,
            loop_state
        )
        return (next_loop_state, epoch_idx + 1), epoch_loss
    
    def fit(init_state, data_loader, max_epochs: int):
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
        start_time = time.time()
        
        (final_loop_state, _), loss_history = lax.scan(
            epoch_step,
            (init_loop_state, 0),
            jnp.arange(effective_epochs)
        )
        
        # Wall-clock timeout enforced OUTSIDE JIT
        if time.time() - start_time > wall_clock_limit_sec:
            logger.warning(f"Wall-clock timeout ({wall_clock_limit_sec}s) reached")
        
        return final_loop_state.state, loss_history
    
    return fit
```

**Critical Safety Rails (Non-Negotiable):**
1. **Patience-based early stopping** inside `lax.scan` (inside XLA graph)
2. **Hard step limit** (`hard_step_limit`) independent of miner `epochs`
3. **Wall-clock timeout** enforced in Python (outside JIT) — kills stuck evaluations
5. **Gradient clipping INSIDE JIT** (prevents recompilation)

---

## 1.3 Precision Policy: bfloat16 + fp32 Physics Gates (Correctness-Critical)

### Problem
bfloat16 reduces VRAM 30-50% but **physics gates fail catastrophically in bf16** (1e-6 residual → 1e-4 error → false FAIL).

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

## 1.4 Mesh-Independent Multi-Fidelity Grid Curriculums

### Problem
Forcing downsampled spatial resolutions validator-side could overwrite valid miner-designed training strategies that rely on high-frequency, fine-mesh features from the first epoch.

### Solution: Miner-Controlled, Validator-Defaulted Curriculum

**Schema Extension** (v1.0+):
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

## 1.5 Gradient Accumulation + Checkpointing (Phase 3-4 Mandatory)

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

## 1.6 Operational Infrastructure (Production Requirements)

### 1. Validator Queue Management

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

### 2. Determinism Lockfile (Pinned)

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

## 2. Physics Gates Implementation

### 2.1 Mass Conservation (Continuity)

```python
# carbon/validator/physics_gates.py
import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
from dataclasses import dataclass
from typing import Callable, Dict, Any

@dataclass
class GateResult:
    gate_id: str
    threshold: float
    result: float
    status: str  # "PASS" | "FAIL"
    worst_case_variant: str = ""
    details: Dict[str, Any] = None

# --- Mass Conservation (Continuity) ---
@jit
def mass_conservation_residual(model_fn: Callable, coords: jnp.ndarray, params: Dict) -> jnp.ndarray:
    """∂ρ/∂t + ∇·(ρu) = 0"""
    # coords: (N, d+1) where last dim is time
    # model_fn outputs (rho, u_x, u_y, u_z)
    
    def continuity_eq(coord):
        rho, ux, uy, uz = model_fn(coord, params)
        # Time derivative of rho
        drho_dt = grad(lambda c: model_fn(c, params)[0])(coord)[-1]
        # Spatial divergence of rho*u
        def flux(c):
            rho, ux, uy, uz = model_fn(c, params)
            return jnp.array([rho*ux, rho*uy, rho*uz])
        div_flux = jnp.trace(jax.jacfwd(flux)(coord)[:, :3])  # spatial dims only
        return drho_dt + div_flux
    
    residuals = vmap(continuity_eq)(coords)
    return jnp.abs(residuals)

# --- Energy Stability ---
@jit
def energy_stability_residual(model_fn: Callable, coords: jnp.ndarray, params: Dict) -> jnp.ndarray:
    """ρ·De/Dt = -∇·q - p∇·u + Φ"""
    # Implementation depends on physics class
    # For compressible NS: total energy E = e + 0.5*|u|^2
    pass

# --- Boundary Satisfaction ---
@jit
def boundary_residual(model_fn: Callable, boundary_coords: jnp.ndarray, 
                      boundary_values: Dict, params: Dict) -> jnp.ndarray:
    """u|_∂Ω = g_D, (σ·n)|_∂Ω = g_N"""
    pred = vmap(model_fn)(boundary_coords, params)
    # Dirichlet
    dirichlet_error = jnp.abs(pred - boundary_values["dirichlet"])
    # Neumann (requires gradient)
    neumann_error = jnp.abs(grad(model_fn)(boundary_coords) - boundary_values["neumann"])
    return jnp.concatenate([dirichlet_error.flatten(), neumann_error.flatten()])

# --- Rollout Stability ---
@jit
def rollout_stability(model_fn: Callable, init_coords: jnp.ndarray, 
                      params: Dict, steps: int = 10000, perturb: float = 0.01) -> bool:
    """Autoregressive rollout with perturbation."""
    state = init_state
    for i in range(steps):
        state = model_fn(state, params)
        if i % 100 == 0:
            state = state + perturb * jax.random.normal(key, state.shape)
        if jnp.any(jnp.isnan(state)) or jnp.any(jnp.abs(state) > 1e6):
            return False
    return True

# --- UQ Calibration (Conformal Prediction) ---
def uq_calibration(model_fn: Callable, calibration_coords: jnp.ndarray,
                   calibration_targets: jnp.ndarray, params: Dict,
                   confidence: float = 0.95) -> float:
    """Split conformal prediction for coverage."""
    # Split calibration set
    n = len(calibration_coords)
    split_idx = n // 2
    
    # Train on first half, calibrate on second
    # Use conformal prediction to get prediction intervals
    # Return coverage probability
    pass

# --- Gate Runner ---
def run_all_gates(model_fn: Callable, challenge: str, params: Dict,
                  stress_data: Dict, generator_version: str) -> List[GateResult]:
    """Run all physics gates for a challenge."""
    gates = []
    
    # 1. Mass Conservation
    mass_residuals = mass_conservation_residual(model_fn, stress_data["coords"], params)
    max_mass_res = float(jnp.max(jnp.abs(mass_residuals)))
    gates.append(GateResult(
        gate_id="mass_conservation",
        threshold=1e-6,
        result=max_mass_res,
        status="PASS" if max_mass_res < 1e-6 else "FAIL",
        worst_case_variant=stress_data["worst_case"]["mass"]
    ))
    
    # 2. Energy Stability
    energy_residuals = energy_stability_residual(model_fn, stress_data["coords"], params)
    max_energy_res = float(jnp.max(jnp.abs(energy_residuals)))
    gates.append(GateResult(...))
    
    # 3. Boundary Satisfaction
    boundary_res = boundary_residual(model_fn, stress_data["boundary_coords"], 
                                     stress_data["boundary_values"], params)
    max_boundary_res = float(jnp.max(jnp.abs(boundary_res)))
    gates.append(GateResult(...))
    
    # 4. Rollout Stability
    rollout_ok = rollout_stability(model_fn, stress_data["init_coords"], params)
    gates.append(GateResult(...))
    
    # 5. UQ Calibration
    uq_coverage = uq_calibration(model_fn, stress_data["cal_coords"],
                                 stress_data["cal_targets"], params)
    gates.append(GateResult(...))
    
    return gates
```

---

## 3. Procedural Generators

### 3.1 Base Generator Interface

```python
# carbon/generators/base.py
import jax
import jax.numpy as jnp
from jax import random
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class GeneratorConfig:
    challenge_id: str
    physics_class: str
    dimension: int
    parameter_ranges: Dict[str, tuple]
    reference_solver: str
    validation_tolerance: str

class ProceduralGenerator(ABC):
    def __init__(self, config: GeneratorConfig):
        self.config = config
    
    @abstractmethod
    def generate_training_data(self, seed: int, n_samples: int) -> Dict:
        """Generate training data: {coords, fields, boundary_conditions}"""
        pass
    
    @abstractmethod
    def generate_stress_variants(self, seed: int, n_variants: int) -> Dict:
        """Generate hidden stress test variants."""
        pass
    
    @abstractmethod
    def generate_benchmark_data(self, seed: int, n_samples: int) -> Dict:
        """Generate held-out benchmark data."""
        pass
    
    def derive_seeds(self, master_seed: int) -> Dict[str, int]:
        """Hierarchical deterministic seed derivation."""
        keys = random.split(random.PRNGKey(master_seed), 5)
        return {
            "data": int(random.randint(keys[0], (), 0, 2**32)),
            "stress": int(random.randint(keys[1], (), 0, 2**32)),
            "init": int(random.randint(keys[2], (), 0, 2**32)),
            "dropout": int(random.randint(keys[3], (), 0, 2**32)),
            "shuffle": int(random.randint(keys[4], (), 0, 2**32)),
        }
```

### 3.2 Phase 0: JAX-FEM Generators (Online)

```python
# carbon/generators/poisson.py
class PoissonGenerator(ProceduralGenerator):
    def generate_training_data(self, seed: int, n_samples: int) -> Dict:
        key = random.PRNGKey(seed)
        
        # 1. Sample coefficient field k(x) ~ LogNormal
        key, k_key = random.split(key)
        log_k = random.normal(k_key, (n_samples, *self.resolution)) * 0.5
        k = jnp.exp(log_k)
        
        # 2. Sample source field f(x) ~ Gaussian Process
        key, f_key = random.split(key)
        f = self._sample_gp(f_key, n_samples, length_scale=0.1)
        
        # 3. Solve -∇·(k∇u) = f using JAX-FEM (differentiable!)
        u = self._solve_poisson(k, f)
        
        return {
            "coords": self.grid,           # (64, 64, 2)
            "inputs": {"coefficient": k, "source": f},
            "targets": {"solution": u},
            "boundary_mask": self.bc_mask
        }
    
    def _solve_poisson(self, k, f):
        """JAX-FEM Poisson solve: -∇·(k∇u) = f"""
        # Differentiable FEM solve using JAX
        pass
```

### 3.3 Phase 1A: Compressible NS (Hybrid Online/Offline)

```python
# carbon/generators/compressible_ns.py
class CompressibleNSGenerator(ProceduralGenerator):
    def generate_training_data(self, seed: int, n_samples: int) -> Dict:
        key = random.PRNGKey(seed)
        
        # 1. Sample flow conditions
        key, mach_key, re_key, aoa_key = random.split(key, 4)
        mach = random.uniform(mach_key, (n_samples,), minval=0.7, maxval=1.2)
        reynolds = 10**random.uniform(re_key, (n_samples,), minval=6, maxval=7)
        aoa = random.uniform(aoa_key, (n_samples,), minval=-2, maxval=4)
        
        # 2. Generate mesh perturbations (geometry variation)
        key, mesh_key = random.split(key)
        mesh_perturb = random.normal(mesh_key, (n_samples, *self.mesh_shape)) * 0.001
        
        # 3. Generate turbulence ICs
        key, turb_key = random.split(key)
        turb_ic = self._sample_turbulence_ic(turb_key, n_samples)
        
        # 4. Load precomputed solutions from cache
        solutions = self._load_precomputed_solutions(mach, reynolds, aoa)
        
        return {
            "coords": self.grid,
            "mach": mach,
            "reynolds": reynolds,
            "aoa": aoa,
            "mesh_perturb": mesh_perturb,
            "turb_ic": turb_ic,
            "targets": {"solution": solutions},
            "boundary_mask": self.bc_mask
        }
```

### 3.4 Phase 1B: Precomputed Generators (Reacting Flow, FSI, 6-DOF, CHT)

```python
# carbon/generators/reacting_ns.py
class ReactingNSGenerator(ProceduralGenerator):
    def generate_training_data(self, seed: int, n_samples: int) -> Dict:
        key = random.PRNGKey(seed)
        
        # Flight conditions
        key, mach_key, re_key, wall_temp_key = random.split(key, 4)
        mach = random.uniform(mach_key, (n_samples,), minval=5.0, maxval=8.0)
        reynolds = 10**random.uniform(re_key, (n_samples,), minval=5, maxval=6)
        wall_temp = random.uniform(wall_temp_key, (n_samples,), minval=300, maxval=2000)
        
        # Chemistry ICs
        key, chem_key = random.split(key)
        species_ic = self._sample_chemistry_ic(chem_key, n_samples)
        
        # Load precomputed solutions from cache
        solutions = self._load_precomputed_solutions(mach, reynolds, wall_temp, species_ic)
        
        return {
            "mach": mach,
            "reynolds": reynolds,
            "wall_temp": wall_temp,
            "species_ic": species_ic,
            "targets": {"solution": solutions},
        }
```

### 3.5 Stress Generator (Hidden, Fresh Every Eval)

```python
# carbon/generators/stress.py

class StressCategory(Enum):
    EXTENDED_ENVELOPE = "extended_envelope"
    SHOCK_PERTURBATION = "shock_perturbation"
    BOUNDARY_LAYER_TRIP = "boundary_layer_trip"
    SEPARATION_TRIGGER = "separation_trigger"
    CHEMISTRY_PERTURBATION = "chemistry_perturbation"
    MESH_PERTURBATION = "mesh_perturbation"
    BOUNDARY_CONDITION = "boundary_condition"
    INITIAL_CONDITION = "initial_condition"
    COUPLING_PERTURBATION = "coupling_perturbation"

@dataclass
class StressVariantSpec:
    category: StressCategory
    weight: float
    params: Dict[str, Any]
    physics_gates: List[str]

STRESS_VARIANT_SPECS = {
    "naca0012_transonic-v1": [
        StressVariantSpec(
            category=StressCategory.EXTENDED_ENVELOPE,
            weight=0.30,
            params={"mach_range": (0.5, 1.5), "reynolds_range": (0.5e6, 20e6)},
            physics_gates=["mass_conservation", "energy_stability", "shock_capture"]
        ),
        StressVariantSpec(
            category=StressCategory.SHOCK_PERTURBATION,
            weight=0.20,
            params={"shock_strength": (0.05, 0.15), "position_perturbation": 0.05},
            physics_gates=["shock_capture", "mass_conservation"]
        ),
        # ... more variants
    ],
}

class StressGenerator(ProceduralGenerator):
    def generate_stress_variants(self, seed: int, n_variants: int) -> Dict:
        key = random.PRNGKey(seed)
        specs = STRESS_VARIANT_SPECS.get(self.config.challenge_id, [])
        
        # Normalize weights
        total_weight = sum(s.weight for s in specs)
        probs = [s.weight / total_weight for s in specs]
        
        variants = []
        for i in range(n_variants):
            key, var_key = random.split(key)
            
            # Select category by weight
            cat_idx = random.choice(var_key, len(specs), p=jnp.array(probs))
            spec = specs[cat_idx]
            
            # Generate variant
            variant = self._generate_variant(var_key, spec)
            variants.append(variant)
        
        return self._collate_variants(variants)
    
    def _generate_variant(self, key: int, spec: StressVariantSpec) -> Dict:
        handlers = {
            StressCategory.EXTENDED_ENVELOPE: self._gen_extended_envelope,
            StressCategory.SHOCK_PERTURBATION: self._gen_shock_perturbation,
            StressCategory.BOUNDARY_LAYER_TRIP: self._gen_bl_trip,
            StressCategory.SEPARATION_TRIGGER: self._gen_separation_trigger,
            StressCategory.MESH_PERTURBATION: self._gen_mesh_perturbation,
            StressCategory.BOUNDARY_CONDITION: self._gen_boundary_condition,
            StressCategory.CHEMISTRY_PERTURBATION: self._gen_chemistry_perturbation,
            StressCategory.COUPLING_PERTURBATION: self._gen_coupling_perturbation,
        }
        return handlers[spec.category](key, spec.params)
```

---

## 5. Validator Training Pipeline

### 5.1 Complete Training Pipeline

```python
# carbon/validator/training.py
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax.training import train_state
from typing import Callable, Dict, Any
import yaml

class TrainState(train_state.TrainState):
    epoch: int
    best_score: float
    rng: jax.Array

class ValidatorTrainer:
    def __init__(self, config: Dict):
        self.config = config
        self.checkpointer = ocp.StandardCheckpointer()
    
    def create_train_state(self, model_fn: Callable, params: Dict, 
                           strategy: Dict, rng: jax.Array) -> TrainState:
        """Initialize training state with optimizer from strategy."""
        tx = self._create_optimizer(strategy["training"])
        return TrainState.create(
            apply_fn=model_fn,
            params=params,
            tx=tx,
            epoch=0,
            best_score=-1.0,
            rng=rng
        )
    
    def _create_optimizer(self, training_config: Dict) -> optax.GradientTransformation:
        lr_schedule = self._create_lr_schedule(training_config)
        tx = optax.chain(
            optax.clip_by_global_norm(training_config.get("gradient_clip", 1.0)),
            optax.adamw(
                learning_rate=lr_schedule,
                weight_decay=training_config.get("weight_decay", 1e-4)
            )
        )
        return tx
    
    def _create_lr_schedule(self, training_config: Dict) -> optax.Schedule:
        schedule_type = training_config.get("lr_schedule", "cosine_warm_restarts")
        if schedule_type == "cosine_warm_restarts":
            return optax.cosine_decay_schedule(
                init_value=training_config["learning_rate"],
                decay_steps=training_config["epochs"],
                alpha=0.01
            )
        # ... other schedules
    
    def train(self, state: TrainState, train_loader, val_loader, 
              strategy: Dict, physics_gates_fn) -> TrainState:
        """Main training loop with checkpointing and adaptive loss reweighting."""
        
        for epoch in range(state.epoch, self.config["training"]["epochs"]):
            # Training step
            state, train_metrics = self._train_epoch(state, train_loader, strategy)
            
            # Validation + physics gates
            if epoch % 10 == 0 or epoch == self.config["training"]["epochs"] - 1:
                val_metrics = self._validate(state, val_loader)
                gate_results = self._run_physics_gates(state)
                
                # Adaptive loss reweighting
                state = self._adaptive_reweight(state, gate_results, strategy)
                
                # Checkpointing
                if self._should_checkpoint(epoch, val_metrics):
                    self._save_checkpoint(state, epoch)
                
                # Early stopping
                if self._should_stop(state):
                    break
        
        return state
    
    def _adaptive_reweight(self, state: TrainState, gate_results: List, 
                          strategy: Dict) -> TrainState:
        """Adaptive loss reweighting within bounds."""
        if not strategy["loss"].get("adaptive_reweighting", {}).get("enabled"):
            return state
        
        bounds = strategy["loss"]["adaptive_reweighting"]["bounds"]
        # Adjust loss weights based on gate margins
        # ... implementation
        return state
    
    def _save_checkpoint(self, state: TrainState, epoch: int):
        """Save checkpoint with Orbax."""
        ckpt = {
            "model": state.params,
            "optimizer": state.opt_state,
            "epoch": epoch,
            "best_score": state.best_score,
            "rng": state.rng
        }
        self.checkpointer.save(f"/checkpoints/epoch_{epoch}", ckpt)
```

---

## 6. Landscape Agent Pipeline

### 6.1 Pipeline Architecture

```python
# carbon/landscape/pipeline.py
class LandscapeAgent:
    def __init__(self, config: Dict):
        self.pysr_config = config.get("pysr", PYSR_CONFIG)
        self.dml_config = config.get("dml", DML_CONFIG)
        self.mt_bridge = ModelingToolkitBridge()
        self.specialist_bank = SpecialistBank()
        self.prior_engine = PriorEngine()
    
    def ingest_model_card(self, model_card: ModelCard):
        """Process new model card, update knowledge base."""
        features = self._extract_features(model_card)
        self.pysr_dataset.append(features, model_card.gate_results)
        self.dml_dataset.append(features, model_card.robustness_score)
        
        # Periodic batch processing
        if len(self.pysr_dataset) % 100 == 0:
            self._run_pysr()
        if len(self.dml_dataset) % 500 == 0:
            self._run_dml()
    
    def _run_pysr(self):
        """Run PySR symbolic regression on accumulated data."""
        equations = pysr_regress(self.pysr_dataset, self.pysr_config)
        for eq in equations:
            # Convert to structured loss term
            loss_term = self.mt_bridge.json_to_loss_term(eq.json)
            self.specialist_bank.add_loss_term(loss_term)
        
        # Update noisy priors
        self.prior_engine.update_from_symbolic(equations)
    
    def _run_dml(self):
        """Run Double ML causal inference."""
        # Treatment: strategy choices (discretized)
        # Outcome: robustness_score
        # Confounders: physics_class, data_seed, backbone
        causal_effects = double_ml(self.dml_dataset, self.dml_config)
        
        # Generate strategic guidance
        guidance = self._generate_guidance(causal_effects)
        self.prior_engine.update_from_causal(guidance)
    
    def get_noisy_prior(self, challenge: str, backbone: str) -> Strategy:
        """Get current best prior for challenge+backbone."""
        base = self.prior_engine.get_base_prior(challenge, backbone)
        return self._add_noise(base, noise_scale=0.1)
```

### PySR Configuration (Phase 0)

```python
# carbon/landscape/pysr_config.py
PYSR_CONFIG = {
    "populations": 50,
    "population_size": 100,
    "ncycles_per_iteration": 500,
    "maxsize": 40,
    "maxdepth": 8,
    "binary_operators": ["+", "-", "*", "/", "^"],
    "unary_operators": ["sin", "cos", "exp", "log", "sqrt", "abs"],
    "constraints": {"pow": (-1, 1)},
    "complexity_of_operators": {"+": 1, "-": 1, "*": 2, "/": 2, "^": 3},
    "feature_names": [
        "loss_data_weight", "loss_physics_weight", "loss_boundary_weight",
        "lr_initial", "lr_decay_rate", "curriculum_phase", "backbone_depth",
        "backbone_width", "activation_type", "normalization_type",
        "physics_gate_margin", "residual_l2", "conservation_l2", "boundary_l2"
    ],
    "target_name": "robustness_score",
    "verbosity": 1,
    "batch_size": 1000,
    "early_stop_condition": "stop_if_no_improvement(50)",
}
```

### ModelingToolkit.jl Bridge (JSON → JAX Loss Terms)

```julia
# carbon/landscape/bridge.jl
module CarbonMTBridge
using ModelingToolkit, Symbolics, JSON3, StructTypes

function json_to_loss_term(json_expr::Dict) 
    @variables t x y z
    @parameters p[1:20]  # strategy params
    
    # Parse PySR expression tree
    expr = parse_pysr_json(json_expr)
    
    # Compile to differentiable function
    loss_fn = eval(build_function(expr, [p...], [t, x, y, z]))
    
    return loss_fn
end

function parse_pysr_json(json::Dict)
    # Recursively parse PySR expression tree to Symbolics expression
    # Handles: +, -, *, /, ^, sin, cos, exp, log, sqrt, abs
    # Returns Symbolics expression
end

function loss_terms_to_jax(loss_fns::Vector) -> String
    """Generate JAX code for compiled loss terms."""
    # Generate: 
    # def structured_loss(params, physics_state):
    #     term1 = λ₁ * (div(u))^2
    #     term2 = λ₂ * (dρ/dt + div(ρu))^2
    #     return sum(terms)
end
end
```

### Double ML Configuration (Phase 2A+, JAX-Native)

```python
# carbon/landscape/dml_config.py
DML_CONFIG = {
    "n_folds": 5,
    "n_repeats": 3,
    "ml_model": "jax_boosting",
    "treatment_types": {
        "loss_weights": "continuous_multivariate",
        "curriculum": "categorical",
        "backbone": "categorical",
        "lr_schedule": "categorical"
    },
    "confounders": ["physics_class", "data_seed", "backbone", "epochs"],
    "target": "robustness_score",
    "confidence_level": 0.95,
    "min_samples_per_treatment": 50
}
```

---

## 7. Miner Toolkit & SDK

### 7.1 Miner Toolkit Docker Image

```dockerfile
# carbon/miner/Dockerfile
FROM nvidia/cuda:12.4-devel-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv git curl wget \
    build-essential cmake libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Python env
RUN python3.11 -m venv /opt/carbon
ENV PATH="/opt/carbon/bin:$PATH"

# Carbon Miner Toolkit
COPY carbon/miner /opt/carbon/miner
COPY carbon/common /opt/carbon/common
COPY requirements-miner.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-miner.txt

# Entry points
COPY docker/entrypoint-miner.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--help"]
```

### CLI Interface

```bash
# carbon/miner/cli.py
@click.group()
def cli():
    """Carbon Miner Toolkit - Local iteration for Carbon Subnet"""
    pass

@cli.command()
@click.option('--challenge', required=True, help='Challenge ID')
@click.option('--backbone', default='fno', type=click.Choice(['fno', 'gino', 'wno', 'transolver']))
@click.option('--strategy', 'strategy_path', required=True, type=click.Path(exists=True))
@click.option('--mode', default='estimation', type=click.Choice(['estimation', 'light_training', 'full_confirmation']))
@click.option('--provider', type=click.Choice(['targon', 'chutes', 'runpod', 'lambda', 'local']))
@click.option('--gpu', default='A100_40GB')
@click.option('--output', default='./carbon_output')
def run(challenge, backbone, strategy_path, mode, provider, gpu, output):
    """Run local iteration loop."""
    # Load strategy, validate, estimate cost, confirm, execute
    pass

@cli.command()
@click.option('--challenge', required=True)
def pull_prior(challenge):
    """Download latest noisy prior for challenge."""
    pass

@cli.command()
@click.option('--challenge', required=True)
@click.option('--strategy', 'strategy_path', required=True)
def submit(challenge, strategy_path):
    """Submit strategy to validator via MCP."""
    pass

@cli.command()
def doctor():
    """Validate environment: GPU, drivers, MCP connectivity, credentials."""
    pass
```

### Python SDK (Agent-Friendly)

```python
# carbon/miner/sdk.py
class CarbonMiner:
    def __init__(self, mcp_endpoint: str = None, hotkey: str = None):
        self.mcp = MCPClient(mcp_endpoint or os.getenv("CARBON_MCP_ENDPOINT"))
        self.hotkey = hotkey or os.getenv("CARBON_HOTKEY")
    
    def get_noisy_prior(self, challenge: str, backbone: str) -> Strategy:
        """Fetch latest noisy prior for challenge+backbone."""
        return self.mcp.call("get_noisy_prior", challenge=challenge, backbone=backbone)
    
    def estimate(self, strategy: Strategy, prior: Strategy) -> EstimationResult:
        """Run local estimation (no GPU needed)."""
        engine = EstimationEngine(prior, strategy.backbone, strategy.challenge)
        return engine.estimate(strategy)
    
    def train_local(self, strategy: Strategy, mode: str = "light", **kwargs) -> TrainingResult:
        """Run local training loop (requires GPU)."""
        runner = LocalTrainingRunner(strategy, mode, **kwargs)
        return runner.run()
    
    def submit(self, strategy: Strategy) -> SubmissionReceipt:
        """Submit to validator via MCP."""
        return self.mcp.call("submit_strategy", strategy=strategy.dict(), hotkey=self.hotkey)
    
    def get_diagnostics(self, submission_id: str) -> Diagnostics:
        """Fetch results from validator."""
        return self.mcp.call("get_diagnostics", submission_id=submission_id)

# Agent-friendly async interface
class AsyncCarbonMiner(CarbonMiner):
    async def propose_train_evaluate_submit(self, challenge: str, backbone: str, n_iterations: int = 100):
        """Full Autoresearch loop: propose → estimate → train → evaluate → submit."""
        prior = await self.get_noisy_prior_async(challenge, backbone)
        for i in range(n_iterations):
            candidate = self.propose_candidate(prior)
            est = await self.estimate_async(candidate, prior)
            if est.confidence > 0.7 and est.estimated_score_delta > 0:
                result = await self.train_local_async(candidate, mode="light")
                if result.local_score > prior.score * 1.05:
                    receipt = await self.submit_async(candidate)
                    if receipt.accepted:
                        prior = candidate  # Update prior
        return prior
```

---

## 8. Genesis Contracts (Immutable, Deployed at Subnet Genesis)

### CarbonTreasury.sol

```solidity
// contracts/CarbonTreasury.sol
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IERC20 {
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

interface IDexRouter {
    function exactInputSingle(ExactInputSingleParams calldata params) external returns (uint256 amountOut);
    struct ExactInputSingleParams {
        address tokenIn;
        address tokenOut;
        uint24 fee;
        address recipient;
        uint256 deadline;
        uint256 amountIn;
        uint256 amountOutMinimum;
        uint160 sqrtPriceLimitX96;
    }
}

contract CarbonTreasury {
    IERC20 public immutable USDC;
    IERC20 public immutable ALPHA;
    IDexRouter public immutable DEX_ROUTER;
    address public immutable BOUNTY_POOL;
    address public immutable OPCO_MULTISIG;
    
    uint256 public immutable BUYBACK_BPS = 2500; // 25% of revenue
    uint256 public immutable TWAP_DURATION = 24 hours;
    
    event BuybackExecuted(uint256 usdcIn, uint256 alphaOut, uint256 blockNumber);
    
    constructor(
        address _usdc,
        address _alpha,
        address _dexRouter,
        address _bountyPool,
        address _opcoMultisig
    ) {
        USDC = IERC20(_usdc);
        ALPHA = IERC20(_alpha);
        DEX_ROUTER = IDexRouter(_dexRouter);
        BOUNTY_POOL = _bountyPool;
        OPCO_MULTISIG = _opcoMultisig;
    }
    
    function depositRevenue(uint256 amount) external {
        require(msg.sender == OPCO_MULTISIG, "Only OpCo");
        USDC.transferFrom(OPCO_MULTISIG, address(this), amount);
        
        uint256 buybackAmount = amount * BUYBACK_BPS / 10000;
        _executeTwapBuyback(buybackAmount);
    }
    
    function _executeTwapBuyback(uint256 usdcAmount) internal {
        // Split into N chunks over TWAP_DURATION
        uint256 chunks = 24; // hourly
        uint256 chunkSize = usdcAmount / chunks;
        
        for (uint256 i = 0; i < chunks; i++) {
            // Execute via DEX with TWAP logic
            // In practice: use Uniswap V3 TWAP oracle + limit orders
            // Simplified here:
            IDexRouter.ExactInputSingleParams memory params = IDexRouter.ExactInputSingleParams({
                tokenIn: address(USDC),
                tokenOut: address(ALPHA),
                fee: 3000, // 0.3%
                recipient: BOUNTY_POOL,
                deadline: block.timestamp + 1 hours,
                amountIn: chunkSize,
                amountOutMinimum: 0, // Protected by TWAP oracle
                sqrtPriceLimitX96: 0
            });
            DEX_ROUTER.exactInputSingle(params);
        }
        
        emit BuybackExecuted(usdcAmount, 0, block.number); // alphaOut tracked off-chain
    }
}
```

### BountyPool.sol

```solidity
// contracts/BountyPool.sol
contract BountyPool {
    IERC20 public immutable ALPHA;
    address public immutable CHALLENGE_WINNER_TRACKER; // Off-chain oracle
    
    mapping(bytes32 => uint256) public challengeBounties; // challenge_id -> alpha amount
    
    event BountyDeposited(bytes32 indexed challengeId, uint256 amount);
    event BountyClaimed(bytes32 indexed challengeId, address indexed miner, uint256 amount);
    
    constructor(address _alpha, address _tracker) {
        ALPHA = IERC20(_alpha);
        CHALLENGE_WINNER_TRACKER = _tracker;
    }
    
    function depositBounty(bytes32 challengeId, uint256 amount) external {
        require(msg.sender == address(CarbonTreasury), "Only Treasury");
        challengeBounties[challengeId] += amount;
        emit BountyDeposited(challengeId, amount);
    }
    
    function claimBounty(bytes32 challengeId, uint256 minerScore, bytes calldata proof) external {
        // Verify miner won challenge via ChallengeWinnerTracker (off-chain oracle)
        // Simplified: check merkle proof against off-chain root
        require(_verifyWinner(challengeId, msg.sender, minerScore, proof), "Invalid claim");
        
        uint256 amount = challengeBounties[challengeId];
        require(amount > 0, "No bounty");
        challengeBounties[challengeId] = 0;
        
        ALPHA.transfer(msg.sender, amount);
        emit BountyClaimed(challengeId, msg.sender, amount);
    }
}
```

### VerificationRegistry.sol

```solidity
// contracts/VerificationRegistry.sol
contract VerificationRegistry {
    struct ModelProof {
        bytes32 modelCardHash;
        string generatorVersion;
        bytes32 operationalEnvelopeHash;
        bytes32 validatorSetHash;
        uint256 blockHeight;
        uint256 timestamp;
    }
    
    mapping(bytes32 => ModelProof) public models; // model_id -> proof
    
    event ModelRegistered(bytes32 indexed modelId, ModelProof proof);
    
    function registerModel(
        bytes32 modelId,
        bytes32 modelCardHash,
        string calldata generatorVersion,
        bytes32 operationalEnvelopeHash,
        bytes32 validatorSetHash
    ) external {
        models[modelId] = ModelProof({
            modelCardHash: modelCardHash,
            generatorVersion: generatorVersion,
            operationalEnvelopeHash: operationalEnvelopeHash,
            validatorSetHash: validatorSetHash,
            blockHeight: block.number,
            timestamp: block.timestamp
        });
        emit ModelRegistered(modelId, models[modelId]);
    }
    
    function getModelProof(bytes32 modelId) external view returns (ModelProof memory) {
        return models[modelId];
    }
    
    function verifyModelCard(bytes32 modelId, bytes32 modelCardHash) external view returns (bool) {
        return models[modelId].modelCardHash == modelCardHash;
    }
}
```

### VerificationGas.sol

```solidity
// contracts/VerificationGas.sol
contract VerificationGas {
    IERC20 public immutable ALPHA;
    AggregatorV3Interface public immutable PRICE_FEED; // Chainlink α/USD
    
    uint256 public constant USD_PRICE_PER_QUERY = 0.001 * 1e18; // $0.001 in 18 decimals
    
    mapping(address => uint256) public prepaidBalance; // Partners stake α
    mapping(address => uint256) public tier; // 0=none, 1=10k/day, 2=100k/day, 3=unlimited
    
    uint256[] public TIER_STAKES = [0, 100_000 * 1e18, 500_000 * 1e18, 2_000_000 * 1e18];
    uint256[] public TIER_QUOTAS = [0, 10_000, 100_000, type(uint256).max];
    
    event Staked(address indexed partner, uint256 tier);
    event QueryExecuted(address indexed caller, bytes32 modelId, uint256 alphaCost);
    
    constructor(address _alpha, address _priceFeed) {
        ALPHA = IERC20(_alpha);
        PRICE_FEED = AggregatorV3Interface(_priceFeed);
    }
    
    function stakeForTier(uint256 tier) external {
        require(tier <= 3, "Invalid tier");
        uint256 amount = TIER_STAKES[tier];
        ALPHA.transferFrom(msg.sender, address(this), amount);
        prepaidBalance[msg.sender] += amount;
        tier[msg.sender] = tier;
        emit Staked(msg.sender, tier);
    }
    
    function query(bytes32 modelId) external payable returns (VerificationData memory) {
        uint256 gasCost = getGasInAlpha();
        
        if (prepaidBalance[msg.sender] >= gasCost) {
            prepaidBalance[msg.sender] -= gasCost;
        } else {
            require(msg.value >= gasCost, "Insufficient gas");
            // Refund excess
            if (msg.value > gasCost) {
                payable(msg.sender).transfer(msg.value - gasCost);
            }
        }
        
        emit QueryExecuted(msg.sender, modelId, gasCost);
        return _getVerificationData(modelId);
    }
    
    function getGasInAlpha() public view returns (uint256) {
        (, int256 price, , , ) = PRICE_FEED.latestRoundData();
        require(price > 0, "Stale price");
        return (USD_PRICE_PER_QUERY * 1e18) / uint256(price);
    }
}
```

---

## 8. MCP Protocol

### Protocol Definition

```python
# carbon/mcp/protocol.py
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
import asyncio
import websockets

class MCPMessageType(Enum):
    # Client → Server
    GET_NOISY_PRIOR = "get_noisy_prior"
    SUBMIT_STRATEGY = "submit_strategy"
    GET_DIAGNOSTICS = "get_diagnostics"
    START_SESSION = "start_session"
    END_SESSION = "end_session"
    
    ESTIMATE = "estimate"
    TRAIN_LOCAL = "train_local"
    
    # Server → Client
    NOISY_PRIOR = "noisy_prior"
    SUBMISSION_RECEIPT = "submission_receipt"
    DIAGNOSTICS = "diagnostics"
    SESSION_STARTED = "session_started"
    STREAMING_UPDATE = "streaming_update"
    ERROR = "error"

@dataclass
class MCPMessage:
    type: MCPMessageType
    payload: Dict[str, Any]
    request_id: str = ""
    session_id: str = ""

class MCPClient:
    def __init__(self, endpoint: str = "wss://mcp.carbon.subnet:8081"):
        self.endpoint = endpoint
        self.ws = None
        self.session_id = None
        self.pending_requests = {}
    
    async def connect(self):
        self.ws = await websockets.connect(self.endpoint)
        asyncio.create_task(self._listen())
    
    async def _listen(self):
        async for message in self.ws:
            msg = MCPMessage(**json.loads(message))
            if msg.request_id in self.pending_requests:
                future = self.pending_requests.pop(msg.request_id)
                future.set_result(msg)
    
    async def call(self, method: str, **kwargs) -> Dict:
        request_id = str(uuid.uuid4())
        future = asyncio.Future()
        self.pending_requests[request_id] = future
        
        msg = MCPMessage(
            type=MCPMessageType(method.upper()),
            payload=kwargs,
            request_id=request_id,
            session_id=self.session_id
        )
        await self.ws.send(json.dumps(msg.__dict__))
        response = await asyncio.wait_for(future, timeout=30.0)
        return response.payload
    
    # Convenience methods
    async def get_noisy_prior(self, challenge: str, backbone: str) -> Strategy:
        return await self.call("get_noisy_prior", challenge=challenge, backbone=backbone)
    
    async def submit_strategy(self, strategy: Strategy) -> SubmissionReceipt:
        return await self.call("submit_strategy", strategy=strategy.dict())
    
    async def get_diagnostics(self, submission_id: str) -> Diagnostics:
        return await self.call("get_diagnostics", submission_id=submission_id)
    
    async def estimate(self, strategy: Strategy, prior: Strategy) -> EstimationResult:
        return await self.call("estimate", strategy=strategy.dict(), prior=prior.dict())
```

---

## 9. Reproducibility & Determinism

### Global Determinism Setup

```python
# carbon/common/determinism.py
import os
import random
import numpy as np

def set_global_determinism(seed: int = 42):
    """Set all random seeds and deterministic flags."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    random.seed(seed)
    np.random.seed(seed)
    jax.config.update("jax_default_prng_impl", "threefry")
    
    # JAX
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

def verify_reproducibility(run_fn, seed: int, n_runs: int = 3) -> ReproducibilityReport:
    """Run function n times, verify identical outputs."""
    set_global_determinism(seed)
    outputs = []
    for i in range(n_runs):
        set_global_determinism(seed)
        output = run_fn()
        outputs.append(hashlib.sha256(str(output).encode()).hexdigest())
    
    all_same = len(set(outputs)) == 1
    return ReproducibilityReport(
        master_seed=seed,
        docker_image_hash=get_docker_image_hash(),
        git_commit=get_git_commit(),
        python_hashseed=seed,
        cublas_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        torch_deterministic=os.environ.get("TORCH_DETERMINISTIC", "1") == "1",
        output_hash=outputs[0],
        passed=all_same
    )
```

### Docker Determinism

```dockerfile
# Pinned base image
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:<pinned>

# Pinned Python packages via requirements-lock.txt
# PYTHONHASHSEED=0
# CUBLAS_WORKSPACE_CONFIG=:4096:8
# torch.use_deterministic_algorithms(True)
```

### Requirements Lockfile (Pinned)

```
jax==0.4.30
jaxlib==0.4.30+cuda12.cudnn89
flax==0.8.4
optax==0.2.1
orbax-checkpoint==0.5.2
numpy==1.26.4
scipy==1.12.0
```

---

## 10. Operational Infrastructure

### Validator Queue Management

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

### Determinism Lockfile (Pinned)

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

### Compilation Cache Persistence

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

---

## Summary: File Organization

```
carbon/
├── validator/
│   ├── main.py                 # Entry point
│   ├── training.py             # lax.scan training loop + early stopping
│   ├── training_phase3.py      # Gradient accumulation + checkpointing
│   ├── losses.py               # unified_loss_fn (boolean masking)
│   ├── physics_gates.py        # 15+ gates (fp32 enforced)
│   ├── monitoring.py           # Online residual monitoring
│   ├── scorer.py               # CarbonScorer (45/30/25)
│   ├── queue.py                # ValidatorQueue (priority + timeouts)
│   └── monitoring.py           # Online residual monitoring
├── generators/
│   ├── base.py                 # ProceduralGenerator ABC
│   ├── poisson.py              # JAX-FEM
│   ├── compressible_ns.py      # Hybrid (mesh online, solutions cached)
│   ├── reacting_ns.py          # Precomputed (DPLR/US3D cache)
│   ├── fsi.py                  # Sequential + coupled (preCICE)
│   ├── stress.py               # StressGenerator (9 categories)
│   └── resolution.py           # Multi-fidelity downsampling
├── backbones/
│   ├── registry.py             # BACKBONE_REGISTRY (FNO, GINO, WNO, Transolver)
│   ├── fno.py                  # SpectralConv + InstanceNorm
│   ├── gino.py                 # Message passing
│   ├── wno.py                  # Wavelet layers
│   ├── transolver.py           # Attention-based
│   ├── precision.py            # bfloat16 + fp32 physics gates context
│   └── checkpointing.py        # Gradient checkpointing policies
├── landscape/
│   ├── pipeline.py             # LandscapeAgent (PySR + MT + DML)
│   ├── pysr_config.py          # PySR configuration
│   └── bridge.jl               # ModelingToolkit.jl bridge
├── miner/
│   ├── cli.py                  # carbon-miner CLI
│   ├── sdk.py                  # CarbonMiner / AsyncCarbonMiner
│   ├── estimator.py            # EstimationEngine (linear sensitivity + proxy)
│   ├── cost_estimator.py       # Provider rate tables
│   └── airgapped.py            # AirgappedMinerToolkit (IL5/IL6)
├── challenges/
│   └── factory.py              # generate_challenge() + create_sponsored_challenge()
├── mcp/
│   └── protocol.py             # MCPClient / MCPServer (WebSocket)
├── landscape/
│   ├── pipeline.py             # LandscapeAgent (PySR + MT + DML)
│   ├── pysr_config.py          # PySR configuration
│   └── bridge.jl               # ModelingToolkit.jl bridge
├── common/
│   ├── determinism.py          # set_global_determinism + verify_reproducibility
│   ├── seeding.py              # Hierarchical seed derivation
│   └── reproducibility.py      # Verification harness
├── contracts/
│   ├── CarbonTreasury.sol
│   ├── BountyPool.sol
│   ├── VerificationRegistry.sol
│   ├── VerificationGas.sol
│   └── PartnerStaking.sol
└── common/
    └── determinism.py          # set_global_determinism + verify_reproducibility
```

---

This implementation document contains all the build-level details needed for Harshdeep to implement the Carbon subnet. Every component is specified with exact JAX patterns, physics gate implementations, generator architectures, training loops, precision policies, contracts, and operational infrastructure.
