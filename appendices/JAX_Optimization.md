# Appendix: JAX Core Training Optimizations (Validator-Side)

**Carbon PDE Subnet**  
**Version:** 1.0 (July 2026)  
**Status:** Core Engineering Appendix

This document specifies the JAX compilation, execution, and data-routing optimizations integrated into the Carbon Validator engine (`carbon/validator/training.py` and supporting modules). The goal is to reduce hardware runtime costs across all execution phases while preserving the mathematical intent of the miner’s submitted `strategy.json`.

These optimizations are a foundational requirement for keeping validator-side compute manageable as the subnet scales from academic PDEs through compressible flow, multi-physics coupling, and 3D turbulence regimes.

---

## 1. Overview & Objectives

### Key Performance Targets

| Target | Description |
|--------|-------------|
| **Zero Re-compilation** | Eliminate XLA compilation overhead during runtime block tempos caused by strategy-dependent control flow. |
| **VRAM Footprint Mitigation** | Reduce peak memory saturation by ~30–50% using unified tensor precision handling (bfloat16). |
| **Deterministic Execution** | Enforce strict hardware constraints and hard resource bounds while still honoring miner-defined optimization paths. |
| **Bounded Evaluation Latency** | Prevent runaway epoch budgets and sequential evaluation queues from monopolizing validator resources. |

### Design Principles

1. **Preserve Miner Intent** — Optimizations must not silently rewrite or degrade a miner’s stated strategy. Defaults may be applied only when fields are omitted.
2. **Static XLA Graphs** — Prefer functional masking, `lax.scan` / `lax.cond`, and `vmap` over Python-level conditionals that trigger recompilation.
3. **Hard Safety Rails** — Absolute step/wall-clock limits and early-stopping floors always take precedence over miner-specified epoch counts.
4. **Opportunistic Parallelism** — Vectorize only when architectures are identical; fall back cleanly otherwise.

---

## 2. Dynamic Loss Masking (Unified XLA Graph)

### Problem

Miners can dynamically modify or omit loss terms (e.g., toggling `physics_residual` or `conservation_penalty`) in `strategy.json`. Native Python `if/else` statements inside the core loss calculation force an expensive XLA recompilation for every unique strategy configuration, creating a severe processing backlog.

### Solution

Execute a single static, unified loss function. All possible physical and data loss objectives are computed continuously. The miner’s choices are expressed purely as floating-point multipliers (0.0 or 1.0) passed as runtime parameters.

```python
# carbon/validator/losses.py
import jax
import jax.numpy as jnp
from typing import Dict, NamedTuple

class LossWeights(NamedTuple):
    data_mse: float
    physics_residual: float
    boundary_mse: float
    conservation_penalty: float

@jax.jit
def unified_loss_fn(
    params: Dict,
    batch_inputs: jnp.ndarray,
    batch_targets: jnp.ndarray,
    weights: LossWeights,
    model_apply_fn
) -> jnp.ndarray:
    """
    Statically compiled loss function. Uses functional masking instead of
    procedural branch statements to maintain a single XLA compilation path.
    """
    # 1. Forward Pass
    predictions = model_apply_fn(params, batch_inputs)

    # 2. Continuous calculation of all objective terms
    loss_data = jnp.mean(jnp.square(predictions - batch_targets))
    loss_phys = compute_pde_residuals(params, batch_inputs, model_apply_fn)
    loss_bound = compute_boundary_residuals(params, batch_inputs, model_apply_fn)
    loss_conserve = compute_conservation_penalties(predictions)

    # 3. Dynamic linear masking via parameter multipliers
    # Hard cutoff prevents tiny floating-point leakage from "zeroed" terms
    w_data = jnp.where(weights.data_mse < 1e-8, 0.0, weights.data_mse)
    w_phys = jnp.where(weights.physics_residual < 1e-8, 0.0, weights.physics_residual)
    w_bound = jnp.where(weights.boundary_mse < 1e-8, 0.0, weights.boundary_mse)
    w_cons = jnp.where(weights.conservation_penalty < 1e-8, 0.0, weights.conservation_penalty)

    total_loss = (
        (w_data * loss_data) +
        (w_phys * loss_phys) +
        (w_bound * loss_bound) +
        (w_cons * loss_conserve)
    )
    return total_loss
```

**Notes**
- All residual functions must themselves be safe when their corresponding weight is zero (no side-effects or NaN paths).
- This pattern is the highest-ROI optimization and should be implemented first.

---

## 3. Functional Early-Stopping Loop via `jax.lax.scan`

### Problem

Miners control the absolute `epochs` parameter. Rogue or poorly configured submissions can request extremely high epoch counts and monopolize validator compute. Traditional Python loop-based training prevents JAX from optimizing the step sequence into a single hardware trace and makes hard resource limits difficult to enforce.

### Solution

Structure the epoch sequence with `jax.lax.scan` and use `jax.lax.cond` for early-termination logic so that the control flow remains inside XLA. A hard absolute step limit and wall-clock timeout act as additional safety rails outside the scan.

```python
# carbon/validator/training.py (excerpt)
import jax
import jax.lax as lax
import jax.numpy as jnp
from typing import Tuple, Dict, NamedTuple

class TrainerState(NamedTuple):
    params: Dict
    opt_state: any
    best_loss: float
    consecutive_no_improve: int
    terminated: bool

def training_step_engine(
    state: TrainerState,
    unused_idx: int,
    data_loader,
    weights: LossWeights,
    patience: int = 50
) -> Tuple[TrainerState, float]:
    """
    Single compiled epoch execution step passed to lax.scan.
    """
    def execution_branch(s: TrainerState) -> Tuple[TrainerState, float]:
        new_params, new_opt_state, epoch_loss = run_epoch_batches(
            s.params, s.opt_state, data_loader, weights
        )

        improved = epoch_loss < s.best_loss
        next_best = jnp.where(improved, epoch_loss, s.best_loss)
        next_count = jnp.where(improved, 0, s.consecutive_no_improve + 1)
        should_abort = next_count >= patience

        return TrainerState(
            params=new_params,
            opt_state=new_opt_state,
            best_loss=next_best,
            consecutive_no_improve=next_count,
            terminated=should_abort
        ), epoch_loss

    def short_circuit_branch(s: TrainerState) -> Tuple[TrainerState, float]:
        return s, s.best_loss

    next_state, running_loss = lax.cond(
        state.terminated,
        short_circuit_branch,
        execution_branch,
        state
    )
    return next_state, running_loss

def fit(
    init_params,
    init_opt,
    max_epochs: int,
    data_loader,
    weights: LossWeights,
    patience: int = 50,
    hard_step_limit: int = None
):
    """
    Compiles the training trajectory into a single execution pass.
    hard_step_limit provides an absolute safety bound independent of miner epochs.
    """
    effective_epochs = max_epochs
    if hard_step_limit is not None:
        effective_epochs = min(max_epochs, hard_step_limit)

    init_state = TrainerState(
        params=init_params,
        opt_state=init_opt,
        best_loss=jnp.inf,
        consecutive_no_improve=0,
        terminated=False
    )

    final_state, loss_history = lax.scan(
        lambda s, i: training_step_engine(s, i, data_loader, weights, patience),
        init_state,
        jnp.arange(effective_epochs)
    )
    return final_state, loss_history
```

**Implementation Guidance**
- Prefer JIT-ing the training *step* and controlling the outer loop in Python with a hard wall-clock timeout for maximum robustness.
- Use the full `lax.scan` form when the entire trajectory must stay inside a single compiled graph.
- Always enforce both a patience-based early stop *and* an absolute step/wall-clock limit.

---

## 4. bfloat16 Mixed-Precision & Memory Sandboxing

### Problem

High-dimensional Fourier Neural Operators and multi-physics models require large amounts of VRAM during full backpropagation. This creates a hardware barrier for smaller validators, especially in later phases.

### Solution

Enforce native XLA bfloat16 mixed precision for inner tensor contractions and spectral convolutions while keeping master weights and critical residual calculations in float32.

```python
# carbon/backbones/precision.py
import jax
import jax.numpy as jnp
from jaxtyping import Array

def enforce_mixed_precision_policy():
    """
    Initializes global XLA hardware compilation flags for tensor-core utilization.
    """
    jax.config.update("jax_default_matmul_precision", "tensorcore")

def cast_to_bfloat16(x: Array) -> Array:
    """
    Downsamples dense representations to bfloat16 while preserving the
    broad dynamic exponent range of FP32.
    """
    return x.astype(jnp.bfloat16)

def selective_cast_for_residuals(x: Array) -> Array:
    """
    Keep residual and conservation calculations in higher precision
    to avoid noise amplification in physics gates.
    """
    return x.astype(jnp.float32)
```

**Notes**
- Apply casting consistently across the model and the loss so the XLA graph remains clean.
- Critical residual functions used by physics gates should remain in float32 or use selective casting.
- Expected VRAM reduction on A100/H100 for spectral models: 30–50% in practice.

---

## 5. Mesh-Independence Multi-Fidelity Grid Curriculums

### Problem

Forcing downsampled spatial resolutions validator-side could overwrite valid miner-designed training strategies that rely on high-frequency, fine-mesh features from the first epoch.

### Solution

Extend the `strategy.json` curriculum schema with an optional `spatial_resolution_scale` field. Miners retain full control. If the field is omitted, the validator applies a default coarse-to-fine multi-fidelity schedule to save compute.

**Schema Extension**

```json
{
  "curriculum": [
    {
      "phase": 1,
      "epochs": 100,
      "spatial_resolution_scale": 0.5
    },
    {
      "phase": 2,
      "epochs": 200,
      "spatial_resolution_scale": 1.0
    }
  ]
}
```

**Rules**
1. Miners retain full control over when and how their models downsample grids.
2. If the parameter is omitted, the validator enforces a default multi-fidelity schedule (coarse → full resolution).
3. Downsampling must use proper restriction operators (simple striding is acceptable only for structured grids; unstructured / Abaqus meshes require interpolation or filtering).

```python
# carbon/generators/resolution.py
import jax
import jax.numpy as jnp
from typing import Tuple

@jax.jit
def downsample_spatial_grid(
    coords: jnp.ndarray,
    fields: jnp.ndarray,
    scale: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
```

**Caution**  
Changing resolution mid-training interacts with normalization statistics and Fourier mode counts. FNO mode counts must be adjusted or padded consistently when resolution changes.

---

## 6. Multi-Miner Vectorized Batched Evaluation (`jax.vmap`)

### Problem

Processing many unique configuration JSONs sequentially creates a severe latency queue inside a standard block tempo.

### Solution

When multiple miners submit identical backbone layout definitions (same modes, width, depth, activation, etc.) and differ only in scalar hyperparameters (learning rate, loss weights, etc.), the validator automatically stacks them into a single parallel tensor and evaluates via `jax.vmap`.

```python
# carbon/validator/evaluation.py
import jax
from typing import Dict

def parallel_miner_batch_execution(
    stacked_params: Dict,
    stacked_weights: LossWeights,
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    model_apply_fn
):
    """
    Vectorizes optimization tasks across a multi-miner cohort that shares
    identical static architecture shapes.
    """
    vectorized_loss_evaluator = jax.vmap(
        lambda p, w: unified_loss_fn(p, inputs, targets, w, model_apply_fn),
        in_axes=(0, 0)
    )
    batch_loss_metrics = vectorized_loss_evaluator(stacked_params, stacked_weights)
    return batch_loss_metrics
```

**Applicability Window**
- Only strategies with *identical* static shapes can be safely `vmap`-ed.
- Fall back to sequential evaluation when architectures diverge.
- Stacking increases peak VRAM; monitor and limit cohort size.

This is an opportunistic optimization, not a general-purpose path.

---

## 7. Validator-Side Compute & Cost Estimates (Per Miner Submission)

All estimates assume the optimizations above are active. Marketplace rates are conservative 2026 on-demand figures and include a ~20% buffer for real-world variance.

### Phase 0: Foundation (Months 0–4)

- **Physics**: 7 Academic 2D/3D PDEs
- **Hardware**: 1× NVIDIA A100 80GB
- **Optimized Runtime**: ~5 minutes
- **Estimated Cost**: ~$0.07–0.10

**Breakdown**
- Deterministic Training (80%): multi-fidelity coarse curriculum + early stopping
- Physics Gates (8%): compiled residual checks
- Robustness / Stress (10%): 10k-step rollout
- Benchmark (2%): held-out inference

### Phase 1A–1B: Compressible & Reacting Flow (Months 4–14)

- **Physics**: Transonic/Hypersonic NS, chemistry, sequential FSI, 6-DOF
- **Hardware**: 1× NVIDIA H100 PCIe
- **Optimized Runtime**: ~20 minutes
- **Estimated Cost**: ~$0.35–0.45

**Breakdown**
- Training (75%): bfloat16 + unified graph
- Gates (10%): adjoint consistency + shock-local residuals
- Robustness (12%): turbulence UQ variants
- Benchmark (3%)

### Phase 2A–2B: Customization, Intelligence & Air-Gap (Months 14–28)

- **Physics**: Abaqus ODB, ModelingToolkit losses, LoRA, IL5 enclaves
- **Hardware**: 2× NVIDIA H100 PCIe (Slurm)
- **Optimized Runtime**: ~30 minutes
- **Estimated Cost**: ~$1.00–1.30

### Phase 3: Multi-Physics Coupling (Months 28–40)

- **Physics**: Full two-way preCICE coupling (Fluid FNO + Solid GINO)
- **Hardware**: 4× NVIDIA H100 SXM5 (NVLink)
- **Optimized Runtime**: ~2 hours
- **Estimated Cost**: ~$20–28

### Phase 4: Production (Months 40–52)

- **Physics**: 3D wall-resolved LES, hypersonic vehicle, ablation
- **Hardware**: 8× NVIDIA H100/H200 SXM
- **Optimized Runtime**: ~4 hours
- **Estimated Cost**: ~$80–110

> **Note**: Later-phase numbers assume near-ideal multi-GPU scaling and aggressive early termination. Real-world multi-physics and 3D turbulence workloads will frequently exceed these figures. Additional economic controls (rate limits, reputation-weighted evaluation depth, paid priority evaluation) will be required to keep validators sustainable.

---

## 8. Implementation Priority & Safeguards

**Recommended Implementation Order**

1. Unified loss masking (highest ROI, lowest risk)
2. bfloat16 mixed-precision policy + selective residual casting
3. Hard wall-clock / absolute step limits + patience early-stopping
4. Default multi-fidelity curriculum (with proper restriction operators)
5. Opportunistic `vmap` cohorting for identical architectures

**Mandatory Safeguards**

- Per-challenge maximum GPU-second budget
- Absolute epoch / step ceiling independent of miner `strategy.json`
- Wall-clock timeout per submission
- XLA compilation-cache hit-rate and peak-VRAM monitoring
- Clear fallback to sequential evaluation when `vmap` shape requirements are not met
- Reputation- or stake-weighted evaluation depth for high-load periods

---

## 9. Integration Notes

These optimizations live primarily in:

- `carbon/validator/losses.py` — unified loss
- `carbon/validator/training.py` — `lax.scan` / early-stop engine
- `carbon/backbones/precision.py` — mixed-precision policy
- `carbon/generators/resolution.py` — multi-fidelity downsampling
- `carbon/validator/evaluation.py` — `vmap` cohort path

They must be fully compatible with the existing Model Card generation, physics-gate pipeline, and deterministic seeding hierarchy defined in the main SPEC.

---

*This appendix is a living engineering document. Cost figures and hardware assumptions should be revisited as marketplace rates and XLA performance characteristics evolve.*
