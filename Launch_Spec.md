# SPEC.md — Hydrogen PDE Subnet Technical Specification v2.4 (Phase 0 MVP)

---

## Hydrogen Phase 0: The Physics Flywheel

*A Bittensor subnet that turns physics simulation into a compounding knowledge economy. Phase 0 = working flywheel with 3 PDEs, 1 validator, 5+ miners, symbolic loss weighting, and a working economic loop.*

---

## 1. The Core Loop (What Actually Runs)

```
Every Epoch (Week):
├── 1. CHALLENGE released (PDE + train/holdout/stress data + symbolic metadata)
├── 2. MINERS submit strategy JSON (backbone, loss vector, curriculum, UQ)
├── 3. VALIDATORS run training on public split → evaluate holdout → run hidden stress test
├── 4. PHYSICS GATES (hard pass/fail): Boundary, Rollout, UQ, Mass*, Energy*
├── 5. SCORE = log(E_baseline) - log(E_submission)
├── 6. CONSENSUS: Median of 3+ validator scores → final rank
├── 7. EMISSIONS: Top 4 split 40/30/20/10% of challenge budget
├── 8. LANDSCAPE AGENT ingests all fragments → DML causal inference → new baseline
├── 9. WEEKLY: Top-K strategies → ONNX specialists → Specialist Bank

```
```
┌─────────────────────────────────────────────────────────────────┐
│                        HYDROGEN ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   MINERS                                                        │
│   Submit strategy JSONs (backbone, loss weights, curriculum)   │
│   Pay fee • No GPU needed • Never upload weights                │
│          │                                                       │
│          ▼                                                       │
│   VALIDATORS                                                     │
│   Pinned Docker • Train on public split • Hidden stress test   │
│   Hard physics gates (mass, energy, boundary, rollout, UQ)     │
│          │                                                       │
│          ▼                                                       │
│   LANDSCAPE AGENT                                               │
│   Double ML on fragment DAG → Causal effects → New baselines   │
│   Distills winners → ONNX specialists → Specialist Bank        │
│          │                                                       │
│          ▼                                                       │
│   SPECIALIST BANK                                               │
│   Composable ONNX specialists + symbolic metadata + CUDA code  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Core Loop (What Actually Runs Every Week)

---

## The Symbolic Layer: Giving Neural Operators a Physics Brain

Neural operators are brilliant pattern matchers but black boxes. They don't *know* physics.

Hydrogen adds a **Symbolic Layer** (built on Julia's ModelingToolkit.jl) between problem definition and neural network. Think of it as giving the neural network a physics textbook before the exam.

| What the Symbolic Layer Does | Why It Changes Everything |
|-----------------------------|---------------------------|
| **Parses PDEs into symbolic math (AST)** | Neural net gets the *structure* of equations, not just data |
| **Extracts symmetries, conservation laws, dimensionless groups** | Network gets the *physics* for free |
| **Auto-computes loss weights** | Physics computes loss weights; miners don't guess |
| **Symbolic Regression** | Discovers PDEs from specialist behavior → new challenges |
| **Symbolic Distillation** | Compresses specialists *without losing physics* |
| **Acausal Composition** | Specialists snap together like LEGO via symbolic interfaces |
| **Code Generation** | Specialists → CUDA / VHDL / Rust for edge deployment |

**Where It Fits:**
```
Challenge → Symbolic Layer → Miner Strategy → Validator → Landscape Agent → Specialist Bank
                ↑                    ↑              ↑              ↑              ↑
           PDE → AST          Auto loss      Physics gates   PDE discovery   ONNX + sym.
             → AST            weights          + sym.          → new challenges   metadata + CUDA
```

---

## Why Bittensor? Why a Subnet?

Because the incentive structure *is* the product.

| Centralized Benchmark | Hydrogen Subnet |
|----------------------|-----------------|
| Optimize for metric | Optimize for **what validators check** (physics) |
| Static holdouts | **Hidden stress tests** — can't overfit what you've never seen |
| Soft physics penalties | **Hard physics gates** — binary pass/fail |
| Centralized trust | **Median consensus** (5 validators, deterministic) |
| Tribal knowledge | **Causal knowledge graph** (compounds autonomously) |
| Centralized compute | **Decentralized validation** (miners bring ideas, validators provide compute) |

**Barrier to entry = an idea, not a GPU cluster.** A researcher with a clever curriculum competes on equal footing with a quant fund running H100s.

---

## The Mechanism (What Actually Runs)

### 1. Open Challenges
Each challenge = PDE problem + public train/holdout + **hidden stress test** (procedural, seeded by challenge ID: shifted Reynolds, resolutions, geometries). No miner has ever seen it.

### 2. Miners Submit Strategy JSON
```json
{
  "challenge_id": "poisson_2d_v1",
  "hotkey": "5F...",
  "backbone": "PINO",
  "resolution": [128, 128],
  "pino": {
    "loss_vector": {
      "pde_residual": 1.0,
      "conservation_mass": 1.5,
      "conservation_momentum": 1.0,
      "boundary": 0.5
    },
    "physics_loss_type": "pde_residual",
    "boundary_handling": "ghost_cells"
  },
  "optimizer": "AdamW",
  "learning_rate": 0.001,
  "scheduler": "CosineAnnealingLR",
  "batch_size": 16,
  "epochs": 200,
  "physics_informed": true,
  "curriculum_learning": {
    "enabled": true,
    "start_resolution": [64, 64],
    "end_resolution": [128, 128],
    "ramp_epochs": 50
  },
  "uq_config": {
    "method": "deep_ensemble",
    "num_members": 4,
    "calibration_target": 0.90
  }
}
```
**Fee:** 0.1 TAO (burned if validation fails). No GPU. No weights uploaded. Just an idea.

### 3. Validators Run the Gauntlet
Pull pinned Docker image (`hydrogen/validator:pino-v24.09`), inject JSON, train on public split, evaluate on holdout, run **hidden stress test** through **hard physics gates**:

| Gate | Formula | Threshold | What It Catches |
|------|---------|-----------|-----------------|
| **Boundary Satisfaction** | `‖u - u_BC‖₂ / ‖u_BC‖₂` | `< 1e-3` | Wrong boundary conditions |
| **Rollout Stability** | `‖E(t=100) - E(0)‖ / E(0)` | `< 0.01` | Blow-up over time |
| **UQ Calibration** | `|coverage - 0.90|` | `< 0.02` | Uncertainty lying |

**Hard failure = score zero. Physics is binary.**

**Score = log(E_baseline) - log(E_submission).** Log-space improvement against current baseline. Median of 3+ validators = final rank.

**Every submission becomes a StrategyFragment.** Config + score + stress result + UQ + lineage → every fragment teaches the Landscape.

---

## The Landscape Agent: Where Knowledge Compounds

The Landscape runs **Double Machine Learning on the fragment DAG** to estimate `P(improvement | do(param))` — the *causal effect* of a config change, not spurious correlation.

It discovers: Fourier modes 32 help *only when* physics loss > 1.0. Curriculum helps *only when* start resolution ≤ 0.5× end resolution. Ghost cells prevent boundary locking *only for* elasticity and NS, not Poisson.

Every challenge cycle: proposes new baseline JSON with strongest causal effects.  
Every distillation cycle: Top-K strategies → **ONNX specialists** via multi-teacher distillation → regression tested against stress tests → published to **Specialist Bank** with validity domains + dual licensing (AGPL-3.0 + commercial).

The Landscape is funded by Owner's allocation + time-locked treasury. The agent is replaceable; **the causal knowledge graph is not.**

---

## The Symbolic Layer: Giving Neural Operators a Physics Brain

Neural operators are brilliant pattern matchers but black boxes. They don't *know* physics.

Hydrogen adds a **Symbolic Layer** (ModelingToolkit.jl, from Chris Rackauckas's team at MIT) between problem definition and neural network. Think: giving the neural network a physics textbook before the exam.

| What the Symbolic Layer Does | Why It Changes Everything |
|-----------------------------|---------------------------|
| **Parses PDEs into symbolic math (AST)** | Net gets *structure* of equations, not just data |
| **Extracts symmetries, conservation laws, dimensionless groups** | Network gets *physics* for free |
| **Auto-computes loss weights** | Physics computes loss weights; miners don't guess |
| **Symbolic Regression** | Discovers PDEs from specialist behavior → new challenges |
| **Symbolic Distillation** | Compresses specialists *without losing physics* |
| **Acausal Composition** | Specialists snap together like LEGO via symbolic interfaces |
| **Code Generation** | Specialists → CUDA / VHDL / Rust for edge deployment |

**Where It Fits:**
```
Challenge → Symbolic Layer → Miner Strategy → Validator → Landscape Agent → Specialist Bank
                ↑                    ↑              ↑              ↑              ↑
           PDE → AST          Auto loss      Physics gates   PDE discovery   ONNX + sym.
             → AST            weights          + sym.          → new challenges   metadata + CUDA
```

---
## 1. The Problems (Phase 0: 3 PDEs, Launch Day)

| ID | Problem | Dimension | Physics Class | Resolution | Reference |
|----|---------|-----------|---------------|------------|-----------|
| 1 | **Poisson** | 2D | Elliptic, constant-coeff | 128×128 | PhysicsNeMo |
| 2 | **Darcy** | 2D | Elliptic, variable-coeff | 128×128 | PhysicsNeMo / PDEBench |
| 3 | **Burgers** | 1D | Nonlinear advection/shocks | 256×100 (x,t) | PhysicsNeMo |

**Each challenge provides:** public train/holdout + hidden stress test (procedural, seeded by challenge ID) + symbolic metadata.

---

## 2. Miner Interface (Dead Simple)

### 2.1 Strategy JSON (What Miners Submit)
```json
{
  "challenge_id": "poisson_2d_v1",
  "hotkey": "5F...",
  "backbone": "PINO",
  "resolution": [128, 128],
  "pino": {
    "loss_vector": {
      "pde_residual": 1.0,
      "conservation_mass": 1.5,
      "conservation_momentum": 1.0,
      "boundary": 0.5
    },
    "physics_loss_type": "pde_residual",
    "boundary_handling": "ghost_cells"
  },
  "optimizer": "AdamW",
  "learning_rate": 0.001,
  "scheduler": "CosineAnnealingLR",
  "batch_size": 16,
  "epochs": 200,
  "physics_informed": true,
  "curriculum_learning": {
    "enabled": true,
    "start_resolution": [64, 64],
    "end_resolution": [128, 128],
    "ramp_epochs": 50
  },
  "uq_config": {
    "method": "deep_ensemble",
    "num_members": 4,
    "calibration_target": 0.90
  }
}
```

### 2.2 Miner CLI (Dead Simple)
```bash
# Install
pip install hydrogen-miner

# Configure
hydrogen-miner config --wallet my_wallet --hotkey my_hotkey --netuid 107

# See active challenges
hydrogen-miner challenges

# Get baseline + symbolic priors
hydrogen-miner baseline poisson_2d_v1
hydrogen-miner priors poisson_2d_v1

# Submit strategy (pays 0.1 TAO fee)
hydrogen-miner submit --challenge poisson_2d_v1 --strategy my_strategy.json

# Check results
hydrogen-miner rewards --days 30
```

### 2.3 Python SDK (For Agent Builders)
```python
from hydrogen_miner import HydrogenClient, Strategy

client = HydrogenClient(hotkey="5F...")

# Get challenge + symbolic priors
challenge = client.get_challenge("poisson_2d_v1")
baseline = client.get_baseline("poisson_2d_v1")
priors = client.get_priors("poisson_2d_v1")  # Symbolic features + suggested weights

# Build strategy
strategy = Strategy(
    backbone="PINO",
    resolution=[128, 128],
    pino=dict(
        loss_vector=dict(
            pde_residual=1.0,
            conservation_mass=priors["suggested_weights"]["conservation_mass"],  # 1.5
            conservation_momentum=priors["suggested_weights"]["conservation_momentum"],  # 1.0
            boundary=0.5
        )
    ),
    optimizer="AdamW",
    learning_rate=0.001,
    curriculum=dict(enabled=True, start_resolution=[64,64], end_resolution=[128,128], ramp_epochs=50)
)

# Local validation before submitting (saves fee)
result = client.validate_locally(strategy, challenge_id="poisson_2d_v1", quick=True)
print(f"Estimated score: {result.estimated_score}")

# Submit
result = client.submit("poisson_2d_v1", strategy)
print(f"Rank: {result.rank}, Score: {result.score}, Reward: {result.emission_reward} TAO")
```

---

## 3. Validator Specification (Single Image, Deterministic)

### 3.1 Single Validator Image (Phase 0)
```dockerfile
# hydrogen/validator:pino-v24.09
FROM nvcr.io/nvidia/pytorch:24.09-py3

RUN apt-get update && apt-get install -y julia=1.10.* && rm -rf /var/lib/apt/lists/*
RUN julia -e 'using Pkg; Pkg.add(["ModelingToolkit", "Symbolics", "DataDrivenDiffEq", "PySR"]); Pkg.precompile()'
RUN pip install juliacall

COPY validator/ /workspace/validator/
WORKDIR /workspace/validator
ENTRYPOINT ["python", "entrypoint.py"]
```

### 3.2 Validator Pipeline (Deterministic)
```python
def validate_submission(challenge_id: str, miner_submission: dict) -> ValidationResult:
    # 1. Load challenge data
    train_data, holdout_data, stress_data = load_challenge_splits(challenge_id)
    challenge = load_challenge_metadata(challenge_id)
    
    # 2. Enrich with symbolic metadata
    symbolic_metadata = enrich_challenge(challenge_id, challenge_data_path)
    
    # 3. Execute
    model = train_backbone(
        image=select_backbone_image(miner_submission["backbone"]),
        config=miner_submission,
        train_data=train_data,
        custom_data=resolve_custom_data(miner_submission.get("custom_data")),
        seed=derive_seed(challenge_id, validator_hotkey)  # DETERMINISTIC
    )
    
    # 4. Evaluate on public holdout
    E_baseline = load_current_baseline_error(challenge_id)
    E_submission = evaluate(model, holdout_data, challenge)
    improvement = log(E_baseline) - log(E_submission)
    
    # 4. Hidden stress test + physics gates
    stress_result = run_stress_test(model, stress_data, challenge)
    
    # 5. UQ calibration check
    uq_metrics = evaluate_uq(model, stress_data, miner_submission.get("uq_config", {}))
    
    # 6. Compute score
    if stress_result.hard_failure:
        score = 0.0
        reason = stress_result.failure_reason
    else:
        base_score = max(0.0, improvement)
        soft_penalty = stress_result.soft_penalty
        uq_bonus = uq_calibration_bonus(uq_metrics, 0)  # Phase 0
        score = (base_score * soft_penalty) + uq_bonus
    
    return ValidationResult(score, improvement, stress_result, uq_metrics)
```

---

## 3.3 Physics Gates (Phase 0: Only 3 Hard Gates)

| Gate | Formula | Threshold | What It Catches |
|------|---------|-----------|-----------------|
| **Boundary Satisfaction** | `‖u - u_BC‖₂ / ‖u_BC‖₂` | `< 1e-3` | Wrong boundary conditions |
| **Rollout Stability** | `‖E(t=100) - E(0)‖ / E(0)` | `< 0.01` | Blow-up over time |
| **UQ Calibration** | `|coverage - 0.90|` | `< 0.02` | Uncertainty lying |

**Hard failure = score zero. Physics is binary.**

**Score = log(E_baseline) - log(E_submission).** Median of 3+ validators = final rank.

---

## 4. Challenge Specification (3 PDEs, Launch Day)

| ID | Problem | Dimension | Physics Class | Resolution | Reference |
|----|---------|-----------|---------------|------------|-----------|
| 1 | **Poisson** | 2D | Elliptic, constant-coeff | 128×128 | PhysicsNeMo |
| 2 | **Darcy** | 2D | Elliptic, variable-coeff | 128×128 | PhysicsNeMo / PDEBench |
| 3 | **Burgers** | 1D | Nonlinear advection/shocks | 256×100 (x,t) | PhysicsNeMo |

**Each challenge provides:** public train/holdout + hidden stress test (procedural, seeded by challenge_id) + symbolic metadata.

### Symbolic Metadata (Per Challenge, Generated Once, Cached Forever)
```json
{
  "governing_pde": "∇²u = f",
  "symmetries": ["translation", "rotation"],
  "conservation_laws": ["mass"],
  "dimensionless_groups": {},
  "boundary_types": ["dirichlet", "neumann"],
  "coupling_terms": [],
  "suggested_loss_weights": {
    "pde_residual": 1.0,
    "conservation_mass": 1.5,
    "boundary": 0.5
  },
  "validity_domain": {}
}
```
---

## 5. Symbolic Layer (Phase 0 MVP: Config-Based, No MTK Runtime)

### 5.1 One Python Call Per Challenge (Cached Forever)
```python
# symbolic_layer/extract_features.py
def extract_symbolic_features(pde_spec: dict) -> dict:
    """
    Input: {"pde": "poisson", "dim": 2, "bc": "dirichlet"}
    Output: Enriched features for miner/validator/landscape
    """
    return {
        "symmetries": ["translation", "rotation"],
        "conservation_laws": ["mass"],
        "dimensionless_groups": {},
        "boundary_types": ["dirichlet", "neumann"],
        "suggested_loss_weights": {
            "pde_residual": 1.0,
            "conservation_mass": 1.5,
            "boundary": 0.5
        },
        "validity_domain": {}
    }
```

**Used by:**
- **Miners:** Auto-fill loss_vector from `suggested_loss_weights`
- **Validators:** Physics gates thresholds adapt to PDE type
- **Landscape:** Features enrich DML causal inference

---

## 6. Landscape Agent (The Brain)

### 6.1 Daily Causal Update (Runs Automatically)
```python
def daily_causal_update(new_fragments: List[StrategyFragment]):
    dag.add_nodes(new_fragments)
    
    for problem_id in PROBLEM_FAMILIES:
        problem_fragments = dag.get_problem_fragments(problem_id)
        
        # Double Machine Learning
        dml = DoubleML(
            model_y=HistGradientBoostingRegressor(),
            model_t=HistGradientBoostingRegressor(),
            n_folds=5
        )
        dml.fit(problem_fragments)
        
        effects = {param: dml.ate(param) for param in TUNABLE_PARAMS}
        interactions = dml.interaction_effects()  # "fourier_modes helps when physics_loss > 1.0"
        
        proposal = propose_baseline(effects, interactions, current_baseline)
        owner_review_queue.put(proposal)
```

### 6.2 Weekly Specialist Distillation
```python
def weekly_distillation():
    for problem_id in PROBLEM_FAMILIES:
        teachers = select_top_fragments(problem_id, k=5)
        
        student = distill_ensemble(
            teachers=teachers,
            student_config=StudentConfig(width_factor=0.5),
            loss_fn=DistillationLoss(
                alpha_output=1.0,
                alpha_physics=0.5,
                alpha_features=0.3,
                alpha_uq=0.2
            ),
            data=load_problem_data(problem_id)
        )
        
        if pass_stress_test(student, get_stress_data(problem_id)):
            specialist = Specialist(
                specialist_id=f"{problem_id}_v{version}",
                onnx_model=export_onnx(student),
                problem_signature=get_signature(problem_id),
                metrics=evaluate_full(student, problem_id),
                validity_domain=estimate_validity_domain(student),
                license="AGPL-3.0 + Commercial Dual-License"
            )
            specialist_bank.publish(specialist)
```

---

## 8. Emission Mechanics (Simple & Hardened)

| Stream | % Budget | Frequency | Recipients | Trigger |
|--------|----------|-----------|------------|---------|
| **Stipend (Salary)** | **70%** | **Every Epoch** | Top 4 (Rank 1-4) | `Score > 0` |
| **Breakthrough Bounty** | **25%** | **Only on Breakthrough** | Breakthrough Miner(s) | `improvement > 5%` |
| **Treasury** | **5%** | Every Epoch | Owner | Always |

**Stipend Split (70% pool → Top 4):**
| Rank | Share | Condition |
|------|-------|-----------|
| 1st | 40% | `Score > 0` |
| 2nd | 30% | `Score > 0` |
| 3rd | 20% | `Score > 0` |
| 4th | 10% | `Score > 0` |

**Breakthrough Bounty (25% pool):**
- Accumulates until `improvement > 5%` (log-space)
- **Half-life: 3 epochs** (decays if no breakthrough)
- **Auto-claims** on next `improvement > 0`

**Stipend Safety Net:**
- 100% forever **UNLESS** 3+ epochs with `improvement ≤ 0`
- Then: 50% → 25% → 10% → Ejected at 5 epochs
- **Any `improvement > 0` resets to 100% instantly**

---

## 12. Phase Definitions (Only What Ships)

### Phase 0: The Physics Flywheel (Launch → Month 3) ✅ **SHIP THIS**
| Metric | Target |
|--------|--------|
| PDEs | 3 (Poisson, Darcy, Burgers) |
| Validators | ≥ 3 operational |
| Miners | ≥ 10 unique |
| Baseline improvement | > 0.02/challenge avg 30 epochs |
| Symbolic layer | Auto loss weights from config for all 3 PDEs 

---

## 13. Validator Docker Image (Buildable Today)

```dockerfile
# hydrogen/validator:pino-v24.09
FROM nvcr.io/nvidia/pytorch:24.09-py3

RUN apt-get update && apt-get install -y julia=1.10.* && rm -rf /var/lib/apt/lists/*
RUN julia -e 'using Pkg; Pkg.add(["ModelingToolkit", "Symbolics", "DataDrivenDiffEq", "PySR"]); Pkg.precompile()'
RUN pip install juliacall

COPY validator/ /workspace/validator/
WORKDIR /workspace/validator
ENTRYPOINT ["python", "entrypoint.py"]
```

---

## 14. Appendix: Minimal Math

### 14.1 Log-Space Improvement
```
improvement = log(E_baseline) - log(E_submission)
E = relative L2 error = ‖u_pred - u_true‖₂ / ‖u_true‖₂
```

### 14.2 Physics Gates (Phase 0)
| Gate | Formula | Threshold |
|------|---------|-----------|
| Boundary Satisfaction | `‖u - u_BC‖₂ / ‖u_BC‖₂` | `< 1e-3` |
| Rollout Stability | `‖E(t=100) - E(0)‖ / E(0)` | `< 0.01` |
| UQ Calibration | `|coverage - 0.90|` | `< 0.02` |

### 14.2 Emission Math
```
improvement = log(E_baseline) - log(E_submission)
Score = (improvement × soft_penalty) + UQ_bonus
Stipend = 70% × [40/30/20/10 split] × decay_multiplier
Bounty: 25% pool, half-life 3 epochs, triggers at improvement > 5%
```

---

## 14. Phase 0 Launch Checklist

- [ ] 3 PDE challenges generated (Poisson, Darcy, Burgers) + symbolic metadata
- [ ] 1 validator docker image built & tested (`hydrogen/validator:pino-v24.09`)
- [ ] 1 miner CLI working (`hydrogen-miner submit`)
- [ ] 1 validator node running & reachable
- [ ] Chain pallet deployed (challenge + commit/reveal + scoring + emissions)
- [ ] Landscape agent running (daily DML → baseline updates)
- [ ] 5+ miners submitting strategies
- [ ] 3+ validators operational
- [ ] 30 epochs without manual intervention

**Launch = Physics flywheel spinning autonomously.**

---

## Links (Only What Exists)

| Resource | Link |
|----------|------|
| **Technical Specification** | `SPEC.md` (this file) |
| **Roadmap** | `ROADMAP.md` |
| **Validator Image** | `docker pull hydrogen/validator:pino-v24.09` |
| **Miner CLI** | `pip install hydrogen-miner` |
| **Python SDK** | `pip install hydrogen-agent` |
| **Repository** | `github.com/hydrogen-subnet` |
| **Discord** | `discord.gg/hydrogen` |

---

*Hydrogen Phase 0: Where every training run teaches the network. Where physics is the only metric that pays. Where the sim-to-real gap becomes a shrinking, measured, attributable quantity.*
