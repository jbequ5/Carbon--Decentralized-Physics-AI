# Carbon Physics Intelligence Subnet

---

## 1. Executive Summary

Carbon is a Bittensor subnet that operates a **decentralized verification layer for physics-informed neural operator surrogates**. It coordinates a network of miners and autonomous agents to discover optimal training strategies for Neural Operators (FNO, GINO, WNO, Transolver) under rigorous, trustless adversarial validation.

**Core Innovation**: Miners submit training strategies (loss configurations, curricula, architectures, data generation parameters). Validators execute full deterministic training from scratch on hidden, procedurally generated data, evaluating against hard physics gates. The Landscape Agent compounds symbolic and causal insights across all evaluations, creating a self-improving intelligence layer.

**Carbon's Position in the Physics-AI Stack**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHYSICS-AI STACK                             │
├─────────────────────────────────────────────────────────────────┤
│  COMPUTE LAYER          │ NVIDIA (H100, Blackwell, CUDA,        │
│                         │  TensorRT, Apollo, Cosmos)            │
│                         │ Demand generator — not competitor      │
├─────────────────────────┼───────────────────────────────────────┤
│  MODEL SUPPLY LAYER     │ **CARBON** (Decentralized, Verified,  │
│                         │  Compounding, Trustless)               │
│                         │ ◄── NO DIRECT COMPETITOR              │
├─────────────────────────┼───────────────────────────────────────┤
│  TOOLING/DEPLOYMENT     │ Ansys, Siemens, Dyad, Dassault,      │
│                         │ nTop, Rescale                          │
│                         │ Consumers of Carbon's model supply    │
├─────────────────────────┼───────────────────────────────────────┤
│  END USERS              │ Aero/Auto/Energy/Defense — Digital    │
│                         │ Twins, HIL, Design Optimization,      │
│                         │ Certification                         │
└─────────────────────────────────────────────────────────────────┘
```

*Implementation details for all components are in `IMPLEMENTATION.md`.*

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    CARBON SUBNET ARCHITECTURE                   │
├─────────────────────────────────────────────────────────────────┤
│  MINERS / AGENTS                                                │
│  ├─ MCP Layer (Model Context Protocol)                          │
│  │  ├─ Estimation Mode (near-zero cost screening)               │
│  │  ├─ Light Training Mode (reduced budget + local eval)        │
│  │  └─ Full Submission (strategy JSON → validator)              │
│  └─ Miner Toolkit (Docker + Python SDK + cost estimation)      │
├─────────────────────────────────────────────────────────────────┤
│  VALIDATORS (5+ for consensus)                                  │
│  ├─ Trustless Procedural Data Generation (seeded by block hash) │
│  ├─ Multi-Fidelity Pipeline:                                    │
│  │  ├─ Tier 1: Fast stress filter                               │
│  │  └─ Tier 2: Full hidden adversarial + physics gates         │
│  ├─ Online physics residual monitoring (adaptive loss re-weight)│
│  └─ Model Card generation (full provenance + diagnostics)      │
├─────────────────────────────────────────────────────────────────┤
│  LANDSCAPE AGENT (Compounding Intelligence)                     │
│  ├─ Symbolic extraction (PySR → ModelingToolkit.jl losses)     │
│  ├─ Causal analysis (Double ML for strategy → outcome)         │
│  ├─ Specialist Bank (distilled reusable components)            │
│  └─ Prior updates → noisy priors distributed to miners         │
├─────────────────────────────────────────────────────────────────┤
│  INCENTIVES (Yuma Consensus + ChallengeWinnerTracker)          │
│  ├─ Winner-heavy + exponential decay                           │
│  ├─ Future: Breakthrough Bounties + Decaying Top stipends      │
│  └─ Treasury for unclaimed allocations                         │
└─────────────────────────────────────────────────────────────────┘
```

*Implementation details for all components are in `IMPLEMENTATION.md`.*

---

## 3. Trustless Verification & Data Generation System

### Core Principles

- **Procedural generation at runtime**: All evaluation data (stress testing and benchmark/held-out) is generated at runtime using open-source generators.
- **Public unpredictable seeding**: Generation seeded by `hash(challenge_id + block_hash + run_nonce)` (Phase 0); moving toward commit-reveal + drand in Phase 1B+.
- **Auditable by anyone**: Generator code is open-source; anyone can reproduce evaluation data given the seed.
- **Scientific credibility**: Generator parameter ranges have documented physical justification; validated against high-fidelity reference solvers (FEniCS, OpenFOAM, SU2, DPLR, US3D).
- **No fixed reference datasets**: Primary evaluation data is procedurally generated to preserve trustlessness; fixed datasets used only for generator validation.

---

## 4. Dual-Regime Architecture (DoD/Regulated Markets)

Carbon operates a **Dual-Regime Model Supply** for defense and regulated domains:

```
┌─────────────────────────────────────────────────────────────────┐
│  PUBLIC REGIME (Carbon Subnet)                                  │
│  ├─ Discovers architectures on public/synthetic data            │
│  ├─ Adversarial verification + physics gates                    │
│  ├─ Outputs: Strategy.json + Model Card + ONNX + Evidence       │
│  └─ Zero ITAR/controlled data                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                    CROSS-DOMAIN SOLUTION / SECURE TRANSFER
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  CLASSIFIED REGIME (Prime Enclave IL5/IL6)                      │
│  ├─ Ingests architecture blueprint (strategy.json)              │
│  ├─ Fine-tunes on classified telemetry / proprietary geometry   │
│  ├─ Deploys ONNX locally (HIL, edge, air-gapped)                │
│  ├─ Runs inference LOCALLY — zero network calls                 │
│  ├─ Applies ITAR/EAR classification (Prime's ECO)               │
│  └─ Packages DI-SESS-82483 deliverable + ATO artifacts          │
└─────────────────────────────────────────────────────────────────┘
```

**Data Handling Phases**:
- **Phase 0-1B**: Public + synthetic data only. No proprietary data enters the network.
- **Phase 2A**: Customer-controlled local fine-tuning with custom datasets (Abaqus ODB). Raw data never leaves customer control.
- **Phase 2B**: Air-Gapped Miner Toolkit v1 for classified enclaves (IL5/IL6). Zero network dependencies.
- **Phase 3+**: Confidential Computing (NVIDIA H100 TEEs) on validator side for sensitive workloads.

---

## 5. Miner Compute, Local Iteration & Submission Model

### Core Philosophy

- **Validator authority**: The validator always performs full deterministic training + hidden adversarial stress evaluation identically for every submission.
- **Miner autonomy**: Miners/agents run local iterative training loops on their own hardware to improve strategies before submission.
- **Zero-friction submission**: Submission is always free; local training is an *optional enhancement*, not a requirement.
- **Moat protection**: Only noisy priors are distributed (never the clean champion). High-value Landscape knowledge (raw causal graphs, DML outputs) remains protected.

### Three-Tier Local System

| Tier | Compute Cost | Anchored To | Purpose | Required Before Submission? |
|------|-------------|-------------|---------|----------------------------|
| **Estimation Mode** | Near-zero | Noisy Prior | Rapid idea screening & filtering | No |
| **Light Training Mode** | Low | Noisy Prior | Main iterative improvement loop | No (recommended) |
| **Validator (Official)** | Network-paid | Full hidden data | Official scoring + emissions | Yes |

**Key Rule**: A miner can submit a strategy JSON at any time with zero local training.

### Data & Stress Separation (Critical Security Boundary)

| Aspect | Miner Local Loops | Validator Official Evaluation |
|--------|-------------------|-------------------------------|
| **Data** | Procedural + custom datasets; different seeds | Procedural (validator config only); hidden seeds |
| **Stress Tests** | Reduced, non-hidden variants | Full hidden stress variant set |
| **Physics Gates** | Optional, for learning signal | Mandatory; hard fail = zero score |
| **Data Visibility** | Miner controls | **Never exposed to miners** |

This separation preserves adversarial integrity: miners optimize for training distribution; validators evaluate on hidden, procedurally generated distribution with hard physics gates.

---

## 5. Three-Tier Participation (MCP Layer)

| Mode | Compute | Purpose | Leaderboard Impact |
|------|---------|---------|-------------------|
| **Estimation Mode** | Near-zero (CPU, <100ms) | Rapid screening via linear sensitivity around noisy prior | None (logged, lower Landscape weight) |
| **Light Training** | Low (1-4 GPU-hrs) | Main iteration loop; full gates on local data | None (logged, lower Landscape weight) |
| **Full Submission** | Network-paid | Official scoring + emissions | **Only path to emissions weight** |

*Implementation details in `IMPLEMENTATION.md`.*

---

## 6. Challenges by Phase (Capability-Gated)

### Phase Transition Criteria (All Gates Must Pass)

| Transition | Entry Gate (ALL Required) | Meaning |
|------------|---------------------------|---------|
| **0 → 1A** | 5 validators (99% uptime), 7 PDEs mesh-converged vs FEniCS, 3 backbones, Model Zoo v1 live, 3 pilot subscribers | Subnet operational — verification layer live |
| **1A → 1B** | 2 defense benchmarks mesh-converged + turb UQ, Factory v1 live, 1+ Tier 2 LOI, Turbulence UQ framework | Compressible flow verified — Factory revenue live |
| **1B → 2A** | 4 defense benchmarks (turb UQ + chem UQ), Factory hardened (2 T2 LOIs, 1 T3 LOI), Prime teaming, SBIR I submitted | Defense breadth + Factory hardened — SBIR pipeline live |
| **2A → 2B** | Schema v1.1 live (LoRA + Custom Data + MT Losses), Specialist Bank v1 (20+ components), DML flowing (3 effects), 10 Tier 1 subs | Customization + Intelligence live — Deep customization revenue |
| **2B → 3** | Air-Gap Toolkit v1 in 2+ enclaves, Air-Gap Validator v1 in IL5, preCICE sidecar tested on sequential FSI, Coupling convergence validated | Classified-ready + Coupling architected — Tier 4 + Coupling ready |
| **3 → 4** | 3 coupled benchmarks live (coupling gates passing), preCICE sidecar production on 5 validators, SBIR II awarded, 2+ Tier 4 composite deals | Coupled physics supply chain live — SBIR II + Tier 4 composite |
| **4 → 5** | 4 3D turbulence benchmarks live (gates passing), 3D curriculum proven (2D→3D→Turb→Coupled), 2+ Tier 4 production contracts, $10M+ ARR run rate | Production-grade 3D turbulence — Weapon-system/Digital Twin ready |

> **Phase jumps are capability-gated, not calendar-gated.** Timelines below are estimates only.

---

## 6. Phase Overview

| Phase | Name | Physics Scope | Challenge Types | Schema | Revenue |
|-------|------|---------------|-----------------|--------|---------|
| **0** | **Foundation** | 7 Academic PDEs | Base (7) | v1.0 | Model Zoo (Tier 1) |
| **1A** | **Compressible Flow** | 7 Academic + 2 Defense | Base (9) + Hosted (Factory v1) | v1.0 | Tier 1 + Tier 2 |
| **1B** | **Reacting Flow + Sequential FSI** | 9 + 4 Defense | Base (13) + Hosted (Factory v1 hardened) | v1.0 | Tier 1-3 + Tier 3 |
| **2A** | **Customization & Intelligence** | 13 + Custom | Base + Hosted + LoRA + Custom Data + MT Losses | v1.1 | Tier 1-3 + LoRA/MT |
| **2B** | **Air-Gap + Coupling Prep** | 13 + Custom | Base + Hosted + Air-Gap + preCICE Arch | v1.1 | Tier 4 Pilot |
| **3** | **Multi-Physics Coupling** | Coupled Physics | Base + Hosted + Composite (v2.0) | v2.0 | Tier 4 + SBIR II |
| **4** | **Production** | 3D + Turbulence | All + 3D/Turbulence | v2.0+ | Tier 4 + Production |

---

## 6. Challenges by Phase

### Phase 0: Foundation (7 Academic PDEs)

| ID | Problem | Dimension | Key Physics |
|----|---------|-----------|-------------|
| 1 | Poisson | 2D/3D | Elliptic, source-driven |
| 2 | Darcy | 2D/3D | Elliptic, heterogeneous media |
| 3 | Burgers | 2D | Hyperbolic, shock formation |
| 4 | Navier-Stokes (laminar) | 2D/3D | Incompressible flow, div-free |
| 5 | Heat | 2D | Parabolic, transient conduction |
| 6 | Linear Elasticity | 2D | Vector mechanics, equilibrium |
| 7 | Thermo-Elasticity | 2D | Coupled thermal-mechanical |

**Mesh/Temporal Convergence Required**: 3-level h-refinement (h, h/2, h/4), 3-level dt-refinement; L2 tolerance 1.0%/0.5%; validated vs FEniCS/OpenFOAM (20 cases, 2% tolerance).

### Phase 1A: Compressible Flow (+2 Defense Benchmarks)

| ID | Benchmark | Physics | Key Physics |
|----|-----------|---------|-------------|
| 8 | NACA 0012 Transonic Flutter | 2D/3D Compressible NS | Shock-BL interaction, flutter |
| 9 | NASA CRM Wing-Body | 3D RANS | Transonic separation, buffet |

**Turbulence UQ Required**: 3 models (SA, k-ω SST, k-ε); uncertainty budget for separation (15%), skin friction (10%), heat flux (15%); gate margins include model-form uncertainty.

### Phase 1B: Reacting Flow + Sequential FSI (+4 Defense Benchmarks)

| ID | Benchmark | Physics | Key Physics |
|----|-----------|---------|-------------|
| 10 | HIFiRE-1 Scramjet Forebody | Reacting NS (5-species) | Hypersonic BL transition, chemistry |
| 11 | Turek/Hron FSI 3D (Sequential) | NS + Elasticity (one-way) | FSI, large displacement |
| 12 | Store Separation (6-DOF) | NS + Rigid Body Dynamics | Moving boundaries, dynamic mesh |
| 13 | Turbine Blade Heat Transfer | Conjugate HT | Film cooling, CHT, rotating frame |

**Chemistry UQ Required**: 3 mechanisms (GRI-Mech 3.0, Park, Guardone); uncertainty budget for heat flux (20%), species (25%), recession (30%).

### Phase 2A: Customization & Intelligence (+ LoRA + Custom Data + MT Losses)

**Schema v1.1 Additions**:
```json
{
  "lora": {"rank": 16, "alpha": 32, "target_modules": ["spectral_conv", "projection"]},
  "custom_dataset": {"source": "abaqus_odb", "storage_uri": "s3://...", "field_mapping": {...}},
  "structured_losses": {"enabled": true, "terms_uri": "ipfs://...", "weights": {...}}
}
```

**Landscape Agent**: PySR + MT Bridge (production) + Double ML (JAX-native DML) → Causal guidance + Specialist Bank v1.

### Phase 2B: Air-Gap + Coupling Prep

| Capability | Delivery |
|------------|----------|
| Air-Gapped Miner Toolkit v1 | Full fine-tuning in IL5/IL6 enclaves (zero network deps) |
| Air-Gapped Validator v1 | Validator in IL5 enclave; pre-provisioned seeds; SLURM integration |
| preCICE Sidecar Architecture | gRPC/Unix socket sidecar; deterministic coupling (shared seed, synchronized checkpointing) |
| Sequential Multi-Physics Ladder | 6-step ladder: CHT(seq) → Thermo-Elasticity(seq) → FSI(seq) → FSI(coupled) → CHT(coupled) → Thermo-Elasticity(coupled) |

### Phase 3: Multi-Physics Coupling (Composite v2.0 Schema)

| ID | Challenge | Physics | Coupling |
|----|-----------|---------|-----------|
| 14 | FSI (Turek/Hron 3D) | NS + Nonlinear Elasticity | preCICE implicit |
| 15 | CHT (Conjugate HT) | NS + Heat (solid) | preCICE explicit |
| 17 | Thermo-Elasticity 3D | NS + Heat + Elasticity | preCICE multi-field |

**Schema v2.0 (Composite)**:
```json
{
  "schema_version": "2.0",
  "composite": true,
  "sub_strategies": {"fluid": {...}, "solid": {...}},
  "coupling": {"method": "preCICE_implicit", "max_iterations": 20, "convergence_tolerance": 1e-6},
  "coupling_gates": [{"gate": "interface_continuity", "threshold": 1e-4}, ...]
}
```

### Phase 4: Production (3D + Turbulence)

| ID | Challenge | Physics |
|----|-----------|---------|
| 17 | 3D FSI + Turbulence | NS (k-ω SST/LES) + Elasticity |
| 18 | 3D CHT + Turbulence | NS (k-ω SST/LES) + Heat |
| 19 | 3D Thermo-Elasticity + Turb | NS (k-ω SST/LES) + Heat + Elasticity |
| 20 | Hypersonic 6-DOF | Reacting NS + 6-DOF + Ablation |

**3D Curriculum**: 2D specialists → 3D initialization → Turbulence (RANS → LES) → Coupled.

---

## 7. Miner Controls (Strategy Schema Evolution)

| Schema | Phase | Key Fields | Backward Compatible |
|--------|-------|------------|---------------------|
| **v1.0** | Phase 0-1B | `backbone`, `training`, `loss` (with `enabled` booleans), `curriculum`, `data` (split, augmentation) | Base |
| **v1.1** | Phase 2A-2B | + `lora`, `custom_dataset` (Abaqus ODB), `structured_losses`, `data_generation` (generator_params, augmentation_policy, curriculum, synthetic_ratio) | ✅ (optional) |
| **v2.0** | Phase 3 | `composite=true`, `sub_strategies` (fluid/solid), `coupling`, `coupling_gates` | ❌ (new schema) |
| **v2.0+** | Phase 4 | + `curriculum.phases`, `turbulence_gates`, `3d_gates` | ✅ |

### Miner Data Generation Controls (v1.1+)

| Control | Scope | Validation |
|---------|-------|------------|
| `generator_params` | Distribution params within challenge envelope | **Entropy floor enforced** (min entropy per param) |
| `augmentation_policy` | Types & magnitudes | Max magnitude capped |
| `curriculum` | Phase definitions, epoch allocation | Total epochs ≤ max_epochs |
| `synthetic_ratio` | [0.0, 1.0] | Clamped to [0, 1] |
| `custom_dataset` | Abaqus ODB / external data | **Full validation** (ref solver + physics check) |

**Entropy Floor** (anti-gaming): Minimum entropy per parameter enforced at submission (e.g., `mach_distribution` ≥ 0.5 bits).

---

## 8. Validation Strategy

### Scoring Weights (45/30/25)

| Component | Weight | Composition |
|-----------|--------|-------------|
| **Physics Fidelity** | 45% | Gate margins (1 - result/threshold) averaged |
| **Robustness** | 30% | Mean performance across stress variants (1/(1+MSE)) |
| **Accuracy** | 25% | Normalized benchmark MSE (1/(1+MSE/var)) |

### Physics Gates (Hard — Zero Score on Failure)

| Gate | Phase | Threshold | Notes |
|------|-------|-----------|-------|
| Mass Conservation | 0+ | 1e-6 L2 + 1e-4 Linf | + shock-local Linf (1A+) |
| Energy Stability | 0+ | 1e-6 | Split: numerical vs physical dissipation |
| Boundary Satisfaction | 0+ | 1e-4 Linf | + catalytic wall (1B+) |
| Rollout Stability | 0+ | 10k steps, 1% perturb | Lyapunov < 0 |
| UQ Calibration | 0+ | Conformal 95% | Split conformal, 500 pts |
| Adjoint Consistency | 1A+ | 1e-4 rel error | Optimization-grade surrogates |
| Shock Capture | 1A+ | Δx/shock_thickness < 0.1 | Resolution-based |
| Turbulence UQ | 1A+ | Model-form budget | Gate margin = num_err + turb_UQ + 3σ |
| Species Conservation | 1B+ | 1e-4 per-species | Per-species + total mass |
| Chemistry UQ | 1B+ | Model-form budget | Gate margin = num_err + chem_UQ + 3σ |
| Sequential FSI Interface | 1B+ | 1e-4 velocity jump | One-way traction continuity |
| Trajectory Accuracy | 1B+ | 0.05m RMS (6-DOF) | — |
| Ablation/Recession | 1B+ | 10% error | HIFiRE |
| Interface Continuity | 3+ | 1e-4 velocity jump | preCICE residual |
| Momentum Conservation | 3+ | 1e-5 | Finite volume balance |
| Energy Conservation | 3+ | 1e-5 | Finite volume balance |
| Coupling Convergence | 3+ | <1e-6 in 20 iters | preCICE residual |
| Vorticity Preservation | 4 | ‖ω_p - ω_r‖/‖ω_r‖ < 0.1 | DNS/LES reference |
| Boundary Layer Resolution | 4 | y+ error < 5% | Wall-resolved LES |
| Turbulence Spectra Match | 4 | E(k) slope = -5/3 ± 0.1 | Experimental/LES |
| Separation Prediction | 4 | Separation point error < 0.05c | Experimental/LES |
| Ablation Recession Rate | 4 | < 10% error | Arc-jet data |

**Hard Gate Rule**: Any FAIL → total score = 0. No partial credit.

### ChallengeWinnerTracker (Emission Weighting)

```python
weight = score * exp(-blocks_since_win / half_life)  # half_life = 30 days (tunable)
```
- Winner-heavy weighting + participation dust for recent contributors
- Only genuine new best combined scores drive strong rewards
- Future: Breakthrough Bounties + Decaying Top stipends; unclaimed → treasury

---

## 9. Data Generation Architecture

### Generator Types

| Type | Use Case | Phases |
|------|----------|--------|
| **ONLINE_JAX** | JAX-FEM / JAX-RANS (differentiable) | Phase 0 |
| **ONLINE_HYBRID** | Mesh online, solutions cached | Phase 1A |
| **PRECOMPUTED** | Fully cached (S3/GCS) | Phase 1B+ |
| **SEQUENTIAL_FSI** | One-way coupling | Phase 1B+ |

### Seed Derivation (Deterministic)

```
master_seed = hash(challenge_id + block_hash + run_nonce)
  ├── data_seed     = splitmix64(master_seed, 0)  → training data
  ├── stress_seed   = splitmix64(master_seed, 1)  → hidden stress variants
  ├── init_seed     = splitmix64(master_seed, 2)  → weight initialization
  ├── dropout_seed  = splitmix64(master_seed, 3)  → dropout RNG
  └── shuffle_seed  = splitmix64(master_seed, 4)  → train/val split
```

**Critical**: `stress_seed` derived from block hash → unknown to miners until evaluation.

### Training vs Evaluation Distribution Separation

| Aspect | Training | Evaluation (Hidden) |
|--------|----------|---------------------|
| Seed | `data_seed` (derived from master) | `stress_seed` (different derivation path) |
| Generator | Miner params + challenge config | **Validator config ONLY** (miner params ignored) |
| Envelope | Training envelope (challenge.config) | **Extended envelope** (wider ranges) |
| Perturbations | Miner-defined augmentation | Physics-gated perturbations (shock, separation, BL trip) |
| Seed Visibility | Known to miner (in strategy) | **Never exposed to miner** |

### Stress Variant Categories

| Category | Weight | Physics Gates Tested |
|----------|--------|---------------------|
| Extended Envelope | 0.30 | Mass, Energy, Shock Capture |
| Shock Perturbation | 0.20 | Shock Capture, Mass |
| Boundary Layer Trip | 0.15 | Boundary, Separation |
| Separation Trigger | 0.15 | Separation, Rollout |
| Mesh Perturbation | 0.10 | Boundary, Mass |
| Boundary Condition | 0.10 | Thermal, Energy |

**Coverage Requirement**: ≥95% categories present per evaluation.

### Custom Dataset Validation (Miner-Provided)

1. **Mesh quality** + field consistency checks
2. **Reference solver validation** (sample vs FEniCS/OpenFOAM)
3. **Physics consistency** (conservation laws, BCs)
3. **Only validated datasets enter training pipeline**

---

## 8. Landscape Agent (Compounding Intelligence)

### Phasing

| Phase | Components | Output |
|-------|------------|--------|
| **0** | PySR symbolic regression | Conservation laws, symmetries → JSON → ModelingToolkit.jl → structured loss terms |
| **1A** | PySR + ModelingToolkit.jl bridge (JSON) | Structured loss terms (production) |
| **2A** | PySR + MT + Double ML (JAX-native) | Causal effects (strategy → robustness) + Specialist Bank v1 |
| **2B** | Cross-domain causal mapping | Cross-physics transferable insights |

### PySR Configuration (Phase 0)

```python
PYSR_CONFIG = {
    "populations": 50, "population_size": 100, "ncycles_per_iteration": 500,
    "maxsize": 40, "maxdepth": 8,
    "binary_operators": ["+", "-", "*", "/", "^"],
    "unary_operators": ["sin", "cos", "exp", "log", "sqrt", "abs"],
    "constraints": {"pow": (-1, 1)},
    "feature_names": ["loss_data_weight", "loss_physics_weight", "loss_boundary_weight",
                       "lr_initial", "lr_decay_rate", "curriculum_phase", "backbone_depth",
                       "backbone_width", "activation_type", "normalization_type",
                       "physics_gate_margin", "residual_l2", "conservation_l2", "boundary_l2"],
    "target_name": "robustness_score", "verbosity": 1,
}
```

### ModelingToolkit.jl Bridge (JSON → JAX Loss Terms)

```julia
function json_to_loss_term(json_expr::Dict)
    @variables x[1:n]  # strategy params
    @parameters p[1:m]  # physics state
    expr = parse_pysr_json(json_expr)
    return eval(build_function(expr, [p...], [t, x, y, z]))
end
```

### Double ML (Phase 2A+, JAX-Native)

```python
DML_CONFIG = {
    "n_folds": 5, "n_repeats": 3, "ml_model": "jax_boosting",
    "treatment_types": {"loss_weights": "continuous_multivariate", "curriculum": "categorical"},
    "confounders": ["physics_class", "data_seed", "backbone", "epochs"],
    "target": "robustness_score", "confidence_level": 0.95,
}
```

---

## 9. Incentives & Tokenomics

### ChallengeWinnerTracker (Emission Weighting)

```python
weight = score * exp(-blocks_since_win / half_life)  # half_life = 30 days (tunable)
```
- Winner-heavy weighting + participation dust for recent contributors
- Only genuine new best combined scores drive strong rewards
- Future: Breakthrough Bounties + Decaying Top stipends; unclaimed → treasury

---

## 9. Miner Toolkit & Submission Interface

### Miner Toolkit (Docker Image)

```dockerfile
FROM nvidia/cuda:12.4-devel-ubuntu22.04
RUN apt-get update && apt-get install -y python3.11 python3.11-venv git curl wget build-essential cmake libopenmpi-dev
RUN python3.11 -m venv /opt/carbon
COPY carbon/miner /opt/carbon/miner
COPY requirements-miner.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-miner.txt
ENTRYPOINT ["/entrypoint.sh"]
```

### CLI / SDK

```bash
carbon-miner run --challenge naca0012_transonic --backbone gino --strategy strategy.json --mode light_training
carbon-miner submit --challenge naca0012_transonic --strategy strategy.json
```

```python
class AsyncCarbonMiner:
    async def propose_train_evaluate_submit(self, challenge, backbone, n_iterations=100):
        prior = await self.get_noisy_prior_async(challenge, backbone)
        for _ in range(n_iterations):
            candidate = self.propose_candidate(prior)
            est = await self.estimate_async(candidate, prior)
            if est.confidence > 0.7 and est.estimated_score_delta > 0:
                result = await self.train_local_async(candidate, mode="light")
                if result.local_score > prior.score * 1.05:
                    receipt = await self.submit_async(candidate)
                    if receipt.accepted: prior = candidate
        return prior
```
---

## 10. Commercial GTM — Three Revenue Engines

| Engine | Product | Price | Buyer | Timeline |
|--------|---------|-------|-------|----------|
| **Tier 1: Specialist Bank** | 7→50 certified specialists (ONNX + Model Card + Gate Certs) | $20-50k/yr per model / $100-200k/yr bundle | Sim teams (Aero/Auto/Energy) | Month 3 |
| **Tier 2-4: Sponsored Challenges** | Custom PDE/geometry challenges | T2: $150-300k (open)<br>T3: $400-800k (IP-licensed)<br>T4: $800k-2M+ (private/on-prem) | Primes, OEMs, Labs | Month 6-12 |
| **DoD Subcontract (SBIR/BAA)** | Evidence Package for IV&V/ATO | Phase I: $250k (6mo)<br>Phase II: $1.5-2M (24mo) | Primes (Shield AI, Anduril, etc.) | Month 12-18 |
| **Verification Gas/Registry** | Programmatic badge resolution, model card API | $0.001-0.01/query (USD-denominated, α-settled) | Tooling platforms (Dyad, Ansys, nTop, Rescale) | Month 12+ |

**Revenue Projections (Conservative)**: Y1: $1.8M | Y2: $5.4M | Y3: $16.6M | Y4: $35.8M

---

## 10. Validator Operations & Economics

### Hardware Requirements by Phase

| Phase | Validators | GPU Config | Monthly Cost (Reserved) |
|-------|------------|------------|-------------------------|
| Phase 0 | 5 | 1× A100 80GB | ~$22k |
| Phase 1A | 5 | 1× H100 | ~$25k |
| Phase 1B | 5 | 1× H100 | ~$40k |
| Phase 2A | 5 | 1× H100 | ~$40k |
| Phase 2B | 6 (5 + 1 air-gap) | 6× H100 | ~$55k |
| Phase 3 | 10 (5 pairs) | 20× H100 (5 pairs) | ~$165k |
| Phase 4 | 20+ | 20× H200 | ~$975k |

### Validator Health Gates (Every Phase Transition)

| Metric | Threshold |
|--------|-----------|
| Uptime | > 99% |
| Miner participation | > 20 active |
| Challenge throughput | > 50/week |
| Multi-GPU stability (Ph 2B+) | Verified |
| α/TAO floor | Validated |
| Validator ROI at floor | > 0 |
| Emission efficiency | > 80% |


## 11. Security & Correctness Guarantees

### Critical Correctness Invariants

| Invariant | Enforcement |
|-----------|-------------|
| **Physics gates run in fp32** | Context manager forces `jax_default_matmul_precision="float32"` |
| **Loss masking uses booleans** | Explicit `enabled: bool` in schema (no fp thresholds) |
| **Gradient clipping inside JIT** | `optax.clip_by_global_norm` inside `@jax.jit` |
| **Determinism** | Pinned JAX/jaxlib/cudnn; `threefry` PRNG; `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| **Compilation cache** | Persistent `/persistent/compile_cache` volume |
| **Validator queue** | Priority queue with timeouts; max depth 100 |

### Security Boundaries

| Boundary | Enforcement |
|-----------|-------------|
| **Eval seed unknown to miners** | `seed = hash(challenge_id + block_hash + run_nonce)` |
| **Eval generator immutable** | Generator config frozen in Challenge Spec (on-chain) |
| **Physics gates hard** | Binary PASS/FAIL; zero score on failure |
| **Eval data never exposed** | Stress variants generated in-validator-memory only |
| **Training ≠ Eval distribution** | Extended envelopes for stress variants |

---

## 11. Operational Infrastructure

### Validator Queue Management

```python
class ValidatorQueue:
    max_concurrent = 3
    max_queue_depth = 100
    submission_timeout = 7200  # 2 hours
    
    # Priority: T4 Sponsored > T3 Sponsored > T2 Sponsored > High Rep > Standard > Estimation
```

### Determinism Lockfile (Pinned)

```
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
ENV JAX_COMPILATION_CACHE_DIR=/persistent/compile_cache
ENV JAX_CACHE_DIR=/persistent/jax_cache
ENV XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda --xla_gpu_per_thread_default_stream=true"
```

### Determinism Lockfile (Pinned)

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

## 12. Phase Timeline Summary (Capability-Gated)

| Phase | Est. Months | Key Deliverable | Revenue |
|-------|-------------|-----------------|---------|
| **0** | 0-4 | Subnet live, 7 PDEs mesh-converged, Model Zoo v1 | Tier 1 ($30k) |
| **1A** | 4-8 | 2 defense benchmarks, Factory v1, Tier 2 LOI | Tier 1-2 ($150k) |
| **1B** | 8-14 | 4 defense benchmarks, Factory hardened, Prime + SBIR I | Tier 1-3 ($500k) |
| **2A** | 14-22 | v1.1, LoRA, Custom Data, MT, DML, Commercial traction | Tier 1-3 + LoRA ($1.5M) |
| **2B** | 22-28 | Air-Gap live, preCICE architected, Sequential ladder | Tier 4 Pilot ($500k) |
| **3** | 28-40 | v2.0 Composite, preCICE verified, SBIR II | Tier 4 + SBIR II ($5M) |
| **4** | 40-52 | 3D Turbulence, Curriculum, Production contracts | $10M+ ARR |

---

## 12. Appendix: Key Schemas

### Strategy Schema v1.0 (Phase 0-1B)

```json
{
  "$schema": "https://carbonsubnet.org/schemas/strategy/v1.0.json",
  "challenge_id": "navier-stokes-laminar-2d-v1",
  "backbone": "fno",
  "backbone_config": {"modes": 32, "width": 64, "depth": 4, "activation": "gelu", "normalization": "instance_norm"},
  "training": {"epochs": 500, "batch_size": 32, "optimizer": "adamw", "learning_rate": 1e-3, "lr_schedule": "cosine_warm_restarts", "lr_warmup_epochs": 10, "weight_decay": 1e-4, "gradient_clip": 1.0, "mixed_precision": true},
  "loss": {
    "data_mse": {"enabled": true, "weight": 1.0},
    "physics_residual": {"enabled": true, "weight": 0.5},
    "boundary_mse": {"enabled": true, "weight": 0.3},
    "conservation_penalty": {"enabled": false, "weight": 0.2}
  },
  "curriculum": [{"phase": 1, "reynolds": [100, 500], "epochs": 100}, ...],
  "data": {"train_split": 0.8, "augmentation": ["rotation", "scaling"], "noise_level": 0.01},
  "checkpointing": {"save_every_n_epochs": 25, "keep_best_n": 5, "metric": "combined_score"}
}
```

### Strategy Schema v1.1 (Phase 2A-2B)

```json
{
  ...v1.0,
  "schema_version": "1.1",
  "lora": {"rank": 16, "alpha": 32, "target_modules": ["spectral_conv", "projection"]},
  "custom_dataset": {"source": "abaqus_odb", "storage_uri": "s3://...", "field_mapping": {...}},
  "structured_losses": {"enabled": true, "terms_uri": "ipfs://...", "weights": {...}},
  "data_generation": {
    "generator_params": {"mach_distribution": "beta(2,2)", ...},
    "augmentation_policy": {"physics_informed_noise": true, ...},
    "curriculum": {"phase_1": {"mach": [0.7, 0.9], "epochs": 100}, ...},
    "synthetic_ratio": 0.8
  }
}
```

### Strategy Schema v2.0 (Phase 3+)

```json
{
  "schema_version": "2.0",
  "composite": true,
  "sub_strategies": {
    "fluid": {"backbone": "fno", "backbone_config": {...}, "loss": {...}},
    "solid": {"backbone": "gino", "backbone_config": {...}, "loss": {...}}
  },
  "coupling": {"method": "preCICE_implicit", "max_iterations": 20, "convergence_tolerance": 1e-6},
  "coupling_gates": [{"gate": "interface_continuity", "threshold": 1e-4}, ...]
}
```

---

## 13. Security Checklist (Launch Requirements)

| Requirement | Status |
|-------------|--------|
| Physics gates run in fp32 (context manager enforced) | ✅ Required |
| Loss masking uses explicit booleans (no fp thresholds) | ✅ Required |
| Gradient clipping inside JIT | ✅ Required |
| Determinism: pinned JAX/jaxlib/cudnn, threefry PRNG, CUBLAS_WORKSPACE_CONFIG | ✅ Required |
| Compilation cache persistence (persistent volume) | ✅ Required |
| Validator queue with priority + timeouts | ✅ Required |
| Determinism lockfile (pinned JAX/jaxlib/cudnn) | ✅ Required |
| XLA compilation cache persistence | ✅ Required |

---

## 13. Phase Gate Template (Every Transition)

| Gate Category | Required Metrics |
|---------------|------------------|
| **Physics Rigor** | Mesh convergence ✓, Turbulence/Chemistry UQ ✓, Gate calibration ✓, Coupling convergence (Ph 3+) ✓ |
| **Validator Health** | Uptime > 99%, Miner participation > 20, Challenge throughput > 50/week, Multi-GPU stability (Ph 2B+) |
| **Economics** | α/TAO floor validated, Validator ROI > 0 at floor, Emission efficiency > 80% |
| **Revenue** | Tier 1 ARR target, Tier 2/3 LOIs, Tier 4 pilots, SBIR milestones |
| **Network** | Miner count > 20 (Ph 1+), Strategy throughput > 100/week, Gate pass rate > 15% |

---

*This specification is written to be scientifically rigorous and buildable. All implementation details (JAX patterns, generator code, gate implementations, training loops, precision policies, contracts, MCP protocol, Landscape Agent pipeline, Miner Toolkit CLI/SDK, Reproducibility harness, Genesis Contracts, MCP Protocol, Reproducibility harness, Genesis Contracts) are in `IMPLEMENTATION.md`.*

---

*This specification is written to be scientifically rigorous and buildable. All implementation details are in `IMPLEMENTATION.md`.*
