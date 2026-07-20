# SPEC.md — Carbon PDE Subnet Technical Specification (Buildable Level with Strategic Emphasis)

**Version:** 4.7 (Updated July 2026) — **Full Strategy JSON Schema**
**Audience**: Researchers and engineers with PhD-level background in Physics, Computational Mechanics, or Scientific Computing.

This specification provides sufficient detail for a domain expert to understand the scientific rationale, implementation logic, and expected behavior of every major component. It is intended to be buildable and scientifically defensible.

---

## 1. Scientific Motivation & Strategic Positioning

High-fidelity simulation remains the bottleneck in engineering design, optimization, digital twins, and real-time control. Traditional solvers scale poorly with design space size or real-time requirements. Pure data-driven ML surrogates are fast but frequently violate conservation laws, stability conditions, or boundary physics, rendering them unreliable for downstream engineering use.

The space for AI-powered physics simulation and Neural Operators is still **nascent**. There is a tremendous amount left to discover in *how* to best build, train, and use these models for real engineering problems.

Centralized teams explore this space linearly. Carbon is designed to explore it in parallel across thousands of strategies with strong selection pressure from hidden adversarial validation and compounding knowledge via the Landscape Agent.

**Core Thesis**: A properly aligned decentralized subnet can discover superior Neural Operator training methodologies faster and cheaper than centralized players, while providing trustless, verifiable robustness.

---

## 2. Validator Docker Image, Backbone Selection, Training, Evaluation & Advanced Features

### 2.1 Purpose and Design Goals
The validator runs inside a hardened Docker container that provides a fully reproducible environment for accepting strategy configurations as JSON, dynamically selecting and instantiating the correct neural operator backbone, executing deterministic training on the appropriate data mixture, running hidden stress tests and physics gates, evaluating on held-out benchmark data, and producing auditable results and artifacts (including rich Model Cards).

### 2.2 Full Strategy JSON Schema (Detailed)

Miners and agents submit strategies as structured JSON. Below is the current detailed schema for Phase 0/1 (extensible in later phases). The validator validates the JSON against this schema.

```json
{
  "schema_version": "1.0",
  "strategy_id": "unique-uuid-or-hash",

  "backbone": {
    "type": "FNO" | "DeepONet" | "U-Net" | "GraphNO" | "PINO" | "Custom",
    "config": {
      // FNO example
      "modes": 12,
      "width": 64,
      "n_layers": 4,
      "padding": 9,
      "fourier_modes": [12, 12, 12],
      "activation": "gelu" | "relu" | "silu",

      // DeepONet example
      "branch_net": { "layers": [128, 128, 64] },
      "trunk_net": { "layers": [128, 128, 64] },
      "output_dim": 1,

      // U-Net / ResNet example
      "channels": [64, 128, 256, 512],
      "kernel_size": 3,
      "dropout": 0.1,

      // PINO-specific
      "physics_loss_weight": 0.5,
      "boundary_loss_weight": 0.3,

      // Common
      "normalization": "instance" | "batch" | "layer" | "none",
      "skip_connections": true
    }
  },

  "training": {
    "optimizer": "AdamW" | "Adam" | "LBFGS" | "SGD",
    "lr": 0.001,
    "weight_decay": 1e-4,
    "lr_schedule": {
      "type": "cosine" | "step" | "exponential" | "reduce_on_plateau" | "constant",
      "params": { "T_max": 100, "eta_min": 1e-6 }
    },
    "epochs": 200,
    "batch_size": 32,
    "gradient_clip_val": 1.0,
    "accumulate_grad_batches": 1,
    "precision": "32" | "16-mixed",

    "loss": {
      "pde_residual": 1.0,
      "boundary": 0.5,
      "initial_condition": 0.3,
      "conservation": 0.2,
      "data_fidelity": 0.8,
      "physics_informed": 0.6,
      "spectral": 0.1,           // spectral consistency
      "symmetry": 0.05,
      "advanced": {
        "causal_weighting": false,
        "curriculum_loss_ramping": true,
        "adaptive_reweighting": true
      }
    },

    "curriculum": {
      "type": "progressive" | "self_paced" | "difficulty_based" | "fixed",
      "params": {
        "start_difficulty": 0.3,
        "end_difficulty": 1.0,
        "ramp_epochs": 80,
        "difficulty_metric": "shock_strength" | "viscosity" | "coupling_strength"
      }
    },

    "data_mixture": {
      "procedural": {
        "weight": 0.65,
        "sampling": "uniform" | "difficulty_based" | "uncertainty_based",
        "max_variants_per_epoch": 500
      },
      "well_slices": {
        "weight": 0.25,
        "dataset_filter": "turbulence" | "viscoelastic" | "acoustic" | "all_relevant",
        "augmentation": "physics_preserving" | "standard"
      },
      "custom_dataset": {
        "weight": 0.10,
        "source": "abaqus" | "user_upload" | "none",
        "validation_required": true
      }
    },

    "early_stopping": {
      "patience": 20,
      "monitor": "val_loss" | "physics_residual" | "combined_score"
    }
  },

  "conditioning": {
    "type": "none" | "FiLM" | "hypernetwork" | "parameter_embedding",
    "config": {
      "embedding_dim": 64,
      "num_params": 5
    }
  },

  "uncertainty": {
    "type": "none" | "evidential" | "ensemble" | "dropout_mc",
    "config": {
      "num_ensembles": 5,
      "dropout_rate": 0.1
    }
  },

  "evaluation_preferences": {
    "multi_fidelity_tier": "auto" | "tier1_only" | "tier2_full",
    "diagnostics_level": "basic" | "detailed" | "full_explainable",
    "return_pareto": false
  },

  "metadata": {
    "description": "Short human-readable description of the strategy",
    "tags": ["fno", "burgers", "shock-capturing"],
    "author": "miner_hotkey_or_agent_id",
    "version": "1.2"
  }
}
```

### 2.3 Backbone Registry & Dynamic Instantiation
The Docker image contains a Backbone Registry. Upon receiving the JSON, the validator looks up the `backbone.type`, instantiates the model with the provided `config`, applies conditioning and uncertainty modules if specified, and seeds all weights deterministically using the hierarchical seeding system.

### 2.4 Data Pipeline & Deterministic Training
The validator builds a fully seeded data pipeline according to `data_mixture`. Procedural data is generated on-the-fly, The Well slices are sampled deterministically, and custom datasets are loaded with seeded data loaders. Training follows the exact optimizer, learning rate schedule, loss weights, curriculum, and early stopping rules defined in the JSON. All random operations are controlled for full reproducibility.

### 2.5 Multi-Fidelity Evaluation + Uncertainty-Aware Stress
The validator supports multi-fidelity evaluation (Tier 1 cheap filter → Tier 2 full stress) and can prioritize stress variants based on model uncertainty when the backbone supports it.

### 2.6 Online Physics Residual Monitoring + Adaptive Behavior
During training the validator monitors PDE residuals and conservation metrics in real time and can apply dynamic loss re-weighting (within JSON-defined bounds) or early stopping on persistent physics violations.

### 2.7 Automated Model Card Generation
Every submission automatically generates a rich Model Card containing the full strategy JSON, training curves (hashed), held-out metrics, stress results, gate violations with physics explanations, symbolic features, and uncertainty statistics. These cards are logged and fed to the Landscape Agent.

### 2.8 Docker Implementation
The container is self-contained with the backbone registry, data generators, stress system, scorer, reproducibility harness, and model card generator. Strategy JSON can be submitted via MCP or mounted volume.

---

## 3. Agent-Friendly MCP Mining Loop, Internal Testing & Advanced Features

### 3.1 Testing Philosophy
Even "test" runs provide real signal. Test modes involve actual (light) training followed by gated evaluation under stress.

### 3.2 Testing Modes
- **Light Training + Gated Evaluation** (default recommended): Reduced training budget + held-out evaluation + hidden stress + full physics gates. Produces a real test score.
- **Simulated / Cached Approximation**: Early prototyping only.
- **Full Production**: Complete training + full adversarial stress.

### 3.3 Prior-Informed Warm Start
Agents can request initialization from current best priors or distilled specialists via the Landscape Agent.

### 3.4 Explainable Failure Diagnostics
Test and production runs return rich diagnostics (residual locations, gate violation types, spectral issues, uncertainty hotspots, comparisons to successful strategies).

### 3.5 Pareto / Multi-Objective Reporting
Light test mode can return the Pareto front across physics fidelity, robustness, and accuracy.

### 3.6 Defensibility
Clear separation of test vs production paths, rate limiting on test modes, full determinism, provenance via Model Cards, and lower weighting for test runs in the Landscape Agent.

---

## 4. Challenges by Phase (Specific Problem Definitions)

### 4.1 Phase 0: Foundational Single-Physics Challenges

| ID | Problem | Dimension | Key Physics | Reference / Notes |
|----|---------|-----------|-------------|-------------------|
| 1 | Poisson | 2D/3D | Elliptic, source-driven | Standard benchmark; variable source amplitude/location, coefficient fields |
| 2 | Darcy | 2D/3D | Elliptic, heterogeneous media | Variable permeability fields (smooth + discontinuous); tests conservation & maximum principle |
| 3 | Burgers | 2D | Hyperbolic, nonlinear advection + viscosity | Shock formation/capturing, conservation, long-time stability |
| 4 | Navier-Stokes (laminar) | 2D/3D | Incompressible flow | Reynolds-number variation in laminar regime; divergence-free constraint, energy stability |
| 5 | Heat | 2D | Parabolic, transient conduction | Time-dependent forcing, variable conductivity, long-term decay |
| 6 | Linear Elasticity | 2D | Vector mechanics, equilibrium | Material property variation (Young's modulus, Poisson ratio), boundary displacement |
| 7 | Thermo-Elasticity | 2D | Coupled thermal-mechanical | Thermal expansion, coupling strength, temperature loading; tests coupled conservation |

Each challenge includes public training/holdout splits and hidden stress configurations. Symbolic metadata is attached.

### 4.2 Phase 1: Customization Layer
Same 7 challenges + custom datasets and LoRA/custom strategy support.

### 4.3 Phase 2: Verified Multi-Physics Composition
FSI (Turek/Hron), CHT, and expanded Thermo-Elasticity with preCICE and reference solutions.

### 4.4 Phase 3: 3D Multi-Physics & Advanced Composition
3D FSI/CHT/Thermo-elasticity with turbulence, 3D-specific gates, and curriculum from 2D specialists.

---

## 5. Validation Strategy — Scientific Rigor & Competitive Edge

Multi-objective scoring (45/30/25), hard/soft physics gates, and hidden stress testing. Multi-fidelity and uncertainty-aware extensions enhance throughput and sophistication.

---

## 6. Determinism & Reproducibility
Hierarchical seeding and Docker-based reproducibility harness ensure all training, stress testing, and scoring are reproducible and auditable.

---

## 7. Landscape Agent — Symbolic & Causal Compounding
Ingests results and Model Cards from production and high-quality test runs. Extracts symbolic features and applies causal analysis to discover effective training methodologies.

---

## 8. Detailed Implementation Components
- Stress Generators & StressEvaluator (multi-fidelity support)
- HydrogenScorer
- Backbone Registry (dynamic instantiation from JSON)
- Validator Docker image (model card generator, residual monitoring)
- MCP layer (multi-mode testing, explainable diagnostics, warm-start support)
- generate_challenge()
- Reproducibility Harness

---

## 9. Phased Roadmap (Build-Level)

**Phase 0**:
- Core pipeline + full JSON schema support
- Multi-fidelity evaluation
- Automated Model Cards
- Explainable diagnostics
- Light training + gated evaluation

**Phase 1**:
- Prior-informed warm starts
- Uncertainty-aware stress
- Pareto reporting
- Enhanced curriculum and adaptive loss features

**Phase 2+**: Multi-physics, 3D, advanced agent-proposed stress variants.

---

## 10. Scientific Defensibility & Competitive Differentiation
All extensions are designed to be scientifically grounded, reproducible, and auditable while enabling faster discovery of superior Neural Operator training methodologies.

---

*This specification is written to be scientifically rigorous and buildable. Reference the implementation in `neurons/` and supporting design documents for concrete code.*
