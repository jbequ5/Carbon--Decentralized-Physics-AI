# SPEC.md — Carbon PDE Subnet Technical Specification (Buildable Level with Strategic Emphasis)

**Version:** 4.8 (Updated July 2026) — **Full Detailed Workflows + Complete JSON Schema**
**Audience**: Researchers and engineers with PhD-level background in Physics, Computational Mechanics, or Scientific Computing.

This specification provides sufficient detail for a domain expert to understand the scientific rationale, implementation logic, and expected behavior of every major component. It is intended to be buildable and scientifically defensible.

---

## 1. Scientific Motivation & Strategic Positioning

High-fidelity simulation remains the bottleneck in engineering design, optimization, digital twins, and real-time control. Traditional solvers scale poorly with design space size or real-time requirements. Pure data-driven ML surrogates are fast but frequently violate conservation laws, stability conditions, or boundary physics, rendering them unreliable for downstream engineering use.

The space for AI-powered physics simulation and Neural Operators is still **nascent**. There is a tremendous amount left to discover in *how* to best build, train, and use these models for real engineering problems (loss formulations, curricula, conditioning, robustness under distribution shift, multi-physics coupling, long-horizon stability, etc.).

Centralized teams explore this space linearly and with limited internal testing. Carbon is designed to explore it in parallel across thousands of strategies with strong selection pressure from hidden adversarial validation.

**Core Thesis**: A properly aligned decentralized subnet can discover superior Neural Operator training methodologies faster and cheaper than centralized players, while gaining significant credibility through trustless, verifiable stress testing.

Carbon addresses this by creating a **decentralized, adversarially validated, self-improving engine** for physics-informed neural operator strategies. Key scientific and strategic advantages:

- Adversarial hidden stress testing forces robustness beyond public benchmarks.
- Hard physics gates enforce non-negotiable physical constraints.
- The Landscape Agent extracts symbolic structure and causal relationships, enabling compounding knowledge.
- Determinism and auditability ensure scientific reproducibility and trust.
- Decentralized parallel exploration replaces centralized linear search.

The system is designed so that a domain expert can verify that submitted strategies are not merely fitting data but are learning physically meaningful operators.

---

## 2. Validator Docker Image, Backbone Selection, Training, Evaluation & Advanced Features

### 2.1 Purpose and Design Goals
The validator runs inside a hardened Docker container that provides a fully reproducible environment for:
- Accepting strategy configurations as JSON.
- Dynamically selecting and instantiating the correct neural operator backbone.
- Executing deterministic training on the appropriate data mixture.
- Running hidden stress tests and physics gates.
- Evaluating on held-out benchmark data.
- Producing auditable, reproducible results and rich artifacts (including Model Cards).

This design ensures that any validator can exactly reproduce another validator’s training, stress testing, and scoring outcomes (within numerical tolerance), which is essential for trust, auditability, and defensibility of the subnet.

### 2.2 Full Strategy JSON Schema (Detailed & Comprehensive)

Miners and agents submit strategies as structured JSON. Below is the detailed schema for Phase 0/1 (extensible in later phases). The validator validates submissions against this schema.

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
      "spectral": 0.1,
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
The Docker image contains a Backbone Registry (Python module with registration decorators). Upon receiving the JSON, the validator:
1. Looks up the backbone type in the registry.
2. Instantiates the model with the provided `config`.
3. Applies any conditioning or uncertainty modules specified.
4. Moves the model to the correct device with deterministic seeding.

This registry pattern allows easy extension of new backbones without changing core validator logic.

### 2.4 Data Pipeline & Deterministic Training
The validator constructs a deterministic data pipeline based on the `data_mixture` in the JSON:

- **Procedural Data**: Generated on-the-fly using the seeded `ProceduralStressGenerator` (or challenge-specific data generator) for the current challenge. Parameters are derived from the master seed + strategy_id.
- **The Well Slices**: Deterministically sampled relevant dataset slices (mapped to physics class) with physics-preserving augmentations.
- **Custom Datasets** (Phase 1+): Loaded from approved sources (Abaqus ODB/.fil, user-uploaded with validation) using seeded data loaders.

Training loop:
- Fully deterministic (hierarchical seeding for weight init, data order, dropout, augmentation).
- Uses the loss formulation and weights from the JSON.
- Curriculum strategy executed as specified.
- Optimizer and learning rate schedule applied exactly as configured.
- Intermediate checkpoints and metrics logged with hashes for auditability.
- Early stopping or convergence criteria can be defined in the JSON (with safeguards).

All random operations inside training are controlled so that two validators running the identical JSON on the same challenge produce bitwise-identical model weights (within floating-point tolerance) and identical training curves.

### 2.5 Multi-Fidelity Evaluation Pipeline
To improve throughput while preserving adversarial signal:
- **Tier 1 (Cheap)**: Low-fidelity stress (fewer variants, lower resolution or reduced rollout length). Quick filter.
- **Tier 2 (Full)**: Only strategies passing Tier 1 thresholds proceed to full hidden stress testing.

Thresholds and tier definitions are challenge-specific and versioned. This enables significantly more parallel strategy exploration without weakening the core adversarial mechanism.

### 2.6 Uncertainty-Aware Stress Prioritization
When the backbone supports uncertainty (evidential outputs or ensemble mode), the validator can prioritize or re-weight stress variants where epistemic uncertainty is high. This focuses adversarial pressure on the regions where the model is least confident.

### 2.7 Online Physics Residual Monitoring + Adaptive Behavior
During training, the validator monitors PDE residuals and conservation metrics in real time. Configurable behaviors include:
- Dynamic loss weight adjustment (within JSON-defined bounds).
- Early stopping on persistent physics violations.
- Logging of residual trajectories for the Landscape Agent.

This makes training itself more physics-aware and supports discovery of robust training dynamics.

### 2.8 Automated Strategy Provenance & Model Card Generation
Every submission (test or production) automatically generates a structured **Model Card** containing:
- Exact strategy JSON and version.
- Backbone, hyperparameters, data mixture, loss weights, curriculum.
- Training curves (hashed).
- Held-out metrics.
- Stress test summary + gate violation details (with physics explanations where possible).
- Key symbolic features extracted.
- Uncertainty/robustness statistics.

These cards are logged, versioned, and fed to the Landscape Agent. They dramatically improve auditability and collective intelligence.

### 2.9 Docker Implementation Notes
- The image contains the full registry of backbones, data generators, stress generators, scorer, reproducibility harness, and model card generator.
- Strategy JSON is accepted via MCP endpoint or mounted volume.
- Training can be GPU-accelerated inside the container with proper device seeding.
- Resource limits and timeouts are enforced to prevent runaway jobs.
- Image is version-pinned; updates require governance approval and image rebuild.

This design makes the validator a self-contained, executable specification of the entire evaluation pipeline.

---

## 3. Agent-Friendly MCP Mining Loop, Internal Testing & Advanced Features

### 3.1 Overview and Innovation Goals
The MCP layer is designed to be highly agent-friendly while remaining scientifically and incentive-defensible. A key innovation is that **even "test" runs are meaningful**: they involve actual (light or simulated) training, are gated by physics constraints, evaluated under stress, and scored. This provides real signal to agents without compromising the adversarial nature of full production submissions.

### 3.2 Strategy Testing Modes in MCP
Agents can request different testing modes via MCP. All modes still execute real training + evaluation pipelines.

**Mode 1: Light Training + Full Evaluation (Recommended for most testing)**
- Reduced training budget (e.g., 20-40% of full epochs, smaller batch size, or accelerated curriculum).
- Same backbone, loss formulation, and data mixture as the production JSON.
- After light training, the model undergoes:
  - Held-out benchmark evaluation.
  - Hidden stress testing (can be on a reduced but still challenging subset of variants for speed, or full set with early stopping).
  - Full physics gate application (hard gates still zero the score on critical violations).
- Produces a **test score** (same 45/30/25 formulation) that is returned quickly to the agent.
- This test score does **not** update the main leaderboard but is logged and can optionally contribute (with lower weight) to the Landscape Agent for prior improvement.

**Mode 2: Simulated / Cached Training Approximation (Early prototyping only)**
- For very rapid iteration, agents can request a simulated run that re-uses cached training dynamics or fast approximations.
- Still applies the same stress testing and physics gates on the resulting approximate model.
- Clearly marked as "simulated" in results and given lower trust weight by the Landscape Agent.
- Useful for hyperparameter sweeps or architecture search before committing to light or full training.

**Mode 3: Full Production Submission**
- Complete training budget as specified in the JSON.
- Full hidden stress testing on the complete StressTestSet.
- Full physics gates and scoring.
- Only these submissions can set new best combined scores and earn strong emissions weight.

### 3.3 Prior-Informed Warm Start from Landscape Agent
In test mode (and optionally production), agents can request initialization from the current best priors or distilled specialist weights for that challenge + backbone combination. This is retrieved from the Landscape Agent’s knowledge base.

This dramatically accelerates discovery and directly leverages the compounding intelligence of the subnet.

### 3.4 Explainable Failure Diagnostics
Test and production runs return rich, interpretable diagnostics in addition to scores:
- Locations and types of high residual or gate violations (e.g., "shock capturing failure in high-frequency region").
- Spectral or conservation drift analysis where applicable.
- Uncertainty hotspots (if uncertainty module is active).
- Comparison against recent successful strategies on similar challenges.

This greatly improves iteration speed for both human miners and autonomous agents.

### 3.5 Pareto / Multi-Objective Reporting in Test Mode
Light test mode can optionally return the Pareto front across physics fidelity, robustness, and accuracy instead of (or in addition to) a single scalar score. This helps agents discover interesting trade-off strategies and provides richer data to the Landscape Agent.

### 3.6 Defensibility and Anti-Gaming Design
- **Clear Separation of Concerns**: Test modes never affect the official leaderboard or primary emissions. Only full production submissions do.
- **Meaningful Signal in Testing**: Because light training + stress/gates still occurs, agents receive genuine feedback on whether their strategy is promising.
- **Rate Limiting & Credit System**: Each agent/hotkey has a limited number of test-mode runs per epoch or per challenge. Production submissions may have higher cost or staking requirements.
- **Determinism**: All test and production runs are fully deterministic and reproducible inside the validator Docker image.
- **Provenance & Auditing**: Every test and production run logs the exact JSON, seeds used, training curves (hashed), stress results, and gate outcomes.
- **Phased Difficulty in Testing**: Early test modes can use easier stress variants or fewer variants while still applying hard physics gates. This gives fast feedback while preserving the adversarial character of the main evaluation.
- **Landscape Agent Integration**: Even test runs can feed the Landscape Agent (with appropriate weighting), turning every agent interaction into collective intelligence.

### 3.7 Benefits for SOTA and Innovative Strategies
This design strongly supports the development of state-of-the-art and innovative strategies:
- Autonomous agents can implement closed-loop optimization (e.g., Bayesian optimization, evolutionary search, or reinforcement learning over strategy JSON space) using the fast but meaningful test-mode feedback as a reward signal.
- Human miners can rapidly prototype novel loss formulations, curricula, conditioning schemes, or backbone modifications and receive gated, stress-tested scores within minutes instead of hours.
- The combination of light-but-real training + adversarial stress testing in test mode creates a high-signal environment that rewards genuine innovation rather than benchmark overfitting.
- Because test results are still physics-gated and stress-evaluated, the subnet maintains scientific integrity even during rapid agent-driven exploration.

### 3.8 MCP Implementation Notes
MCP exposes endpoints for submitting JSON strategies in different modes (test_light, test_simulated, production). Streaming of intermediate training metrics, gate status, and final scores is supported for long-running jobs. Persistent sessions allow agents to maintain state across multiple test iterations. Authentication is hotkey-based with optional higher quotas for high-performing or staked agents.

This MCP + testing design makes Carbon one of the most agent-friendly yet rigorously defensible subnets, enabling the rapid discovery of superior Neural Operator methodologies that centralized platforms cannot easily replicate.

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

Each challenge includes public training/holdout splits and hidden stress configurations. Symbolic metadata (conservation laws, symmetries, boundary types) is attached.

### 4.2 Phase 1: Customization Layer
Same 7 challenges + support for custom datasets (Abaqus ODB/.fil ingestion) and LoRA/custom strategy parameters. Focus remains on single-physics robustness with richer data diversity.

### 4.3 Phase 2: Verified Multi-Physics Composition
**Phase 2A — Benchmark Problems**:
- FSI (Fluid-Structure Interaction): Turek/Hron 2D benchmarks (FSI-1/2/3). Reference: partitioned coupling via preCICE + OpenFOAM/CalculiX or equivalent. Key physics: fluid-structure interaction, added mass, vortex shedding.
- Conjugate Heat Transfer (CHT): Solid-fluid thermal coupling (electronics cooling, heat exchangers). Reference solutions from PDEBench/OpenFOAM.

**Phase 2B — Thermo-Elasticity Reference Suite**:
Generate ~48 mesh-converged Tier-1 reference cases at 256^{2} using FEniCS monolithic solver, varying thermal expansion coefficient (β), conductivity (κ), and geometry. Cost target: moderate compute for ground truth.

**Phase 2C — Variant Expansion**:
New Reynolds numbers, geometries, coupling strengths on the above. Three-track leaderboard (Monolithic / Composition / Specialist-Only).

### 4.4 Phase 3: 3D Multi-Physics & Advanced Composition
- 3D FSI (cylinder/flap with turbulence)
- 3D Thermo-Elasticity (bimetal, engine, turbine components)
- 3D CHT (electronics, battery, turbine cooling)

Requires 3D turbulence bridge (proper Kolmogorov spectrum initialization, not simple zero-pad), 3D-specific gates (energy spectrum, Q-criterion, wall shear, Nusselt distribution), and curriculum from 2D specialists.

Reference solutions: preCICE partitioned, FEniCS/OpenFOAM monolithic where appropriate.

---

## 5. Validation Strategy — Scientific Rigor & Competitive Edge

### 5.1 Multi-Objective Scoring (45/30/25)

Final combined score = w_p · PhysicsFidelity + w_r · Robustness + w_a · Accuracy, with w = [0.45, 0.30, 0.25].

Only new best combined scores on a challenge receive strong weight from the ChallengeWinnerTracker.

**Physics Fidelity (45%)**:
- Residual norms (PDE residual, divergence error for incompressible flow)
- Conservation errors (mass, momentum, energy)
- Boundary condition satisfaction (relative L2 or max norm)
- Stability metrics (energy growth/decay rates, rollout stability)

**Robustness (30%)**:
- Performance degradation under hidden stress (procedural + data-driven)
- Long-horizon rollout stability
- Generalization across parameter variations and out-of-distribution slices

**Accuracy (25%)**:
- Hold-out / benchmark error (relative L2, max error)

### 5.2 Physics Gates (Enforcement Layer)

**Hard Gates** (score = 0 on violation — non-negotiable physical constraints):
- Mass conservation: ‖∇·u‖ / ‖u‖ < threshold (typically 1e-3)
- Energy dissipation/stability: |dE/dt| or long-term energy drift below threshold
- Boundary satisfaction: relative error on Dirichlet/Neumann conditions
- Rollout stability: bounded energy growth over long horizons
- UQ calibration (when applicable)

Phase-field specific: crack irreversibility (∂d/∂t ≥ 0), length-scale consistency, valid degradation function.

**Soft Gates** (multiplicative penalty):
- Symmetry violation, spectral fidelity, conservation drift (graded penalties).

These gates are designed so that a PhD-level reviewer can verify they correspond to fundamental physical requirements of the PDE class.

### 5.3 Hidden Stress Testing (Adversarial Robustness — Core Moat)

Stress is generated per physics class with explicit scientific justification. Number of variants and parameter ranges scale with difficulty. All variants carry `physics_justification` metadata.

**Elliptic (Poisson/Darcy)**:
- Source amplitude & spatial variation (tests maximum principle & conservation)
- Boundary condition strength/type variation
- Coefficient field regularity (smooth → discontinuous; tests solution regularity)

**Hyperbolic (Burgers)**:
- Shock strength (initial condition steepness)
- High-frequency perturbation amplitude (tests shock capturing & stability)
- Rollout horizon length (long-time behavior after shock formation)
- Viscosity variation (within physically relevant range)
- Initial condition noise

**Parabolic (Heat)**:
- Time-dependent forcing amplitude & frequency
- Conductivity field variation
- Rollout length (long-term decay)
- Initial condition roughness

**Incompressible Flow (NS laminar)**:
- Reynolds number (laminar regime)
- Geometry scale/perturbation
- Boundary condition perturbation
- Initial vorticity noise
- Weak body forcing

**Elasticity**:
- Young's modulus variation
- Poisson ratio range
- Boundary displacement magnitude
- Material anisotropy
- Body force strength

**Thermo-Elasticity (Multi-physics)**:
- Thermal expansion coefficient
- Thermo-mechanical coupling strength
- Temperature loading amplitude
- Mechanical damping
- Deformation-induced heat source strength

**Difficulty Scaling**: Higher difficulty increases number of variants, range of parameter excursions, and rollout lengths while remaining physically plausible.

**Data-Driven Stress (The Well)**: Relevant dataset slices (turbulence, viscoelastic, active matter, acoustic scattering, etc.) mapped to physics class. Physics-preserving augmentations applied where possible.

**Strategic Advantage**: Hidden adversarial stress testing is extremely difficult for centralized platforms to match at scale. It forces genuine robustness and provides verifiable, trustless evaluation — a major credibility advantage for engineering and regulated applications.

See `docs/STRESS_TEST_DESIGN.md` and current `neurons/stress/procedural_generator.py` for exact parameter ranges and justification strings.

---

## 6. Determinism & Reproducibility (Scientific Trust & Credibility Moat)

Every evaluation must be reproducible given only public inputs + validator identity. Hierarchical seeding + framework controls achieve this.

Master seed derived from challenge_id + validator hotkey. Sub-seeds control data loading, augmentation, training, stress generation, and scoring.

PyTorch determinism flags + environment provenance recording enable cross-validator verification. A Reproducibility Harness compares scores, key tensors (within tolerance), and gate outcomes.

**Strategic Importance**: This level of determinism is required for credible claims of robustness and for audit/dispute resolution. It provides a trustless verification layer that centralized "black-box" AI platforms struggle to match, especially in safety-critical or regulated domains.

---

## 7. Landscape Agent — Symbolic & Causal Compounding (Core Innovation Engine)

The Landscape Agent is the component that turns decentralized evaluation into collective intelligence — enabling Carbon to discover better *ways* to train Neural Operators.

**Ingestion**: StrategyFragments containing strategy config, training metrics, full stress results (per-variant performance + gate outcomes), and symbolic features. Model Cards from both production and high-quality test runs are also ingested.

**Symbolic Processing**: Enrichment with conservation laws, symmetries, boundary types, coupling terms (extracted via rule-based Phase 0; ModelingToolkit + PySR in later phases).

**Causal Analysis**: Double Machine Learning (DML) to estimate heterogeneous treatment effects of strategy features (loss weights, backbone choice, curriculum parameters, etc.) on outcomes (combined score or robustness). This is central to discovering which training methodologies actually work.

**Knowledge Compounding**: Updated priors, causal graphs, and performance history are used to generate better challenge priors and specialist candidates. This creates compounding returns on every evaluation submitted to the network.

**Outputs**: Improved agent priors, specialist distillation candidates, causal insights for roadmap/challenge design, inputs to Symbolic Gauntlet.

**Strategic Role**: This is how Carbon turns parallel exploration into superior methodologies faster than centralized teams. The compounding effect (better data → better insights → better strategies → richer data) is a core part of the thesis that a decentralized subnet can outperform centralized players.

**Future**: Integration with multi-physics composition and Foundation Operator (LPM with FiLM conditioning, evidential UQ).

See design notes for DML implementation sketch and data schemas.

---

## 8. Detailed Implementation Components

(See current code in `neurons/` for reference implementations that match the interfaces below.)

**Stress Generators**: Registry-based with physics-class-specific deep implementations (parameter variation logic as described in Section 5.3), with multi-fidelity support.

**StressEvaluator**: Runs model on StressTestSet, applies gates, returns structured results (hard failures, soft penalties, per-variant metrics, stress contribution).

**HydrogenScorer**: Integrates base metrics + stress results; computes weighted scores; modulates final combined score by stress contribution.

**generate_challenge()**: Deterministic function returning Challenge object with training/holdout references, stress_config, and attached SymbolicMetadata.

**MCP Layer**: Handles strategy JSON submission in multiple modes (test_light, test_simulated, production), streaming results, built-in testing with light training + gated evaluation, explainable diagnostics, prior warm-start support, and agent authentication (detailed in Section 3).

**Backbone Registry**: Dynamic model instantiation from JSON `backbone` specification inside the validator Docker image.

---

## 9. Phased Roadmap (Build-Level)

**Phase 0**:
- Stress generators (procedural deep per class + Well) with determinism.
- StressEvaluator + full scoring integration.
- Determinism utilities and reproducibility harness.
- MCP basics + testing loop with light training + gated evaluation.
- Symbolic extractor + PySR skeleton.
- ChallengeWinnerTracker.
- generate_challenge() with symbolic attachment.
- Validator Docker image with backbone registry, full JSON schema support, training pipeline, held-out + stress evaluation, multi-fidelity, model card generation, residual monitoring.
- Explainable diagnostics in test mode.

**Phase 1**:
- Abaqus ODB/.fil parser (mesh + field outputs: stress/strain/displacement/history).
- CustomDataset mixing with procedural data.
- Initial Landscape Agent (DML causal updates + symbolic enrichment).
- Expanded determinism in data paths.
- Enhanced MCP testing modes, credit system, and simulated training approximations.
- Prior-informed warm start from Landscape Agent.
- Uncertainty-aware stress prioritization.
- Pareto / multi-objective test reporting.
- LoRA / custom backbone plugin support.

**Phase 2**:
- preCICE orchestration for FSI (Turek/Hron) and CHT benchmarks.
- Thermo-elasticity reference generation (~48 cases, mesh-converged FEniCS).
- Specialist pipeline schema + execution.
- Three-track leaderboard and stress testing on multi-physics problems.

**Phase 3**:
- 3D turbulence initialization (Kolmogorov spectrum) and 3D-specific gates (energy spectrum, Q-criterion, wall shear, Nu).
- 3D FSI/CHT/Thermo-elasticity with appropriate references (preCICE, OpenFOAM, FEniCS).
- Advanced Landscape Agent outputs and Foundation Operator foundations.
- Full MCP production features, agent governance, and advanced testing modes.

---

## 10. Scientific Defensibility & Competitive Differentiation

Every stress dimension is chosen because it directly probes a fundamental physical property of the PDE class (conservation, stability, shock capturing, coupling strength, etc.). Hard gates correspond to non-negotiable physical requirements. The Landscape Agent's causal analysis provides interpretable insights into what actually improves physical fidelity and robustness. Determinism ensures results are reproducible and auditable by domain experts.

**Competitive Positioning**:

The broader industry is rapidly advancing toward **Software Defined Machines** and **Living Digital Twins**, where high-fidelity models serve as the single source of truth across the entire product lifecycle — from design and embedded control to ongoing operation and predictive maintenance, continuously refined by real-world sensor data. Leading platforms such as JuliaHub’s Dyad are building modern acausal modeling environments, deep SciML integration (neural surrogates, model discovery, differentiable programming), generative AI assistance, and cloud-native deployment workflows to realize this vision for industrial engineering (aerospace, automotive, energy, etc.).

**Carbon plays a distinct and complementary role** in this ecosystem:

- While centralized platforms focus on usability, integration, model generation, and accelerating surrogate creation within a single environment, Carbon is the **decentralized discovery and robustness engine**.
- Carbon enables massively parallel exploration of Neural Operator training strategies across a global network of agents and miners.
- It applies rigorous, hidden adversarial stress testing with physics-class-specific gates that centralized systems struggle to replicate at scale.
- The Landscape Agent extracts symbolic features and causal relationships, turning individual evaluations into compounding collective intelligence about *how* to best train and validate these surrogates.

This makes Carbon particularly valuable for producing surrogates that are not only fast but genuinely robust, physically trustworthy, and auditable — critical for safety-critical and regulated applications within the Software Defined Machines and Living Digital Twins paradigm.

In essence: Platforms like Dyad modernize the modeling and surrogate *environment*; Carbon discovers and validates the superior *methodologies* that power higher-performing, more reliable digital twins across the network.

A PhD reviewer should be able to verify that the system is not merely optimizing for benchmark scores but is enforcing and learning physically meaningful behavior while leveraging decentralized parallel exploration to innovate faster than centralized alternatives.

---

*This specification is written to be scientifically rigorous and buildable. Reference the implementation in `neurons/` and supporting design documents for concrete code.*
