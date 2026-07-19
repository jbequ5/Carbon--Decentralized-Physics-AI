# SPEC.md — Hydrogen PDE Subnet Technical Specification

**Version:** 3.2 (Updated July 2026)
**Status:** Active Development

---

## 1. System Overview

### 1.1 Core Loop

```
Challenge Release → Strategy Submission → Validator Training + Hidden Stress Testing → Physics-Gated Scoring → ChallengeWinnerTracker → Weights (Yuma) → Landscape Learning
```

### 1.2 Design Philosophy
Physics fidelity first, adversarial robustness via hidden stress + hard gates, decentralized strategy discovery, and information asymmetry.

### 1.3 Current Economics
Standard Yuma Consensus + `ChallengeWinnerTracker` (winner-heavy + exponential decay + dust). Hybrid bounty/stipend model is planned but not active.

---

## 2. Scoring & Tracker

### 2.1 Multi-Objective Scoring (45/30/25)
Physics Fidelity, Robustness, Accuracy → combined score. Only new combined bests on a challenge receive strong weight.

### 2.2 Physics Gates
Hard gates (mass conservation, energy dissipation, boundary satisfaction, rollout stability, UQ calibration) and soft gates with multiplicative penalties. Phase-field specific gates also defined.

### 2.3 ChallengeWinnerTracker
Per-challenge best score + leader tracking with exponential decay. Produces winner-heavy + participation dust weights.

---

## 3. Symbolic Layer (Planned)

**ModelingToolkit.jl** for symbolic PDE representation and auto loss weights.

**DataDrivenDiffEq.jl + PySR** for the Symbolic Regression track (discover governing equations from trajectories, validated on hidden stress data).

Symbolic metadata will be preserved through specialist distillation (Symbolic Gauntlet).

---

## 4. Phased Roadmap

### Phase 0 (Current — Completed)
Core infrastructure: `ChallengeWinnerTracker`, `StrategyStore`, multi-objective physics-gated scorer, validator, MCP server, determinism.

### Phase 1: Customization & Data Ingestion
- Same 7 core PDE challenges (Poisson, Darcy, Burgers, NS laminar, Heat, Elasticity, Thermo-elasticity)
- Miners submit LoRA adapters + custom datasets
- **Abaqus ODB / .fil Ingestion Pipeline**:
  - Miner submits `custom_data` with IPFS `data_uri` + checksum
  - Validator verifies checksum, downloads, parses ODB or .fil
  - Parses mesh + field outputs (stress, strain, displacement, history)
  - Caches parsed `CustomDataset` and mixes with procedural data according to `weight` parameter
- Expanded symbolic regression track using **PySR** + DataDrivenDiffEq

### Phase 2: Multi-Physics Composition

**Phase 2A** — Verified multi-physics benchmarks (first 3–6 months)
- FSI challenges (Turek/Hron 2D-1/2/3) with preCICE reference
- Conjugate Heat Transfer (CHT) challenges with OpenFOAM/PDEBench references
- Three-track leaderboard: Monolith / Composition / Specialist-Only

**Phase 2B** — Thermo-Elasticity
- Generate ~48 Tier-1 mesh-converged references (varying β, κ, geometry) at 256^{2} using FEniCS monolithic solver
- Add thermo-elasticity challenges with `loss_vector` coupling terms

**Phase 2C** — Variant expansion
- New Reynolds numbers, geometries, and coupling strengths on FSI/CHT/Thermo-elasticity
- Specialist Bank grows; reuse rate target >80%
- Adapter innovation tracked as a key metric

Specialist pipeline JSON schema + staggered coupling execution become first-class supported submission format.

### Phase 3: 3D Multi-Physics (Post-Turbulence Bridge)

**Phase 3.1** — 3D Turbulence Bridge (prerequisite)
- 3D Spectral Initialization Protocol (proper Kolmogorov spectrum, not simple zero-pad)
- Curriculum from 2D specialists → 3D (zero-pad + controlled noise → progressive resolution)
- `ns_3d_turbulent` specialist with verified k^(-5/3) energy spectrum
- 3D-specific stress gates: energy spectrum, Q-criterion, wall shear stress distribution, Nu distribution at corners

**Phase 3.2** — 3D Multi-Physics Rollout
- 3D FSI (cylinder/flap, turbulent) using `ns_3d_turbulent` + `elasticity_3d` + `fsi_3d_adapter` (preCICE partitioned reference)
- 3D Thermo-Elasticity using `elasticity_3d` + `heat_3d` + `thermal_expansion_3d` (FEniCS reference)
- 3D CHT using `ns_3d_turbulent` + `heat_3d` + `cht_3d_adapter` (OpenFOAM/COMSOL reference)
- Same three-track leaderboard and stress testing applied to all 3D multi-physics challenges

**Phase 3.3** — Foundation Operator (LPM)
- Multi-teacher distillation across entire Specialist Bank (2D + 3D)
- FiLM conditioning on ProblemSignature + SymbolicMetadata
- Evidential UQ head
- Commercial fine-tuning API (TEE decryption → LoRA adaptation → stress-test verification → encrypted ONNX return)

---

## 5. Validator & Scoring Details

(Physics gates tables, multi-objective formula, ChallengeWinnerTracker behavior, and determinism requirements are defined in earlier sections and remain unchanged.)

---

## 6. Future Domains

See `docs/FUTURE_DOMAINS.md` for the full analysis of additional high-value domains (Electromagnetics, Photonics, Acoustics, Plasmas/Fusion, Quantum-informed modeling, Climate, Nuclear, Biological systems, etc.).

---

## 7. Current Limitations

- Abaqus ingestion pipeline and full Phase 1–3 roadmap are planned but not yet implemented
- Hybrid emissions model is defined but not active
- Landscape Agent, specialist distillation, and Symbolic Gauntlet are future work
- Multi-validator consistency and canonical ranking not yet stress-tested

---

*This specification reflects both current implementation and the planned phased roadmap.*
