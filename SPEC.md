# SPEC.md — Hydrogen PDE Subnet Technical Specification

**Version:** 3.5 (Updated July 2026)
**Status:** Active Development

---

## 1. System Overview & Core Agentic Loop

Hydrogen is a decentralized, agentic system for discovering high-quality, physics-respecting training strategies for neural operator surrogates in engineering and scientific simulation.

**Core Loop**:
MCP-enabled Agent/Miner Submission → Validator (deterministic training + hidden stress testing + physics gates) → Multi-Objective Scoring & Detailed Feedback → Results returned to agent + ingestion into Landscape Agent → Causal/symbolic analysis & knowledge base update → Improved priors for future submissions → Compounding discovery.

**Design Philosophy**:
- Physics fidelity first via hard/soft gates and multi-objective scoring.
- Adversarial robustness through hidden stress testing.
- Agent-friendly participation via MCP with built-in testing/feedback loops.
- Information asymmetry to prevent gaming while preserving learning signals.
- Landscape Agent for symbolic feature extraction and causal (DML) knowledge compounding.
- Determinism and auditability for trust.
- Decentralized strategy discovery with aligned incentives.

**Current Economics**: Standard Yuma + ChallengeWinnerTracker (per-challenge leader tracking with exponential decay and winner-heavy + dust weighting). Hybrid bounty/stipend model planned for later phases.

---

## 2. MCP / Agent Interface & Built-in Testing Loop

**MCP (Miner/Agent Communication Protocol)** enables seamless, agent-friendly interaction:
- Persistent sessions and streaming validation/results.
- Easy local or remote testing of strategies against the live system or subsets.
- Low-friction submission and feedback loop designed specifically for autonomous agents and human-in-the-loop workflows.

This design dramatically lowers the barrier to participation and accelerates the rate of strategy iteration and discovery.

---

## 3. Validation Mechanism (Detailed)

**Pipeline**:
1. Strategy retrieval (via StrategyStore).
2. Deterministic training (seeded from challenge_id + validator identity).
3. Hidden Stress Test Generation & Evaluation (procedural + Well-based; future adversarial tiers).
4. Physics Gate Application (hard gates = 0 on violation; soft penalties).
5. Multi-Objective Scoring (Physics Fidelity 45%, Robustness 30%, Accuracy 25%).
6. Results + diagnostics returned; full data ingested by Landscape Agent.

**Determinism Guarantees**:
Hierarchical seeding (master seed from challenge + validator hotkey → sub-seeds for data, training, stress, scoring). PyTorch/JAX determinism flags, DataLoader generators, and environment provenance recording. Reproducibility Harness for verification.

**Stress Test System** (see docs/STRESS_TEST_DESIGN.md for full spec):
- Tier 1: Procedural parametric variation (physics-class specific, difficulty-adaptive).
- Tier 2: The Well dataset slices with physics-preserving augmentations.
- Tier 3 (future): Adversarial/uncertainty-guided/Pareto stress.
- Full access control (validator-private), auditability, and physical justification metadata.

**Information Asymmetry**: Agents see scores and useful diagnostics but not exact hidden stress conditions or internal gate implementations.

---

## 4. Scoring System

Multi-objective with combined score. Only new best combined scores on a challenge drive strong weighting via ChallengeWinnerTracker.

Physics gates (hard = 0 on critical violations like mass conservation, energy stability, boundary satisfaction; soft multiplicative penalties). Detailed metrics returned for feedback and Landscape ingestion.

---

## 5. ChallengeWinnerTracker & Incentives

Per-challenge tracking with exponential decay. Produces winner-heavy + participation dust weights. Only genuine improvements on the physics + robustness + accuracy metric are strongly rewarded.

---

## 6. Landscape Agent Architecture (Symbolic & Causal Compounding)

**Core Function**: Ingests evaluation results (strategy configs, performance under stress, symbolic features, metrics) and compounds knowledge.

**Components**:
- **Symbolic Processing**: Extracts/uses conservation laws, symmetries, dimensionless groups, boundary types (via PySR, ModelingToolkit integration planned). Enriches data and supports auto loss weighting.
- **Causal Learning**: Double Machine Learning (DML) to estimate heterogeneous treatment effects of strategy choices (e.g., loss weight schedules, backbone choices, curricula) on outcomes.
- **Knowledge Base & Prior Updating**: Maintains evolving priors and causal insights that improve future strategy generation and evaluation.
- **Outputs**: Better priors for agents, specialist distillation candidates, causal reports, Symbolic Gauntlet inputs.

This creates compounding returns: better data → better insights → better strategies → even richer data.

**Integration**: Feeds into specialist promotion, multi-physics composition, and long-term Foundation Operator development.

---

## 7. Symbolic Layer

- ModelingToolkit.jl for symbolic PDE representation, conservation laws, and auto loss weights.
- DataDrivenDiffEq.jl + PySR for symbolic regression track (discover governing equations validated on hidden stress data).
- SymbolicMetadataExtractor (rule-based Phase 0; advanced later).
- Symbolic Gauntlet during specialist distillation to preserve symbolic properties.

See docs/SYMBOLIC_LAYER_DESIGN.md for details.

---

## 8. Challenge Generation

Deterministic `generate_challenge()` function that produces training/holdout references, stress configuration, and attaches SymbolicMetadata. Supports future advanced stress and multi-physics definitions.

See docs/CHALLENGE_GENERATION_DESIGN.md.

---

## 9. Validator Implementation

Orchestrates MCP handling, deterministic training/stress generation (using centralized seeding utilities), StressEvaluator integration, scoring, and weight submission via ChallengeWinnerTracker.

Full determinism setup (PyTorch flags, sub-seeds) applied per evaluation.

---

## 10. Phased Roadmap (Build-Level)

**Phase 0 (Current Foundations)**: Core scoring, stress testing (procedural + Well for all physics classes), determinism utilities, MCP basics, ChallengeWinnerTracker, symbolic skeleton (PySR runner + metadata), StressEvaluator integration.

**Phase 1**: Abaqus ODB/.fil custom data ingestion, expanded symbolic capabilities, LoRA/custom dataset support, deeper determinism across data loading/training, initial Landscape Agent causal/symbolic updates.

**Phase 2**: Verified multi-physics composition using preCICE (FSI, CHT, Thermo-elasticity benchmarks and variant expansion). Specialist pipelines and staggered coupling. Growing Specialist Bank.

**Phase 3**: 3D multi-physics (FSI, CHT, Thermo-elasticity with turbulence), 3D-specific gates and curriculum from 2D. Advanced Landscape Agent compounding and Foundation Operator/LPM foundations.

---

## 11. Future Domains

See docs/FUTURE_DOMAINS.md for detailed analysis of Electromagnetics, Photonics, Acoustics, Plasmas/Fusion, Quantum-informed modeling, Climate, Nuclear, Biological systems, and others.

---

## 12. Current Limitations

Full adversarial stress tiers, complete Landscape Agent causal/symbolic compounding loop, preCICE-orchestrated multi-physics, and advanced determinism in data/training loops are in active development or planned for near-term phases.

---

*This specification captures both implemented foundations and the detailed build roadmap for the agentic, compounding physics surrogate discovery engine.*
