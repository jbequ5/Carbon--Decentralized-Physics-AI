# Hydrogen

**Decentralized Agentic Engine for Robust Physics-Informed Neural Operator Surrogates**

Hydrogen is a Bittensor subnet that builds a decentralized, agentic system where miners and autonomous agents collaboratively discover fast, robust, and physics-respecting training strategies for high-fidelity engineering surrogates. It combines an **MCP-powered participation layer with built-in testing loops**, a **rigorous hidden adversarial validation mechanism**, and a **Landscape Agent that compounds symbolic and causal knowledge** over time. The result is an accelerating engine that produces surrogates far more effectively than centralized labs.

---

## Vision

A self-improving decentralized intelligence layer for physical modeling — where agents and a central Landscape Agent together create a compounding library of reusable, physics-grounded surrogates and specialists that accelerate engineering design, digital twins, real-time simulation, and scientific discovery. Long-term, this evolves toward foundational models of physics that are fast, robust, and interpretable across domains.

---

## Value Proposition & Market Opportunity

Traditional high-fidelity simulation is too slow and expensive for large design spaces or real-time use. Pure data-driven ML surrogates are fast but often violate fundamental physical constraints, limiting their reliability in engineering.

Hydrogen produces **physics-informed neural operator surrogates** through a unique engine that delivers:

- **Valuable Causal + Symbolic Dataset**: Every evaluation under hidden stress generates rich, physics-grounded data, symbolic features (conservation laws, symmetries), and causal relationships between strategy choices and outcomes. This dataset is far more informative than standard benchmarks and compounds over time.

- **Custom Surrogate Design Engine**: Via MCP and the agent-friendly loop (with built-in testing and fast feedback), participants can rapidly develop and validate problem-specific surrogates tailored to specific geometries, coupling strengths, or domains — not generic one-size-fits-all models.

**Broad Market Opportunities**:
- Core CAE/simulation acceleration (aerospace, automotive, energy, manufacturing)
- Hardware-in-the-Loop (HIL) and real-time testing
- Robotics and digital twins of complex systems
- Multi-physics problems (FSI, CHT, thermo-elasticity)
- High-impact domains like fusion/plasma, nuclear, and advanced energy systems
- Quantum-hybrid modeling
- Long-term foundational physics models

The **Landscape Agent** creates a compounding moat by turning every submission into better priors and reusable specialists through symbolic and causal analysis.

---

## How the Engine Works — Clear Mechanism Walkthrough

### 1. Participation via MCP (Agent-Friendly with Built-in Testing Loop)
Miners and agents interact through MCP, which supports persistent sessions, streaming validation/results, and easy local or remote testing of strategies. Agents can quickly iterate ideas against the live system or subsets of challenges and get immediate diagnostics.

### 2. Challenges by Phase
- **Phase 0**: 7 core single-physics PDE challenges (Poisson, Darcy, Burgers, Navier-Stokes laminar, Heat, Elasticity, Thermo-elasticity).
- **Phase 1**: Same challenges + custom datasets (including Abaqus ODB/.fil ingestion) and LoRA/custom strategy support.
- **Phase 2**: Verified multi-physics benchmarks (FSI using Turek/Hron, Conjugate Heat Transfer) with preCICE composition, plus Thermo-elasticity reference cases and variant expansion.
- **Phase 3**: 3D multi-physics (FSI, CHT, Thermo-elasticity with turbulence), 3D-specific gates, and curriculum progression from 2D specialists.

### 3. Validation Strategy (The Heart of Robustness)
Every submission goes through a rigorous, hidden validation process:

- **Benchmarking**: Performance on public holdout data (accuracy component).
- **Hidden Stress Testing**: Adversarial evaluation under conditions miners cannot see or target. Includes procedural parametric variation (physics-class specific) and slices from The Well dataset. Future tiers add adversarial/uncertainty-guided stress.
- **Physics Gates**: Hard gates (e.g., mass conservation, energy dissipation/stability, boundary satisfaction, rollout stability, UQ calibration) that zero the score on critical violations. Soft gates apply multiplicative penalties for finer issues (symmetry, spectral fidelity, etc.). Phase-field specific gates are also defined.
- **Multi-Objective Scoring (45/30/25)**: 
  - Physics Fidelity (45%): Residuals, conservation laws, boundary conditions, stability.
  - Robustness (30%): Performance under hidden stress, long-term rollout, generalization.
  - Accuracy (25%): Benchmark/hold-out performance.
  Only strategies that set a new best *combined score* on a challenge receive meaningful weight.

This combination ensures surrogates are not just accurate on known data but genuinely robust and physically trustworthy under unseen conditions.

### 4. Feedback, Landscape Agent & Knowledge Compounding
Agents receive detailed scores, gate outcomes, and diagnostics. All results are ingested by the **Landscape Agent**, which:
- Extracts symbolic features (conservation laws, symmetries, etc., via PySR and planned ModelingToolkit integration).
- Applies causal analysis (Double Machine Learning) to understand which strategy choices causally improve outcomes (e.g., certain loss-weight schedules causally improve long-term rollout stability under hidden stress variations).
- Updates a compounding knowledge base and priors.
- Drives specialist distillation and better future challenges.

This creates accelerating returns: better data → better insights → better strategies → even richer data.

### 5. Emissions & Incentives
Current model uses standard Yuma Consensus with the **ChallengeWinnerTracker**:
- Per-challenge leader tracking with exponential decay on old performance.
- Winner-heavy weighting + participation dust for recent contributors.
- Only genuine new best combined scores drive strong rewards.

Future phases add a hybrid model with Breakthrough Bounties (for record-setting improvements) and Decaying Top stipends, with unclaimed allocations rolling to a treasury.

---

## Why This Design Matters
- **Robustness without centralization**: Hidden stress + physics gates create trustworthy evaluation at scale.
- **Fast agent iteration**: MCP + built-in testing loop lowers barriers and speeds discovery.
- **Compounding intelligence**: The Landscape Agent turns individual effort into collective, accelerating progress via symbolic and causal knowledge.
- **Trust & Auditability**: Full determinism (hierarchical seeding, framework controls) and reproducibility make results credible and auditable across validators.
- **Aligned incentives**: Winner-heavy tracking with decay keeps focus on continuous, genuine improvement.
- **Compounding Moat**: The Specialist Bank and knowledge base create network effects and data royalties.

---

## Current State & Roadmap
Phase 0 foundations (scoring, stress testing across all physics classes, determinism utilities, MCP basics, symbolic skeleton, full integration of stress into scoring) are advancing rapidly. Phase 1 adds customization, Abaqus ingestion, and deeper symbolic/causal capabilities. Phase 2 brings verified multi-physics with preCICE. Phase 3 expands to 3D and more advanced composition.

See `SPEC.md`, `docs/FUTURE_DOMAINS.md`, and other docs in the repository for full technical details.

---

*Hydrogen is building the decentralized agentic infrastructure for trustworthy, compounding physical intelligence in engineering and science.*
