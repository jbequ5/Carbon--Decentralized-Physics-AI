# Hydrogen

**Decentralized Physical Intelligence through Robust Physics-Informed Neural Operator Trained Surrogates**

Hydrogen builds a decentralized, incentive system on Bittensor where miners and agents collaboratively discover fast, physics-respecting training strategies for high-fidelity engineering surrogates. It combines an **MCP-powered participation layer with built-in testing loops**, a **rigorous hidden validation mechanism**, and a **Landscape Agent that compounds symbolic and causal knowledge** to create accelerating, trustworthy discovery.

---

## Vision

A self-improving decentralized engine that turns collective strategy exploration into a compounding library of reusable, physics-grounded surrogates — eventually contributing to foundational models of physics that are fast, robust, and interpretable across domains.

---


## How the Engine Works — Clear Mechanism Walkthrough

### 1. Participation via MCP (Agent-Friendly with Built-in Testing Loop)

Miners and autonomous agents interact through MCP, which supports persistent sessions, streaming validation/results, and easy local or remote testing of strategies. This design makes participation seamless and enables rapid iteration — agents can test ideas quickly against the system or subsets of challenges.

### 2. Challenges by Phase

- **Phase 0**:
7 core single-physics PDE challenges (Poisson, Darcy, Burgers, Navier-Stokes laminar, Heat, Elasticity, Thermo-elasticity).
- **Phase 1**:
Same challenges + custom datasets (including Abaqus ODB/.fil ingestion) and LoRA/custom strategy support.
- **Phase 2**:
Verified multi-physics benchmarks (FSI using Turek/Hron, Conjugate Heat Transfer) with preCICE composition, plus Thermo-elasticity reference cases and variant expansion.
- **Phase 3**:
3D multi-physics (FSI, CHT, Thermo-elasticity with turbulence), 3D-specific gates, and curriculum progression from 2D specialists.

### 3. Validation Strategy (The Heart of Robustness)

Every submission goes through a rigorous, hidden validation process:

- **Benchmarking**: Performance on public holdout data (accuracy component).
- **Hidden Stress Testing**: Adversarial evaluation under conditions miners cannot see or target during development. Includes procedural parametric variation (physics-class specific) and slices from The Well dataset. Future tiers add adversarial/uncertainty-guided stress.
- **Physics Gates**: Hard gates (e.g., mass conservation, energy dissipation/stability, boundary satisfaction, rollout stability, UQ calibration) that zero the score on critical violations. Soft gates apply multiplicative penalties for finer issues (symmetry, spectral fidelity, etc.). Phase-field specific gates are also defined.
- **Multi-Objective Scoring (45/30/25)**: 
  - Physics Fidelity (45%): Residuals, conservation laws, boundary conditions, stability.
  - Robustness (30%): Performance under hidden stress, long-term rollout, generalization.
  - Accuracy (25%): Benchmark/hold-out performance.
  Only strategies that set a new best *combined score* on a challenge receive meaningful weight.

This combination ensures surrogates are not just accurate on known data but genuinely robust and physically trustworthy under unseen conditions — critical for engineering adoption.

### 4. Feedback, Landscape Agent & Knowledge Compounding

Agents receive detailed scores, gate outcomes, and diagnostics. All results are ingested by the **Landscape Agent**, which:
- Extracts symbolic features (conservation laws, symmetries, etc., via PySR and planned ModelingToolkit integration).
- Applies causal analysis (Double Machine Learning) to understand which strategy choices causally improve outcomes.
- Updates a compounding knowledge base and priors.
- Drives specialist distillation and better future challenges.

This creates accelerating returns: better data → better insights → better strategies → richer data.

### 5. Emissions & Incentives
Current model uses standard Yuma Consensus with the **ChallengeWinnerTracker**:
- Per-challenge leader tracking with exponential decay on old performance.
- Challenge Winner-heavy weighting + participation dust for recent contributors.
- Only genuine new best combined score gets strong rewards.

---

## Why This Design Matters

- **Robustness without centralization**: Hidden stress + physics gates create trustworthy evaluation at scale.
  
- **Fast agent iteration**: MCP + built-in testing loop lowers barriers and speeds discovery.
  
- **Compounding intelligence**: The Landscape Agent turns individual effort into collective, accelerating progress via symbolic and causal knowledge.
  
- **Trust & Auditability**: Full determinism (hierarchical seeding, framework controls) and reproducibility make results credible and auditable across validators.
  
- **Aligned incentives**: Winner-heavy tracking with decay keeps focus on continuous, genuine improvement.

---
## Value Proposition & Market Opportunity

Traditional high-fidelity simulation is too slow and costly for large design spaces or real-time applications. Pure ML surrogates are fast but often fail basic physics, limiting their use in engineering.

Hydrogen delivers **physics-informed neural operator surrogates** through a unique engine that produces:

- **Valuable Causal + Symbolic Dataset**: Every evaluation under hidden stress generates rich physics-grounded data, symbolic features, and causal relationships between strategy choices and outcomes. This compounds into one of the most informative datasets in scientific ML.

- **Custom Surrogate Design at Scale**: Via MCP and the agent-friendly loop (with built-in testing and fast feedback), participants rapidly develop and validate problem-specific surrogates rather than generic models.

**Market Opportunities** include:
- Core CAE/simulation acceleration (aerospace, automotive, energy, manufacturing)
- Hardware-in-the-Loop (HIL) and real-time testing
- Robotics, digital twins, and complex system simulation
- Multi-physics problems (FSI, CHT, thermo-elasticity)

**Long Term "Moonshots"**
- High-impact domains like fusion/plasma, nuclear, and advanced energy systems
- Quantum-hybrid modeling
- Long-term foundational physics models

The **Landscape Agent** creates a compounding moat by turning every submission into better priors and reusable specialists.

---

## Current State & Roadmap
Phase 0 foundations (core scoring, stress testing across all physics classes, determinism utilities, MCP basics, symbolic skeleton, full integration of stress into scoring) are advancing rapidly. Phase 1 adds custom data ingestion and deeper symbolic/causal capabilities. Phase 2 brings verified multi-physics with preCICE. Phase 3 expands to 3D and advanced composition.

See `SPEC.md`, `docs/FUTURE_DOMAINS.md`, and other docs in the repository for full technical details.

---

## Getting Started
```bash
git clone https://github.com/jbequ5/Hydrogen.git
cd Hydrogen
python neurons/validator.py --dry_run true   # Explore in dry-run mode
```

See the `docs/` folder for detailed guides (including MCP/agent usage and validator configuration).

---

## Contributing
We welcome contributions in stress testing, determinism, symbolic integration, MCP enhancements, multi-physics composition, and testing infrastructure.

---

*Hydrogen is building the decentralized agentic infrastructure for trustworthy, compounding physical intelligence in engineering and science.*
