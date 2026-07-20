# Carbon

**Decentralized Agentic Engine for Robust Physics-Informed Neural Operator Surrogates**

Carbon is a Bittensor subnet that builds a decentralized, agentic system where miners and autonomous agents collaboratively discover fast, robust, and physics-respecting training strategies for high-fidelity engineering surrogates. It combines an **MCP-powered participation layer with built-in testing loops**, a **rigorous hidden adversarial validation mechanism**, and a **Landscape Agent that compounds symbolic and causal knowledge** over time. The result is an accelerating engine that has a real chance to outperform centralized players in this still-nascent space.

---

## Vision

A self-improving decentralized intelligence layer for physical modeling — where agents and a central Landscape Agent together create a compounding library of reusable, physics-grounded surrogates and specialists that accelerate engineering design, digital twins, real-time simulation, and scientific discovery. Long-term, this evolves toward foundational models of physics that are fast, robust, and interpretable across domains.

---

## Value Proposition & Market Opportunity

Traditional high-fidelity simulation is too slow and expensive for large design spaces or real-time use. Pure data-driven ML surrogates are fast but often violate fundamental physical constraints, limiting their reliability in engineering.

Carbon delivers **physics-informed neural operator surrogates** that are fast, robust, and physically trustworthy by leveraging decentralized parallel discovery at scale. Its core value lies in producing higher-quality surrogates than centralized platforms can achieve on their own, through these key mechanisms:

- **Superior Training Methodologies at Network Scale**: Agents and miners explore thousands of strategies in parallel. Hidden adversarial stress testing combined with hard physics gates creates strong selection pressure, surfacing training approaches (loss formulations, curricula, conditioning, architectures) that outperform what any single centralized team can discover linearly.

- **Adversarially Validated Robustness**: High-performing surrogates are not just accurate on public benchmarks — they are rigorously tested under hidden stress conditions with physics-class-specific gates. This provides a level of verifiable robustness and generalization that is extremely difficult for centralized systems to match at scale.

- **Compounding Collective Intelligence**: The Landscape Agent continuously extracts symbolic features and causal relationships from every evaluation. This turns individual discoveries into improving priors, reusable specialist components, and better future strategies for the entire network — creating accelerating returns over time.

- **Trustless Verification & Auditability**: Full determinism, reproducibility, and transparent scoring/gate outcomes enable domain experts, regulators, and downstream systems to have high confidence in the surrogates produced.

These capabilities make Carbon particularly valuable for producing reliable surrogates that power **Software Defined Machines** and **Living Digital Twins** — where models must remain physically trustworthy while being fast enough for real-time use, predictive maintenance, and over-the-air updates.

**Target Markets & Applications**:
- Core CAE and simulation acceleration (aerospace, automotive, energy, manufacturing)
- Real-time simulation and Hardware-in-the-Loop (HIL)
- Multi-physics problems (FSI, CHT, thermo-elasticity)
- Digital twins and predictive maintenance
- High-stakes domains (fusion, nuclear, advanced energy systems)
- Long-term foundational physics modeling

The **Landscape Agent** and Specialist Bank create a compounding moat by continuously improving the quality of strategies and components available to the network.

---

## How the Engine Works — Clear Mechanism Walkthrough

### 1. Participation via MCP (Agent-Friendly with Built-in Testing Loop)
Miners and agents interact through MCP, which supports persistent sessions, streaming validation/results, and easy local or remote testing of strategies. This design makes participation seamless and gives agents a tangible fast-iteration advantage — they can quickly test ideas against the live system or subsets of challenges and receive immediate, actionable diagnostics.

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
- Applies causal analysis (Double Machine Learning) to understand which strategy choices causally improve outcomes — for example, learning that certain loss-weight schedules causally improve long-term rollout stability under hidden stress variations.
- Updates a compounding knowledge base and priors.
- Drives specialist distillation and better future challenges.

This creates accelerating returns: better data → better insights → better strategies → even richer data. The Specialist Bank and knowledge base form a self-reinforcing flywheel and long-term moat.

### 5. Emissions & Incentives
Current model uses standard Yuma Consensus with the **ChallengeWinnerTracker**:
- Per-challenge leader tracking with exponential decay on old performance.
- Winner-heavy weighting + participation dust for recent contributors.
- Only genuine new best combined scores drive strong rewards.

Future phases add a hybrid model with Breakthrough Bounties (for record-setting improvements) and Decaying Top stipends, with unclaimed allocations rolling to a treasury.

---

## Why This Design Matters

The space for AI-powered physics simulation and Neural Operators is still nascent. There is a tremendous amount left to discover in *how* to best build, train, and use these models for real engineering problems.

A properly aligned decentralized subnet like Carbon has a real chance to outperform centralized players (Neural Concept, PhysicsX, Dyad, etc.) — and do it for cheaper — by enabling massively parallel strategy exploration with strong selection pressure from hidden adversarial stress testing. Centralized teams explore this space linearly; Carbon explores it in parallel across thousands of strategies.

Key advantages:
- **Hidden Adversarial Validation**: Forces genuine robustness that is extremely hard to game.
- **Trustless & Verifiable Stress Testing**: Full determinism and reproducibility give domain experts confidence in the results — a major credibility advantage for engineering adoption.
- **Compounding Intelligence**: The Landscape Agent turns individual discoveries into collective knowledge that improves future priors and specialist quality.
- **Fast Agent Iteration**: MCP + built-in testing loop dramatically speeds discovery.
- **Aligned Incentives**: Winner-heavy tracking with decay keeps focus on continuous, genuine improvement.

This combination of decentralized exploration, rigorous hidden testing, and compounding knowledge gives Carbon a structural edge in this still-early field.

---

## Competitive Positioning

The broader industry is moving rapidly toward **Software Defined Machines** and **Living Digital Twins** — where models serve as the single source of truth across design, embedded control, deployment, and ongoing operation, continuously refined by real-world data. Leading efforts such as JuliaHub’s Dyad platform are building modern acausal modeling environments, SciML-powered surrogates, generative AI assistance, and cloud-native workflows to make this vision practical for industrial engineering.

**Carbon occupies a distinct and complementary role**: while these platforms focus on making high-quality modeling and surrogate generation accessible and integrated, Carbon is the **decentralized discovery and robustness engine** that finds superior Neural Operator training methodologies, validates them under hidden adversarial stress with physics gates, and compounds that knowledge across the network via the Landscape Agent. This makes the surrogates produced for Software Defined Machines and Living Digital Twins more trustworthy, robust, and performant — especially for safety-critical and regulated applications.

In short: Dyad and similar platforms modernize the modeling *environment*; Carbon discovers the *best ways* to train and validate the physics-informed surrogates that power the next generation of digital twins.

---

## Current State & Roadmap

Phase 0 foundations (scoring, stress testing across all physics classes, determinism utilities, MCP basics, symbolic skeleton, full integration of stress into scoring) are advancing rapidly. Phase 1 adds customization, Abaqus ingestion, and deeper symbolic/causal capabilities. Phase 2 brings verified multi-physics with preCICE. Phase 3 expands to 3D and more advanced composition.

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

*Carbon is building the decentralized agentic infrastructure for trustworthy, compounding physical intelligence in engineering and science.*
