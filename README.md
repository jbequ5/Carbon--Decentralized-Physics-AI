# Hydrogen

**Decentralized Discovery of Physics-Informed Neural Operator Strategies**

Hydrogen is a Bittensor subnet (SN63) building a decentralized network for discovering high-quality, physics-respecting training strategies for complex engineering simulations.

At its core, Hydrogen combines **adversarial evaluation**, **strong physics constraints**, and **decentralized strategy iteration** to produce fast, robust neural operator surrogates that outperform both traditional high-fidelity solvers (in speed) and black-box ML models (in physical consistency).

---

## Vision

To create a living, evolving library of reusable, physics-grounded surrogate models and specialists that dramatically accelerate engineering design, digital twins, real-time simulation, and scientific discovery — built through decentralized, adversarial, physics-constrained competition.

Longer term, Hydrogen aims to contribute to **foundational models of physics** that generalize across domains (fluids, solids, electromagnetics, plasmas, quantum-informed problems, etc.) while maintaining rigorous physical consistency.

---

## Value Proposition

Traditional engineering simulation is slow and expensive for large design spaces or real-time use. Pure data-driven ML surrogates are fast but often lack physical trustworthiness.

Hydrogen produces **physics-informed neural operator surrogates** that are:

- **Fast** — orders of magnitude faster than traditional solvers once trained
- **Physically consistent** — trained and stress-tested against hard physics constraints
- **Robust** — evaluated under hidden stress conditions, making gaming and overfitting much harder
- **Discoverable at scale** — decentralized competition surfaces better strategies than any single lab could explore

This is particularly valuable for real-time digital twins, large-scale design optimization, HIL testing, and multi-physics problems.

---

## Core Design Principles

- Multi-objective physics-gated scoring (Physics Fidelity / Robustness / Accuracy)
- Per-challenge leader tracking with exponential decay (`ChallengeWinnerTracker`)
- Information asymmetry for anti-gaming (miners do not see hidden stress conditions or exact gate logic)
- Decentralized adversarial strategy discovery
- Future: Specialist distillation + verified multi-physics composition

---

## Current State (July 2026)

Hydrogen has completed foundational Phase 0 infrastructure:

- `ChallengeWinnerTracker` with exponential decay and participation dust
- `StrategyStore` abstraction
- Multi-objective `HydrogenScorer`
- Validator with dry-run mode
- MCP server with persistent sessions and streaming validation

Emissions currently follow standard Yuma Consensus. A hybrid model with Breakthrough Bounties and Decaying Stipends is planned but not yet active.

---

## Phased Roadmap

### Phase 0 (Current)
Core infrastructure: per-challenge tracking, multi-objective physics-gated scoring, strategy storage, validator, and MCP tooling.

### Phase 1: Customization & Data Ingestion
- Same 7 core PDE challenges
- Miners add LoRA adapters and custom datasets
- Abaqus ODB / .fil ingestion pipeline (miner submits `custom_data` with IPFS URI; validator parses and mixes with procedural data)
- Expanded symbolic regression track (PySR + DataDrivenDiffEq)

### Phase 2: Multi-Physics Composition
**Phase 2A** — Verified multi-physics benchmarks:
- FSI (Turek/Hron 2D)
- Conjugate Heat Transfer (CHT)

**Phase 2B** — Thermo-Elasticity (generate ~48 Tier-1 mesh-converged references)

**Phase 2C** — Variant expansion on FSI/CHT/Thermo-elasticity (new Re, geometries, coupling strengths)

Specialist pipelines + staggered coupling become first-class submission format.

### Phase 3: 3D Multi-Physics
- 3D FSI, 3D Thermo-Elasticity, 3D CHT with turbulence
- Specialist composition using preCICE / FEniCS / OpenFOAM references
- 3D-specific stress gates (energy spectrum, Q-criterion, wall shear, Nu distribution)
- Curriculum from 2D → 3D specialists

See `docs/FUTURE_DOMAINS.md` for the full domain expansion analysis (Electromagnetics, Photonics, Acoustics, Plasmas/Fusion, Quantum-informed, etc.).

---

## Documentation

| Document | Description |
|----------|-------------|
| `SPEC.md` | Full technical specification |
| `docs/FUTURE_DOMAINS.md` | Future domains, how they fit, why they matter, market context |
| `docs/VALIDATOR_GUIDE.md` | How to run and understand the validator |
| `docs/EMISSIONS.md` | Current emissions model |
| `docs/AGENT_USAGE.md` | MCP / Agent interaction guide |

---

## Getting Started

```bash
git clone https://github.com/jbequ5/Hydrogen.git
cd Hydrogen
pip install -r requirements.txt
python neurons/validator.py --dry_run true
```

See `docs/VALIDATOR_GUIDE.md` for details.

---

## Contributing

Contributions welcome in:
- New challenges and physics gates
- ChallengeWinnerTracker and scoring improvements
- Agent tooling and MCP enhancements
- Documentation

---

## License

MIT License
