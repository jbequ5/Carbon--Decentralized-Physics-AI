# Hydrogen

**Decentralized Physics-Informed Neural Operator Subnet for Bittensor**

Hydrogen is a Bittensor subnet where miners compete to discover better physics-informed neural operators for solving Partial Differential Equations (PDEs). The system emphasizes **verifiable physics correctness** through hidden stress testing, hard physics gates, and causal reasoning.

## Current Architecture

```
Miners
  ↓ submit strategy JSON (backbone, loss weights, curriculum, UQ)
Validators
  ↓ run get_evaluation_plan() → train + hidden stress test + physics gates
Landscape Agent
  ↓ causal inference (Double ML + CATE) + novelty scoring
  ↓ distill top candidates → Specialist Bank
  ↓ auto-generate improved published priors
Specialist Bank
  ↓ versioned, validated specialists
```

## Key Components

| Component              | Status     | Description |
|------------------------|------------|-------------|
| **Backbone Registry**  | Complete   | Supports PhysicsNeMo + NeuralOperator models (FNO, DeepONet, UNO) |
| **Unified Trainer**    | Complete   | Multi-backbone training with validation loop |
| **Benchmark Loader**   | Complete   | PDEBench + fallback + NeuralOperator extensibility |
| **Evaluation Plans**   | Complete   | Team-controlled routing for train / stress / benchmark |
| **Landscape Agent**    | Advanced   | Causal inference, novelty-aware distillation, auto prior generation |
| **Specialist Bank**    | Functional | Register and query distilled specialists |
| **Emission Mechanics** | Complete   | 75% Breakthrough Bounties + 25% Decaying Top-2 Stipend |

## How It Works

1. **Miners** submit training strategies (not model weights).
2. **Validators** execute team-defined evaluation plans:
   - Train on benchmark data
   - Run hidden procedural stress tests with physics gates
   - Score using real ground truth from benchmark hold-outs
3. **Landscape Agent** uses causal inference to identify valuable strategies and distills them into reusable specialists.
4. Improved priors are automatically published back to miners.

Miners do **not** control data splits or stress conditions — these are defined by the team in the validator code.

## Anti-Gaming Features

- Hidden procedural stress tests
- Hard physics gates (divergence, energy stability, boundary conditions)
- Adaptive stress difficulty with anti-sandbagging (EMA + noise)
- Real ground truth from benchmark test splits
- No miner control over evaluation data

## Getting Started

```bash
# Clone
git clone https://github.com/jbequ5/Hydrogen.git
cd Hydrogen

# Install
pip install -e .

# Run validator (example)
python -m neurons.validator
```

## Project Structure

- `hydrogen/` — Core library (backbones, training, evaluation, landscape, specialist bank)
- `neurons/` — Bittensor validator/miner nodes
- `docker/` — Validator Docker images
- `scripts/` — Utility scripts

## Status

Hydrogen is in active development. The core evaluation pipeline, multi-backbone support, benchmark loading, and Landscape Agent with causal reasoning are functional.

## License

To be determined.
