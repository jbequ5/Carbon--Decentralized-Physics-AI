# Hydrogen

**A Bittensor subnet for decentralized discovery of physics-informed neural operators.**

Hydrogen turns the problem of solving complex Partial Differential Equations (PDEs) into a competitive, self-improving knowledge economy. Miners compete by proposing better training strategies. Validators evaluate them under hidden stress conditions with real physics constraints. A causal reasoning layer (the Landscape Agent) identifies what actually works and distills it into reusable specialists.

## The Problem

Traditional scientific machine learning is centralized:
- One team trains one model on fixed benchmarks
- Easy to overfit to public data
- Physics constraints are often soft penalties rather than hard requirements
- Knowledge doesn't compound across teams

Hydrogen decentralizes this process using Bittensor incentives while enforcing **verifiable physics correctness**.

## How It Works

### 1. Miners Submit Strategies
Miners don't upload model weights. They submit **training strategies** — JSON configs that specify:
- Which neural operator backbone to use (FNO, DeepONet, PhysicsNeMo, etc.)
- Loss weight vectors
- Curriculum learning schedule
- Uncertainty quantification method

This lowers the barrier to entry dramatically. A clever researcher with a good idea can compete without needing a large GPU cluster.

### 2. Validators Run Team-Defined Evaluation Plans
Validators execute a clear evaluation pipeline defined by the team:

- Train on public benchmark data
- Run **hidden procedural stress tests** with hard physics gates (divergence-free, energy stability, boundary conditions, rollout stability)
- Score using real ground truth from benchmark hold-out sets

Miners have no visibility into the stress conditions or data splits used for evaluation. This prevents overfitting to public data.

### 3. The Landscape Agent (The Brain)
This is Hydrogen's most distinctive component.

The Landscape Agent continuously analyzes all submitted strategies using causal inference (Double Machine Learning + Heterogeneous Treatment Effects). It answers questions like:

> "Does increasing the PDE residual weight actually cause better stress performance — and under what conditions?"

It only proposes distillation when there is positive causal evidence of improvement. Successful strategies are distilled into compact, reusable **specialists** and stored in the Specialist Bank. Improved priors are automatically published back to miners.

### 4. Specialist Bank
High-quality distilled models are versioned, regression-tested, and made available with metadata (validity domains, performance metrics). This creates a growing library of composable, physics-aware components.

### 5. Emissions (75/25 Model)
- **75%** goes to Breakthrough Bounties — large rewards when someone sets a new record on hidden stress tests.
- **25%** goes to a Decaying Top-2 Stipend — only current leaders earn this, and it decays without continued improvement.

Strong anti-gaming measures are built in: hidden procedural stress, hard physics gates, adaptive difficulty with anti-sandbagging, and no miner control over evaluation data.

## Key Technical Features

- **Multi-backbone support**: Works with both NVIDIA PhysicsNeMo and the open-source NeuralOperator library (FNO, DeepONet, UNO).
- **Benchmark-driven training**: Real PDEBench data with fallback to synthetic.
- **Explicit evaluation plans**: Team-controlled routing for training, stress testing, and benchmarking.
- **Causal reasoning**: Double ML + CATE to understand what actually improves performance.
- **Automatic prior improvement**: The system generates better published priors after successful distillation.

## Current Status

Hydrogen has a working multi-backbone training pipeline, benchmark loading, explicit evaluation plans with adaptive stress testing, a Landscape Agent with causal inference, and a functional Specialist Bank. The emission mechanics and anti-gaming layers are implemented.

The system is designed so that **only verifiable, physics-correct generalization** reliably earns rewards.

## Getting Started

```bash
git clone https://github.com/jbequ5/Hydrogen.git
cd Hydrogen
pip install -e .
```

See `SPEC.md` for a more detailed technical specification of the current implementation.

## Philosophy

Hydrogen is built on the principle that the incentive structure *is* the product. By making hidden stress testing, hard physics gates, and causal reasoning the things that pay, we align decentralized participation with genuine scientific progress.
