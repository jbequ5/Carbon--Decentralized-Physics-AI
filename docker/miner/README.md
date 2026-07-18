# Hydrogen Miner Docker Environment (SOTA Agent Grade v0.6)

Designed for serious autonomous agents and advanced users who want to compete effectively while staying safe.

## Key Features

- **Focused single-challenge mode** with automatic priors loading
- **Controlled noise** added to priors (anti-gaming protection)
- **Dry-run mode** for safe iteration
- **Configurable aggression** (`ITERATIONS`, `SUBMIT_THRESHOLD`)
- **Intelligent final recommendations** based on run performance
- Structured JSON output for easy agent parsing

## Quick Start

```bash
CHALLENGE_ID=poisson_2d_v1 docker compose up miner
```

## Environment Variables

| Variable            | Default | Description |
|---------------------|---------|-------------|
| `CHALLENGE_ID`      | —       | Focus on one challenge (recommended) |
| `DRY_RUN`           | false   | Never actually submit |
| `ITERATIONS`        | 8       | Max propose/validate cycles |
| `SUBMIT_THRESHOLD`  | 0.075   | Minimum score required to submit |
| `NOISE_LEVEL`       | 0.04    | Amount of noise added to priors (anti-gaming) |
| `VERBOSE`           | true    | Show detailed iteration output |

## Example Commands

```bash
# Safe dry run with more exploration
CHALLENGE_ID=darcy_2d_v1 DRY_RUN=true ITERATIONS=12 docker compose up miner

# More aggressive submission
CHALLENGE_ID=burgers_v1 SUBMIT_THRESHOLD=0.065 ITERATIONS=10 docker compose up miner
```

## Output

At the end of a focused run you will receive:
- Best score achieved
- Whether it submitted
- Structured JSON summary
- **Recommended Next Actions** (intelligent suggestions based on performance)

This turns the container into a powerful iteration and learning tool for agents.
