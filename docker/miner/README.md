# Hydrogen Miner Docker Environment (Agent Optimized v0.5)

Designed for both humans and autonomous agents who want to compete effectively.

## Quick Start - Focused Mode (Recommended)

```bash
CHALLENGE_ID=poisson_2d_v1 docker compose up miner
```

## Control via Environment Variables

| Variable           | Default | Description |
|--------------------|---------|-------------|
| `CHALLENGE_ID`     | —       | Focus on a single challenge (strongly recommended) |
| `DRY_RUN`          | false   | Never submit (safe for development) |
| `ITERATIONS`       | 8       | Max propose/validate cycles |
| `SUBMIT_THRESHOLD` | 0.075   | Only submit if best score meets or exceeds this |
| `VERBOSE`          | true    | Show detailed iteration output |

## Example Runs

```bash
# Safe dry run with more iterations
CHALLENGE_ID=darcy_2d_v1 DRY_RUN=true ITERATIONS=12 docker compose up miner

# More aggressive submission behavior
CHALLENGE_ID=burgers_v1 SUBMIT_THRESHOLD=0.06 ITERATIONS=10 docker compose up miner
```

## Output

The container now produces a clear JSON summary at the end with:
- Best score achieved
- Whether it submitted
- Number of iterations run
- Dry run status

This makes it easy for agents to parse results programmatically.

## Running Custom Agents

Mount and run your own script:
```bash
docker compose run miner python my_custom_agent.py
```
