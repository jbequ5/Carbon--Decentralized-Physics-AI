# Hydrogen Miner Docker Environment (Agent Optimized)

This environment is designed to be highly usable for both humans and autonomous agents.

## Quick Start (Focused Mode - Recommended)

```bash
CHALLENGE_ID=poisson_2d_v1 docker compose up miner
```

This will:
- Load priors for the chosen challenge
- Run an internal testing loop
- Submit only when the estimated score looks good

## Environment Variables (Control Behavior)

| Variable            | Default     | Description |
|---------------------|-------------|-------------|
| `CHALLENGE_ID`      | (none)      | Focus on one specific challenge (highly recommended) |
| `DRY_RUN`           | false       | If true, never actually submit (great for testing) |
| `ITERATIONS`        | 6           | Max number of propose/validate iterations |
| `SUBMIT_THRESHOLD`  | 0.07        | Only submit if estimated score is at or above this |
| `HYDROGEN_HOTKEY`   | (required)  | Your Bittensor hotkey |
| `HYDROGEN_WALLET`   | default     | Your wallet name |
| `HYDROGEN_API_KEY`  | (optional)  | API key if using MCP server with auth |

## Examples

```bash
# Focused run on one challenge
CHALLENGE_ID=darcy_2d_v1 docker compose up miner

# Dry run (test without submitting)
CHALLENGE_ID=burgers_v1 DRY_RUN=true docker compose up miner

# More aggressive iteration
CHALLENGE_ID=poisson_2d_v1 ITERATIONS=12 SUBMIT_THRESHOLD=0.08 docker compose up miner
```

## Running Custom Agents

```bash
docker compose run miner python my_custom_agent.py
```

## Tips for Agents

- Always start from priors when possible
- Use `DRY_RUN=true` while developing your agent logic
- Tune `ITERATIONS` and `SUBMIT_THRESHOLD` based on how aggressive you want to be
- Check your recent results after runs to improve future strategies
