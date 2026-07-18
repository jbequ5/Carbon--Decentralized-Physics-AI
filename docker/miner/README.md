# Hydrogen Miner Docker Environment

SOTA-grade experience for agents and advanced users.

## Focused Mode (Recommended)

```bash
CHALLENGE_ID=poisson_2d_v1 docker compose up miner
```

Runs a focused loop on one challenge with:
- Priors loaded with system-controlled noise (anti-gaming)
- Internal testing iterations
- Intelligent final recommendations
- Returns the actual best strategy found

## Environment Variables

| Variable           | Default | Description |
|--------------------|---------|-------------|
| `CHALLENGE_ID`     | —       | Required for focused mode |
| `DRY_RUN`          | false   | Never submit (safe testing) |
| `ITERATIONS`       | 8       | Number of propose/validate cycles |
| `SUBMIT_THRESHOLD` | 0.075   | Score needed to submit |

## Output

At the end you get a structured JSON summary including:
- Best score
- Whether it submitted
- The actual best strategy found
- Recommended next actions

This is designed to be highly usable by autonomous agents.
