# Hydrogen Miner Docker Environment

Clean, powerful, and agent-friendly.

## Focused Mode

```bash
CHALLENGE_ID=poisson_2d_v1 docker compose up miner
```

Runs a focused loop with:
- Challenge-specific priors (with system noise)
- Controllable iterations and threshold
- Returns the actual best strategy
- Gives intelligent recommendations

## Key Environment Variables

- `CHALLENGE_ID` — Focus on one challenge
- `DRY_RUN` — Safe testing mode
- `ITERATIONS` — Max cycles (default 8)
- `SUBMIT_THRESHOLD` — Score needed to submit

## Output

Structured JSON at the end including the best strategy and recommended next actions.
