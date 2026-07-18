# validator_private/

This directory is intended for **validator-only** logic and data.

## Purpose

Keep sensitive components here so they are not easily accessible to miners or public agents:

- Hidden stress test conditions and data
- Exact physics gate implementations and thresholds
- Internal scoring logic and weight calculation details
- Landscape Agent raw data and full learned model
- Any other validator-private configuration

## Recommended Structure

```
validator_private/
├── hidden_stress/          # Hidden evaluation data and conditions
├── physics_gates/          # Exact gate implementations
├── scoring_internal/       # Private scoring formulas and logic
├── landscape/              # Raw Landscape Agent data and model
└── config/                 # Validator-only configuration
```

## Notes

- This folder should **not** be imported by public miner/agent code.
- Keep the main `neurons/` and `hydrogen/` folders clean for public use.
- Update `SPEC.md` if you add new private components.
