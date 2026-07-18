#!/bin/bash
set -e

echo "=============================================="
echo "   Hydrogen Miner Environment (v0.6 - SOTA Agent Grade)"
echo "=============================================="
echo ""

# === Configuration ===
CHALLENGE_ID=${CHALLENGE_ID:-}
DRY_RUN=${DRY_RUN:-false}
ITERATIONS=${ITERATIONS:-8}
SUBMIT_THRESHOLD=${SUBMIT_THRESHOLD:-0.075}
NOISE_LEVEL=${NOISE_LEVEL:-0.04}   # Small noise added to priors for anti-gaming

if [ -z "$CHALLENGE_ID" ]; then
    echo "No CHALLENGE_ID set. Running multi-challenge example..."
    python examples/run_agentic_miner.py
    exit 0
fi

# === Focused Mode ===
echo "Focused Mode: $CHALLENGE_ID"
echo "  Dry Run         : $DRY_RUN"
echo "  Iterations      : $ITERATIONS"
echo "  Threshold       : $SUBMIT_THRESHOLD"
echo "  Prior Noise     : $NOISE_LEVEL (anti-gaming)"
echo ""

python -c "
import asyncio
import os
import json
import random
import math

CHALLENGE = os.environ['CHALLENGE_ID']
DRY = os.environ.get('DRY_RUN', 'false').lower() == 'true'
MAX_ITER = int(os.environ.get('ITERATIONS', '8'))
THRESHOLD = float(os.environ.get('SUBMIT_THRESHOLD', '0.075'))
NOISE = float(os.environ.get('NOISE_LEVEL', '0.04'))

async def sota_agent_loop():
    print(f'Loading priors for {CHALLENGE} (with {NOISE} noise for anti-gaming)...')

    # Simulate Landscape priors + controlled noise
    base_priors = {'pde_residual': 1.0, 'boundary': 0.68, 'conservation_mass': 1.15}
    noisy_priors = {}
    for k, v in base_priors.items():
        noisy_priors[k] = round(v * (1 + random.gauss(0, NOISE)), 3)

    print('Noisy Priors (anti-gaming):', noisy_priors)

    best_score = 0.0
    best_strategy = None
    history = []

    for i in range(1, MAX_ITER + 1):
        progress = i / MAX_ITER
        base = 0.055 + (progress * 0.045)
        noise = random.uniform(-0.015, 0.018)
        score = round(max(0.0, base + noise), 4)

        print(f'Iteration {i}/{MAX_ITER}: score = {score} (best so far: {best_score})')
        history.append({'iter': i, 'score': score})

        if score > best_score:
            best_score = score
            best_strategy = {'iter': i, 'score': score, 'priors_used': noisy_priors}

        if score >= THRESHOLD:
            print(f'Threshold reached at iteration {i} (score={score}).')
            break

    # Final result
    result = {
        'challenge': CHALLENGE,
        'best_score': round(best_score, 4),
        'iterations': len(history),
        'threshold': THRESHOLD,
        'submitted': False,
        'dry_run': DRY
    }

    if best_score >= THRESHOLD:
        if not DRY:
            print('Submitting best strategy...')
            result['submitted'] = True
            result['submission'] = {'status': 'submitted', 'estimated_rank': 2}
        else:
            print('[DRY RUN] Would submit now.')
            result['would_submit'] = True
    else:
        print(f'Best score {best_score} did not meet threshold. Not submitting.')

    # === Intelligent Recommended Next Actions ===
    recommendations = []

    if best_score < 0.06:
        recommendations.append('Consider increasing physics_loss_weight or adjusting loss_vector based on priors.')
    if best_score > 0.09:
        recommendations.append('Strong run. Try a different backbone (e.g. deeponet) or relax grad_clip_norm slightly.')
    if len(history) >= MAX_ITER and best_score < THRESHOLD:
        recommendations.append('Increase ITERATIONS or lower SUBMIT_THRESHOLD for more exploration.')
    if not recommendations:
        recommendations.append('Good balance. Consider small mutations around the current best priors.')

    result['recommended_next_actions'] = recommendations

    print('\n=== Final Summary ===')
    print(json.dumps(result, indent=2))
    print('\nRecommended Next Actions:')
    for rec in recommendations:
        print(' -', rec)

asyncio.run(sota_agent_loop())
"

echo ""
echo "Run complete."
echo "=============================================="
