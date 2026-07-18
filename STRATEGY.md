# STRATEGY.md — Miner Strategy Guide for Hydrogen

This guide explains how to write effective strategies for the Hydrogen subnet.

## What is a Strategy?

A strategy is a JSON configuration that tells the validator **how** to train your model. You do **not** upload model weights — only the training recipe.

The validator will:
1. Train a model using your strategy on public benchmark data.
2. Test it under hidden stress conditions with hard physics gates.
3. Score you based on improvement + physics correctness.

Your goal is to produce a strategy that generalizes well under unseen conditions while respecting physics.

---

## Strategy Fields Reference

### Core Fields

| Field              | Type     | Description                                      | Recommended Range / Notes                  |
|--------------------|----------|--------------------------------------------------|--------------------------------------------|
| `backbone`         | string   | Neural operator architecture                     | `fno`, `deeponet`, `uno`, `physicsnemo_fno` |
| `resolution`       | list     | Training resolution                              | Usually `[128, 128]` or challenge default  |

### Loss & Physics

| Field                    | Type   | Description                                      | Notes                                      |
|--------------------------|--------|--------------------------------------------------|--------------------------------------------|
| `pino.loss_vector`       | dict   | Weights for different loss terms                 | Use priors from `hydrogen-miner priors`    |
| `physics_loss_weight`    | float  | Overall multiplier for physics loss              | 0.5 – 2.0 (default ~1.0)                   |

### Optimizer & Training

| Field                | Type   | Description                           | Recommended                          |
|----------------------|--------|---------------------------------------|--------------------------------------|
| `optimizer`          | str    | Optimizer type                        | `AdamW` (default), `Adam`, `SGD`     |
| `learning_rate`      | float  | Learning rate                         | 1e-4 – 5e-3                          |
| `weight_decay`       | float  | Weight decay (L2 regularization)      | 1e-5 – 1e-3                          |
| `scheduler`          | str    | Learning rate scheduler               | `CosineAnnealingLR` (recommended)    |
| `batch_size`         | int    | Batch size                            | 4 – 16 (depends on GPU memory)       |
| `epochs`             | int    | Number of training epochs             | 80 – 200                             |

### Advanced Training Controls

| Field                    | Type    | Description                                      | Notes                                      |
|--------------------------|---------|--------------------------------------------------|--------------------------------------------|
| `weight_init`            | str     | Weight initialization method                     | `kaiming_normal`, `xavier_uniform`, etc.   |
| `grad_clip_norm`         | float   | Gradient clipping norm                           | 0.5 – 2.0 (or null to disable)             |
| `accumulation_steps`     | int     | Gradient accumulation steps                      | 1 – 8                                      |
| `use_amp`                | bool    | Use mixed precision (AMP)                        | `true` for speed on modern GPUs            |
| `early_stop_patience`    | int     | Early stopping patience (epochs)                 | 10 – 30 (or null to disable)               |

### Curriculum Learning

| Field                        | Type   | Description                              | Recommended                              |
|------------------------------|--------|------------------------------------------|------------------------------------------|
| `curriculum_learning.enabled`     | bool   | Enable curriculum                        | Usually `true`                           |
| `curriculum_learning.start_resolution` | list | Starting resolution for curriculum     | `[64, 64]` or lower                      |
| `curriculum_learning.ramp_epochs`     | int    | How many epochs to ramp resolution     | 30 – 60                                  |

### Uncertainty Quantification (UQ)

| Field                        | Type   | Description                              | Recommended                              |
|------------------------------|--------|------------------------------------------|------------------------------------------|
| `uq_config.enabled`          | bool   | Enable UQ                                | `true` for most challenges               |
| `uq_config.method`           | str    | UQ method                                | `deep_ensemble` (recommended)            |
| `uq_config.num_members`      | int    | Number of ensemble members               | 3 – 5                                    |
| `uq_config.calibration_target` | float | Target coverage for calibration        | 0.85 – 0.95                              |

### Model-Specific Parameters

Use `model_kwargs` to pass extra parameters to the backbone:

```json
"model_kwargs": {
  "modes": 32,
  "width": 64
}
```

---

## Example Strategies

### Basic Strategy (Good Starting Point)

```json
{
  "backbone": "fno",
  "optimizer": "AdamW",
  "learning_rate": 0.001,
  "weight_decay": 1e-4,
  "scheduler": "CosineAnnealingLR",
  "epochs": 100,
  "batch_size": 8,
  "use_amp": true,
  "curriculum_learning": {
    "enabled": true,
    "start_resolution": [64, 64],
    "ramp_epochs": 40
  }
}
```

### Advanced Strategy (More Control)

```json
{
  "backbone": "fno",
  "optimizer": "AdamW",
  "learning_rate": 0.0008,
  "weight_decay": 1e-4,
  "weight_init": "kaiming_normal",
  "grad_clip_norm": 1.0,
  "accumulation_steps": 4,
  "use_amp": true,
  "physics_loss_weight": 0.9,
  "early_stop_patience": 20,
  "scheduler": "CosineAnnealingLR",
  "curriculum_learning": {
    "enabled": true,
    "start_resolution": [64, 64],
    "ramp_epochs": 50
  },
  "uq_config": {
    "enabled": true,
    "method": "deep_ensemble",
    "num_members": 4
  },
  "model_kwargs": {
    "modes": 32,
    "width": 64
  }
}
```

---

## Tips for Writing Good Strategies

1. **Start with the priors**
   - Run `hydrogen-miner priors <challenge_id>` to get suggested loss weights and features.

2. **Use curriculum learning**
   - Almost always helpful for PDE problems. Start at lower resolution and ramp up.

3. **Don't over-tune early**
   - The Landscape Agent will learn which combinations actually work. Start reasonable, then iterate based on results.

4. **Pay attention to physics_loss_weight**
   - Too low → model may violate physics.
   - Too high → model may underfit the data.

5. **Mixed precision (AMP) is usually worth it**
   - Faster training with minimal accuracy loss on modern GPUs.

6. **Gradient clipping helps stability**
   - Especially useful for stiff PDEs (Navier-Stokes, Burgers).

---

## Common Pitfalls

- Setting learning rate too high → unstable training or NaNs.
- Disabling curriculum too early → poor convergence on fine resolution.
- Ignoring UQ calibration → lower score even if error looks good.
- Using very exotic optimizer settings without testing locally first.

---

## Local Validation (Recommended)

Before submitting (and paying the fee), test your strategy locally:

```bash
hydrogen-miner validate --challenge poisson_2d_v1 --strategy my_strategy.json
```

This runs a quick local evaluation so you can iterate without burning fees.

---

## Philosophy

The strategy is your **idea**. The more thoughtfully you design the training process (loss weighting, curriculum, regularization, optimization), the better your model will generalize under hidden stress.

The Landscape Agent exists to discover which ideas actually work across many miners. Your job is to propose good ones.
