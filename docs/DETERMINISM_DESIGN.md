# Determinism Design Specification

**Document:** `docs/DETERMINISM_DESIGN.md`
**Version:** 1.0
**Date:** July 2026
**Focus:** Full determinism guarantees across the Hydrogen evaluation pipeline

---

## 1. Purpose

Determinism is a foundational requirement for Phase 0. Without it, the system cannot credibly claim robustness, reproducibility, or auditability. Every evaluation run for the same submission on the same validator must produce identical results.

This document defines a rigorous, state-of-the-art approach to achieving **full pipeline determinism** in Hydrogen.

---

## 2. Scope

Determinism must cover the entire evaluation pipeline:

- Data loading and preprocessing
- Data augmentation (when used)
- Model training / fine-tuning
- Stress test generation and evaluation
- Physics gate computation
- Scoring and `ChallengeWinnerTracker` updates

Future extensions (preCICE coupling, Landscape Agent) must also respect determinism.

---

## 3. Sources of Non-Determinism

Modern ML frameworks and hardware introduce many sources of non-determinism:

| Source | Example | Impact |
|--------|---------|--------|
| PyTorch / JAX random ops | `torch.randn`, dropout, weight init | High |
| GPU operations | Atomic operations, cuDNN algorithms | High |
| Data loading order | Multiprocessing, shuffling | High |
| External libraries | NumPy, SciPy, custom CUDA kernels | Medium |
| System-level | CUDA version, cuDNN version, driver | Medium |
| Distributed training | NCCL, all-reduce order | High (if used) |

A SOTA determinism strategy must systematically close all these vectors.

---

## 4. Core Design Principles

| Principle | Requirement |
|-----------|-------------|
| **Hierarchical Seeding** | Every random operation must derive from a single, challenge-specific master seed. |
| **Framework-Level Control** | Use all available determinism flags in PyTorch/JAX. |
| **Container & Environment Pinning** | Pin CUDA, cuDNN, and library versions. |
| **Reproducibility Harness** | Provide a way to verify that a full run is deterministic. |
| **Auditability** | Any validator must be able to reproduce another validator’s results. |

---

## 5. Hierarchical Seeding Strategy (SOTA Approach)

We adopt a **hierarchical, challenge-bound PRNG key system** inspired by modern JAX practices and reproducible scientific computing.

### 5.1 Master Seed Derivation

```python
def get_master_seed(challenge_id: str, validator_hotkey: str) -> int:
    combined = f"{challenge_id}:{validator_hotkey}"
    return hash(combined) & 0xFFFFFFFF
```

### 5.2 Sub-Key Hierarchy

```python
master_seed = get_master_seed(challenge_id, validator_hotkey)

seeds = {
    "data_loading": hash(master_seed + "data"),
    "augmentation": hash(master_seed + "aug"),
    "training": hash(master_seed + "train"),
    "stress_generation": hash(master_seed + "stress"),
    "noise": hash(master_seed + "noise"),
}
```

Each subsystem receives its own sub-seed. This prevents cross-contamination and makes debugging easier.

### 5.3 JAX Best Practice (Recommended Long-Term)

For JAX-based backbones, use the modern PRNG key system:

```python
key = jax.random.PRNGKey(master_seed)
key, subkey = jax.random.split(key)
```

This is considered state-of-the-art for reproducible scientific ML.

### 5.4 PyTorch Determinism Flags

```python
import torch

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Note: `use_deterministic_algorithms(True)` can reduce performance. This is acceptable for evaluation determinism.

---

## 6. Data Loading & Augmentation Determinism

- Use `torch.Generator` with explicit seeds for all `DataLoader` shuffling.
- Fix multiprocessing worker seeds.
- Make augmentation pipelines (if used) fully deterministic by seeding every random transform.
- Prefer deterministic alternatives to non-deterministic operations (e.g., avoid `torch.rand` in favor of seeded generators).

---

## 7. Reproducibility Harness

We will implement a `ReproducibilityHarness` that can:

1. Run a full evaluation twice with the same inputs.
2. Compare all intermediate and final outputs (scores, stress results, gate outcomes).
3. Report any non-determinism with high precision.

This harness will be used both during development and for validator audits.

---

## 8. Environment & Container Pinning

For true cross-validator reproducibility:

- Use pinned Docker images with exact CUDA + cuDNN versions.
- Record full environment (Python, library versions) in every evaluation run.
- Consider using `conda-lock` or `pip freeze` + image digest for full provenance.

---

## 9. Integration with Stress Testing

The stress generation system (see `docs/STRESS_TEST_DESIGN.md`) must use the same master seed hierarchy. This ensures that stress conditions themselves are deterministic and auditable.

---

## 10. Success Criteria

| Criterion | Target |
|-----------|--------|
| Same submission, same validator → identical scores | 100% |
| Same submission, different validator (same image) → identical scores | 100% |
| Full pipeline reproducibility test passes | Yes |
| All random operations derive from challenge-specific seed | Yes |
| Environment is fully recorded per evaluation | Yes |

---

## 11. Phased Rollout

| Phase | Scope | Notes |
|-------|-------|-------|
| Phase 0 | Core pipeline (data, training, scoring, basic stress) | Must be solid |
| Phase 1 | Add preCICE / multi-physics runs | Higher complexity |
| Phase 2+ | Advanced adversarial stress generation | Requires careful seeding |

---

## 12. Summary

This design adopts modern best practices from reproducible scientific machine learning:

- Hierarchical challenge-bound seeding
- Framework-level determinism flags (PyTorch/JAX)
- Explicit reproducibility harness
- Environment pinning and provenance

Combined with the stress test design, this gives Hydrogen one of the strongest determinism and auditability guarantees among decentralized evaluation systems.

---

*This document should be treated as the authoritative reference for determinism in Hydrogen and updated as implementation evolves.*
