# Carbon Subnet Data Management Specification

**Version**: 1.0  
**Status**: Implementation Specification  
**Classification**: Core Protocol — Security Critical  

---

## 1. Executive Summary

This document specifies Carbon's complete data management architecture. The central security invariant is **strict separation between training data (miner-influenced) and evaluation data (validator-controlled, hidden, physics-gated)**. This separation is the foundation of Carbon's trustless verification claim.

**Security Invariant**: *Miners optimize for training distribution. Validators evaluate on hidden, procedurally generated distribution with hard physics gates. These distributions are cryptographically separated by block-hash seeding.*

---

## 2. Core Security Architecture

### 2.1 Threat Model

| Adversary | Capability | Goal |
|-----------|------------|------|
| **Malicious Miner** | Controls training data distribution, submits custom datasets, chooses strategy | Pass evaluation without genuine physics compliance |
| **Colluding Validators** | Control evaluation seed, stress generation | Favor specific miners |
| **External Attacker** | Manipulates reference data, precomputed cache | Poison ground truth |

### 2.2 Security Boundaries

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TRUST BOUNDARIES                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  MINER REALM (Untrusted)              VALIDATOR REALM (Trusted)         │
│  ┌─────────────────────────┐          ┌─────────────────────────────┐  │
│  │ Strategy JSON           │          │ Challenge Spec (Immutable)  │  │
│  │ • data_generation params│          │ Generator Config (Frozen)   │  │
│  │ • custom_dataset ref    │          │ Gate Thresholds (Frozen)    │  │
│  │ • custom_dataset URI    │          │ Gate Logic (Immutable)      │  │
│  └───────────┬─────────────┘          └──────────────┬──────────────┘  │
│               │                                       │                │
│               │ Strategy JSON (v1.1)                  │                │
│               ▼                                       ▼                │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │              VALIDATOR EXECUTION ENVIRONMENT                     │  │
│  │  ┌─────────────────┐  ┌─────────────────────────────────────┐  │  │
│  │  │ TRAINING PIPE   │  │ EVALUATION PIPE (SEPARATE PROCESS)  │  │  │
│  │  │ • Miner params  │  │ • Validator-controlled generator    │  │  │
│  │  │ • Custom dataset│  │ • Hidden seed (block hash)          │  │
│  │  │ • Miner augment │  │ • Extended envelopes                │  │
│  │  │ • Miner curriculum    │ • Hard physics gates              │  │
│  │  └────────┬────────┘  └──────────────────┬──────────────────┘  │  │
│  │           │                                │                  │  │
│  │           ▼                                ▼                  │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │           SCORING ENGINE (Immutable)                     │  │  │
│  │  │  Physics Gates (Hard) → Score → Emissions               │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Critical Separation Invariants

| Invariant | Enforcement Mechanism | Violation Consequence |
|-----------|----------------------|----------------------|
| **Eval seed unknown to miners** | `seed = hash(challenge_id + block_hash + run_nonce)` | Miners cannot pre-compute eval distribution |
| **Eval generator immutable** | Generator config frozen in Challenge Spec (on-chain) | Validators cannot bias eval for specific miners |
| **Physics gates are hard** | Binary PASS/FAIL, zero score on failure | No gradient hacking possible |
| **Eval data never exposed** | Stress variants generated in-validator-memory only | Miners cannot train on eval distribution |
| **Training ≠ Evaluation distribution** | Extended envelopes for stress variants | Overfitting to training = failing eval |

---

## 3. Data Generation Architecture

### 3.1 Generator Taxonomy

```python
# carbon/generators/registry.py
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Callable
from abc import ABC, abstractmethod

class GeneratorType(Enum):
    """Generator execution mode"""
    ONLINE_JAX = "online_jax"           # JAX-FEM/JAX-RANS, differentiable
    ONLINE_HYBRID = "online_hybrid"     # Mesh online, solutions cached
    PRECOMPUTED = "precomputed"         # Fully cached (S3/GCS)
    SEQUENTIAL_FSI = "sequential_fsi"   # One-way coupling

@dataclass
class GeneratorConfig:
    """Immutable generator configuration (frozen in Challenge Spec)"""
    generator_type: GeneratorType
    challenge_id: str
    physics_class: str
    pde: str
    dimension: int
    parameter_ranges: Dict[str, tuple]      # Envelope for procedural generation
    reference_solver: str                   # Validation reference
    validation_tolerance: str               # e.g., "L2 < 2% vs SU2"
    precomputed_bucket: str = None          # S3/GCS bucket for precomputed
    precomputed_manifest: str = None        # Manifest hash for integrity

# Frozen at challenge registration — miners CANNOT modify
CHALLENGE_GENERATOR_CONFIGS = {
    "poisson-2d-v1": GeneratorConfig(
        generator_type=GeneratorType.ONLINE_JAX,
        challenge_id="poisson-2d-v1",
        physics_class="elliptic",
        pde="poisson",
        dimension=2,
        parameter_ranges={
            "coefficient_field": ("log_normal", {"mean": 0, "std": 0.5}),
            "source_field": ("gaussian_process", {"length_scale": 0.1}),
            "resolution": (64, 64),
            "n_samples": 1000,
        },
        reference_solver="FEniCS",
        validation_tolerance="L2 < 2% vs FEniCS",
    ),
    "naca0012_transonic-v1": GeneratorConfig(
        generator_type=GeneratorType.ONLINE_HYBRID,
        challenge_id="naca0012_transonic-v1",
        physics_class="compressible_ns",
        pde="compressible_navier_stokes",
        dimension=2,
        parameter_ranges={
            "mach": (0.7, 1.2),
            "reynolds": (1e6, 10e6),
            "angle_of_attack": (-2.0, 4.0),
        },
        reference_solver="SU2 v7.5.0",
        validation_tolerance="Cd < 3% vs SU2",
        precomputed_bucket="carbon-precomputed/naca0012",
    ),
    # ... all challenge configs
}
```

### 3.2 Generator Interface

```python
# carbon/generators/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
import jax.random as random
import jax.numpy as jnp

@dataclass
class GeneratedData:
    """Standardized data container"""
    coords: jnp.ndarray              # Spatial/temporal coordinates
    inputs: Dict[str, jnp.ndarray]   # Input fields (coeffs, BCs, params)
    targets: Dict[str, jnp.ndarray]  # Target fields (solutions)
    boundary_mask: jnp.ndarray       # Boundary condition mask
    metadata: Dict[str, Any]         # Generation metadata

class ProceduralGenerator(ABC):
    """Base class for all procedural generators"""
    
    def __init__(self, config: GeneratorConfig):
        self.config = config
        self._validate_config()
    
    @abstractmethod
    def generate_training_data(self, seed: int, n_samples: int) -> GeneratedData:
        """Generate training dataset (deterministic given seed)"""
        pass
    
    @abstractmethod
    def generate_stress_variants(self, seed: int, n_variants: int) -> GeneratedData:
        """Generate stress evaluation variants (extended envelopes)"""
        pass
    
    @abstractmethod
    def generate_benchmark_data(self, seed: int, n_samples: int) -> GeneratedData:
        """Generate held-out benchmark data"""
        pass
    
    def derive_seeds(self, master_seed: int) -> Dict[str, int]:
        """Hierarchical deterministic seed derivation"""
        keys = random.split(random.PRNGKey(master_seed), 5)
        return {
            "data": int(random.randint(keys[0], (), 0, 2**32)),
            "stress": int(random.randint(keys[1], (), 0, 2**32)),
            "init": int(random.randint(keys[2], (), 0, 2**32)),
            "dropout": int(random.randint(keys[3], (), 0, 2**32)),
            "shuffle": int(random.randint(keys[4], (), 0, 2**32)),
        }
    
    def _validate_config(self):
        """Validate generator config integrity"""
        # Check parameter ranges are valid
        # Check reference solver exists
        # Validate bucket/manifest if precomputed
        pass


# carbon/generators/poisson.py
class PoissonGenerator(ProceduralGenerator):
    """Poisson: -∇·(k∇u) = f"""
    
    def generate_training_data(self, seed: int, n_samples: int) -> GeneratedData:
        key = random.PRNGKey(seed)
        
        # Sample coefficient field k(x) ~ LogNormal
        key, k_key = random.split(key)
        log_k = random.normal(k_key, (n_samples, *self.resolution)) * 0.5
        k = jnp.exp(log_k)
        
        # Sample source field f(x) ~ Gaussian Process
        key, f_key = random.split(key)
        f = self._sample_gp(f_key, n_samples, length_scale=0.1)
        
        # Solve -∇·(k∇u) = f using JAX-FEM
        u = self._solve_poisson(k, f)
        
        return GeneratedData(
            coords=self.grid,
            inputs={"coefficient": k, "source": f},
            targets={"solution": u},
            boundary_mask=self.bc_mask,
            metadata={"pde": "poisson", "generator": "jax_fem"}
        )
    
    def generate_stress_variants(self, seed: int, n_variants: int) -> GeneratedData:
        """Extended envelope: higher contrast coefficients, sharper sources"""
        # Sample from EXTENDED envelope (beyond training distribution)
        pass


# carbon/generators/compressible_ns.py
class CompressibleNSGenerator(ProceduralGenerator):
    """Compressible Navier-Stokes (RANS)"""
    
    def generate_training_data(self, seed: int, n_samples: int) -> GeneratedData:
        # Hybrid: mesh online, solutions from precomputed cache
        pass
    
    def generate_stress_variants(self, seed: int, n_variants: int) -> GeneratedData:
        """Extended envelope: wider Mach/Re/AoA ranges, shock perturbations"""
        pass
```

---

## 4. Training vs Evaluation Data Separation

### 4.1 Seed Derivation & Separation

```python
# carbon/common/seeding.py
import hashlib
import jax.random as random

def derive_master_seed(challenge_id: str, block_hash: str, run_nonce: int) -> int:
    """Derive master seed from public unpredictable entropy"""
    seed_str = f"{challenge_id}:{block_hash}:{run_nonce}"
    return int(hashlib.sha256(seed_str.encode()).hexdigest()[:16], 16)


def derive_pipeline_seeds(master_seed: int) -> Dict[str, int]:
    """Derive pipeline-specific seeds from master seed"""
    keys = random.split(random.PRNGKey(master_seed), 5)
    return {
        "data_seed": int(random.randint(random.PRNGKey(master_seed), (), 0, 2**32)),
        "stress_seed": int(random.randint(random.PRNGKey(master_seed), (), 0, 2**32)),
        "init_seed": int(random.randint(random.PRNGKey(master_seed), (), 0, 2**32)),
        "dropout_seed": int(random.randint(random.PRNGKey(master_seed), (), 0, 2**32)),
        "shuffle_seed": int(random.randint(random.PRNGKey(master_seed), (), 0, 2**32)),
    }


# In validator evaluation loop:
def run_evaluation(submission: StrategySubmission, block_hash: str, run_nonce: int):
    # 1. Derive master seed from public entropy
    master_seed = derive_master_seed(
        submission.challenge_id, 
        block_hash, 
        run_nonce
    )
    
    # 2. Derive pipeline seeds
    seeds = derive_pipeline_seeds(master_seed)
    
    # 3. TRAINING PIPELINE (miner-influenced)
    train_seed = seeds["data_seed"]
    init_seed = seeds["init_seed"]
    dropout_seed = seeds["dropout_seed"]
    shuffle_seed = seeds["shuffle_seed"]
    
    # Generate training data using MINER'S generator params
    train_data = generate_training_data_with_miner_params(
        challenge_id=submission.challenge_id,
        seed=train_seed,
        n_samples=challenge.config.n_training_samples,
        miner_params=submission.strategy.data_generation.generator_params
        if submission.strategy.data_generation else None
    )
    
    # Split train/val
    train_data, val_data = split_data(train_data, seeds["shuffle_seed"])
    
    # Train model
    model = train_model(
        strategy=submission.strategy,
        train_data=train_data,
        val_data=val_data,
        init_seed=seeds["init_seed"],
        dropout_seed=seeds["dropout_seed"]
    )
    
    # 4. EVALUATION PIPELINE (VALIDATOR-CONTROLLED, HIDDEN)
    stress_seed = seeds["stress_seed"]  # MINER NEVER SEES THIS
    
    stress_data = generate_stress_variants(
        challenge_id=submission.challenge_id,
        seed=stress_seed,                    # UNKNOWN to miner
        n_variants=challenge.config.n_stress_variants,
        extended_envelope=True              # EXTENDED envelopes
    )
    
    # Run physics gates
    gate_results = run_physics_gates(model, stress_data)
    
    return EvaluationResult(gate_results, stress_data)
```

### 4.2 Training vs Evaluation Distribution Separation

| Aspect | Training Distribution | Evaluation Distribution |
|--------|----------------------|------------------------|
| **Seed** | `data_seed` (derived from master) | `stress_seed` (different derivation path) |
| **Generator** | Miner params + challenge config | **Validator config ONLY** (miner params ignored) |
| **Envelope** | Training envelope (challenge.config) | **Extended envelope** (wider ranges) |
| **Perturbations** | Miner-defined augmentation | Physics-gated perturbations (shock, separation, etc.) |
| **Perturbation Magnitude** | Miner-controlled | Fixed by challenge spec (gate thresholds) |
| **Seed Visibility** | Known to miner (in strategy) | **Never exposed to miner** |

```python
# carbon/generators/envelopes.py

# TRAINING ENVELOPE (from challenge spec)
TRAINING_ENVELOPES = {
    "naca0012_transonic-v1": {
        "mach": (0.7, 1.2),
        "reynolds": (1e6, 10e6),
        "angle_of_attack": (-2.0, 4.0),
    },
    "hifire1_forebody-v1": {
        "mach": (5.0, 8.0),
        "reynolds": (1e5, 1e6),
        "wall_temperature": (300, 2000),
    },
}

# EVALUATION ENVELOPE (Extended for stress testing)
STRESS_ENVELOPES = {
    "naca0012_transonic-v1": {
        "mach": (0.5, 1.5),           # EXTENDED: wider range
        "reynolds": (0.5e6, 20e6),    # EXTENDED: wider range
        "angle_of_attack": (-5.0, 8.0), # EXTENDED
        "shock_perturbation": 0.1,     # ADDED: shock perturbations
        "boundary_layer_trip": 0.05,   # ADDED: BL trip
    },
    "hifire1_forebody-v1": {
        "mach": (4.0, 9.0),           # EXTENDED
        "reynolds": (0.5e5, 2e6),     # EXTENDED
        "wall_temperature": (200, 2500), # EXTENDED
        "chemistry_perturbation": 0.15, # ADDED
    },
}
```

### 4.3 Custom Dataset Handling (Miner-Provided Data)

```python
# carbon/validator/custom_dataset.py

class CustomDatasetValidator:
    """Validates miner-provided datasets before training"""
    
    @staticmethod
    def validate(dataset_uri: str, challenge: Challenge) -> ValidationResult:
        """Validate miner-provided dataset before training"""
        
        # 1. Load dataset
        dataset = load_dataset(dataset_uri)
        
        # 2. Basic sanity checks
        checks = [
            ("mesh_quality", check_mesh_quality(dataset.mesh)),
            ("field_consistency", check_field_consistency(dataset)),
            ("boundary_completeness", check_boundary_conditions(dataset)),
            ("physics_consistency", check_physics_consistency(dataset, challenge)),
        ]
        
        for name, result in checks:
            if not result.passed:
                return ValidationResult(False, f"Failed {name}: {result.message}")
        
        # 2. Reference solver validation (sample)
        if challenge.config.get("validate_custom_dataset", True):
            ref_result = validate_against_reference_solver(
                dataset, 
                challenge.config.reference_solver,
                sample_size=min(10, len(dataset))
            )
            if not ref_result.passed:
                return ValidationResult(False, f"Reference validation failed: {ref_result.message}")
        
        # 3. Physics consistency check
        physics_check = validate_physics_consistency(dataset, challenge)
        if not physics_check.passed:
            return ValidationResult(False, f"Physics violation: {physics_check.message}")
        
        return ValidationResult(True, "Custom dataset validated")


# In validator pipeline:
def prepare_training_data(strategy: Strategy, challenge: Challenge, seed: int):
    if strategy.custom_dataset:
        # Validate first
        validation = CustomDatasetValidator.validate(strategy.custom_dataset.uri, challenge)
        if not validation.passed:
            raise ValidationError(f"Custom dataset rejected: {validation.message}")
        
        # Load and merge with procedural data
        custom_data = load_custom_dataset(strategy.custom_dataset.uri)
        procedural_data = generator.generate_training_data(seed, n_samples)
        merged_data = merge_datasets(procedural_data, custom_data, strategy.custom_dataset)
        return merged_data
    else:
        return generator.generate_training_data(seed, n_samples)
```

---

## 5. Stress Data / Test Generation

### 5.1 Stress Variant Categories

```python
# carbon/generators/stress_categories.py
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any

class StressCategory(Enum):
    """Categories of stress variants"""
    EXTENDED_ENVELOPE = "extended_envelope"       # Wider parameter ranges
    SHOCK_PERTURBATION = "shock_perturbation"     # Shock capturing
    BOUNDARY_LAYER_TRIP = "boundary_layer_trip"   # BL transition
    SEPARATION_TRIGGER = "separation_trigger"     # Flow separation
    CHEMISTRY_PERTURBATION = "chemistry_perturbation"  # Reacting flow
    MESH_PERTURBATION = "mesh_perturbation"       # Geometry variation
    BOUNDARY_CONDITION = "boundary_condition"     # BC variation
    INITIAL_CONDITION = "initial_condition"       # IC perturbation
    COUPLING_PERTURBATION = "coupling_perturbation"  # Multi-physics coupling

@dataclass
class StressVariantSpec:
    category: StressCategory
    weight: float                    # Probability weight
    params: Dict[str, Any]           # Category-specific parameters
    physics_gates: List[str]         # Gates this variant tests

STRESS_VARIANT_SPECS = {
    "naca0012_transonic-v1": [
        StressVariantSpec(
            category=StressCategory.EXTENDED_ENVELOPE,
            weight=0.30,
            params={"mach_range": (0.5, 1.5), "reynolds_range": (0.5e6, 20e6)},
            physics_gates=["mass_conservation", "energy_stability", "shock_capture"]
        ),
        StressVariantSpec(
            category=StressCategory.SHOCK_PERTURBATION,
            weight=0.20,
            params={"shock_strength": (0.05, 0.15), "position_perturbation": 0.05},
            physics_gates=["shock_capture", "mass_conservation"]
        ),
        StressVariantSpec(
            category=StressCategory.BOUNDARY_LAYER_TRIP,
            weight=0.15,
            params={"trip_location": (0.3, 0.7), "trip_height": (0.01, 0.05)},
            physics_gates=["boundary_satisfaction", "separation_capture"]
        ),
        StressVariantSpec(
            category=StressCategory.SEPARATION_TRIGGER,
            weight=0.15,
            params={"adverse_pressure_gradient": (1.5, 3.0)},
            physics_gates=["separation_capture", "rollout_stability"]
        ),
        StressVariantSpec(
            category=StressCategory.MESH_PERTURBATION,
            weight=0.10,
            params={"perturbation_magnitude": 0.005, "frequency": (1, 5)},
            physics_gates=["boundary_satisfaction", "mass_conservation"]
        ),
        StressVariantSpec(
            category=StressCategory.BOUNDARY_CONDITION,
            weight=0.10,
            params={"wall_temp_factor": (0.5, 2.0), "catalytic_factor": (0.1, 10.0)},
            physics_gates=["thermal_protection", "energy_stability"]
        ),
    ],
}
```

### 5.2 Stress Generator Implementation

```python
# carbon/generators/stress.py

class StressGenerator(ProceduralGenerator):
    """Generates hidden stress variants for evaluation"""
    
    def generate_stress_variants(self, seed: int, n_variants: int) -> GeneratedData:
        """Generate hidden stress variants for evaluation"""
        key = random.PRNGKey(seed)
        specs = STRESS_VARIANT_SPECS.get(self.config.challenge_id, [])
        
        # Normalize weights
        total_weight = sum(s.weight for s in specs)
        probs = [s.weight / total_weight for s in specs]
        
        variants = []
        for i in range(n_variants):
            key, var_key = random.split(key)
            
            # Select category by weight
            cat_idx = random.choice(var_key, len(specs), p=jnp.array([s.weight for s in specs]))
            spec = specs[cat_idx]
            
            # Generate variant
            variant = self._generate_variant(var_key, spec)
            variants.append(variant)
        
        return self._collate_variants(variants)
    
    def _generate_variant(self, key: int, spec: StressVariantSpec) -> GeneratedData:
        """Generate single stress variant per category"""
        handlers = {
            StressCategory.EXTENDED_ENVELOPE: self._gen_extended_envelope,
            StressCategory.SHOCK_PERTURBATION: self._gen_shock_perturbation,
            StressCategory.BOUNDARY_LAYER_TRIP: self._gen_bl_trip,
            StressCategory.SEPARATION_TRIGGER: self._gen_separation_trigger,
            StressCategory.MESH_PERTURBATION: self._gen_mesh_perturbation,
            StressCategory.BOUNDARY_CONDITION: self._gen_boundary_condition,
            StressCategory.CHEMISTRY_PERTURBATION: self._gen_chemistry_perturbation,
            StressCategory.COUPLING_PERTURBATION: self._gen_coupling_perturbation,
        }
        return handlers[spec.category](key, spec.params)
    
    def _gen_extended_envelope(self, key, params) -> GeneratedData:
        """Sample from extended parameter envelope"""
        # Sample from EXTENDED ranges (beyond training)
        pass
    
    def _gen_shock_perturbation(self, key, params) -> GeneratedData:
        """Perturb shock position/strength"""
        pass
    
    def _gen_bl_trip(self, key, params) -> GeneratedData:
        """Add boundary layer trip"""
        pass
    
    def _gen_separation_trigger(self, key, params) -> GeneratedData:
        """Create adverse pressure gradient for separation"""
        pass
```

### 5.3 Stress Variant Distribution & Coverage

```python
# carbon/generators/stress_coverage.py

def validate_stress_coverage(stress_data: GeneratedData, challenge_id: str) -> CoverageReport:
    """Validate stress variants cover required physics regimes"""
    
    specs = STRESS_VARIANT_SPECS[challenge_id]
    required_categories = set(s.category for s in STRESS_VARIANT_SPECS[challenge_id])
    present_categories = set()
    
    for variant in stress_data.variants:
        # Classify variant (could be stored in metadata)
        cat = classify_variant(variant)
        present_categories.add(cat)
    
    missing = required_categories - present_categories
    
    return CoverageReport(
        challenge_id=challenge_id,
        required_categories=required_categories,
        present_categories=present_categories,
        missing_categories=missing,
        coverage_pct=len(present_categories) / len(required_categories) * 100
    )

# In validator: reject evaluation if coverage < 95%
def evaluate_with_stress(model, challenge_id: str, stress_seed: int):
    stress_data = stress_generator.generate_stress_variants(stress_seed, 60)
    
    coverage = validate_stress_coverage(stress_data, challenge_id)
    if coverage.coverage_pct < 95.0:
        raise EvaluationError(f"Insufficient stress coverage: {coverage.coverage_pct}%")
    
    return run_physics_gates(model, stress_data)
```

---

## 6. Miner Controls (v1.1 Strategy Schema)

### 6.1 Complete Data Generation Control Surface (v1.1)

```json
{
  "schema_version": "1.1",
  "challenge_id": "naca0012_transonic-v1",
  "backbone": "gino",
  "backbone_config": { ... },
  "training": { ... },
  "loss": { ... },
  "curriculum": [ ... ],
  
  "data_generation": {
    "generator_params": {
      "mach_distribution": "beta(2, 2)",
      "reynolds_distribution": "log_uniform([1e6, 10e6])",
      "aoa_distribution": "uniform([-5, 10])",
      "turbulence_ic_perturbation": "log_normal(0, 0.3)",
      "mesh_perturbation_std": 0.001
    },
    "augmentation_policy": {
      "rotation": true,
      "scaling": true,
      "reflection": false,
      "physics_informed_noise": true,
      "noise_level": 0.01
    },
    "curriculum": {
      "phase_1": {"mach_range": [0.7, 0.9], "epochs": 100},
      "phase_2": {"mach_range": [0.9, 1.1], "epochs": 150},
      "phase_3": {"mach_range": [1.1, 1.2], "epochs": 200}
    },
    "synthetic_ratio": 0.8
  },
  
  "custom_dataset": {
    "source": "abaqus_odb",
    "storage_uri": "s3://bucket/case.odb",
    "field_mapping": {
      "displacement": "U",
      "stress": "S", 
      "strain": "E"
    },
    "mesh_handling": "interpolate_to_uniform"
  }
}
```

### 6.2 Miner Control Surface Summary

| Control | Scope | Validation | Security Impact |
|---------|-------|------------|-----------------|
| `generator_params` | Distribution parameters within challenge envelope | **Entropy floor enforced** (min entropy per param) | Low — affects training only |
| `augmentation_policy` | Augmentation types & magnitudes | Max magnitude capped | Low — training only |
| `curriculum` | Phase definitions, epoch allocation | Total epochs ≤ max_epochs | Low — training schedule only |
| `synthetic_ratio` | [0.0, 1.0] | Clamped to [0, 1] | Low — training mix only |
| `custom_dataset` | External data (Abaqus ODB, etc.) | **Full validation pipeline** (ref solver, physics check) | Medium — validated before use |

### 6.3 Entropy Floor Enforcement (Anti-Gaming)

```python
# carbon/validation/entropy_floor.py

MIN_ENTROPY_PER_PARAM = {
    "mach_distribution": 0.5,           # bits
    "reynolds_distribution": 0.5,
    "aoa_distribution": 0.3,
    "turbulence_ic_perturbation": 0.4,
    "mesh_perturbation_std": 0.2,
    "wall_temp_distribution": 0.4,
    "chemistry_perturbation": 0.3,
}

def validate_generator_params(params: Dict[str, Any]) -> ValidationResult:
    """Enforce minimum entropy on miner generator params"""
    
    for param_name, min_entropy in MIN_ENTROPY_PER_PARAM.items():
        if param_name not in params:
            continue
            
        param_config = params[param_name]
        entropy = compute_distribution_entropy(param_config)
        
        if entropy < min_entropy:
            return ValidationResult(
                passed=False,
                message=f"Parameter '{param_name}' entropy {entropy:.3f} < "
                        f"minimum {min_entropy}. Miner must cover full physics envelope."
            )
    
    return ValidationResult(passed=True)


def compute_distribution_entropy(param_config: Any) -> float:
    """Compute Shannon entropy of distribution parameter"""
    if isinstance(param_config, str):
        # Parse distribution string: "uniform(a,b)", "beta(a,b)", etc.
        dist_type, params = parse_distribution_string(param_config)
        return analytical_entropy(dist_type, params)
    elif isinstance(param_config, dict):
        # Discrete distribution
        probs = np.array(list(param_config.values()))
        probs = probs / probs.sum()
        return float(-np.sum(probs * np.log(probs + 1e-12)))
    else:
        return 0.0
```

---

## 7. Data Credibility & Verification

### 7.1 Generator Validation Pipeline

```python
# carbon/validation/generator_validation.py

class GeneratorValidator:
    """Validates procedural generators against reference solvers"""
    
    @staticmethod
    def validate_generator(
        generator: ProceduralGenerator,
        challenge: Challenge,
        n_validation_cases: int = 50
    ) -> GeneratorValidationReport:
        """Comprehensive generator validation"""
        
        results = {
            "accuracy": [],
            "conservation": [],
            "stability": [],
            "mesh_convergence": [],
            "temporal_convergence": [],
        }
        
        for i in range(n_validation_cases):
            seed = hash(f"{challenge.challenge_id}:validation:{i}")
            
            # Generate test case
            test_data = generator.generate_benchmark_data(seed, n_samples=1)
            
            # Run reference solver
            ref_solution = run_reference_solver(
                challenge.config.reference_solver,
                test_data.inputs
            )
            
            # Compare
            error = compute_relative_error(
                test_data.targets, 
                ref_solution
            )
            
            results["accuracy"].append(error)
            
            # Check conservation
            cons_error = check_conservation_laws(test_data.targets, challenge.pde)
            results["conservation"].append(cons_error)
            
            # Check stability
            results["stability"].append(check_stability(test_data.targets))
        
        # Mesh convergence study (3 levels)
        mesh_conv = run_mesh_convergence_study(generator, challenge)
        
        # Temporal convergence study (3 levels)
        temp_conv = run_temporal_convergence_study(generator, challenge)
        
        return GeneratorValidationReport(
            challenge_id=challenge.challenge_id,
            generator_version=generator.config.version,
            accuracy_stats=compute_stats(results["accuracy"]),
            conservation_stats=compute_stats(results["conservation"]),
            mesh_convergence=mesh_conv,
            temporal_convergence=temp_conv,
            passed=all_passed(results, mesh_conv, temp_conv)
        )
```

### 7.2 Reference Solver Validation Matrix

| Challenge | Reference Solver | Validation Cases | Tolerance |
|-----------|-----------------|------------------|-----------|
| Poisson | FEniCS | 50 | L2 < 2% |
| Darcy | FEniCS | 50 | L2 < 2% |
| Burgers | Analytical / FEniCS | 50 | L2 < 3% |
| NS Laminar | OpenFOAM (icoFoam) | 30 | L2 < 3% |
| Heat | FEniCS | 50 | L2 < 2% |
| Elasticity | FEniCS / CalculiX | 30 | L2 < 3% |
| Thermo-Elasticity | CalculiX | 20 | L2 < 5% |
| NACA 0012 | SU2 v7.5.0 | 50 | Cd < 3%, Cl < 5% |
| CRM Wing-Body | OpenFOAM (simpleFoam) | 30 | Cd < 3%, Cl < 5% |
| HIFiRE-1 | DPLR / US3D | 20 | Heat flux < 10% |
| FSI 3D | preCICE + OF + CalculiX | 15 | Disp < 5%, Drag < 3% |
| Store Separation | OVERFLOW + 6DOF | 10 | Trajectory < 0.5m |
| Turbine CHT | ANSYS Fluent / OF | 20 | Eta < 0.05 RMS |

### 7.3 Continuous Validation (Runtime)

```python
# carbon/validator/runtime_validation.py

class RuntimeValidator:
    """Validates generator output during live evaluation"""
    
    @staticmethod
    def validate_training_batch(batch: GeneratedData, challenge: Challenge) -> bool:
        """Quick validation on each training batch"""
        checks = [
            # NaN/Inf check
            lambda d: not jnp.any(jnp.isnan(d.targets["solution"])),
            lambda d: not jnp.any(jnp.isinf(d.targets["solution"])),
            # Range check
            lambda d: jnp.all(d.targets["solution"] > -1e6),
            lambda d: jnp.all(d.targets["solution"] < 1e6),
            # Conservation spot-check (sample 10%)
            lambda d: check_conservation_sample(d, challenge, sample_pct=0.1),
        ]
        return all(c(batch) for c in checks)
    
    @staticmethod
    def validate_stress_batch(stress_data: GeneratedData, challenge: Challenge) -> bool:
        """Validate stress variants before gate evaluation"""
        # Check all variants present
        coverage = validate_stress_coverage(stress_data, challenge.challenge_id)
        if coverage.coverage_pct < 95.0:
            return False
        
        # Check no NaN/Inf
        for variant in stress_data.variants:
            if jnp.any(jnp.isnan(variant.targets["solution"])):
                return False
        
        return True
```

---

## 8. Security Analysis: Miner-Controlled Training Data

### 8.1 Attack Surface Matrix

| Attack Vector | Vector Description | Feasibility | Impact | Mitigation |
|---------------|-------------------|-------------|--------|------------|
| **Train on eval distribution** | Miner guesses eval seed/distribution | **Impossible** | N/A | Eval seed = block hash (unpredictable) |
| **Overfit to physics gates** | Train to pass specific gate-specific data to pass gates | **Impossible** | N/A | Gates = physics laws, not patterns |
| **Data poisoning (custom)** | Malicious Abaqus ODB | **Detected** | Medium | Validator validates vs ref solver |
| **Generator param overfitting** | Tune params to current stress dist | **Possible** | Medium | **Self-correcting** (see below) |
| **Curriculum gaming** | Easy curriculum → high score | Low | Low | Eval ignores curriculum |
| **Degenerate generator** | Collapse to single point | Medium | Medium | **Entropy floor enforced** |
| **Augmentation overfitting** | Augment to create "easy" samples | Low | Low | Gates test unaugmented physics |

### 8.2 Generator Param Overfitting: The Only Real Vector

```python
# The attack:
# 1. Miner observes current stress variants cluster around Mach 0.9
# 2. Miner sets mach_distribution = "uniform(0.85, 0.95)"
# 3. Model trains on narrow Mach range
# 4. Next eval: new stress variants cover full [0.5, 1.5] range
# 5. Model FAILS on Mach 0.5 and 1.5 (never trained there)

# Why this self-corrects:
# 1. Miner gets diagnostics: "Failed gate 'shock_capture' at Mach=1.5"
# 2. Miner sees: "Operational envelope: Mach [0.5, 1.5]"
# 3. Miner expands: mach_distribution = "uniform(0.7, 1.3)"
# 4. Next eval: PASSES

# This is a FEATURE: System trains miners to cover FULL physics envelope
```

### 8.3 Entropy Floor: Preventing Degenerate Distributions

```python
# Enforced at strategy submission validation
def validate_strategy_submission(strategy: Strategy) -> ValidationResult:
    if strategy.data_generation and strategy.data_generation.generator_params:
        entropy_result = validate_generator_params(
            strategy.data_generation.generator_params
        )
        if not entropy_result.passed:
            return ValidationResult(
                passed=False,
                message=f"Generator params rejected: {entropy_result.message}"
            )
    return ValidationResult(passed=True)
```

---

## 9. Data Pipeline Implementation Checklist

### 9.1 Phase 0 Launch Requirements

- [ ] 7 PDE generators implemented (JAX-FEM)
- [ ] FEniCS validation harness (50 cases each)
- [ ] Mesh/temporal convergence framework
- [ ] 5 physics gates implemented (JAX)
- [ ] Seed derivation hierarchy implemented
- [ ] Stress generator with 5 base categories
- [ ] Model Card generator + Registry write
- [ ] Determinism harness (3-run verification)

### 9.2 Phase 1A Requirements

- [ ] 2 compressible NS generators (hybrid)
- [ ] Turbulence UQ framework (3 models)
- [ ] Shock capture gate
- [ ] Factory v1 (compressible NS only)
- [ ] Adjoint consistency gate
- [ ] SU2/OpenFOAM validation harness

### 9.2 Phase 1B Requirements

- [ ] 4 defense generators (reacting, FSI, 6-DOF, CHT)
- [ ] Chemistry UQ framework
- [ ] Sequential FSI generator
- [ ] Factory v1 hardened (all 6 physics)
- [ ] DPLR/US3D/OVERFLOW validation

### 9.3 Phase 2A Requirements

- [ ] Schema v1.1 (LoRA, custom_dataset, structured_losses, data_generation)
- [ ] Miner data_generation controls in SDK
- [ ] Entropy floor validation
- [ ] MT bridge production (PySR → MT → JAX)
- [ ] DML pipeline (JAX-native)
- [ ] Specialist Bank v1
- [ ] Air-Gapped Toolkit design

### 9.4 Phase 2B Requirements

- [ ] Air-Gapped Miner Toolkit v1
- [ ] Air-Gapped Validator v1
- [ ] preCICE sidecar architecture
- [ ] Sequential multi-physics ladder (6 steps)
- [ ] Coupling convergence framework

---

## 10. Appendix: Configuration Schemas

### 10.1 Challenge Spec (On-Chain / Registry)

```json
{
  "challenge_id": "string",
  "name": "string",
  "physics_class": "string",
  "pde": "string",
  "dimension": "2D|3D",
  "generator_config": {
    "generator_type": "online_jax|online_hybrid|precomputed",
    "parameter_ranges": {},
    "reference_solver": "string",
    "validation_tolerance": "string",
    "precomputed_bucket": "string",
    "precomputed_manifest_hash": "string"
  },
  "backbone_whitelist": ["fno", "gino", "wno"],
  "gate_thresholds": {},
  "scoring_weights": {"physics_fidelity": 45, "robustness": 30, "accuracy": 25},
  "n_training_samples": 1000,
  "n_stress_variants": 60,
  "stress_envelope_multiplier": 1.5,
  "deployment": {
    "generator_version": "v1.3.2",
    "validator_image": "carbon/validator:v1.3.2",
    "min_validators": 5
  }
}
```

### 10.2 Strategy Schema v1.1 (Complete)

```json
{
  "$schema": "https://carbonsubnet.org/schemas/strategy/v1.1.json",
  "type": "object",
  "required": ["challenge_id", "backbone", "backbone_config", "training", "loss"],
  "properties": {
    "schema_version": {"const": "1.1"},
    "challenge_id": {"type": "string"},
    "backbone": {"enum": ["fno", "gino", "wno", "transolver"]},
    "backbone_config": { "type": "object" },
    "training": { "type": "object" },
    "loss": { "type": "object" },
    "curriculum": { "type": "array" },
    "data_generation": {
      "type": "object",
      "properties": {
        "generator_params": { "type": "object" },
        "augmentation_policy": { "type": "object" },
        "curriculum": { "type": "object" },
        "synthetic_ratio": { "type": "number", "minimum": 0, "maximum": 1 }
      }
    },
    "custom_dataset": {
      "type": "object",
      "properties": {
        "source": {"enum": ["abaqus_odb", "ansys_rst", "openfoam", "custom"]},
        "storage_uri": {"type": "string", "format": "uri"},
        "field_mapping": {"type": "object"},
        "mesh_handling": {"enum": ["interpolate_to_uniform", "native"]}
      }
    }
  }
}
```

---

## 11. Sign-Off

This document defines the complete data management architecture for Carbon Subnet. All implementation must conform to the security invariants, separation boundaries, and validation requirements specified herein.

**Reviewed by**: 
- Physics Lead: ________________ Date: ________
- Security Lead: ________________ Date: ________
- Protocol Lead: ________________ Date: ________
- Implementation Lead: ________________ Date: ________

**Version History**:
- v1.0 (2026-07-24): Initial rigorous specification incorporating all security reviews
