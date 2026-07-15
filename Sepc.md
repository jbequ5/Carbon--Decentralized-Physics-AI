# SPEC.md — Hydrogen PDE Subnet Technical Specification v2.3 (Complete with Symbolic Layer + Agent-Native Architecture + Phase-Field Physics + JAX/PyTorch Interop + Abaqus Ingestion + Agent-Native Architecture)

---

## 1. System Overview

### 1.1 Core Loop
```
Open Challenge → Miner Strategy JSON → Validator Training → Physics-Gated Scoring → Landscape Causal Update → Next Baseline
```

### 1.2 Roles & Economics
| Role | Emission Share | Primary Function |
|------|----------------|------------------|
| **Top 4 Miners (per challenge)** | **41%** (split 40%/30%/20%/10% of challenge budget) | Highest verified improvement |
| Validators | 41% | Median consensus + physics-check completeness + UQ audit |
| Owner | 18% | Landscape Agent, Specialist Bank, Challenge Infra, Treasury |

**Emission Budget per Challenge:** `total_subnet_emission / number_of_active_challenges` (capped at 10 active challenges)

**Emission Distribution per Challenge (Top 4):**
| Rank | Share of Challenge Budget |
|------|---------------------------|
| 1st | 40% |
| 2nd | 30% |
| 3rd | 20% |
| 4th | 10% |
| 5+ | 0% |

**Novelty Bonus:** 5% of challenge budget awarded for strategies with high physics-informed embedding-space distance from recent winners, **only if improvement > 0**.

**Symbolic Bonus:** 2% of challenge budget awarded for strategies using auto-generated symbolic features that improve baseline by >5%.

**Bounty Accumulation:** Emissions accumulate until a submission meaningfully beats the current baseline (log-improvement > 0).

**Submission fee:** 0.1 TAO (burned if validation fails pre-checks; covers validator marginal cost ~0.08 TAO/run)

**Miner burn:** 0%

**Private-until-proven:** Strategies revealed only after earning rewards.

**Warm-up:** <10 distinct submissions/challenge → top 3 split 50/30/20 (only if improvement > 0)  
**Competitive:** Top-4 split as above (40/30/20/10) + novelty bonus (5%, improvement-gated) + symbolic bonus (2%, improvement >5%)

---

## 2. Miner Interface

### 2.1 Strategy JSON Schema (Phase 0-1)
```json
{
  "backbone": "PINO",
  "resolution": [128, 128, 64],
  "pino": {
    "loss_vector": {
      "pde_residual": 1.5,
      "conservation": 0.8,
      "boundary": 0.5,
      "symmetry": 0.3,
      "coupling_thermal_strain": 0.5,
      "coupling_heat_source": 0.3
    },
    "physics_loss_type": "pde_residual",
    "boundary_handling": "ghost_cells"
  },
  "optimizer": "AdamW",
  "learning_rate": 0.001,
  "scheduler": "CosineAnnealingLR",
  "batch_size": 16,
  "epochs": 200,
  "physics_informed": true,
  "curriculum_learning": {
    "enabled": true,
    "start_resolution": [64, 64, 32],
    "end_resolution": [128, 128, 64],
    "ramp_epochs": 80
  },
  "uq_config": {
    "method": "deep_ensemble",
    "num_members": 4,
    "calibration_target": 0.90
  },
  "custom_data": {
    "data_uri": "ipfs://...",
    "checksum": "sha256:...",
    "usage": "augment",
    "weight": 0.25
  },
  "auto_loss_weights": true,
  "symbolic_constraints": [
    "divergence_free: hard_constraint",
    "energy_dissipation: hard_constraint"
  ],
  // NEW: JAX/PyTorch backend selection
  "backend": "pytorch",  // or "jax"
  "jax_config": {
    "platform": "gpu",  // "gpu" | "tpu" | "cpu"
    "precision": "bf16",  // "bf16" | "fp32" | "fp64"
    "compile": true
  }
}
```

### 2.2 Specialist Pipeline JSON Schema (Phase 2+)
```json
{
  "specialist_pipeline": [
    {
      "specialist_id": "ns_2d_v4",
      "role": "primary",
      "inputs": ["forcing", "boundary_conditions", "material_props"]
    },
    {
      "specialist_id": "heat_2d_v3", 
      "role": "secondary",
      "inputs": ["heat_source", "boundary_conditions", "material_props.kappa"]
    },
    {
      "adapter_id": "cht_coupling_v2",
      "role": "coupling",
      "params": {"iterations": 5, "tolerance": 1e-4}
    }
  ],
  "execution_schedule": "staggered",
  "max_coupling_iterations": 5,
  "coupling_tolerance": 1e-4
}
```

### 2.3 Submission Rules
- **Fee:** 0.1 TAO (burned if validation fails pre-checks)
- **Format:** JSON only, ≤ 64 KB
- **Frequency:** 1 submission/miner/challenge (last submission counts)
- **Deadline:** 20 hours after challenge release (4-hour validation window)

---

## 3. Validator Specification

### 3.1 Backbone Images (Pre-built, Pinned) — **UPDATED WITH JAX SUPPORT**

| Image | Backbones | Framework | Base | PhysicsNeMo | NeuralOperator | PyTorch/JAX | CUDA |
|-------|-----------|-----------|------|-------------|----------------|-------------|------|
| `hydrogen/validator:fno-v24.09` | FNO | PyTorch | nvcr.io/nvidia/pytorch:24.09-py3 | 24.09 | 0.3.0 | 2.3.0 | 12.4 |
| `hydrogen/validator:pino-v24.09` | PINO | PyTorch | same | same | same | same | same |
| `hydrogen/validator:deeponet-v24.09` | DeepONet | PyTorch | same | same | same | same | same |
| `hydrogen/validator:gno-v24.09` | GNO | PyTorch | same | same | same | same | same |
| `hydrogen/validator:oformer-v24.09` | OFormer | PyTorch | same | same | same | same | same |
| **`hydrogen/validator:jax-fno-v24.09`** | **FNO** | **JAX (Equinox)** | **nvcr.io/nvidia/jax:24.09-py3** | **N/A** | **NeuralOperators.jl 0.3.0** | **JAX 0.4.30** | **12.4** |
| **`hydrogen/validator:jax-pino-v24.09`** | **PINO** | **JAX (Equinox)** | **same** | **N/A** | **NeuralOperators.jl 0.3.0** | **JAX 0.4.30** | **12.4** |
| **`hydrogen/validator:jax-deeponet-v24.09`** | **DeepONet** | **JAX (Equinox)** | **same** | **N/A** | **NeuralOperators.jl 0.3.0** | **JAX 0.4.30** | **12.4** |

**Hardware requirement:** 
- Phase 0-2: NVIDIA GPU ≥ 16 GB VRAM (RTX 3080/3090/4080/4090, A100)
- Phase 3 (3D): NVIDIA GPU ≥ 24 GB VRAM (RTX 3090/4090, A100 40GB, H100)

**Framework Selection:** Miners specify `"backend": "pytorch"` or `"jax"` in strategy JSON. Validator pulls corresponding image. JAX validators use `NeuralOperators.jl` (Julia) via Python-Julia bridge (`juliacall`) or native JAX (Equinox/Flax) implementations. Cross-framework validation via ONNX export/import.

### 3.2 Validation Pipeline (Deterministic)
```python
def validate_submission(challenge_id: str, miner_submission: dict) -> ValidationResult:
    # 1. Load challenge data
    train_data, holdout_data, stress_data = load_challenge_splits(challenge_id)
    challenge = load_challenge_metadata(challenge_id)
    
    # 2. Enrich challenge with symbolic metadata
    symbolic_metadata = enrich_challenge(challenge_id, challenge_data_path)
    
    # 3. Determine submission type and execute
    if "specialist_pipeline" in miner_submission:
        # Phase 2+: Composition track
        model = execute_specialist_pipeline(
            pipeline=miner_submission["specialist_pipeline"],
            challenge=challenge,
            train_data=train_data,
            seed=derive_seed(challenge_id, validator_hotkey)
        )
    else:
        # Phase 0-1: Strategy JSON
        # Detect framework from submission
        framework = miner_submission.get("backend", "pytorch")  # "pytorch" | "jax"
        model = train_backbone(
            image=select_backbone_image(miner_submission["backbone"], framework),
            config=miner_submission,
            train_data=train_data,
            custom_data=resolve_custom_data(miner_submission.get("custom_data")),
            seed=derive_seed(challenge_id, validator_hotkey)
        )
    
    # 4. Evaluate on public holdout
    E_baseline = load_current_baseline_error(challenge_id)
    E_submission = evaluate(model, holdout_data, challenge)
    improvement = log(E_baseline) - log(E_submission)
    
    # 5. Hidden stress test + physics gates
    stress_result = run_stress_test(model, stress_data, challenge)
    
    # 6. UQ calibration check
    uq_metrics = evaluate_uq(model, stress_data, miner_submission.get("uq_config", {}))
    
    # 7. Compute score
    if stress_result.hard_failure:
        score = 0.0
        reason = stress_result.failure_reason
    else:
        base_score = max(0.0, improvement)
        soft_penalty = stress_result.soft_penalty
        uq_bonus = uq_calibration_bonus(uq_metrics, challenge.phase)
        symbolic_bonus = 0.02 if symbolic_features_used and improvement > 0.05 else 0.0
        score = (base_score * soft_penalty) + uq_bonus + symbolic_bonus
    
    return ValidationResult(score, improvement, stress_result, uq_metrics)
```

### 3.3 Physics Gates (Stress Test) — **UPDATED WITH PHASE-FIELD GATES**

#### Hard Gates (Score = 0 if failed)
| Check | Formula | Threshold | PDE Classes |
|-------|---------|-----------|-------------|
| **Mass Conservation** | `‖∇·u‖₁ / ‖u‖₁` | `< 1e-3` | NS, Darcy, Elasticity |
| **Energy Dissipation** | `dE/dt` | `≤ 1e-4` | NS, Heat, Burgers |
| **Boundary Satisfaction** | `‖u - u_BC‖₂ / ‖u_BC‖₂` | `< 1e-3` | All with BCs |
| **Rollout Stability** | `‖E(t=T) - E(t=0)‖ / E(t=0)` | `< 0.01` | All transient |
| **UQ Calibration** | `|coverage - target|` | `< 0.02` | All (UQ mandatory) |

#### Phase-Field Specific Hard Gates (Phase 2B+)
| Check | Formula | Threshold | Physics Meaning |
|-------|---------|-----------|-----------------|
| **Crack Irreversibility** | `min(∂d/∂t)` | `≥ 0` | Crack irreversibility (∂d/∂t ≥ 0) |
| **Length Scale ℓ Enforcement** | `‖ℓ_pred - ℓ_true‖₂ / ‖ℓ_true‖₂` | `< 0.05` | Phase-field length scale consistency |
| **Degradation Function g(d)** | `‖g(d)_pred - g(d)_true‖₂ / ‖g(d)_true‖₂` | `< 0.05` | Degradation function validity |
| **History Variable H** | `‖H_pred - H_true‖₂ / ‖H_true‖₂` | `< 0.1` | History variable tracking |
| **Crack Irreversibility (Integral)** | `∫(∂d/∂t)_- dt` | `= 0` | No healing (crack irreversibility) |

#### Soft Gates (Multiplicative Penalty)
| Check | Formula | Penalty Function |
|-------|---------|------------------|
| **Symmetry** | `‖u - u_flipped‖₂ / ‖u‖₂` | `max(0, 1 - 20 × (ratio - 0.05))` |
| **Spectral Fidelity** | `‖E_pred(k) - E_true(k)‖ / ‖E_true(k)‖` | `max(0, 1 - 10 × (error - 0.1))` |
| **Conservation Drift** | `‖dM/dt‖` | `max(0, 1 - 100 × drift)` |

#### Phase-Gated Stress Floors & Targets
| PDE Class | Mass Cons | Energy Diss | Rollout Steps | UQ Target | Spectral Gate |
|-----------|-----------|-------------|---------------|-----------|---------------|
| Poisson/Darcy (2D/3D) | 1e-4 | N/A | N/A | 90% (P0-1), 95% (P2) | N/A |
| Burgers | 1e-3 | 1e-3 | 100 | 90% (P0-1), 95% (P2) | Shock capture |
| NS 2D / 3D Laminar | 1e-3 | 1e-3 | 100 | 90% (P0-1), 95% (P2) | N/A |
| **NS 3D Turbulent** | 1e-3 | 1e-3 | **1000+** | **99% (P3)** | **k^(-5/3) scaling** |
| Heat | N/A | 1e-4 | 100 | 90% (P0-1), 95% (P2) | N/A |
| Elasticity (2D/3D) | 1e-3 | N/A | N/A | 90% (P0-1), 95% (P2) | Locking test |
| Thermo-elasticity | 1e-3 | 1e-4 | 100/1000* | 90% (P0-1), 95% (P2) | Coupling consistency |
| **3D FSI** | 1e-3 | 1e-3 | 1000+ | **99% (P3)** | **Added-mass tensor symmetry** |
| **3D CHT** | 1e-3 | 1e-4 | 1000+ | **99% (P3)** | **Nu distribution (3D corners)** |

*\*Thermo-elasticity: 100 steps (P0-1), 1000+ steps (P2-3)*

#### 3D-Specific Gates (Phase 3+)
| Gate | Formula | Threshold | Purpose |
|------|---------|-----------|---------|
| **Energy Spectrum** | `‖E_pred(k) - C·k^(-5/3)‖ / ‖E_true(k)‖` | `< 0.1` | Kolmogorov scaling |
| **Q-Criterion** | `‖Q_pred - Q_true‖₂ / ‖Q_true‖₂` | `< 0.15` | Vortex topology |
| **Wall Shear Stress** | `‖τ_w_pred - τ_w_true‖₂ / ‖τ_w_true‖₂` | `< 0.1` | 3D separation topology |
| **Nu Distribution (Corners)** | `‖Nu_pred - Nu_true‖₂ / ‖Nu_true‖₂` | `< 0.2` | 3D corner heat transfer |
| **Added-Mass Tensor** | `‖A_pred - A_true‖_F / ‖A_true‖_F` | `< 0.1` | FSI coupling stability |

### 3.4 Scoring Formula
```
If hard_failure: score = 0.0
Else:
    base_improvement = max(0, log(E_baseline) - log(E_submission))
    soft_penalty = Π(soft_gate_penalties)  ∈ [0, 1]
    uq_bonus = uq_calibration_bonus(uq_metrics, challenge.phase)
    symbolic_bonus = 0.02 if symbolic_features_used and improvement > 0.05 else 0.0
    score = (base_improvement * soft_penalty) + uq_bonus + symbolic_bonus

def uq_calibration_bonus(uq_metrics, phase):
    if phase <= 1: target = 0.90
    elif phase == 2: target = 0.95
    else: target = 0.99
    if abs(uq_metrics.coverage - target) < 0.02:
        return 0.05
    return 0.0
```

### 3.5 Consensus
- Each validator submits `score`, `improvement`, `stress_passed`, `uq_calibrated`
- Chain computes **median** of validator scores
- Validators with `|score - median| > 0.1` flagged for audit
- 3+ validators required per challenge

---

## 4. Challenge Specification

### 4.1 Phase 0: Single-Physics PDEs (7 Problems)

| ID | Problem | Dimension | Physics Class | Resolution | Reference |
|----|---------|-----------|---------------|------------|-----------|
| 1 | Poisson | 2D / 3D | Elliptic, constant-coeff | 128² / 64³ | PhysicsNeMo |
| 2 | Darcy | 2D / 3D | Elliptic, variable-coeff | 128² / 64³ | PhysicsNeMo / PDEBench |
| 3 | Burgers | 2D | Nonlinear advection/shocks | 256×100 (x,t) | PhysicsNeMo |
| 4 | Navier-Stokes | 2D / 3D | Incompressible (2D vortex / 3D laminar Re≤100) | 128² / 64³×32 | PhysicsNeMo / JHTDB |
| 5 | Heat | 2D | Transient, variable κ | 128²×50 | PhysicsNeMo |
| 6 | Elasticity | 2D | Vector, tensor physics | 128² | PhysicsNeMo |
| 7 | Thermo-elasticity | 2D | Multi-physics, loss_vector | 128²×50 | Generated (48 Tier 1) |

**Each challenge provides:** public training split, public holdout set, hidden stress test (procedural + Well data, seeded by challenge_id), **symbolic metadata**.

### 4.2 Phase 1: Same Challenges + Customization + Abaqus Ingestion
Same 7 problems. Miners add LoRA adapters and custom datasets.

#### 4.2.1 Abaqus ODB/fil Ingestion Pipeline (NEW — Phase 1)
```python
def ingest_abaqus_data(data_uri: str, checksum: str) -> CustomDataset:
    """
    Ingests Abaqus ODB (.odb) and FORTRAN output (.fil) files.
    Returns standardized CustomDataset for validator ingestion.
    """
    # 1. Verify checksum
    verify_sha256(data_uri, expected_checksum)
    
    # 2. Download and cache
    local_path = download_to_cache(data_uri)
    
    # 3. Parse based on file extension
    if local_path.endswith('.odb'):
        return parse_abaqus_odb(local_path)
    elif local_path.endswith('.fil'):
        return parse_abaqus_fil(local_path)
    else:
        raise ValueError("Unsupported Abaqus format")
    
def parse_abaqus_odb(odb_path: str) -> CustomDataset:
    """
    Parses Abaqus ODB using abapy/abaqus2py.
    Extracts: mesh, field outputs (stress, strain, displacement), history outputs.
    """
    import abapy
    odb = abapy.postproc.from_odb(odb_path)
    
    # Extract mesh
    nodes = odb.mesh.nodes
    elements = odb.mesh.elements
    
    # Extract field outputs at each frame
    field_data = {}
    for step_name, step in odb.steps.items():
        for frame in step.frames:
            for field_name, field in frame.field_outputs.items():
                if field_name not in field_data:
                    field_data[field_name] = []
                field_data[field_name].append(field.bulk_data)
    
    # Interpolate to challenge grid if needed
    return CustomDataset(
        mesh={"nodes": nodes, "elements": elements},
        fields=field_data,
        metadata={"source": "abaqus_odb", "format": "odb"}
    )

def parse_abaqus_fil(fil_path: str) -> CustomDataset:
    """
    Parses Abaqus FORTRAN output (.fil) files.
    Extracts: nodal displacements, stresses, strains at each increment.
    """
    # .fil is ASCII - parse line by line
    with open(fil_path, 'r') as f:
        lines = f.readlines()
    
    # Parse header, then increment blocks
    # Format: INCREMENT, TIME, NODE, DOF, VALUE...
    data = parse_fil_blocks(lines)
    
    return CustomDataset(
        field_data=data,
        metadata={"source": "abaqus_fil", "format": "fil"}
    )
```

**Validator Integration:** Miner submits `custom_data` with `data_uri` pointing to IPFS-hosted ODB/fil. Validator downloads, runs ingestion pipeline, caches parsed `CustomDataset`, mixes with procedural data per `weight` parameter.

### 4.2 Phase 1: Same Challenges + Customization + Abaqus Ingestion
Same 7 problems. Miners add LoRA adapters and custom datasets (including Abaqus ODB/fil via new ingestion pipeline).

### 4.2 Phase 1: Same Challenges + Customization
Same 7 problems. Miners add LoRA adapters and custom datasets (including Abaqus ODB/fil via new ingestion pipeline).

### 4.3 Phase 2: Multi-Physics Composition (Verified Benchmarks First)

**Phase 2A (Months 1-3): Verified Benchmarks Only**
| Challenge | Source | Physics | Specialist Pair |
|-----------|--------|---------|-----------------|
| FSI 2D-1/2/3 | Turek/Hron | Fluid-Structure Interaction | `ns_2d` + `elasticity_2d` + `fsi_coupling` |
| CHT: Solid cooling / Electronics | PDEBench | Conjugate Heat Transfer | `ns_2d` + `heat_2d` + `cht_coupling` |

**Phase 2B (Month 3):** Thermo-Elasticity. Generate 48 Tier-1 references (β×κ×geometry) at 256² with FEniCS monolithic, mesh-converged. Cost: ~$3K.

**Phase 2C (Months 4-5):** Variant expansion (new Re, geometries, coupling strengths) on FSI/CHT/thermo-elasticity using existing references.

### 4.4 Phase 3: 3D Multi-Physics (Post-Turbulence Bridge)

| Phase | Challenges | Specialist Composition | Reference |
|-------|------------|------------------------|-----------|
| **3.2A** 3D FSI | Cylinder, flap, turbulent | `ns_3d_turbulent` + `elasticity_3d` + `fsi_3d_adapter` | preCICE partitioned |
| **3.2B** 3D Thermo-Elasticity | Bimetal, engine, turbine | `elasticity_3d` + `heat_3d` + `thermal_expansion_3d` | FEniCS monolithic |
| **3.2C** 3D CHT | Electronics, turbine, battery | `ns_3d_turbulent` + `heat_3d` + `cht_3d_adapter` | OpenFOAM/COMSOL |

### 4.5 Challenge Generation (Deterministic)
```python
def generate_challenge(challenge_id: str, problem_id: int) -> Challenge:
    seed = hash(challenge_id + str(problem_id))
    rng = np.random.default_rng(seed)
    
    train, holdout = load_fixed_splits(problem_id)
    stress = generate_stress_test(problem_id, rng)
    baseline = get_current_baseline(problem_id)
    
    # NEW: Generate symbolic metadata
    symbolic_metadata = generate_symbolic_metadata(problem_id, rng)
    
    return Challenge(
        challenge_id=challenge_id,
        problem_id=problem_id,
        train_data=train,
        holdout_data=holdout,
        stress_data=stress,
        baseline_config=baseline,
        submission_deadline=now() + 20_hours,
        stress_floors=get_stress_floors(problem_id),
        symbolic_metadata=symbolic_metadata  # NEW
    )
```

### 4.6 Symbolic Metadata Generation (Per Challenge)
```python
def generate_symbolic_metadata(problem_id: int, rng) -> SymbolicMetadata:
    """
    Generates symbolic metadata for a challenge using ModelingToolkit.
    Runs once per challenge type, cached forever.
    """
    # Parse PDE → symbolic system
    # Extract: symmetries, conservation laws, dimensionless groups, 
    # boundary types, coupling terms, validity domain
    # Return structured SymbolicMetadata protobuf
```

### 4.7 Hidden Stress Test Generation with The Well
```python
def generate_stress_test(problem_id: int, rng: np.random.Generator) -> StressTestData:
    procedural_variants = generate_procedural_variants(problem_id, rng)
    well_slices = get_well_slices_for_problem(problem_id, rng)
    return StressTestData(procedural=procedural_variants, well_slices=well_slices, seed=rng.bit_generator.state)

def get_well_slices_for_problem(problem_id: int, rng: np.random.Generator) -> List[WellSlice]:
    well_mapping = {
        1: ["active_matter"],
        2: ["active_matter"],
        3: ["turbulence", "viscoelastic"],
        4: ["turbulence", "mhd"],
        5: ["acoustic_scattering"],
        6: ["active_matter", "viscoelastic"],
        7: ["active_matter", "viscoelastic"],
    }
    datasets = well_mapping.get(problem_id, [])
    return [sample_well_dataset(ds, rng, num_samples=50) for ds in datasets]
```

---

## 5. Six Competition Tracks (Phase 2+)

| Track | Submission Format | What It Proves |
|-------|-------------------|----------------|
| **Monolith** | Single strategy JSON (end-to-end training config) | Can a monolithic model beat composition? |
| **Composition** | Specialist pipeline with adapters | Does composition beat monolith? |
| **Specialist-Only** | Single specialist ID (no adapter) | How much does the adapter matter? |
| **Symbolic Regression** | Discovered PDE string + basis | Can the agent discover governing PDE from data? |
| **Symbolic Composition** | MTK component graph + adapters | Can symbolic components compose to beat monolith? |
| **Symbolic Distillation** | ONNX + symbolic metadata + CUDA kernel | Can specialist be compressed with symbolic metadata preserved? |

**Same hidden stress test, same physics gates, six parallel leaderboards.**

---

## 6. Specialist Promotion Gauntlet (Phase 2+)

**Team-side only. Validators are NOT involved.**

### 6.1 Promotion Pipeline
```
1. Multi-teacher distillation → candidate specialist
2. Regression test: MUST pass SAME stress tests as miners
3. Judge Panel: 3 judges (team-side, different backbones) re-run + vote
4. Repair Loop: If any judge fails → specialist rejected → feedback to Landscape
5. Grounding Gate: Explicit lineage to validated fragments + physics gate passes
6. Decontamination Check: Verify no memorization of public holdout sets
7. Triple Crown Consistency: Pass across 3+ challenge variations
8. Symbolic Gauntlet: Verify symbolic metadata preserved + CUDA codegen works
9. Promote to Specialist Bank with validity domain
```

### 6.2 Judge Panel
- **3 judges** (team-side, different backbones: FNO, PINO, DeepONet)
- Unanimous pass required
- Failed judge → repair loop → feedback to Landscape Agent

### 6.3 Grounding Gate
- `distilled_from` field lists Fragment IDs of teacher strategies
- Each fragment must have passed physics gates
- Causal lineage traceable to baseline improvements

### 6.4 Decontamination Check
- Test on held-out Well slices not used in training
- Check for exact memorization vs. generalization

### 6.5 Triple Crown Consistency
Specialist must pass stress tests across **3+ challenge variations** (different Re, geometries, coupling strengths).

### 6.6 Symbolic Gauntlet (NEW)
- Verify symbolic metadata preserved in distillation
- Verify CUDA codegen produces valid kernels
- Verify symbolic metadata matches teacher ensemble
- Verify validity domain matches teacher ensemble

---

## 7. Landscape Agent Specification

### 7.1 Data Model
```protobuf
message StrategyFragment {
  string fragment_id = 1;
  int32 miner_uid = 2;
  string challenge_id = 3;
  int32 problem_id = 4;
  string config_json = 5;
  float improvement = 6;
  bool stress_passed = 7;
  UQMetrics uq_metrics = 8;
  float score = 9;
  uint64 timestamp = 10;
  repeated string causal_parents = 11;
  map<string, float> param_values = 12;
  // NEW: Symbolic features extracted from config
  map<string, float> symbolic_features = 13;
}

message UQMetrics {
  float calibration_error = 1;
  float sharpness = 2;
  float coverage = 3;
  bool calibrated = 4;
}
```

### 7.2 Daily Causal Update (DML) — ENHANCED WITH SYMBOLIC
```python
def daily_causal_update(new_fragments: List[StrategyFragment]):
    dag.add_nodes(new_fragments)
    
    # ENRICH FRAGMENTS WITH SYMBOLIC FEATURES
    for fragment in new_fragments:
        fragment.symbolic_features = extract_symbolic_features(fragment.config_json)
    
    for problem_id in PROBLEM_FAMILIES:
        problem_fragments = dag.get_problem_fragments(problem_id)
        
        # Double Machine Learning for heterogeneous treatment effects
        # Y = improvement, T = config_param, X = other_params + problem_context + SYMBOLIC_FEATURES
        dml = DoubleML(
            model_y=HistGradientBoostingRegressor(),  # Faster, incremental
            model_t=HistGradientBoostingRegressor(),
            n_folds=5
        )
        dml.fit(problem_fragments)
        
        # Estimate ATE for each tunable parameter
        effects = {}
        for param in TUNABLE_PARAMS:
            effects[param] = dml.ate(param)
        
        # Conditional effects: "fourier_modes helps when physics_loss > 1.0"
        interactions = dml.interaction_effects()
        
        # Propose baseline update
        proposal = propose_baseline(effects, interactions, current_baseline)
        owner_review_queue.put(proposal)
```

### 7.3 Weekly Specialist Distillation (Team-Side) — WITH SYMBOLIC GAUNTLET
```python
def weekly_distillation():
    for problem_id in PROBLEM_FAMILIES:
        # 1. Select top-K fragments (score > 90th percentile, stress_passed)
        teachers = select_top_fragments(problem_id, k=5)
        
        # 2. Distill into student (same backbone, 50% width)
        student = distill_ensemble(
            teachers=teachers,
            student_config=StudentConfig(width_factor=0.5),
            loss_fn=DistillationLoss(
                alpha_output=1.0,      # Output MSE
                alpha_physics=0.5,     # Physics loss on student
                alpha_features=0.3,    # Feature matching
                alpha_uq=0.2,          # UQ matching
                alpha_symbolic=0.2     # NEW: preserve symbolic metadata
            ),
            data=load_problem_data(problem_id)
        )
        
        # Regression test: specialist must pass same stress tests
        stress_result = run_stress_test(student, get_stress_data(problem_id))
        if stress_result.hard_failure:
            log_failure("Distilled specialist failed stress test")
            continue
        
        # SYMBOLIC GAUNTLET
        if pass_symbolic_gauntlet(student, problem_id):
            specialist = Specialist(
                specialist_id=f"{problem_id}_v{version}",
                onnx_model=export_onnx(student),
                problem_signature=get_signature(problem_id),
                metrics=evaluate_full(student, problem_id),
                validity_domain=estimate_validity_domain(student),
                symbolic_metadata=extract_symbolic_metadata(teachers),  # NEW
                cuda_kernel=generate_cuda_kernel(student),  # NEW: MTK → CUDA
                license="AGPL-3.0 + Commercial Dual-License"
            )
            specialist_bank.publish(specialist)
```

### 7.4 Landscape Treasury & Governance
- **Funding:** 18% Owner emissions + 10% time-locked (6-month cliff, 2-year vest, 3/5 multi-sig)
- **Spending:** Agent compute, specialist distillation GPU, challenge generation, audits
- **Audit:** Quarterly public report

---

## 7.5 Agent-Native Architecture (NEW SECTION)

### 7.5.1 Agent Identity & Stake
```protobuf
message AgentIdentity {
  string did = 1;                    // did:hydrogen:agent:xyz
  string hotkey = 2;                 // Bittensor hotkey
  uint64 stake = 3;                  // Staked TAO (nanoTAO)
  uint64 reputation = 4;             // Scaled 1e6
  repeated string capabilities = 4;  // ["ns_solver", "phase_field", "optimizer_tuning"]
  string parent_agent = 5;           // Parent DID for lineage/forking
  uint64 created_at = 6;
  uint64 stake_locked_until = 7;     // Block number
}

message AgentRegistration {
  string did = 1;
  uint64 stake = 2;
  repeated string capabilities = 3;
  bytes signature = 4;  // Signed by hotkey
}
```

### 7.5.2 Agent-to-Agent Protocol (A2A)
```protobuf
message AgentMessage {
  string from_did = 1;
  string to_did = 2;
  MessageType type = 3;
  bytes payload = 4;
  uint64 nonce = 5;
  bytes signature = 6;
  uint64 timestamp = 7;
}

enum MessageType {
  PROPOSE = 0;        // Propose strategy/config
  CRITIQUE = 1;       // Critique peer's strategy
  KNOWLEDGE_SHARE = 2; // Share PDE discovery/insight
  CHALLENGE = 3;      // Challenge peer's result
  VOTE = 4;           // Vote on specialist promotion
  FORK_KNOWLEDGE = 5; // Fork knowledge graph
  MERGE_KNOWLEDGE = 6; // Merge knowledge graphs
}
```

**A2A Flow:**
1. Agent A → Agent B: `PROPOSE` strategy for challenge X
2. Agent B → Agent A: `CRITIQUE` with physics-gate feedback
3. Agent A → Agent B: `KNOWLEDGE_SHARE` (discovered PDE pattern)
3. Swarm votes on specialist promotion via `VOTE` messages

### 7.5.3 Swarm Intelligence & Coordination
```protobuf
message SwarmConfig {
  string challenge_family = 1;  // e.g., "navier_stokes", "phase_field"
  repeated string member_dids = 2;
  uint32 min_quorum = 3;
  uint32 max_members = 20;
  VoteMechanism vote_mechanism = 4;
}

enum VoteMechanism {
  MAJORITY = 0;
  WEIGHTED_BY_REPUTATION = 1;
  UNANIMOUS = 2;
}

message SwarmVote {
  string specialist_id = 1;
  string voter_did = 2;
  bool approve = 3;
  string rationale = 4;
  uint64 timestamp = 5;
}
```

**Swarm Coordination:**
- Agents form swarms per challenge family (NS, phase-field, elasticity, etc.)
- Swarm votes on specialist promotion (weighted by reputation)
- Fork/merge knowledge graphs for parallel exploration
- Knowledge graph stored as IPFS Merkle DAG (Merkle DAG of StrategyFragments)

### 7.5.3 Agent Incentives
| Incentive | Mechanism | Reward |
|-----------|-----------|--------|
| **Stake-to-Participate** | Stake TAO to join; slashed for invalid physics | Access to challenges |
| **Discovery Reward** | Novel PDE discovery (validated by Landscape) | 0.5% of challenge budget |
| **Architecture Innovation** | Novel architecture beating baseline by >5% | 1% of challenge budget |
| **Validation Reward** | Verifying peer's specialist (independent reproduction) | 0.2% of challenge budget |
| **Curation Reward** | Curating high-quality custom datasets | Data royalty pool (5% emissions) |
| **Reputation Bonus** | Top 10% reputation | Priority challenge access, fee discount |

**Slashing Conditions:**
| Offense | Slash Amount |
|---------|-------------|
| Invalid physics (gate failure) | 10% stake |
| Spam/flooding | 5% stake |
| Malicious critique (proven false) | 20% stake |
| Knowledge poisoning (false PDE) | 50% stake + ban |

### 7.5.3 Human-in-the-Loop (HITL) API
```python
# REST API for human supervisors
POST /api/v1/human/approve
{
  "agent_did": "did:hydrogen:agent:xyz",
  "submission_id": "sub_abc123",
  "decision": "approve|reject|request_changes",
  "comments": "Increase physics loss weight for mass conservation"
}

POST /api/v1/human/intervene
{
  "challenge_id": "ns_2d_v1_0042",
  "intervention_type": "override_physics_gate|adjust_baseline|halt_challenge",
  "parameters": {"mass_conservation_threshold": 1e-4},
  "reason": "Stricter mass conservation needed for this regime"
}

GET /api/v1/human/audit/{agent_did}
# Returns: agent history, slashing events, discoveries, reputation trajectory
```

### 7.5.4 Agent Lifecycle
```
SPAWN → STAKE → COMPETE → REPRODUCE/DIE
   │           │           │           │
   │           │           │           └── Death: stake returned (minus slashes), knowledge archived
   │           │           │
   │           │           └── Compete: submit strategies, earn reputation
   │           │
   │           └── Stake: lock TAO, register DID, join swarms
   │
   └── Spawn: generate DID, register capabilities, stake minimum TAO
```

**Reproduction:** High-reputation agents can "spawn" child agents with inherited knowledge graph (forked Merkle DAG), inheriting 10% parent stake.

### 7.5.4 Human-in-the-Loop (HITL) API
```yaml
# REST API for human supervisors
POST /api/v1/human/approve
{
  "agent_did": "did:hydrogen:agent:xyz",
  "submission_id": "sub_abc123",
  "decision": "approve|reject|request_changes",
  "comments": "Increase physics loss weight for mass conservation"
}

POST /api/v1/human/intervene
{
  "challenge_id": "ns_2d_v1_0042",
  "intervention_type": "override_physics_gate|adjust_baseline|halt_challenge",
  "parameters": {"mass_conservation_threshold": 1e-4},
  "reason": "Stricter mass conservation needed for this regime"
}

GET /api/v1/human/audit/{agent_did}
# Returns: agent history, slashing events, discoveries, reputation trajectory

POST /api/v1/human/override_gate
{
  "challenge_id": "ns_2d_v1_0042",
  "gate": "mass_conservation",
  "new_threshold": 5e-4,
  "duration_blocks": 1000
}
```

### 7.5.4 Agent Lifecycle & Incentives

**Lifecycle:** `SPAWN → STAKE → COMPETE → REPRODUCE/DIE`

| Phase | Action | Stake/Slash |
|---------|--------|-------------|
| **SPAWN** | Generate DID, register capabilities, stake minimum TAO (1000 TAO) | Lock stake |
| **STAKE** | Register capabilities, join swarms | Stake locked |
| **COMPETE** | Submit strategies, critique peers, vote on specialists | Earn rewards, risk slashing |
| **REPRODUCE** | Fork knowledge graph, spawn child agent with 10% stake inheritance | Parent stakes additional 100 TAO |
| **DIE** | Voluntary exit or slashing to zero | Stake returned (minus slashes) |

**Incentive Structure:**
| Action | Reward | Source |
|--------|--------|--------|
| Novel PDE discovery (validated) | 0.5% challenge budget | Novelty pool |
| Novel architecture (>5% improvement) | 1% challenge budget | Innovation pool |
| Peer validation (reproduce specialist) | 0.2% challenge budget | Validation pool |
| Curation (high-quality custom data) | Data royalty pool (5% emissions) | Data royalty pool |
| Swarm participation | Reputation points | Reputation |

**Slashing Conditions:**
| Offense | Slash |
|---------|-------|
| Invalid physics (gate failure) | 10% stake |
| Spam/flooding | 5% stake |
| Malicious critique (proven false) | 20% stake |
| Knowledge poisoning (false PDE) | 50% stake + ban |

---

## 8. Specialist Bank & Marketplace (UPDATED)

### 8.1 Specialist Metadata
```protobuf
message Specialist {
  string specialist_id = 1;
  string problem_signature = 2;
  bytes onnx_model = 3;
  Metrics7D metrics = 3;
  ValidityDomain validity = 4;
  repeated string known_failures = 5;
  License license = 6;
  uint64 created_at = 7;
  string distilled_from = 8;
  // NEW: Symbolic metadata
  SpecialistSymbolicMetadata symbolic_metadata = 9;
  bytes cuda_kernel = 10;  // MTK → CUDA codegen
}

message SpecialistSymbolicMetadata {
  string governing_pde = 1;
  repeated string symmetries = 2;
  repeated string conservation_laws = 2;
  ValidityDomain validity_domain = 3;
  repeated string symmetry_features = 4;
  repeated string conservation_features = 5;
  repeated string boundary_types_supported = 6;
}
```

### 8.2 Miner Usage (Phase 2+)
```json
{
  "specialist_id": "navier_stokes_2d_v3",
  "adapter": { "type": "lora", "rank": 4, "target_layers": ["pino_layers.3"] }
}
```

### 8.3 Licensing Model
- **AGPL-3.0:** Free for research, open-source, internal use
- **Commercial License:** Per-seat or per-inference API with indemnification, SLA, fine-tuning support

### 8.4 Agent Participation Interface (Phase 0+)

#### 8.4.1 REST API Endpoints
```
GET  /api/v1/challenges
GET  /api/v1/challenges/{challenge_id}
GET  /api/v1/challenges/{challenge_id}/baseline
GET  /api/v1/challenges/{challenge_id}/priors
GET  /api/v1/specialists
GET  /api/v1/specialists/{specialist_id}
POST /api/v1/submit
GET  /api/v1/submissions/{submission_id}
```

#### 8.4.2 Submission Payloads
**Phase 0-1 (Strategy JSON):** `{"challenge_id": "...", "hotkey": "...", "strategy": {...}}`  
**Phase 2+ (Specialist Pipeline):** `{"challenge_id": "...", "hotkey": "...", "specialist_pipeline": {...}}`

#### 8.4.3 Structured Feedback Response
```json
{
  "submission_id": "sub_abc123",
  "status": "accepted",
  "rank": 3,
  "score": 0.047,
  "improvement_vs_baseline": 0.012,
  "novelty_bonus": 0.005,
  "symbolic_bonus": 0.02,
  "emission_reward": 124.5,
  "physics_gates": { ... },
  "stress_test_summary": { ... },
  "causal_insights": [{"parameter": "pino_loss_weight", "effect": "+0.008", "confidence": 0.87}],
  "suggestions": ["Increase conservation weight to 1.0", "Start curriculum at 32²"],
  "validator_consensus": "strong"
}
```

#### 8.4.4 hydrogen-agent Python SDK
```python
from hydrogen_agent import HydrogenClient
client = HydrogenClient(hotkey=agent_hotkey)
challenges = client.list_challenges()
baseline = client.get_baseline("ns_2d_v1")
priors = client.get_priors("ns_2d_v1")
strategy = agent.generate_strategy(baseline, priors)
result = client.submit(strategy)
```

---

## 9. Foundation Operator (Phase 3+)

### 9.1 Architecture
- **Backbone:** FNO/PINO hybrid (spectral + local kernels)
- **Width:** 128 channels, 6 spectral blocks
- **Conditioning:** ProblemSignature → FiLM layers modulate backbone
- **NEW:** SymbolicMetadata → FiLM layers modulate backbone
- **UQ:** Built-in evidential head (μ, σ, ν, α for Student-t)

### 9.2 Training
```
L = Σ w_i * MSE(foundation(x), specialist_i(x)) 
  + λ_physics * PhysicsLoss(foundation)
  + λ_uq * UQ_Calibration_Loss
  + λ_consistency * CrossSpecialistConsistency
```
**Data:** Union of all specialist training data + synthetic ProblemSignature sampling

### 9.3 Fine-Tuning API (Commercial)
```python
def fine_tune_foundation(client_data: EncryptedBlob, problem_signature: ProblemSignature) -> Specialist:
    data = tee_decrypt(client_data)
    adapted = lora_adapt(foundation, data, steps=20)
    if not verify(adapted, problem_signature): raise AdaptationFailed
    return tee_encrypt(export_onnx(adapted))
```

---

## 10. Consensus & Slashing

### 10.1 Validator Slashing Conditions
| Offense | Evidence | Slash |
|---------|----------|-------|
| Score deviation > 0.15 from median (3+ consecutive) | On-chain scores | 5% stake |
| Missed validation deadline (3+ consecutive) | Missed block window | 2% stake |
| Failed reproducibility audit | Auditor rerun ≠ submitted score | 10% stake |
| Physics check manipulation | Auditor finds disabled checks | 20% stake + ban |

### 10.2 Miner Penalties
| Offense | Penalty |
|---------|---------|
| Invalid JSON / schema violation | 0.1 TAO fee burned |
| Custom data checksum mismatch | 0.1 TAO fee burned |
| Spam (>5 submissions/challenge) | 24hr submission cooldown |

---

## 11. Emission Distribution

### 11.1 Per-Challenge Budget (Capped at 10 Active Challenges)
```
challenge_budget = total_subnet_emission / min(active_challenges, 10)
```

### 11.2 Distribution (Competitive Phase)
| Rank | Share of Challenge Budget |
|------|---------------------------|
| 1st | 40% |
| 2nd | 30% |
| 3rd | 20% |
| 4th | 10% |
| 5+ | 0% |

**Novelty Bonus:** 5% of challenge budget for physics-informed embedding distance from recent winners, **only if improvement > 0**.

**Symbolic Bonus:** 2% of challenge budget for using auto-generated symbolic features that improve baseline by >5%.

**Bounty Accumulation:** Emissions pool until improvement > 0.

**Private-until-proven:** Strategies revealed only after earning rewards.

**Miner burn:** 0%

**Warm-up (<10 distinct submissions/challenge):** Top 3 split 50/30/20 (only if improvement > 0).  
**Competitive:** Top-4 split (40/30/20/10) + novelty bonus (5%, improvement-gated) + symbolic bonus (2%, improvement >5%).

**Validators (41%):** Paid per validation via median consensus scoring completeness + physics-check audit.

**Owner (18%):** Funds Landscape Agent, Specialist Bank, Challenge Infra, Treasury (10% time-locked).

---

## 12. Phase Definitions (Condensed)

### Phase 0: The Causal Baseline (Launch → Month 3)
**Challenges:** 7 single-physics PDEs (Poisson 2D/3D, Darcy 2D/3D, Burgers, NS 2D/3D laminar, Heat, Elasticity, Thermo-elasticity)

**Symbolic Milestones:**
- [ ] ModelingToolkit integration in validator pipeline
- [ ] Symbolic feature extraction per challenge (symmetries, conservation, dimless groups)
- [ ] Automatic loss weight computation from symbolic metadata
- [ ] Symbolic-aware physics gates

### Phase 1: Specialist Bank & Data Markets (Months 3-6)
**Symbolic Milestones:**
- [ ] Specialist distillation with symbolic metadata preservation
- [ ] Symbolic metadata extraction from teacher ensemble
- [ ] Symbolic metadata attached to distilled specialists
- [ ] Symbolic regression (DataDrivenDiffEq) for PDE discovery

### Phase 1: Specialist Bank & Data Markets (Months 3-6)
**Challenges:** Same 7 problems. Miners add LoRA adapters and custom datasets.

**What Happens:** Validators apply adapters, cache custom data, measure data impact. Landscape pays data royalties (5% of emissions). Weekly distillation → ONNX specialists (AGPL-3.0 + commercial).

**Product:** 20-30 verified ONNX specialists with validity domains, calibrated UQ, dual licensing. Data royalty pipeline.

**Revenue:** $10-50M/yr specialist licensing, data royalties, fine-tuning API.

**Symbolic Milestones:**
- [ ] Specialist distillation with symbolic metadata preservation
- [ ] Symbolic metadata extraction from teacher ensemble
- [ ] Symbolic metadata attached to distilled specialists
- [ ] Symbolic regression (DataDrivenDiffEq) for PDE discovery

### Phase 1: Abaqus Ingestion (NEW)
- [ ] Abaqus ODB/fil ingestion pipeline operational
- [ ] Abaqus data integrated into custom data pipeline
- [ ] Abaqus data usable for custom_data.custom_data.usage = "augment"

### Phase 2: Composition Engine & Specialist Marketplace (Months 6-18)
**Phase 2A (Months 6-9): Verified Benchmarks**
- [ ] FSI challenges (Turek/Hron 2D-1/2/3) with preCICE reference
- [ ] CHT challenges (PDEBench solid cooling + electronics) with OpenFOAM reference
- [ ] Three-track leaderboard implementation: Monolith / Composition / Specialist-Only
- [ ] `specialist_pipeline` JSON schema support in validator
- [ ] `execute_specialist_pipeline()` validator logic (staggered coupling, adapter support)

**Phase 2B (Month 9): Thermo-Elasticity**
- [ ] Generate 48 Tier-1 thermo-elasticity references (FEniCS monolithic, mesh-converged)
- [ ] Add thermo-elasticity challenges with `loss_vector` coupling terms

**Phase 2C (Months 10-14): Variant Expansion**
- [ ] FSI/CHT/Thermo-elasticity variants (new Re, geometries, coupling strengths)
- [ ] Specialist Bank: ≥50 specialists, reuse rate >80%
- [ ] Composition win rate >60% on multi-physics challenges
- [ ] Adapter innovation >30% (novel coupling adapters)
- [ ] Go/No-Go gate evaluation for 3D transition

**Symbolic Milestones:**
- [ ] Symbolic Composition: Acausal specialist composition via MTK
- [ ] Symbolic metadata enables automatic compatibility checking
- [ ] Acausal composition → auto-generates coupled PDEs
- [ ] Symbolic Composition track in leaderboard

### Phase 2C Exit Criteria (Go/No-Go for 3D Transition)
| Metric | Target | If Missed |
|--------|--------|-----------|
| Composition win rate | >60% (Composition > Monolith) | Pivot: deepen single-physics specialist depth |
| Specialist reuse | >80% compositions use ≥1 Bank specialist | Extend Phase 1 |
| Adapter innovation | >30% novel adapters | Expand adapter design space |
| Stress test pass rate | >70% compositions pass | Coupling brittle → simplify adapter design |

### Phase 3: 3D Transition & Foundation Operator (Months 18+)

### Phase 3 Entry Gates (All Required)
| Gate | Metric | Threshold |
|------|--------|-----------|
| 2D composition proven | Composition win rate >60% | Phase 2 complete |
| 3D single-physics specialists exist | ≥6 specialists in Bank | Phase 3.0 prereq |
| Curriculum validated | 2D→3D fine-tune <50% scratch cost | Ablation study |
| 3D reference pipeline works | Tier 1 3D reference generated, mesh-converged | Pipeline test |
| Validator quorum | ≥5 validators with 24GB+ GPUs | Infrastructure ready |

**Phase 3.0: 3D Single-Physics Foundations (Prerequisite)**
- [ ] 3D curriculum distillation pipeline: 2D specialist → 3D (zero-pad Fourier + noise → curriculum 32³→64³→128³)
- [ ] 3D single-physics specialists in Bank: `poisson_3d`, `darcy_3d`, `ns_3d_laminar`, `heat_3d`, `elasticity_3d`
- [ ] All 3D single-physics specialists pass stress tests (mass, energy, rollout, UQ)

**Phase 3.1: 3D Turbulence Bridge (Critical - Months 1-3 of Phase 3)**
- [ ] 3D Spectral Initialization Protocol (proper 3D energy spectrum priors, not zero-pad)
- [ ] 3D Turbulence Curriculum: Re=50→100→200→500 on channel/cylinder
- [ ] `ns_3d_turbulent_v1` specialist with verified k^(-5/3) energy spectrum
- [ ] 3D-specific stress gates: energy spectrum (k^(-5/3)), Q-criterion, wall shear, Nu distribution
- [ ] Gate: `ns_3d_turbulent_v1` passes all 3D turbulence stress tests

### Phase 3.1 Gate (All Required Before 3.2)
| Metric | Threshold |
|--------|-----------|
| Energy spectrum | Verified k^(-5/3) scaling |
| Q-criterion gate | Pass |
| Wall shear stress distribution | Pass |
| Nu distribution (3D corners) | Pass |
| Stress test pass rate | >70% |

**Phase 3.2: 3D Multi-Physics Rollout**
- [ ] 3D FSI: `ns_3d_turbulent` + `elasticity_3d` + `fsi_3d_adapter` (preCICE reference)
- [ ] 3D Thermo-Elasticity: `elasticity_3d` + `heat_3d` + `thermal_expansion_3d` (FEniCS reference)
- [ ] 3D CHT: `ns_3d_turbulent` + `heat_3d` + `cht_3d_adapter` (OpenFOAM/COMSOL reference)
- [ ] Three-track leaderboard + same stress tests for all 3D multi-physics

**Phase 3.3: Foundation Operator (LPM)**
- [ ] Multi-teacher distillation across entire Specialist Bank (2D + 3D)
- [ ] FiLM conditioning on ProblemSignature
- [ ] Evidential UQ head (μ, σ, ν, α for Student-t)
- [ ] Commercial fine-tuning API: TEE decryption → LoRA rank=8 (10-50 steps) → stress test verification → encrypted ONNX return
- [ ] LPM fine-tuning API commercial launch

**Symbolic Milestones:**
- [ ] Multi-teacher distillation across entire Specialist Bank (2D + 3D)
- [ ] FiLM conditioning on ProblemSignature + SymbolicMetadata
- [ ] Evidential UQ head (μ, σ, ν, α for Student-t)
- [ ] Commercial fine-tuning API: TEE decryption → LoRA rank=8 (10-50 steps) → stress test verification → encrypted ONNX return
- [ ] LPM fine-tuning API commercial launch

---

## 14. Appendix: Mathematical Definitions

### 14.1 Log-Space Improvement
```
improvement = log(E_baseline) - log(E_submission)
where E = relative L2 error = ‖u_pred - u_true‖₂ / ‖u_true‖₂
```
*Properties:* Rewards consistent orders-of-magnitude improvement; invariant to baseline magnitude.

### 14.2 Double Machine Learning (DML) for Causal Effects
```
Y = improvement, T = target_param, X = other_params + context
1. Regress Y ~ X → Ŷ, regress T ~ X → T̂
2. Residuals: Ỹ = Y - Ŷ, T̃ = T - T̂
3. ATE = E[Ỹ * T̃] / E[T̃²]
4. Heterogeneous effects: τ(x) = E[Ỹ | X=x] / E[T̃ | X=x]
```
*Robust to:* High-dimensional X, non-linear relationships, partial confounding.

### 14.3 UQ Calibration Metric
```
calibration_error = |(1/N) Σ 1{y ∈ PI_α(x)} - (1-α)|
sharpness = (1/N) Σ width(PI_α(x))
```
*Target:* calibration_error < 0.02 at phase-gated α (90%/95%/99%)

### 14.4 3D Spectral Initialization (Correct)
```python
def init_3d_spectral_from_2d(weights_2d, spectrum="kolmogorov"):
    # 1. Initialize 3D modes with target spectrum E(k) ~ k^(-5/3)
    # 2. Project 2D modes onto k_z=0 plane with energy scaling
    # 3. Add controlled noise in orthogonal directions (σ ~ 0.01)
    # 4. DO NOT copy 2D weights directly — 2D spectrum is k^(-3), not k^(-5/3)
```

### 14.5 Thermo-Elasticity Coupling Terms
```json
"loss_vector": {
  "elasticity_residual": 1.0,
  "heat_residual": 0.8,
  "coupling_thermal_strain": 0.5,    // α * ΔT * I (thermal → mechanical)
  "coupling_heat_source": 0.3,       // β * ∂(∇·u)/∂t (mechanical → thermal)
  "boundary": 0.5
}
```

### 14.6 3D Spectral Initialization (Correct)
```python
def init_3d_spectral_from_2d(weights_2d, spectrum="kolmogorov"):
    # 1. Initialize 3D modes with target spectrum E(k) ~ k^(-5/3)
    # 2. Project 2D modes onto k_z=0 plane with energy scaling
    # 2. Add controlled noise in orthogonal directions (σ ~ 0.01)
    # 4. DO NOT copy 2D weights directly — 2D spectrum is k^(-3), not k^(-5/3)
```

### 14.7 Phase-Gated UQ & Rollout
```python
def get_phase_targets(phase):
    if phase <= 1: return {"uq_target": 0.90, "rollout_steps": 100}
    elif phase == 2: return {"uq_target": 0.95, "rollout_steps": 100}
    else: return {"uq_target": 0.99, "rollout_steps": 1000}
```

### 14.8 Symbolic Layer Mathematical Definitions

#### 14.8.1 Symbolic PDE Representation
```
Given PDE: ∂u/∂t + ∇·(u⊗u) = -∇p + ν∇²u + f
Symbolic representation: ODESystem(eqs, t, states, params)
where eqs = [D(u,t) + ∇·(u⊗u) + ∇p - ν∇²u - f, ∇·u]
```

#### 14.8.2 Feature Extraction Mapping
```
Φ: ODESystem → FeatureDict
Φ(sys) = {
  symmetries: detect_symmetries(sys),
  conservation_laws: detect_conservation_laws(sys),
  dimensionless_groups: extract_dimensionless_groups(sys),
  boundary_conditions: extract_boundary_conditions(sys),
  coupling_terms: extract_coupling_terms(sys)
}
```

#### 14.8.3 Symbolic Regression (PDE Discovery)
```
Given: trajectories {u(x,t)}, basis functions B = {u, u_x, u_xx, u*u_x, ...}
Find: sparse coefficients ξ such that ∂u/∂t ≈ B·ξ
Method: STLSQ (Sequential Thresholded Least Squares)
Objective: min ‖∂u/∂t - B·ξ‖₂ + λ‖ξ‖₁
```

#### 14.8.4 Symbolic Distillation Loss
```
L_distill = α·MSE(student, teacher_ensemble) 
          + β·PhysicsLoss(student) 
          + γ·FeatureMatching(student, teachers)
          + δ·UQ_Calibration_Loss
          + ε·SymbolicLoss(student, teachers)  // NEW
          
SymbolicLoss = ‖Φ(student) - Φ(teacher_ensemble)‖₂²
where Φ extracts symbolic features from model predictions
```

#### 14.8.5 Phase-Gated UQ & Rollout
```
Phase 0-1: UQ target 90%, rollout 100 steps
Phase 2:   UQ target 95%, rollout 100 steps  
Phase 3:   UQ target 99%, rollout 1000 steps + spectral stationarity
```
