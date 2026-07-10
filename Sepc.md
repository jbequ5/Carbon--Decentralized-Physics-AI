You're right - I restructured it instead of just updating. Here's **v2.0 in the exact same v1.0 format** with all updates integrated:

---

# SPEC.md — Hydrogen PDE Subnet Technical Specification v2.0

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
| Owner | 18% | Landscape Agent, Specialist Bank, Challenge Infra, Treasury (10% time-locked) |

**Emission Budget per Challenge:** `total_subnet_emission / number_of_active_challenges`

**Emission Distribution per Challenge (Top 4):**
| Rank | Share of Challenge Budget |
|------|---------------------------|
| 1st | 40% |
| 2nd | 30% |
| 3rd | 20% |
| 4th | 10% |
| 5+ | 0% |

**Submission fee:** 0.1 TAO (burned if validation fails pre-checks; covers validator marginal cost ~0.08 TAO/run)

**Warm-up:** <10 distinct submissions/challenge → top 3 split 50/30/20  
**Competitive:** Top-4 split as above (40/30/20/10)

---

## 2. Miner Interface

### 2.1 Strategy JSON Schema (Phase 0-1)
```json
{
  "backbone": "PINO",                    // Enum: FNO, PINO, DeepONet, GNO, OFormer
  "resolution": [128, 128, 64],          // Spatial grid [Nx, Ny, Nz]
  "pino": {                              // Backbone-specific config
    "loss_vector": {                     // Per-physics-term weights (required for multi-physics)
      "pde_residual": 1.5,
      "conservation": 0.8,
      "boundary": 0.5,
      "symmetry": 0.3,
      "coupling": 0.3                    // Thermo-elasticity coupling term
    },
    "physics_loss_type": "pde_residual", // Enum: pde_residual, conservation, symmetry
    "boundary_handling": "ghost_cells"   // Enum: ghost_cells, penalty, lagrange
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
    "method": "deep_ensemble",           // Enum: deep_ensemble, conformal, evidential
    "num_members": 4,
    "calibration_target": 0.90
  },
  "custom_data": {                       // Optional (Phase 1+)
    "data_uri": "ipfs://...",
    "checksum": "sha256:...",
    "usage": "augment",                  // Enum: augment, curriculum, label_only
    "weight": 0.25
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
  "execution_schedule": "staggered",     // Enum: staggered, monolithic
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

### 3.1 Backbone Images (Pre-built, Pinned)
| Image | Backbones | Base | PhysicsNeMo | NeuralOperator | PyTorch | CUDA |
|-------|-----------|------|-------------|----------------|---------|------|
| `hydrogen/validator:fno-v24.09` | FNO | nvcr.io/nvidia/pytorch:24.09-py3 | 24.09 | 0.3.0 | 2.3.0 | 12.4 |
| `hydrogen/validator:pino-v24.09` | PINO | same | same | same | same | same |
| `hydrogen/validator:deeponet-v24.09` | DeepONet | same | same | same | same | same |
| `hydrogen/validator:gno-v24.09` | GNO | same | same | same | same | same |
| `hydrogen/validator:oformer-v24.09` | OFormer | same | same | same | same | same |

**Hardware requirement:** 
- Phase 0-2: NVIDIA GPU ≥ 16 GB VRAM (RTX 3080/3090/4080/4090, A100)
- Phase 3 (3D): NVIDIA GPU ≥ 24 GB VRAM (RTX 3090/4090, A100 40GB, H100)

### 3.2 Validation Pipeline (Deterministic)
```python
def validate_submission(challenge_id: str, miner_submission: dict) -> ValidationResult:
    # 1. Load challenge data
    train_data, holdout_data, stress_data = load_challenge_splits(challenge_id)
    challenge = load_challenge_metadata(challenge_id)
    
    # 2. Determine submission type and execute
    if "specialist_pipeline" in miner_submission:
        # Phase 2+: Composition track
        model = execute_specialist_pipeline(
            pipeline=miner_submission["specialist_pipeline"],
            challenge=challenge,
            train_data=train_data,  # For adapter training if present
            seed=derive_seed(challenge_id, validator_hotkey)
        )
    else:
        # Phase 0-1: Strategy JSON
        model = train_backbone(
            image=select_backbone_image(miner_submission["backbone"]),
            config=miner_submission,
            train_data=train_data,
            custom_data=resolve_custom_data(miner_submission.get("custom_data")),
            seed=derive_seed(challenge_id, validator_hotkey)
        )
    
    # 3. Evaluate on public holdout
    E_baseline = load_current_baseline_error(challenge_id)
    E_submission = evaluate(model, holdout_data, challenge)
    improvement = log(E_baseline) - log(E_submission)  # Log-space improvement
    
    # 4. Hidden stress test + physics gates
    stress_result = run_stress_test(model, stress_data, challenge)
    
    # 5. UQ calibration check
    uq_metrics = evaluate_uq(model, stress_data, miner_submission.get("uq_config", {}))
    
    # 7. Compute score
    if stress_result.hard_failure:
        score = 0.0
        reason = stress_result.failure_reason
    else:
        base_score = max(0.0, improvement)
        soft_penalty = stress_result.soft_penalty  # ≤ 1.0
        uq_bonus = 0.05 if uq_metrics.calibrated else 0.0
        # No stress_pass bonus - physics gates are pass/fail
        score = (base_score * soft_penalty) + uq_bonus
    
    return ValidationResult(score, improvement, stress_result, uq_metrics)
```

### 3.3 Physics Gates (Stress Test)

#### Hard Gates (Score = 0 if failed)
| Check | Formula | Threshold | PDE Classes |
|-------|---------|-----------|-------------|
| **Mass Conservation** | `‖∇·u‖₁ / ‖u‖₁` | `< 1e-3` | NS, Darcy, Elasticity |
| **Energy Dissipation** | `dE/dt` | `≤ 1e-4` | NS, Heat, Burgers |
| **Boundary Satisfaction** | `‖u - u_BC‖₂ / ‖u_BC‖₂` | `< 1e-3` | All with BCs |
| **Rollout Stability** | `‖E(t=100) - E(t=0)‖ / E(t=0)` | `< 0.01` | All transient |
| **UQ Calibration** | `|coverage - 0.90|` | `< 0.02` | All (UQ mandatory) |

#### Soft Gates (Multiplicative Penalty)
| Check | Formula | Penalty Function |
|-------|---------|------------------|
| **Symmetry** | `‖u - u_flipped‖₂ / ‖u‖₂` | `max(0, 1 - 20 × (ratio - 0.05))` |
| **Spectral Fidelity** | `‖E_pred(k) - E_true(k)‖ / ‖E_true(k)‖` | `max(0, 1 - 10 × (error - 0.1))` |
| **Conservation Drift** | `‖dM/dt‖` | `max(0, 1 - 100 × drift)` |

**PDE-Specific Stress Floors** (baseline_stress_error × 0.9 floor):
| PDE Class | Mass Cons | Energy Diss | Rollout | UQ Cal |
|-----------|-----------|-------------|---------|--------|
| Poisson/Darcy | 1e-4 | N/A | N/A | 0.02 |
| Burgers | 1e-3 | 1e-3 | 0.02 | 0.03 |
| NS (Re≤100) | 1e-3 | 1e-3 | 0.01 | 0.02 |
| Heat | N/A | 1e-4 | 0.01 | 0.02 |
| Elasticity | 1e-3 | N/A | N/A | 0.02 |
| Thermo-elasticity | 1e-3 | 1e-4 | 0.02 | 0.03 |

### 3.4 Scoring Formula
```
If hard_failure: score = 0.0
Else:
    base_improvement = max(0, log(E_baseline) - log(E_submission))
    soft_penalty = Π(soft_gate_penalties)  ∈ [0, 1]
    uq_bonus = 0.05 if UQ calibrated else 0.0
    # No stress_pass bonus - physics gates are pass/fail
    score = (base_improvement * soft_penalty) + uq_bonus
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
| 8 | Thermo-elasticity | 2D | Multi-physics, loss_vector | 128²×50 | Generated (48 Tier 1) |

**Each challenge provides:** public training split, public holdout set, hidden stress test (procedural parameter/geometry shifts seeded by challenge_id).

### 4.2 Phase 1: Same Challenges + Customization
Same 7 problems. Miners add LoRA adapters and custom datasets.

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
    
    # Public splits (fixed per problem)
    train, holdout = load_fixed_splits(problem_id)
    
    # Hidden stress test (procedural)
    stress = generate_stress_test(problem_id, rng)
    
    # Baseline config (from Landscape)
    baseline = get_current_baseline(problem_id)
    
    return Challenge(
        challenge_id=challenge_id,
        problem_id=problem_id,
        train_data=train,
        holdout_data=holdout,
        stress_data=stress,
        baseline_config=baseline,
        submission_deadline=now() + 20_hours,
        stress_floors=get_stress_floors(problem_id)
    )
```

---

## 5. Landscape Agent Specification

### 5.1 Data Model
```protobuf
message StrategyFragment {
  string fragment_id = 1;              // SHA256(config_json + challenge_id + timestamp)[:16]
  int32 miner_uid = 2;
  string challenge_id = 3;
  int32 problem_id = 4;
  string config_json = 5;
  float improvement = 6;               // log(E_baseline) - log(E_submission)
  bool stress_passed = 7;
  UQMetrics uq_metrics = 8;
  float score = 9;
  uint64 timestamp = 10.
  repeated string causal_parents = 11; // Fragment IDs this derived from
  map<string, float> param_values = 12; // Flattened config for analysis
}

message UQMetrics {
  float calibration_error = 1;         // |coverage - target|
  float sharpness = 2;                 // Mean prediction interval width
  float coverage = 3;                  // Empirical coverage
  bool calibrated = 4;                 // |coverage - 0.90| < 0.02
}
```

### 5.2 Daily Causal Update (Post-Challenge)
```python
def daily_causal_update(new_fragments: List[StrategyFragment]):
    # 1. Add to fragment DAG
    dag.add_nodes(new_fragments)
    
    # 2. Estimate causal effects per problem family
    for problem_id in PROBLEM_FAMILIES:
        problem_fragments = dag.get_problem_fragments(problem_id)
        
        # Double Machine Learning for heterogeneous treatment effects
        # Y = improvement, T = config_param, X = other_params + problem_context
        dml = DoubleML(
            model_y=RandomForestRegressor(),
            model_t=RandomForestRegressor(),
            n_folds=5
        )
        dml.fit(problem_fragments)
        
        # Estimate ATE for each tunable parameter
        effects = {}
        for param in TUNABLE_PARAMS:
            effects[param] = dml.ate(param)
        
        # Conditional effects: "fourier_modes helps when physics_loss > 1.0"
        interactions = dml.interaction_effects()
        
        # 3. Propose baseline update
        proposal = propose_baseline(effects, interactions, current_baseline)
        owner_review_queue.put(proposal)
```

### 5.3 Weekly Specialist Distillation
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
                alpha_uq=0.2           # UQ matching
            ),
            data=load_problem_data(problem_id)
        )
        
        # 3. Regression test: specialist must pass same stress tests
        stress_result = run_stress_test(student, get_stress_data(problem_id))
        if stress_result.hard_failure:
            log_failure("Distilled specialist failed stress test")
            continue
        
        # 4. Export ONNX + metadata
        specialist = Specialist(
            specialist_id=f"{problem_id}_v{version}",
            onnx_model=export_onnx(student),
            problem_signature=get_signature(problem_id),
            metrics=evaluate_full(student, problem_id),
            validity_domain=estimate_validity_domain(student, problem_id),
            license="AGPL-3.0 + Commercial Dual-License"
        )
        specialist_bank.publish(specialist)
```

### 5.4 Landscape Treasury & Governance
- **Funding:** 18% Owner emissions + 10% time-locked (6-month cliff, 2-year vest)
- **Multi-sig:** 3/5 (Owner, Lead Validator, Lead Researcher, Community Rep, Legal)
- **Spending:** Agent compute, specialist distillation GPU, challenge generation, audits
- **Audit:** Quarterly public report of Landscape Treasury spend + causal graph snapshots

---

## 6. Specialist Bank & Marketplace

### 6.1 Specialist Metadata
```protobuf
message Specialist {
  string specialist_id = 1;           // e.g., "navier_stokes_2d_v3"
  string problem_signature = 2;       // Hash of ProblemSignature protobuf
  bytes onnx_model = 3;               // ONNX model bytes
  Metrics7D metrics = 3;              // fidelity, accuracy, efficiency, ...
  ValidityDomain validity = 4;        // PDE params where specialist works
  repeated string known_failures = 5; // e.g., "fails at Re>500"
  License license = 6;                // DUAL: AGPL-3.0 + Commercial
  uint64 created_at = 7.
  string distilled_from = 8;          // Fragment IDs of teachers
}
```

### 6.2 Miner Usage (Phase 2+)
```json
{
  "specialist_id": "navier_stokes_2d_v3",
  "adapter": {                        // Optional micro-tuning
    "type": "lora",
    "rank": 4,
    "target_layers": ["pino_layers.3"]
  }
}
```
Validator loads ONNX → runs inference on holdout + stress test → scores.

### 6.3 Licensing Model
- **AGPL-3.0:** Free for research, open-source, internal use
- **Commercial License:** Per-seat or per-inference API, includes:
  - Indemnification
  - SLA (uptime, latency)
  - Fine-tuning support on private data
  - Validity domain extension consulting

---

## 7. Foundation Operator (Phase 3+)

### 7.1 Architecture
- **Backbone:** FNO/PINO hybrid (spectral + local kernels)
- **Width:** 128 channels, 6 spectral blocks
- **Conditioning:** ProblemSignature → FiLM layers modulate backbone
- **UQ:** Built-in evidential head (outputs μ, σ, ν, α for Student-t)

### 7.2 Training
```
Multi-teacher distillation from Specialist Bank:
  L = Σ w_i * MSE(foundation(x), specialist_i(x)) 
    + λ_physics * PhysicsLoss(foundation)
    + λ_uq * UQ_Calibration_Loss
    + λ_consistency * CrossSpecialistConsistency
```
**Data:** Union of all specialist training data + synthetic ProblemSignature sampling

### 7.3 Fine-Tuning API (Commercial)
```python
def fine_tune_foundation(client_data: EncryptedBlob, problem_signature: ProblemSignature) -> Specialist:
    # 1. Decrypt in TEE (NVIDIA CC / AMD SEV)
    data = tee_decrypt(client_data)
    
    # 2. Few-shot adaptation (10-50 steps, LoRA rank=8)
    adapted = lora_adapt(foundation, data, steps=20)
    
    # 3. Verify on client holdout (if provided) + stress tests
    if not verify(adapted, problem_signature):
        raise AdaptationFailed
    
    # 4. Return encrypted ONNX
    return tee_encrypt(export_onnx(adapted))
```

---

## 8. Consensus & Slashing

### 8.1 Validator Slashing Conditions
| Offense | Evidence | Slash |
|---------|----------|-------|
| Score deviation > 0.15 from median (3+ consecutive) | On-chain scores | 5% stake |
| Missed validation deadline (3+ consecutive) | Missed block window | 2% stake |
| Failed reproducibility audit | Auditor rerun ≠ submitted score | 10% stake |
| Physics check manipulation | Auditor finds disabled checks | 20% stake + ban |

### 8.2 Miner Penalties
| Offense | Penalty |
|---------|---------|
| Invalid JSON / schema violation | 0.1 TAO fee burned |
| Custom data checksum mismatch | 0.1 TAO fee burned |
| Spam (>5 submissions/challenge) | 24hr submission cooldown |

---

You're right - the checklist still has the old timeline. Here's the **updated Section 9** matching the new roadmap:

---

## 9. Implementation Checklist

### Phase 0: The Causal Baseline (Launch → Month 3)
- [ ] 5 backbone Docker images built & tested (`fno`, `pino`, `deeponet`, `gno`, `oformer`)
- [ ] 7 challenge datasets generated + stress floors calibrated (Poisson 2D/3D, Darcy 2D/3D, Burgers, NS 2D/3D laminar, Heat, Elasticity, Thermo-elasticity)
- [ ] Validator pipeline: train → evaluate → stress test → score (median consensus, 3+ validators)
- [ ] Miner CLI: submit JSON, pay 0.1 TAO fee, query results
- [ ] Chain pallet: challenge management, scoring consensus, per-challenge emission budget (40/30/20/10 split)
- [ ] Landscape Agent: fragment store, DML causal inference, daily baseline proposer
- [ ] Dashboard: live leaderboard, fragment explorer, causal graph

### Phase 1: Specialist Bank & Data Markets (Months 3-6)
- [ ] LoRA adapter support in validator (rank-4-8, target layer selection)
- [ ] Custom data ingestion + caching + data royalty pool (5% of emissions)
- [ ] Specialist distillation pipeline: multi-teacher → ONNX export → regression test (same stress tests)
- [ ] Dual-license legal framework (AGPL-3.0 + Commercial)
- [ ] Specialist Bank on-chain registry (specialist_id, onnx_model, validity_domain, license)
- [ ] Miner CLI: `custom_data` submission, `uq_config` support

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
- [ ] 3D Thermo-Elasticity: `elasticity_3d` + `heat_3d` + `thermal_expansion_3d_adapter` (FEniCS reference)
- [ ] 3D CHT: `ns_3d_turbulent` + `heat_3d` + `cht_3d_adapter` (OpenFOAM/COMSOL reference)
- [ ] Three-track leaderboard + same stress tests for all 3D multi-physics

**Phase 3.3: Foundation Operator (LPM)**
- [ ] Multi-teacher distillation across entire Specialist Bank (2D + 3D)
- [ ] FiLM conditioning on ProblemSignature
- [ ] Evidential UQ head (μ, σ, ν, α for Student-t)
- [ ] Commercial fine-tuning API: TEE decryption → LoRA rank=8 (10-50 steps) → stress test verification → encrypted ONNX return
- [ ] LPM fine-tuning API commercial launch

## 10. Appendix: Mathematical Definitions

### 10.1 Log-Space Improvement
```
improvement = log(E_baseline) - log(E_submission)
where E = relative L2 error = ‖u_pred - u_true‖₂ / ‖u_true‖₂
```
*Properties:* Rewards consistent orders-of-magnitude improvement; invariant to baseline magnitude.

### 10.2 Double Machine Learning (DML) for Causal Effects
```
Y = improvement, T = target_param, X = other_params + context
1. Regress Y ~ X → Ŷ, regress T ~ X → T̂
2. Residuals: Ỹ = Y - Ŷ, T̃ = T - T̂
3. ATE = E[Ỹ * T̃] / E[T̃²]
4. Heterogeneous effects: τ(x) = E[Ỹ | X=x] / E[T̃ | X=x]
```
*Robust to:* High-dimensional X, non-linear relationships, partial confounding.

### 10.3 UQ Calibration Metric
```
calibration_error = |(1/N) Σ 1{y ∈ PI_α(x)} - (1-α)|
sharpness = (1/N) Σ width(PI_α(x))
```
*Target:* calibration_error < 0.02 at α=0.90

---

*End of SPEC.md v2.0*
