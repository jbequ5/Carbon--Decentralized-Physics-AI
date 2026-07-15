# ROADMAP.md — Hydrogen Subnet Technical Roadmap v2.3

---

## Overview

This document defines the phased execution plan for Hydrogen, from launch through Foundation Operator. Each phase has clear entry/exit criteria, technical deliverables, and revenue milestones. Phases are sequential gates — not timeboxes. We advance when criteria are met.

---

## Phase 0: The Causal Baseline (Launch → Month 3)

### Objective
Establish the causal flywheel: miners submit strategies → validators verify with physics gates → Landscape distills causal knowledge → baseline improves.

### Scope: 7 Problems, Full PDE Taxonomy (2D + 3D Elliptic/Parabolic + Nonlinear + Multi-Physics)

| # | Problem | Dimension | Physics Class | Reference Source |
|---|---------|-----------|---------------|------------------|
| 1 | Poisson | 2D / 3D | Elliptic, constant-coeff | PhysicsNeMo |
| 2 | Darcy | 2D / 3D | Elliptic, variable-coeff | PhysicsNeMo / PDEBench |
| 3 | Burgers | 2D | Nonlinear advection/shocks | PhysicsNeMo |
| 4 | Navier-Stokes | 2D / 3D | Incompressible (2D vortex / 3D laminar Re≤100) | PhysicsNeMo / JHTDB |
| 5 | Heat | 2D | Transient, variable κ | PhysicsNeMo |
| 6 | Elasticity | 2D | Vector, tensor physics | PhysicsNeMo |
| 7 | Thermo-elasticity | 2D | **Multi-physics, loss_vector** | Generated (48 Tier 1 refs) |

**Each challenge provides:** public training split, public holdout set, hidden stress test (procedural parameter/geometry shifts + **The Well data slices**, seeded by challenge_id). **The Well data slices are custom per challenge** (see SPEC §4.6).

### Technical Deliverables

**Validator Infrastructure**
- 5 pinned Docker images: `fno`, `pino`, `deeponet`, `gno`, `oformer` (PhysicsNeMo 24.09 + NeuralOperator 0.3.0, PyTorch 2.3, CUDA 12.4)
- **NEW:** JAX/Equinox validator images (`hydrogen/validator:jax-fno-v24.09`, `jax-pino-v24.09`, `jax-deeponet-v24.09`) via `NeuralOperators.jl` + `juliacall`
- Validation pipeline: train → public holdout eval → hidden stress test → physics gates → score
- Physics gates: mass (‖∇·u‖₁<1e-3), energy (dE/dt≤1e-4), boundary, rollout (100 steps), UQ calib (90% PI ∈ [0.88,0.92])
- Log-space improvement score: `log(E_baseline) - log(E_submission)`

**Miner Interface**
- Strategy JSON: backbone, loss_vector, optimizer, curriculum, UQ config, optional custom_data
- 0.1 TAO submission fee, 1 submission/challenge, last submission counts

**Landscape Agent (Owner-funded, 18% emissions + 10% time-locked treasury)**
- Fragment store: StrategyFragment (config, score, stress_pass, UQ, causal_parents)
- Daily DML causal update: `P(improvement | do(param))` per problem family
- Baseline proposal: updated JSON with strongest causal effects
- Fragment DAG with `causal_parents` lineage tracking

**Agent Infrastructure (Phase 0 MVP)**
- **REST API**: `/challenges`, `/baseline`, `/priors`, `/specialists`, `/submit`
- **hydrogen-agent Python SDK**: local validation, baseline/prior loading, mutation helpers, submission wrapper
- **Structured feedback**: physics gate breakdown, stress test details, causal insights, suggestions

**The Well Integration (Phase 0)**
- **Custom per-challenge stress test slices** from The Well (15TB multi-domain physics dataset)
- **Well mapping per problem**: Poisson→active_matter, Darcy→active_matter, Burgers→turbulence/viscoelastic, NS→turbulence/mhd, Heat→acoustic_scattering, Elasticity→active_matter/viscoelastic, Thermo-elasticity→active_matter/viscoelastic
- **1-2 optional "The Well Track" challenges** as non-core stretch goals

### Edge HIL Integration (Phase 0 Foundation)
- **Symbolic metadata generation** per challenge (symmetries, conservation laws, dimensionless groups) — foundation for later edge deployment
- **Symbolic metadata schema** defined in SPEC §4.6 — cached per challenge, drives validator physics gates and miner loss weights
- **Well data slices** integrated into hidden stress tests per challenge (custom per PDE)

### Exit Criteria (All Required)
- [ ] Baseline log-improvement > 0.02/challenge averaged over 30 challenges
- [ ] ≥30 unique miners submitting
- [ ] ≥5 validators operational
- [ ] Landscape proposing baseline updates that improve next-challenge baseline
- [ ] All 7 problems have active competition
- [ ] Agent SDK (`hydrogen-agent`) published and documented

### Revenue at Phase 0 Exit
- **Asset:** Causal Knowledge Graph (500+ causally-validated interventions across 7 PDE problems)
- **TAM:** licensing to NVIDIA/ANSYS/Siemens for "training best practices" datasets
- **Revenue model:** API access to causal graph + baseline configs

---

## Phase 1: Specialist Bank & Data Markets (Months 3-6)

### Objective
Enable fine-grained innovation (adapters), reward data contribution, produce first verified specialists.

### New Miner Capabilities
- **LoRA adapters:** Rank-4-8 weight deltas applied to frozen backbone (`target_layers`, `rank`, `alpha`)
- **Custom datasets:** High-fidelity DNS, curated permeability fields, shock-capturing data
  - `data_uri` (IPFS), `checksum`, `usage` (augment/curriculum/label_only), `weight`
  - Encrypted submission option (landscape pubkey)

### Validator Updates
- Adapter application: load backbone → apply LoRA → train
- Custom data: download → checksum → optional decrypt → sanity probe → cache → mix per `usage`/`weight`
- Data royalty pool: 5% of emissions distributed by measured impact (Δerror when data added)

### Landscape Agent Extensions
- Adapter parameter correlation in DML
- Data impact measurement: `Impact = max(0, (E_without - E_with)/E_without)`
- **Specialist Distillation Pipeline (weekly):**
  1. Select top-K fragments per problem (score > 90th percentile, stress_passed)
  2. Multi-teacher distillation → Student (same backbone, 50% width)
  3. Loss: `α·Output_MSE + β·Physics_Loss + γ·Feature_Match + δ·UQ_Match`
  5. **Regression test:** Specialist MUST pass SAME stress tests as miners
  6. Publish ONNX + metadata + validity_domain + dual license (AGPL-3.0 + Commercial)

### Specialist Schema
```json
{
  "specialist_id": "darcy_2d_v3",
  "problem_signature": "...",
  "onnx_model": "...",
  "validity_domain": {"permeability_contrast": "≤10^4", "geometry": "structured"},
  "stress_test_results": {"mass": "pass", "energy": "pass", "rollout": "pass", "uq_calib": "pass"},
  "metrics_7d": {...},
  "license": "DUAL: AGPL-3.0 + Commercial",
  "provenance": ["fragment_id_1", "fragment_id_2", "..."]
}
```

### Exit Criteria
- [ ] ≥20 specialists in bank across ≥6 problem families
- [ ] Data royalty pool >5% of emissions
- [ ] Specialist reuse rate >80% (compositions use Bank specialists)
- [ ] ≥3 novel adapters with measurable impact (>5% improvement)

### Revenue at Phase 1 Exit
- **Asset:** Specialist Bank (20-30 verified ONNX specialists, dual-licensed)
- **TAM:** 
  - Specialist licensing (AGPL-3.0 ecosystem + commercial dual-license)
  - Fine-tuning services on encrypted client data (TEE)

---

## Phase 1.5: Abaqus ODB/fil Ingestion & Phase-Field Gates (Months 3-6)

### Abaqus ODB/fil Ingestion Pipeline (Phase 1 Deliverable)
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

### Phase-Field Specific Physics Gates (Phase 1+)
**NEW PHYSICS GATES for phase-field fracture (AT1/AT2, CZM, PF-CZM):**

| Gate | Formula | Threshold | Physics Meaning |
|------|---------|-----------|-----------------|
| **Crack Irreversibility** | `min(∂d/∂t)` | `≥ 0` | Crack irreversibility (∂d/∂t ≥ 0) |
| **Length Scale ℓ Enforcement** | `‖ℓ_pred - ℓ_true‖₂ / ‖ℓ_true‖₂` | `< 0.05` | Phase-field length scale consistency |
| **Degradation Function g(d)** | `‖g(d)_pred - g(d)_true‖₂ / ‖g(d)_true‖₂` | `< 0.05` | Degradation function validity |
| **History Variable H** | `‖H_pred - H_true‖₂ / ‖H_true‖₂` | `< 0.1` | History variable tracking |
| **Crack Irreversibility (Integral)** | `∫(∂d/∂t)_- dt` | `= 0` | No healing (crack irreversibility) |

These gates activate for phase-field fracture challenges (AT1/AT2, CZM, PF-CZM) and are enforced in the validator stress test pipeline.

---

## Phase 2: Composition Engine & Specialist Marketplace (Months 6-18)

### Objective
Prove composition > monolith on multi-physics. Build composable specialist marketplace.

### Phase 2A: Verified Benchmarks Only (Months 1-3)
**Leverage existing verified 2D benchmarks — zero new data generation.**

| Problem | Source | Status | Specialist Pair |
|---------|--------|--------|-----------------|
| FSI 2D-1/2/3 | Turek/Hron | Verified, mesh-converged | `ns_2d` + `elasticity_2d` + `fsi_coupling` |
| CHT: Solid cooling / Electronics | PDEBench | Verified, mesh-converged | `ns_2d` + `heat_2d` + `cht_coupling` |

### Phase 2B: Thermo-Elasticity (Month 3)
**Only new data generation needed.** Generate 48 Tier 1 references (β×κ×geometry) at 256² with FEniCS monolithic, mesh-converged. Cost: ~$3K.

### Phase 2C: Variant Expansion (Months 4-5)
New Re, geometries, coupling strengths on FSI/CHT/thermo-elasticity using existing references.

### Miner Submission (Phase 2)
```json
{
  "specialist_pipeline": [
    {"specialist_id": "ns_2d_v4", "role": "primary"},
    {"specialist_id": "heat_2d_v3", "role": "secondary"},
    {"adapter_id": "cht_coupling_v2", "role": "coupling"}
  ],
  "execution_schedule": "staggered",
  "max_coupling_iterations": 5,
  "coupling_tolerance": 1e-4
}
```

### Six-Track Leaderboard (Every Multi-Physics Challenge)
| Track | Submission | What It Proves |
|-------|------------|----------------|
| **Monolith** | Single strategy JSON (end-to-end training config for coupled problem) | Can a monolithic model beat composition? |
| **Composition** | Specialist pipeline: `{"specialist_pipeline": [{"specialist_id": "ns_2d_v4"}, {"specialist_id": "heat_2d_v3"}, {"adapter_id": "cht_coupling_v2"}]}` | Does composition beat monolith? |
| **Specialist-Only** | Single specialist ID (no adapter) | How much does the coupling adapter matter? |
| **Symbolic Regression** | Discovered PDE string + basis | Can the agent discover governing PDE from data? |
| **Symbolic Composition** | MTK component graph + adapters | Can symbolic components compose to beat monolith? |
| **Symbolic Distillation** | ONNX + symbolic metadata + CUDA kernel | Can specialist be compressed with symbolic metadata preserved? |

**Six parallel leaderboards.** Same hidden stress test, same physics gates.

### Phase 2C Exit Criteria (Go/No-Go for 3D)
| Metric | Target | If Missed |
|--------|--------|-----------|
| Composition win rate | >60% (Composition > Monolith) | Pivot: deepen single-physics specialist depth |
| Specialist reuse | >80% compositions use ≥1 Bank specialist | Extend Phase 1 |
| Adapter innovation | >30% novel adapters | Expand adapter design space |
| Stress test pass rate | >70% compositions pass | Coupling brittle → simplify adapter design |

### Revenue at Phase 2 Exit
- **Asset:** Composition Engine + Specialist Bank (50+ specialists, composable)
- **TAM:** 
  - Specialist licensing (AGPL-3.0 + commercial)
  - Composition engine licensing (COMSOL, Ansys, Siemens)
  - Custom multi-physics pipelines for enterprise ($10M-100M contracts)

---

## Phase 3: 3D Transition & Foundation Operator (Months 18+)

### Phase 3 Entry Gates (All Required)
| Gate | Metric | Threshold |
|------|--------|-----------|
| 2D composition proven | Composition win rate >60% | Phase 2 complete |
| 3D single-physics specialists exist | ≥6 specialists in Bank | Phase 3.0 prereq |
| Curriculum validated | 2D→3D fine-tune <50% scratch cost | Ablation study |
| 3D reference pipeline works | Tier 1 3D reference generated, mesh-converged | Pipeline test |
| Validator quorum | ≥5 validators with 24GB+ GPUs | Infrastructure ready |

---

## Phase 3.0: 3D Single-Physics Foundations (Prerequisite)
**Before any 3D multi-physics, these MUST exist in Specialist Bank:**

| 3D Specialist | Source | Validation |
|---------------|--------|------------|
| `poisson_3d_v1` | Phase 0 3D Poisson | Stress test passed |
| `darcy_3d_v1` | Phase 0 3D Darcy | Stress test passed |
| `ns_3d_laminar_v1` | Phase 0 3D NS (Re≤100) | Stress test passed |
| `heat_3d_v1` | Phase 0 3D Heat | Stress test passed |
| `elasticity_3d_v1` | 2D→3D curriculum distillation | Stress test passed |
| `heat_3d_v1` | 2D→3D curriculum distillation | Stress test passed |

**Curriculum Distillation Protocol:**
```yaml
stage_1: zero_pad_fourier + noise(0.01), freeze backbone, train adapter (32³, 5 epochs)
stage_2: unfreeze last 2 spectral blocks, 64³, 20 epochs
stage_3: unfreeze all, 128³, 50 epochs, physics_informed + distillation_loss(teacher_2d)
stage_4: 128³ stress test validation + must_pass_2d_stress_tests_in_3d
```

---

## Phase 3.1: 3D Turbulence Bridge (Critical - Months 1-3)
**2D NS does not prepare for 3D turbulence.** Dedicated phase to learn 3D turbulence FROM SCRATCH.

| Week | Activity | Deliverable |
|------|----------|-------------|
| 1-2 | 3D Spectral Initialization Protocol | Proper 3D spectral init (NOT zero-pad) using 3D energy spectrum priors |
| 3-4 | 3D Turbulence Curriculum | Re=50→100→200→500 on channel/cylinder |
| 5-6 | 3D Turbulence Specialist | `ns_3d_turbulent_v1` with verified k^(-5/3) spectrum |
| 7-8 | 3D FSI Bridge | `ns_3d_turbulent` + `elasticity_3d` + FSI adapter |
| 9-10 | 3D Turbulence Stress Gates | Energy spectrum, Q-criterion, wall shear, Nu distribution |

**Gate:** `ns_3d_turbulent_v1` passes 3D turbulence stress tests → open 3D multi-physics.

### Phase 3.1 Gate (All Required Before 3.2)
| Metric | Threshold |
|--------|-----------|
| Energy spectrum | Verified k^(-5/3) scaling |
| Q-criterion gate | Pass |
| Wall shear stress distribution | Pass |
| Nu distribution (3D corners) | Pass |
| Stress test pass rate | >70% |

---

## Phase 3.2: 3D Multi-Physics Rollout

| Phase | Challenges | Specialist Composition | Reference |
|-------|------------|------------------------|-----------|
| **3.2A** 3D FSI | Cylinder, flap, turbulent inflow | `ns_3d_turbulent` + `elasticity_3d` + `fsi_3d_adapter` | preCICE partitioned |
| **3.2B** 3D Thermo-Elasticity | Bimetal, engine block, turbine blade | `elasticity_3d` + `heat_3d` + `thermal_expansion_3d` | FEniCS monolithic |
| **3.2C** 3D CHT | Electronics, turbine cooling, battery | `ns_3d_turbulent` + `heat_3d` + `cht_3d_adapter` | OpenFOAM/COMSOL |

**Three-track leaderboard continues. Same stress test infrastructure.**

---

## Phase 3.3: Foundation Operator (LPM)
**Multi-teacher distillation across entire Specialist Bank (2D + 3D).**

```
LPM Architecture:
  Backbone: FNO/PINO hybrid (spectral + local kernels), 128 channels, 6 blocks
  Conditioning: ProblemSignature → FiLM layers modulate backbone
  UQ Head: Evidential (μ, σ, ν, α for Student-t)
  Router: Optional learned gating to specialist sub-networks

Training:
  L = Σ w_i * MSE(LPM(x; sig_i), Specialist_i(x))
    + λ_physics * PhysicsLoss(LPM)
    + λ_uq * UQ_Calibration_Loss
    + λ_consistency * CrossSpecialistConsistency
```

**Fine-tuning API (Commercial):**
```python
def fine_tune_lpm(client_encrypted_data, problem_signature):
    # 1. Decrypt in TEE (NVIDIA CC / AMD SEV)
    data = tee_decrypt(client_data)
    # 2. Few-shot adaptation (10-50 steps, LoRA rank=8)
    adapted = lora_adapt(foundation, data, steps=20)
    # 3. Verify on client holdout + stress tests
    if not verify(adapted, problem_signature): raise AdaptationFailed
    # 4. Return encrypted ONNX
    return tee_encrypt(export_onnx(adapted))
```

---

## HIL Integration Across Phases

| Phase | HIL Capability | Target Hardware | Key Enabler |
|-------|----------------|-----------------|-------------|
| **Phase 0** | **Symbolic metadata generation** per challenge (symmetries, conservation laws, validity domain) — foundation for edge deployment | — | ModelingToolkit.jl symbolic extraction |
| **Phase 1** | **Symbolic metadata attached to specialists** (symmetries, conservation laws, validity domain, CUDA kernel) | — | Specialist distillation with symbolic metadata |
| **Phase 1.5** | **Abaqus ODB/fil ingestion** → custom data pipeline; **Phase-field physics gates** added | — | Abaqus ODB/fil ingestion; Phase-field physics gates |
| **Phase 2** | **Specialist composition** via symbolic interfaces; **Symbolic distillation** preserves metadata | — | ModelingToolkit acausal composition |
| **Phase 2.5** | **Edge HIL deployment begins:** Specialist → ONNX + symbolic metadata → CUDA kernel → Jetson Orin deployment | Jetson Orin / AGX Orin | ModelingToolkit → CUDA codegen; TensorRT custom plugins for physics gates |
| **Phase 3.0** | **3D single-physics specialists** with full symbolic metadata + CUDA kernels | — | 3D curriculum distillation from 2D specialists |
| **Phase 3.1** | **3D Turbulence Bridge** — dedicated 3-month phase to learn 3D turbulence from scratch | — | 3D spectral init protocol, turbulence curriculum |
| **Phase 3.2** | **3D Multi-physics HIL**: 3D FSI, Thermo-elasticity, CHT specialists deployed to Jetson Orin / AGX Orin | Jetson Orin / AGX Orin | ModelingToolkit → CUDA codegen; TensorRT custom plugins for physics gates |
| **Phase 3.2+** | **3D HIL Deployment**: Specialists → ONNX + symbolic metadata → CUDA kernel → Jetson Orin / AGX Orin deployment | Jetson Orin / AGX Orin | TensorRT custom plugins for physics gates; ModelingToolkit → CUDA codegen |
| **Phase 3.2+** | **3D HIL Deployment:** Specialists → ONNX + symbolic metadata → CUDA kernel → Jetson Orin / AGX Orin deployment | Jetson Orin / AGX Orin | TensorRT custom plugins for physics gates; ModelingToolkit → CUDA codegen |
| **Phase 3.2+** | **3D HIL Deployment:** Specialists → ONNX + symbolic metadata → CUDA kernel → Jetson Orin / AGX Orin deployment | Jetson Orin / AGX Orin | TensorRT custom plugins for physics gates; ModelingToolkit → CUDA codegen |
| **Phase 3.3** | **Foundation Operator (LPM)** fine-tuning API (TEE, LoRA) for custom edge deployment | — | TEE fine-tuning, LoRA adapters |

### HIL Deployment Targets by Phase

| Phase | Target Hardware | Deployment Path | Physics Compliance |
|-------|-----------------|-----------------|-------------------|
| **Phase 1-2** | **Jetson Orin / AGX Orin** | ONNX + Symbolic Metadata → TensorRT + Custom Physics Plugins | Physics gates enforced via custom TensorRT plugins |
| **Phase 3.0-3.2** | **Jetson Orin / AGX Orin** | ONNX + Symbolic Metadata → ModelingToolkit → CUDA Kernel → TensorRT | Physics gates enforced via CUDA kernels + TensorRT plugins |
| **Phase 3.2+** | **Industrial PLC / Beckhoff / Siemens** | ONNX + Symbolic Metadata → ModelingToolkit → C/C++ → PLC Runtime | Symbolic metadata drives constraint enforcement |
| **Phase 3.3+** | **FPGA/ASIC (Versal, Agilex)** | ONNX → ModelingToolkit → C → HLS → VHDL/Verilog → Bitstream | **Phase 3+ Research** (not Phase 0-2 deliverable) |

---

## Revenue & TAM Summary by Phase

| Phase | Primary Asset | Revenue Streams | TAM | Timeline |
|-------|---------------|-----------------|-----|----------|
| **Phase 0** | Causal Knowledge Graph | Licensing to NVIDIA/ANSYS/Siemens | Month 3 |
| **Phase 1** | Specialist Bank (20-30) | Licensing (dual), data royalties, fine-tuning API | Month 6 |
| **Phase 2** | Composition Engine + Bank (50+) | Licensing, custom pipelines, enterprise contracts | Month 12-18 |
| **Phase 3** | Foundation Operator (LPM) | API ($500-5K/model), custom surrogates ($50K-500K), enterprise platform | Year 2-3 |

---

## Risk Mitigation & Pivot Points

| Risk | Probability | Mitigation | Pivot Trigger |
|------|-------------|------------|---------------|
| 3D turbulence bridge fails | High | 2D composition thesis proven; 3D elliptic/parabolic specialists valuable | Pivot: double down on Phase 2 composition engine + elliptic/parabolic specialists |
| 3D reference costs exceed budget | Medium | Tiered fidelity (gold/silver/bronze); 2D→3D curriculum reduces training cost | Reduce 3D challenge frequency; focus on 2D/3D elliptic |
| Validator centralization (24GB+ GPU requirement) | Medium | Fee adjustment; validator emission increase; cloud validator program | Increase submission fee; validator emission share increase |
| Reference solver uncertainty | High | Tiered fidelity (gold/silver/bronze); cross-code verification for Tier 1 | Accept lower-fidelity stress tests; flag in metadata |
| Miner exodus | Low | Landscape has 500+ fragments → specialists → Foundation Operator survives | Emissions become "innovation bonus"; revenue from IP sustains |

---

## The Honest Summary

| Phase | Business Status | Technical Risk | Revenue Independence |
|-------|-----------------|----------------|----------------------|
| **Phase 0** | Proven flywheel, validated asset | Low | ✅ Yes |
| **Phase 1** | Specialist marketplace, recurring revenue | Low | ✅ Yes |
| **Phase 2** | Composition engine, high-value IP | Medium | ✅ Yes (2D multi-physics) |
| **Phase 3** | Moonshot (LPM) | **High (3D turbulence)** | ❌ Depends on 3D turbulence |

**The business is real at Phase 1. Profitable at Phase 2. Transformative at Phase 3.**

## Key Technical Specifications Summary

### Emission Mechanics (All Phases)
- **Challenge Budget:** `total_emission / min(active_challenges, 10)`
- **Top-4 Split:** 40% / 30% / 20% / 10%
- **Bounty Accumulation:** Emissions pool until log-improvement > 0
- **Private-until-proven:** Strategies revealed only after earning rewards
- **Miner burn:** 0%

### Phase-Gated Physics Requirements

| Requirement | Phase 0-1 | Phase 2 | Phase 3 |
|-------------|-----------|---------|---------|
| **UQ Calibration** | 90% | 95% | 99% |
| **Rollout Steps** | 100 | 100 | 1000+ + spectral stationarity |
| **UQ Bonus** | 5% at 90% | 5% at 95% | 5% at 99% |
| **NS Rollout** | 100 steps | 100 steps | 1000+ steps + spectral stationarity |
| **Thermo-elasticity Rollout** | 100 steps | 100 steps | 1000+ steps |

### 3D-Specific Stress Gates (Phase 3+)
| Gate | Formula | Threshold |
|------|---------|-----------|
| Energy Spectrum | `‖E_pred(k) - C·k^(-5/3)‖ / ‖E_true(k)‖` | `< 0.1` |
| Q-Criterion | `‖Q_pred - Q_true‖₂ / ‖Q_true‖₂` | `< 0.15` |
| Wall Shear Stress | `‖τ_w_pred - τ_w_true‖₂ / ‖τ_w_true‖₂` | `< 0.1` |
| Nu Distribution (Corners) | `‖Nu_pred - Nu_true‖₂ / ‖Nu_true‖₂` | `< 0.2` |
| Added-Mass Tensor | `‖A_pred - A_true‖_F / ‖A_true‖_F` | `< 0.1` |

### Thermo-Elasticity Coupling (Bidirectional)
```json
"loss_vector": {
  "elasticity_residual": 1.0,
  "heat_residual": 0.8,
  "coupling_thermal_strain": 0.5,    // α * ΔT * I (thermal → mechanical)
  "coupling_heat_source": 0.3,       // β * ∂(∇·u)/∂t (mechanical → thermal)
  "boundary": 0.5
}
```

### 3D Spectral Initialization (Correct)
```python
def init_3d_spectral_from_2d(weights_2d, spectrum="kolmogorov"):
    # 1. Initialize 3D modes with target spectrum E(k) ~ k^(-5/3)
    # 2. Project 2D modes onto k_z=0 plane with energy scaling
    # 3. Add controlled noise in orthogonal directions (σ ~ 0.01)
    # 4. DO NOT copy 2D weights directly — 2D spectrum is k^(-3), not k^(-5/3)
```

### Phase-Gated UQ & Rollout
```python
def get_phase_targets(phase):
    if phase <= 1: return {"uq_target": 0.90, "rollout_steps": 100}
    elif phase == 2: return {"uq_target": 0.95, "rollout_steps": 100}
    else: return {"uq_target": 0.99, "rollout_steps": 1000}
```

