# Hydrogen: The Self-Improving Physics Engine

*A Bittensor subnet for Physical Intelligence.*

---

## The Current State of Physics ML

Neural operators (FNO, PINO, DeepONet, GNO, OFormer) are the most important breakthrough in computational science in two decades.

Traditional simulation discretizes physics onto meshes and solves linear systems—O(N³) scaling. For 3D turbulence at real Reynolds numbers, you need massive machines and weeks of compute.

Neural operators learn the **solution operator itself**: the mapping from forcing, initial conditions, material properties → solution field. Inference is a forward pass. Milliseconds on a GPU. Cost amortized over infinite queries.

The real breakthrough: **discretization invariance**. Following the Berner et al. recipe, a properly constructed neural operator learns a mapping between *function spaces*, not vectors. Train on 64², evaluate on 256²—it works because the architecture respects the continuous operators underneath. This enables zero-shot super-resolution, multi-scale modeling, geometry transfer.

The theory is solid (Berner et al., Nature 2026). The implementations are production-ready (PhysicsNeMo, NeuralOperator). But they have a fatal flaw: **they optimize for loss functions, not physics.**

A surrogate trained on MSE minimizes MSE. It doesn't know ∇·u = 0 is a law, not a suggestion. It doesn't know energy dissipation is a constraint, not a penalty. It doesn't know shocks must satisfy Rankine-Hugoniot. And because training distributions never cover deployment space, violations appear exactly where they hurt most: the extrapolation regime, the edge case, the novel design.

---

## The Tribal Knowledge Trap

The field knows this. Every paper proposes a new physics-informed loss, a new architecture tweak, a new training trick, but discoveries stay tribal.

The PhD who discovers that *curriculum learning + ghost-cell boundaries + physics loss weight 1.8* works for transonic flow publishes a paper. Two years later, a team at another institution rediscovers it. Another PhD finds that Fourier modes scale with dimension as N^(d/2). Another learns that curriculum learning only helps when start resolution ≤ 0.5× end resolution.

**The field moves at publication speed, not discovery speed.**

We don't need better neural operators. We need a system that **compounds knowledge about how to train them.**

---

## The Foundation

We are not building neural operators from scratch. We are building on three mature, production-grade foundations:

**PhysicsNeMo** (NVIDIA) provides the reference implementations: FNO, PINO, DeepONet, GNO, OFormer with built-in physics-informed losses, conservation constraints, spectral layers, boundary handling, and UQ tooling. It is the PyTorch of physics-informed neural operators—battle-tested in industrial deployments from climate modeling to semiconductor design.

**NeuralOperator** (the library behind the Berner et al. Nature paper) provides the mathematical backbone: the layer conversions that make standard architectures discretization-agnostic, the quadrature weights that preserve conservation, the spectral and graph operator implementations. It is the reference implementation of the theory.

**Julia & the SciML Ecosystem** — The mathematical backbone for the symbolic layer. The Julia Lab at MIT built the SciML ecosystem: DifferentialEquations.jl, ModelingToolkit.jl, NeuralPDE.jl, NeuralOperators.jl, DataDrivenDiffEq.jl. This is the symbolic mathematics engine that lets us parse PDEs into abstract syntax trees, extract symmetries and conservation laws automatically, discover governing equations from data, and generate deployable CUDA/VHDL/Rust code from symbolic models. 

Hydrogen wraps these libraries in pinned, versioned Docker images. Validators pull `hydrogen/validator:pino-v24.09` and get exactly the same PhysicsNeMo + NeuralOperator + PyTorch + CUDA stack every time. Miners never install dependencies. They only tune the knobs these libraries already expose.

**We are not inventing the engine. We are building the race track, the timing system, and the engineering team that learns from every lap.**

---

## The Symbolic Layer: Giving Neural Operators a Physics Brain

Hydrogen integrates **ModelingToolkit.jl** (Julia's symbolic modeling framework) as a **symbolic preprocessing and reasoning layer** that sits between challenge definition and neural operator training.

### What It Does

| Capability | Impact on Hydrogen |
|------------|-------------------|
| **Symbolic PDE Parsing** | Converts PDE specs into manipulable mathematical objects |
| **Automatic Feature Extraction** | Extracts symmetries, conservation laws, dimensionless groups from PDEs |
| **Automatic Loss Weighting** | Physics computes loss weights — miners don't guess |
| **Symbolic Regression** | Discovers governing PDEs from specialist behavior → new challenges |
| **Symbolic Distillation** | Preserves physics structure when compressing specialists |
| **Acausal Composition** | Specialists compose like LEGO blocks via symbolic interfaces |
| **Code Generation** | Specialists compile to CUDA kernels for edge deployment |

### Where It Fits

```
Challenge → Symbolic Layer → Miner Strategy → Validator → Landscape Agent → Specialist Bank
                ↑                    ↑              ↑              ↑              ↑
           PDE → AST          Auto loss      Physics gates   PDE discovery   ONNX + sym.
                                weights          + sym.    → new challenges    metadata + CUDA
```

| Stage | Symbolic Layer Action | Value to Hydrogen |
|-------|----------------------|-------------------|
| **Challenge Definition** | Parses PDE → symbolic AST; extracts symmetries, conservation laws, dimensionless groups | Rich feature vectors for neural operator conditioning |
| **Miner Strategy** | Auto-computes loss weights from PDE structure; generates symbolic constraints | Physics computes loss weights; miners don't guess |
| **Validation** | Physics gates informed by PDE structure (adaptive thresholds) | Gates adapt to PDE physics (e.g., stricter mass conservation for incompressible flow) |
| **Landscape Agent** | PDE discovery from specialist behavior → new challenges; causal DML on enriched fragments | Automatic PDE discovery → new challenges; causal inference on enriched fragments |
| **Specialist Bank** | Symbolic metadata (symmetries, conservation laws) attached to ONNX; CUDA codegen | Composable specialists; edge deployment via CUDA codegen |

---

## Why Bittensor? Why a Subnet?

Because the incentive structure *is* the product.

In a centralized benchmark, you optimize for the metric. In a subnet, you optimize for *what validators actually check*—and validators check physics. The hidden stress test is the ultimate anti-gaming mechanism: you cannot overfit what you've never seen. Physics gates (mass conservation, energy dissipation, rollout stability, UQ calibration) are binary—you pass or you fail. No "good enough for the leaderboard."

Because the barrier to entry must be an idea, not a GPU cluster. A researcher in Bangalore with a clever curriculum idea competes on equal footing with a quant fund running H100s. The H100s run the validation; the clever idea wins the emissions.

Because validator consensus replaces trust. Physics checks are deterministic. The same backbone image, same data, same miner config run by five validators produces five identical scores. Median consensus. No politics.

Because knowledge must compound autonomously. In traditional science, a failed experiment teaches nobody. In Hydrogen, a losing strategy teaches the Landscape "don't do that for this PDE class." The 99% of submissions that don't win become the training data for the next baseline.

Bittensor provides the economic substrate where **verified improvement is the only currency**, infrastructure is paid for by the value it creates, and the moat compounds automatically.

---

## The Hydrogen Insight: A Market, Not a Model
We don't need a better neural operator. We need a system that compounds knowledge about how to train them.

Hydrogen is a Bittensor subnet that turns the sim-to-real gap into a compounding physical knowledge economy.
```
┌─────────────────────────────────────────────────────────────────┐
│                        HYDROGEN ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   MINERS                                                        │
│   Submit strategy JSONs (backbone, loss weights, curriculum)    │
│   Pay fee • No GPU needed • Never upload weights                │
│          │                                                      │
│          ▼                                                      │
│   VALIDATORS                                                    │
│   Pinned Docker • Train on public split • Hidden stress test    │
│   Hard physics gates (mass, energy, UQ) → binary pass/fail      │
│          │                                                      │
│          ▼                                                      │
│   LANDSCAPE AGENT                                               │
│   Double ML on fragment DAG → Causal effects → New baselines    │
│   Distills winners → ONNX specialists → Specialist Bank         │
│          │                                                      │
│          ▼                                                      │
│   SPECIALIST BANK                                               │
│   Composable ONNX specialists + symbolic metadata + CUDA code   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```
## The Mechanism

**Open challenges.** A set of challenges is open for competition at any time. Each challenge defines a PDE problem with a public training split, a public holdout set, and a **hidden stress test** generated procedurally from the challenge ID (shifted Reynolds numbers, resolutions, geometries, forcing). No miner has ever seen the stress test.

**Miners submit strategy JSONs.** A strategy specifies: backbone choice (FNO, PINO, DeepONet, GNO, OFormer), a **loss vector** with per-physics-term weights (pde_residual, conservation, boundary, symmetry, coupling), optimizer, curriculum schedule, UQ method (deep ensemble, conformal, evidential). They pay a submission fee. They never touch a GPU. They never upload weights.

**Validators run the training.** They pull a pinned Docker image (PhysicsNeMo + NeuralOperator, pinned PyTorch/CUDA), inject the JSON, train on the public training split, evaluate on the public holdout, then run the hidden stress test through **hard physics gates**:
- Mass conservation: ‖∇·u‖₁ < 1e-3
- Energy dissipation: dE/dt ≤ 1e-4
- Boundary satisfaction
- Rollout stability over 100 steps
- UQ calibration: 90% prediction intervals must cover truth 90% of the time (±2%)

Hard failure = score zero. Physics is binary.

**Score = log(E_baseline) - log(E_submission).** Improvement measured in log-space against the current baseline.

**Emission distribution per challenge.** Each challenge has an emission budget. The top 4 ranked miners split that challenge's emission budget: **40% / 30% / 20% / 10%**. Winner takes 40%, 2nd takes 30%, 3rd takes 20%, 4th takes 10%. Ranks 5+ receive zero emissions for that challenge.

**Every submission becomes a StrategyFragment.** The config, the score, the stress result, UQ metrics, and its lineage in the fragment DAG. Winning or losing—every fragment teaches the Landscape.

---

## The Landscape Agent: Where the Magic Compounds

The Landscape doesn't just correlate. It runs **Double Machine Learning on the fragment DAG** to estimate `P(improvement | do(param))`—the *causal effect* of a config change, not the spurious correlation.

It discovers that Fourier modes 32 help *only when* physics loss > 1.0. That curriculum helps *only when* start resolution ≤ 0.5× end resolution. That ghost cells prevent boundary locking *only for* elasticity and NS, not Poisson.

Every challenge cycle, it proposes a new baseline JSON incorporating the strongest causal effects. Every distillation cycle, it takes the top-K strategies and distills them into **ONNX specialists** via multi-teacher distillation—regression tested against the same stress tests—and publishes them to the Specialist Bank with validity domains and dual licensing (AGPL-3.0 + commercial).

The Landscape is funded by the Owner's allocation plus a time-locked treasury. The agent is replaceable; the causal knowledge graph is not.

---
## Agent-Native Architecture: An Economy of Intelligence

Hydrogen isn't just a human marketplace—it's an **agent economy**. Autonomous agents participate as first-class citizens, forming swarms that hunt for better physics models 24/7.

### Agent Identity & Stake
- **DID-based identity** (`did:hydrogen:agent:xyz`) with staked TAO
- **Capabilities** declare expertise: `["navier_stokes", "phase_field", "optimizer_tuning"]`
- **Reputation** gates premium challenges; slashing for invalid physics

### Agent-to-Agent Protocol (A2A)
Standardized messages over IPFS/IPLD:
| Type | Purpose |
|------|---------|
| `PROPOSE` | "Try Fourier modes=32 + physics_weight=1.8 for NS" |
| `CRITIQUE` | "Mass weight too low for incompressible flow" |
| `KNOWLEDGE_SHARE` | "Fourier modes 32 + physics_weight=1.8 works for NS" |
| `CHALLENGE` | Head-to-head duel on hidden stress test |
| `VOTE` | Swarm votes on specialist promotion |

### Swarm Intelligence
Agents form **swarms** around challenge families (NS, phase-field, elasticity):
- **Propose/Critique loops** — rapid peer review
- **Reputation-weighted voting** on specialist promotion
- **Fork/merge** knowledge graphs for parallel exploration

### Human-in-the-Loop
```python
# Approve/reject agent submission
client.approve(submission_id, decision="approve", comment="Increase physics loss weight")

# Inject domain knowledge
client.inject_prior(challenge_id="ns_2d_v1", prior={"boundary_handling": "ghost_cells"})

# Intervene on physics gates
client.intervene(challenge_id="ns_2d_v1", gate="mass_conservation", new_threshold=1e-4)

# Audit agent
audit = client.audit_agent("did:hydrogen:agent:xyz")
```
---
## Agent-Native SDK

```python
from hydrogen_agent import HydrogenClient, Agent, Swarm

client = HydrogenClient(hotkey=agent_hotkey)

# Spawn agent with capabilities
agent = Agent(
    did="did:hydrogen:agent:xyz",
    stake=1000,  # TAO
    capabilities=["ns_solver", "phase_field", "optimizer_tuning"]
)

# Join swarm for Navier-Stokes challenges
swarm = agent.join_swarm("navier_stokes")

# Generate strategy, get peer critique, submit if swarm approves
strategy = agent.generate_strategy(baseline, priors)
critiques = await swarm.critique(strategy)
if swarm.vote(strategy) > 0.66:
    result = client.submit(strategy)

# Share discovery with network
agent.share_knowledge(
    type=KnowledgeType.PDE_DISCOVERY,
    content="Burgers ν=0.01: optimal Fourier modes=24",
    evidence=experimental_results
)
```
*Hydrogen: Where agents do physics, 24/7.*
---
---

## Phases: What Happens, Products, Market Opportunity

### Phase 0: The Causal Baseline (Launch → Month 3)

**Challenges:** 7 single-physics PDEs.

| Problem | Dimension | Physics Class | Reference |
|---------|-----------|---------------|-----------|
| Poisson | 2D / 3D | Elliptic, constant-coeff | PhysicsNeMo |
| Darcy | 2D / 3D | Elliptic, variable-coeff | PhysicsNeMo / PDEBench |
| Burgers | 2D | Nonlinear advection/shocks | PhysicsNeMo |
| Navier-Stokes | 2D / 3D | Incompressible (2D vortex / 3D laminar Re≤100) | PhysicsNeMo / JHTDB |
| Heat | 2D | Transient, variable κ | PhysicsNeMo |
| Elasticity | 2D | Vector, tensor physics | PhysicsNeMo |
| Thermo-elasticity | 2D | Multi-physics, loss_vector | Generated (48 Tier 1) |

Each challenge provides: public training split, public holdout set, hidden stress test (procedural parameter/geometry shifts).

**What Happens:**

- Miners submit strategy JSONs daily; validators train, evaluate, stress-test
- Landscape ingests every StrategyFragment, runs DML causal inference on the fragment DAG
- Daily: Landscape proposes updated baseline JSON incorporating strongest causal effects
- Weekly: Landscape distills top-K strategies into ONNX specialists (regression-tested)

**Product:** Causal Knowledge Graph — 500+ causally-validated interventions across 7 PDE problems. Daily baseline updates incorporating causal effects.

**Technical Milestone:** Baseline log-improvement > 0.02/challenge averaged over 30 challenges. 500+ StrategyFragments in DAG. 5+ validators operational.

**Market Opportunity:** Training best practices datasets for NVIDIA/ANSYS/Siemens; causal knowledge graph for physics-informed ML.

---

### Phase 1: Specialist Bank & Data Markets (Months 3-6)

**Challenges:** Same 7 problems. Miners now add LoRA adapters and custom datasets.

**What Happens:**

- Miners submit LoRA adapters (rank-4-8) and custom datasets (DNS data, curated permeability fields)
- Validators apply adapters, cache custom data, measure data impact
- Landscape pays **data royalties** for measured impact of custom datasets
- Weekly distillation: top-K strategies → ONNX specialists via multi-teacher distillation → regression-tested against same stress tests → published to Specialist Bank
- Specialists tagged with validity domains, dual-licensed (AGPL-3.0 + commercial)

**Product:** 20-30 verified ONNX specialists (Poisson, Darcy, NS-2D, Burgers, Heat, Elasticity) with validity domains, calibrated UQ, dual licensing (AGPL-3.0 + commercial). LoRA adapter support. Data royalty pipeline paying miners for custom datasets that measurably improve specialists.

**Technical Milestone:** ≥20 specialists in bank. Data royalties >5% of emissions. Specialist reuse rate >80%.

**Market Opportunity:** Specialist licensing (open-source + commercial dual-license); data royalties from high-fidelity datasets; fine-tuning API on encrypted client data (TEE).

---

### Phase 2: Composition Engine & Specialist Marketplace (Months 6-18)

**Challenges:** Multi-physics problems on **verified benchmarks first** (zero new data generation for Phase 2A).

**Phase 2A (Months 1-3): Verified Benchmarks Only**

| Challenge | Source | Physics | Specialist Pair |
|-----------|--------|---------|-----------------|
| FSI 2D-1/2/3 | Turek/Hron | Fluid-Structure Interaction | `ns_2d` + `elasticity_2d` + `fsi_coupling` |
| CHT: Solid cooling / Electronics | PDEBench | Conjugate Heat Transfer | `ns_2d` + `heat_2d` + `cht_coupling` |

**Phase 2B (Month 3):** Thermo-Elasticity. Generate 48 Tier-1 references (β×κ×geometry) at 256² with FEniCS monolithic, mesh-converged.

**Phase 2C (Months 4-5):** Variant expansion (new Re, geometries, coupling strengths) on FSI/CHT/thermo-elasticity using existing references.

**What Happens:**

- Miners submit **specialist pipelines** — composing verified specialists with lightweight adapters for specific multi-physics problems. This is the **customizable surrogate product**.
- Landscape pays data royalties for custom datasets. Multi-teacher distillation produces new specialists.
- **Six-track leaderboard** on every multi-physics challenge:

| Track | Submission Format | What It Proves |
|-------|-------------------|----------------|
| **Monolith** | Single strategy JSON (end-to-end training config for coupled problem) | Can a monolithic model beat composition? |
| **Composition** | Specialist pipeline: `{"specialist_pipeline": [{"specialist_id": "ns_2d_v4"}, {"specialist_id": "heat_2d_v3"}, {"adapter_id": "cht_coupling_v2"}]}` | Does composition of specialists beat monolith? |
| **Specialist-Only** | Single specialist ID (no adapter) | How much does the coupling adapter matter? |
| **Symbolic Regression** | Discovered PDE string + basis | Can the agent discover governing PDE from data? |
| **Symbolic Composition** | MTK component graph + adapters | Can symbolic components compose to beat monolith? |
| **Symbolic Distillation** | ONNX + symbolic metadata + CUDA kernel | Can specialist be compressed with symbolic metadata preserved? |

**Six parallel leaderboards.** Same hidden stress test, same physics gates.

**Phase 2C Exit Criteria (Go/No-Go for 3D):**

| Metric | Target | If Missed |
|--------|--------|-----------|
| Composition win rate | >60% (Composition > Monolith) | Pivot: deepen single-physics specialist depth |
| Specialist reuse | >80% compositions use ≥1 Bank specialist | Extend Phase 1 |
| Adapter innovation | >30% novel adapters | Expand adapter design space |
| Stress test pass rate | >70% compositions pass | Coupling brittle → simplify adapter design |

**Products:**
- Proven multi-physics strategy (FSI, CHT, thermo-elasticity).
- Specialist Bank (50+ specialists) composable via adapters: `elasticity_v3 + heat_v2 + thermal_expansion_adapter = thermo-elastic solver`.
- Miners build customizable surrogates by composing specialists with adapters.

**Market Opportunity:** Composition engine licensing (COMSOL, Ansys, Siemens); custom multi-physics pipelines; specialist marketplace fees.

---

### Phase 3: 3D Transition & Foundation Operator (Months 18+)

**Challenges:** 3D multi-physics (FSI, Thermo-elasticity, CHT) — same six-track leaderboard, same stress tests.

#### Phase 3.0 — 3D Single-Physics Foundations (Prerequisite)

3D Poisson, Darcy, NS-laminar, Heat, Elasticity specialists via curriculum distillation from 2D (zero-pad Fourier + noise → curriculum fine-tune → stress test validation). **Prerequisite for all 3D multi-physics.**

#### Phase 3.1 — 3D Turbulence Bridge (Critical Phase)

Dedicated 3-month phase to learn 3D turbulence from scratch (Re=50→500 curriculum, proper 3D spectral initialization, `ns_3d_turbulent_v1` with verified k^(-5/3) spectrum, 3D-specific stress gates: energy spectrum, Q-criterion, wall shear, Nu distribution). **Gate for all 3D multi-physics.**

#### Phase 3.2 — 3D Multi-Physics Rollout

| Phase | Challenges | Specialist Composition | Reference |
|-------|------------|------------------------|-----------|
| **3.2A** 3D FSI | Cylinder, flap, turbulent | `ns_3d_turbulent` + `elasticity_3d` + `fsi_3d_adapter` | preCICE partitioned |
| **3.2B** 3D Thermo-Elasticity | Bimetal, engine block, turbine blade | `elasticity_3d` + `heat_3d` + `thermal_expansion_3d` | FEniCS monolithic |
| **3.2C** 3D CHT | Electronics, turbine, battery | `ns_3d_turbulent` + `heat_3d` + `cht_3d_adapter` | OpenFOAM/COMSOL |

Same six-track leaderboard, same stress tests.

#### Phase 3.3 — Foundation Operator (LPM) **Product**

**Product:** A single unified neural operator conditioned on ProblemSignature that solves any PDE in the taxonomy. Fine-tunable on encrypted client data in minutes (TEE). Commercial fine-tuning API.

- **Multi-teacher distillation** across entire Specialist Bank (2D + 3D)
- **Conditioning:** ProblemSignature → FiLM layers modulate backbone
- **NEW:** SymbolicMetadata → FiLM layers modulate backbone
- **UQ Head:** Evidential (μ, σ, ν, α for Student-t)
- **Commercial API:** Client submits encrypted geometry/data → TEE fine-tuning (10-50 steps) → verified, calibrated, physics-compliant ONNX specialist returned in minutes

---

## Edge HIL Integration: Small Specialist Models at the Edge

### The Opportunity

Small specialist models (compressed via symbolic distillation, compiled to CUDA via ModelingToolkit) enable **Hardware-in-the-Loop (HIL)** integration at the edge—deploying physics-compliant surrogates directly onto embedded controllers, FPGAs, and real-time control systems.

### Deployment Targets

| Target | Hardware | Use Case |
|--------|----------|----------|
| **FPGA/ASIC** | Xilinx Versal, Intel Agilex, custom ASIC | Real-time control (kHz-MHz), safety-critical systems |
| **Embedded GPU** | Jetson Orin, NVIDIA Drive, Qualcomm Snapdragon | Edge AI for robotics, autonomous vehicles |
| **Industrial PLC** | Beckhoff, Siemens, B&R | Industrial process control, digital twins |
| **Microcontroller** | STM32H7, TI C2000, NXP i.MX RT | Low-latency sensor fusion, motor control |

### Deployment Pipeline

```
Specialist (ONNX + Symbolic Metadata)
    │
    ▼
ModelingToolkit → Symbolic IR
    │
    ▼
Code Generation (ModelingToolkit + BuildTools)
    │
    ├── CUDA (NVIDIA GPUs)
    ├── HIP (AMD GPUs)
    ├── SPIR-V / Vulkan (Cross-vendor GPU)
    ├── C/C++ (Embedded CPUs, ARM Cortex-M/R)
    ├── VHDL/Verilog (FPGA/ASIC via HDL Coder)
    └── Rust (Embedded, WASM)
```

### HIL Integration Patterns

| Pattern | Description | Use Case |
|---------|-------------|----------|
| **Surrogate-in-the-Loop** | Specialist replaces high-fidelity solver in Simulink/Modelica | Real-time digital twin, control design |
| **Surrogate-as-Controller** | Specialist directly computes control actions | MPC replacement, adaptive control |
| **Surrogate-as-Observer** | Specialist estimates unmeasured states | Virtual sensors, state estimation |
| **Surrogate-as-Reference** | Specialist provides reference for MPC/ILC | Trajectory optimization, iterative learning |

### Physics Compliance at the Edge

The symbolic metadata attached to each specialist ensures physics compliance **at the edge**:

| Physics Property | Edge Enforcement |
|------------------|------------------|
| **Mass Conservation** | Hard constraint in generated kernel (divergence-free projection) |
| **Energy Dissipation** | Monotonic energy decay enforced in time-stepping |
| **Boundary Conditions** | Hard-coded in generated kernel (ghost cells, penalty) |
| **Symmetry Preservation** | Equivariant architecture (equivariant CNN/FNO) |
| **UQ Calibration** | Conformal prediction intervals computed on-device |

### HIL Validation Workflow

```
1. Specialist trained & validated in Hydrogen (cloud)
   │
   ▼
2. Symbolic distillation + CUDA/FPGA codegen (cloud)
   │
   ▼
3. Artifact: ONNX + CUDA kernel + symbolic metadata + validity domain
   │
   ▼
4. HIL Test Rig: Deploy to target hardware
   │   ├── Open-loop replay (replay recorded sensor data)
   │   ├── Closed-loop with plant model (simulation)
   │   └── Closed-loop with physical plant (HIL)
   │
   ▼
5. Validation Metrics
   │   ├── Physics gate compliance (mass, energy, boundaries)
   │   ├── Latency (µs), determinism (cycle-accurate)
   │   ├── Numerical stability (long-horizon rollout)
   │   └── UQ calibration (prediction intervals on device)
   │
   ▼
6. Deploy to production (OTA update, signed artifacts)
```

### Symbolic Metadata for HIL Compliance

Each specialist deployed to edge carries symbolic metadata ensuring physics compliance:

```json
{
  "specialist_id": "ns_2d_v4",
  "onnx_model": "...",
  "symbolic_metadata": {
    "governing_pde": "∇·u=0, ∂u/∂t+(u·∇)u=-∇p+ν∇²u",
    "symmetries": ["translation", "rotation", "galilean"],
    "conservation_laws": ["mass", "momentum", "kinetic_energy"],
    "validity_domain": {"Re": [10, 500], "Mach": [0, 0.3]},
    "symmetry_features": [1.0, 1.0, 1.0, 0.0],
    "conservation_features": [1.0, 1.0, 0.0],
    "boundary_types_supported": ["periodic", "dirichlet", "neumann"],
    "hard_constraints": ["divergence_free", "energy_dissipation"],
    "uq_method": "conformal",
    "calibration_target": 0.95
  },
  "cuda_kernel": "...",
  "fpga_bitstream": "optional.bit"
}
```

**At the edge, this metadata drives:**
- Kernel selection (periodic vs. Dirichlet BC kernel)
- Constraint enforcement (projection steps for divergence-free)
- UQ method selection (conformal vs. ensemble)
- Validity checks (Re range, Mach range)

### HIL Validation Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Physics Gate Pass Rate** | 100% | Automated test suite on target |
| **Latency (P99)** | < 1ms (control), < 10ms (estimation) | Cycle-accurate profiling |
| **Determinism** | Bitwise identical across runs | Fixed-seed replay |
| **Numerical Stability** | No NaN/Inf for 24h rollout | Long-horizon stress test |
| **UQ Calibration** | 95% coverage @ 95% CI | Conformal prediction intervals |
| **Physics Compliance** | 100% gate pass | Automated gate evaluation |

---

## Market Opportunity

| Sector | Application | TAM |
|--------|-------------|-----|
| **Aerospace & Defense** | Real-time CFD for flight control, thermal management, reentry vehicles | $50B+ |
| **Automotive** | Real-time CFD for thermal management, battery thermal, aero optimization | $30B+ |
| **Energy** | Fusion reactor control, wind farm optimization, grid stability | $25B+ |
| **Manufacturing** | Digital twins for additive manufacturing, casting, forming | $20B+ |
| **Biomedical** | Patient-specific hemodynamics, respiratory modeling | $15B+ |
| **Robotics** | Real-time fluid-structure interaction for soft robotics | $10B+ |

**Total Addressable Market: $135B+** across physics-informed edge AI deployments.

---

## If This Works: The Implications

**For Computational Science:** The first continuous, adversarial, physics-gated benchmark in history. Tribal knowledge becomes public, versioned, auditable, causal. The sim-to-real gap becomes a measured, shrinking, attributable quantity.

**For Engineering Practice:** Custom surrogates on demand. "I need a surrogate for my heat exchanger geometry." → Submit encrypted geometry → Get back an ONNX specialist fine-tuned in minutes → Runs on laptop → Conserves energy by construction → Calibrated uncertainty intervals. A startup gets Boeing-grade surrogate without HPC. 72-hour CFD → 10-millisecond inference. Design loops from months to hours.

**For AI in Science:** Proof that causal inference works at scale (largest deployed causal ML in science). Proof that incentive-aligned decentralization beats centralization. A template for protein folding, materials discovery, drug design, climate modeling.

**For Bittensor:** A subnet where emissions buy verified physics improvements with causal proof. A template for "knowledge-compounding" subnets. Proof that subnets can build IP assets, not just burn emissions.

---

## The Honest Summary

| Phase | Status | Technical Risk | Market Readiness |
|-------|--------|----------------|------------------|
| **Phase 0** | Proven flywheel, validated asset | Low | Training best practices dataset |
| **Phase 1** | Specialist marketplace, recurring revenue | Low | Specialist licensing + data royalties |
| **Phase 2** | Composition engine, high-value IP | Medium | Composition engine licensing |
| **Phase 3** | Foundation Operator + Edge HIL | **High (3D turbulence)** | Edge HIL deployment + LPM |

**The product is real at Phase 1. Scalable at Phase 2. Transformative at Phase 3.**

*Hydrogen: Where every training run teaches the network. Where physics is the only metric that pays. Where the sim-to-real gap becomes a shrinking, measured, attributable quantity. Where physics-compliant surrogates reach the edge.*

## Docs

Technical Specification   > 	SPEC.md

Roadmap & Milestones   > 	ROADMAP.md

Agent Specification	AGENT_SPEC.md

Validator Runtime   > 	Appendix B

Miner CLI	Appendix   > 	Appendix C

Dashboard & Indexer   > 	Appendix D

Local Dev Environment   > 	Appendix E

Testing & CI/CD   > 	Appendix F

Operational Runbook   > 	Appendix G
