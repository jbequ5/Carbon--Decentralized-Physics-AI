# Carbon - A Physics Intelligence Subnet
---

## Trustless Verification & Data Generation System

Carbon uses a dedicated **Trustless Verification and Data Generation System** as a core architectural component. Full details are documented in `TRUSTLESS_VERIFICATION_AND_DATA_GENERATION.md`.

### Key Principles
- All evaluation data (stress testing and benchmark/held-out) is generated **procedurally at runtime** using an open-source generator.
- Generation is seeded by public, unpredictable information (Challenge ID + Block Hash in Phase 0; moving toward commit-reveal + drand evaluation in Phase 1).
- The system is designed to be **auditable by anyone** while remaining unpredictable to miners and agents.
- Benchmark data quality is established through strong scientific justification of the generator + ongoing validation against high-fidelity reference solvers + physics gates as an independent filter.
- We do **not** rely on fixed known reference datasets as the primary evaluation data (to preserve the trustless property).

See `TRUSTLESS_VERIFICATION_AND_DATA_GENERATION.md` for the full design, including the Proprietary Data Handling Plan (Section 8).

---

## Proprietary Data Handling Plan (Summary)

Carbon's long-term vision includes safely incorporating signals from proprietary customer data. The current phased plan prioritizes customer control and privacy:

**Phase 0 (Current)**: Public + synthetic data only. No proprietary data enters the network.

**Phase 1 (Near-term)**: Customer-controlled local fine-tuning. Customers run Carbon priors locally on their own infrastructure via the Air-Gapped Miner Toolkit. Raw data never leaves customer control. Optional privacy-preserved signal contribution back to the network.

**Phase 2 (Medium-term)**: Introduce Confidential Computing (NVIDIA stack) on the validator side for sensitive workloads. Only aggregated or differentially private signals leave protected enclaves.

**Phase 3 (Longer-term)**: Explore federated learning + Confidential Computing patterns for more advanced privacy-preserving contributions.

### Dual-Regime Architecture (DoD/Regulated Markets)

Carbon operates a **Dual-Regime Model Supply** for defense and regulated domains:

```
┌─────────────────────────────────────────────────────────────────┐
│  PUBLIC REGIME (Carbon Subnet)                                  │
│  ├─ Discovers architectures on public/synthetic data            │
│  ├─ Adversarial verification + physics gates                    │
│  ├─ Outputs: Strategy.json + Model Card + ONNX + Evidence       │
│  └─ Zero ITAR/controlled data                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                    CROSS-DOMAIN SOLUTION / SECURE TRANSFER
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  CLASSIFIED REGIME (Prime Enclave IL5/IL6)                      │
│  ├─ Ingests architecture blueprint (strategy.json)              │
│  ├─ Fine-tunes on classified telemetry / proprietary geometry   │
│  ├─ Deploys ONNX locally (HIL, edge, air-gapped)                │
│  ├─ Runs inference LOCALLY — zero network calls                 │
│  ├─ Applies ITAR/EAR classification (Prime's ECO)               │
│  └─ Packages DI-SESS-82483 deliverable + ATO artifacts          │
└─────────────────────────────────────────────────────────────────┘
```
- Public Regime: Subnet mechanics, evidence generation, zero ITAR
- Transfer Layer: Cross-domain solution, secure media, CDS
- Classified Regime: Air-Gapped Miner Toolkit, fine-tuning loop, ITAR classification, DI-SESS-82483 packaging, ATO evidence
- Prime Onboarding Checklist (clearance, facility, contract vehicle)
- Evidence Package Schema (Model Card → DI-SESS-82483 manifest)

## Commercial GTM — Three Revenue Engines

- **Engine 1 — Specialist Bank (Tier 1)**: Product, pricing ($/yr), buyer, motion, differentiation, Phase 0 delivery
- **Engine 2 — Sponsored Challenges (Tiers 2-4)**: Product, pricing (T2 $$, T3 $$$, T4 $$$$), buyer, motion, Challenge Factory CLI, standardized agreements
- **Engine 3 — DoD Subcontract (SBIR/BAA)**: Positioning, entry vehicle, prime targets, Carbon deliverable, Prime deliverable, Phase 0.5 requirement
- **Engine 4 — Verification Gas/Registry**: Asset, metering, pricing, partners, scale
- **Revenue Projections**: Y1-Y4 table

---

## Initial Design Priorities (Phase 0 Focus)

The following three capabilities are prioritized for the initial design because they directly enable low-friction, scalable iteration while protecting the integrity of the hidden validation system:

### 1. Black-Box Diagnostics with Clear Diagnostic Tiers (Reputation-Gated)

MCP diagnostics returned to miners and agents will be deliberately limited. Only objective scores, gate pass/fail status, and high-level category feedback will be provided. Precise geometric, spectral, or spatial hotspot information will not be exposed. This is a core architectural decision to protect the hidden stress distribution and prevent reverse-engineering of the validator's evaluation data.

**Diagnostic Tier Definitions (Reputation-Gated, Not Stake-Gated):**

| Tier | Access | Gate | Reputation Threshold |
|------|--------|------|---------------------|
| **Basic** | Objective scores + overall gate status + high-level category | Free (all miners) | 0 |
| **Intermediate** | High-level failure categories + spectral hints + envelope warnings | Reputation ≥ 0.65 | 0.65 |
| **Rich** | Spatial hotspots + causal snippets + priority queue | Reputation ≥ 0.85 + validator nomination | 0.85 |

**Reputation Score Formula:**
```
reputation = 0.4 × gate_pass_rate_30d + 0.3 × challenge_win_rate_90d + 0.2 × consistency_score + 0.1 × recency_weight
```
- `gate_pass_rate_30d`: Fraction of submissions passing all gates in last 30 days
- `challenge_win_rate_90d`: Fraction of challenges where miner held top-3 position in last 90 days
- `consistency_score`: 1 - (std_dev_of_scores / mean_score) over last 20 submissions
- `recency_weight`: Exponential decay, half-life = 14 days

Staking reserved for Partners/Enterprises only (see Partner Staking mechanism).

### 2. Noisy Prior Distribution + Estimation Mode

The system will distribute only noisy / perturbed versions of the current best strategies (never the clean champion). An Estimation Mode will be provided that allows near-zero-cost screening of new strategy ideas using fast approximations anchored to the noisy prior. This enables both human miners and autonomous agents to run fast, Autoresearch-style iteration loops with minimal friction while maintaining strong protection of the collective intelligence moat.

Estimation Mode outputs will be clearly labeled as estimates with confidence scores. They are intended for rapid filtering and idea generation, not as a replacement for actual training or official validator evaluation.

### 3. ModelingToolkit.jl Integration for Structured Losses

PySR symbolic constraints extracted by the Landscape Agent will be compiled into structured loss equations using ModelingToolkit.jl. This integration will be pursued early so that symbolic insights can be turned into concrete, differentiable loss terms that agents can directly use in their local training loops. This significantly increases the actionability of the feedback loop between the Landscape Agent and participating miners/agents.

These three priorities are viewed as foundational for enabling iteration at scale while preserving scientific rigor and long-term defensibility.

---

## Miner Compute, Local Iteration & Submission Model

### 1. Core Philosophy

- The **validator always performs full deterministic training + hidden adversarial stress evaluation** the same way for every submission.
- Miners and autonomous agents run **local iterative training loops** on their own hardware (or rented machines) to improve strategies before submission.
- **Submission is always free**. Local training is an *optional enhancement*, not a requirement.
- The system distributes **noisy priors** only (never the clean champion) to enable compounding while protecting the moat.
- High-quality Landscape knowledge (causal insights, symbolic patterns) is protected and shared selectively via strategic guidance.

### 2. Three-Tier Local System

| Tier                    | Compute Cost      | Anchored To          | Purpose                              | Required Before Submission? | Cost Estimate Provided? |
|-------------------------|-------------------|----------------------|--------------------------------------|-----------------------------|-------------------------|
| **Estimation Mode**     | Near-zero         | Noisy Prior          | Rapid idea screening & filtering     | No                          | Yes (if renting)        |
| **Light Training Mode** | Low               | Noisy Prior          | Main iterative improvement loop      | No (recommended)            | Yes                     |
| **Validator (Official)**| Network-paid      | Full hidden data     | Official scoring + emissions         | Yes                         | N/A                     |

**Key Rule**: A miner can submit a strategy JSON to the validator at any time with zero local training. Training is purely optional to help them submit stronger strategies.

### 3. Data, Stress Tests, and Physics Gates in Internal Mining Loops

#### 3.1 Core Principle
Internal mining loops (Estimation Mode and Light Training Mode) must use **different data and stress conditions** from the validator's hidden evaluation set. This preserves the adversarial integrity of official scoring while still providing useful signal for iteration.

Miners and agents **never** have access to the validator's hidden test data or the full hidden stress variant set during local loops.

#### 3.2 Training Data in Local Loops
- Local loops may use procedural data generation and The Well slices.
- Data generation should use different random seeds or subsampling strategies than the validator.
- Custom datasets are allowed if properly validated.
- The goal is to enable meaningful training signal without replicating the validator's exact data distribution.

#### 3.3 Local Stress Testing and Evaluation
- **Estimation Mode**: Uses fast approximations anchored to the noisy prior. No actual stress rollout is performed.
- **Light Training Mode**: May run a **reduced, non-hidden set** of stress-like variants for local evaluation. These variants must be different from the validator's hidden stress set.
- Local stress testing should still apply physics-aware metrics (residuals, conservation, stability) for signal quality.
- Full hard physics gates **can** be applied during Light Training Mode for better learning signal, but this does not replace the validator's official gated evaluation.

#### 3.4 Why This Separation Matters
- Prevents miners from gaming the official hidden stress by tuning against it locally.
- Maintains strong defensibility of the validator pipeline.
- Still allows fast, high-signal local iteration.
- Ensures that only strategies evaluated under the true hidden adversarial conditions receive official credit and emissions.

### 4. Estimation Mode (Noisy-Prior Only)

**Purpose**: Allow very fast, near-zero-cost screening of ideas.

**Rules**:
- Must be based **only on the latest noisy prior** for the challenge + backbone.
- Never uses the clean champion model.
- Returns estimated deltas, confidence, and risk notes.
- Clearly labeled as an *estimate* (not a substitute for actual training).

**Implementation Specification:**

```python
# carbon/estimation/estimator.py
class EstimationEngine:
    """
    Linear sensitivity approximation around noisy prior.
    Optional: Small proxy model (3-layer MLP) for nonlinear correction.
    """
    
    def __init__(self, noisy_prior: Strategy, backbone: str, challenge: str):
        self.noisy_prior = noisy_prior
        self.backbone = backbone
        self.challenge = challenge
        self.proxy_model = self._load_proxy_model()  # 3-layer MLP, 64 hidden
    
    def estimate(self, candidate_strategy: Strategy) -> EstimationResult:
        # 1. Compute parameter delta from noisy prior
        delta = self._compute_delta(candidate_strategy, self.noisy_prior)
        
        # 2. Linear sensitivity (Jacobian wrt strategy params at noisy prior)
        jacobian = self._compute_jacobian(self.noisy_prior)
        linear_estimate = jacobian @ delta
        
        # 3. Proxy model correction (optional)
        if self.proxy_model:
            proxy_input = self._encode_strategy(candidate_strategy)
            proxy_correction = self.proxy_model(proxy_input)
            estimate = linear_estimate + proxy_correction
        else:
            estimate = linear_estimate
        
        # 4. Confidence via ensemble variance (5 proxy models with dropout)
        confidence = self._compute_confidence(candidate_strategy)
        
        # 5. Risk flags
        risk_flags = self._check_risk_flags(candidate_strategy)
        
        return EstimationResult(
            estimated_score_delta=estimate,
            confidence=confidence,
            risk_flags=risk_flags,
            method="linear_sensitivity_proxy",
            compute_time_ms=<100
        )
```

**Output Schema:**
```json
{
  "estimated_score_delta": -0.023,
  "confidence": 0.78,
  "risk_flags": ["high_lr_risk", "gate_margin_thin"],
  "method": "linear_sensitivity_proxy",
  "compute_time_ms": 47,
  "label": "ESTIMATE_ONLY"
}
```

### 5. Local Training (Optional Enhancement)

Miners may optionally run actual training loops starting from the noisy prior:

- **Light Training Mode** (recommended default): Reduced budget with multi-fidelity local evaluation + physics-residual monitoring.
- **Full Local Confirmation**: Longer runs for final validation before submission.

Training is provided as a convenience to improve submission quality. It is not required.

**Light Training Mode Spec:**
```yaml
# carbon/miner/configs/light_training.yaml
budget:
  max_gpu_hours: 4
  max_epochs: 50
  early_stop_patience: 10
  
local_eval:
  stress_variants: 12  # Non-hidden, different seeds from validator
  physics_gates: true  # Full gates for learning signal
  metrics: ["residual_l2", "conservation_l2", "boundary_l2", "rollout_stability"]
  
checkpointing:
  every_n_epochs: 5
  keep_best_n: 3
  metric: "combined_score"
```

### 6. Submission Model (Zero Friction)

- Miners can submit a strategy JSON to the validator **at any time**.
- No local training is required to submit.
- The validator will perform full training from the submitted JSON + full hidden stress testing + physics gates.
- Only submissions that set a new best combined score on the validator side receive strong weight in the ChallengeWinnerTracker.

### 7. Cost Estimation

When a miner chooses to use rented compute (Targon, Chutes, RunPod, etc.), the Miner Toolkit must provide clear upfront cost estimates before execution.

**Cost Estimation Engine:**
```python
# carbon/miner/cost_estimator.py
PROVIDER_RATES = {
    "targon": {"A100_40GB": 1.20, "A100_80GB": 1.80, "H100_80GB": 3.50},  # $/hr
    "chutes": {"A100_40GB": 1.10, "H100_80GB": 3.20},
    "runpod": {"A100_40GB": 1.35, "H100_80GB": 3.80},
    "lambda": {"A100_40GB": 1.50, "H100_80GB": 4.00},
}

def estimate_cost(tier: str, provider: str, gpu_type: str, gpu_count: int = 1) -> CostEstimate:
    """Returns USD estimate with 20% buffer."""
    rate = PROVIDER_RATES[provider][gpu_type]
    hours = {"estimation": 0.1, "light_training": 4.0, "full_confirmation": 24.0}[tier]
    base = rate * hours * gpu_count
    return CostEstimate(
        tier=tier,
        provider=provider,
        gpu_type=gpu_type,
        estimated_hours=hours,
        estimated_usd=base * 1.2,
        buffer_pct=20
    )
```

### 8. Miner Toolkit (Docker Image + Interface)

A dedicated **Miner Toolkit** Docker image will be provided with:

**Dockerfile:**
```dockerfile
# carbon/miner/Dockerfile
FROM nvidia/cuda:12.4-devel-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv git curl wget \
    build-essential cmake libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Python env
RUN python3.11 -m venv /opt/carbon
ENV PATH="/opt/carbon/bin:$PATH"

# Carbon Miner Toolkit
COPY carbon/miner /opt/carbon/miner
COPY carbon/common /opt/carbon/common
COPY requirements-miner.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-miner.txt

# Entry points
COPY docker/entrypoint-miner.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--help"]
```

**CLI Interface:**
```bash
# carbon/miner/cli.py
@click.group()
def cli():
    """Carbon Miner Toolkit - Local iteration for Carbon Subnet"""
    pass

@cli.command()
@click.option('--challenge', required=True, help='Challenge ID')
@click.option('--backbone', default='fno', type=click.Choice(['fno', 'gino', 'wno', 'transolver']))
@click.option('--strategy', 'strategy_path', required=True, type=click.Path(exists=True))
@click.option('--mode', default='estimation', type=click.Choice(['estimation', 'light_training', 'full_confirmation']))
@click.option('--provider', type=click.Choice(['targon', 'chutes', 'runpod', 'lambda', 'local']))
@click.option('--gpu', default='A100_40GB')
@click.option('--output', default='./carbon_output')
def run(challenge, backbone, strategy_path, mode, provider, gpu, output):
    """Run local iteration loop."""
    # Load strategy, validate, estimate cost, confirm, execute
    pass

@cli.command()
@click.option('--challenge', required=True)
def pull_prior(challenge):
    """Download latest noisy prior for challenge."""
    pass

@cli.command()
@click.option('--challenge', required=True)
@click.option('--strategy', 'strategy_path', required=True)
def submit(challenge, strategy_path):
    """Submit strategy to validator via MCP."""
    pass

@cli.command()
def doctor():
    """Validate environment: GPU, drivers, MCP connectivity, credentials."""
    pass
```

**Python SDK:**
```python
# carbon/miner/sdk.py
class CarbonMiner:
    def __init__(self, mcp_endpoint: str = None, hotkey: str = None):
        self.mcp = MCPClient(mcp_endpoint or os.getenv("CARBON_MCP_ENDPOINT"))
        self.hotkey = hotkey or os.getenv("CARBON_HOTKEY")
    
    def get_noisy_prior(self, challenge: str, backbone: str) -> Strategy:
        """Fetch latest noisy prior for challenge+backbone."""
        return self.mcp.call("get_noisy_prior", challenge=challenge, backbone=backbone)
    
    def estimate(self, strategy: Strategy, prior: Strategy) -> EstimationResult:
        """Run local estimation (no GPU needed)."""
        engine = EstimationEngine(prior, strategy.backbone, strategy.challenge)
        return engine.estimate(strategy)
    
    def train_local(self, strategy: Strategy, mode: str = "light", **kwargs) -> TrainingResult:
        """Run local training loop (requires GPU)."""
        runner = LocalTrainingRunner(strategy, mode, **kwargs)
        return runner.run()
    
    def submit(self, strategy: Strategy) -> SubmissionReceipt:
        """Submit to validator via MCP."""
        return self.mcp.call("submit_strategy", strategy=strategy.dict(), hotkey=self.hotkey)
    
    def get_diagnostics(self, submission_id: str) -> Diagnostics:
        """Fetch results from validator."""
        return self.mcp.call("get_diagnostics", submission_id=submission_id)

# Agent-friendly async interface
class AsyncCarbonMiner(CarbonMiner):
    async def propose_train_evaluate_submit(self, challenge: str, backbone: str, n_iterations: int = 100):
        """Full Autoresearch loop: propose → estimate → train → evaluate → submit."""
        prior = await self.get_noisy_prior_async(challenge, backbone)
        for i in range(n_iterations):
            candidate = self.propose_candidate(prior)
            est = await self.estimate_async(candidate, prior)
            if est.confidence > 0.7 and est.estimated_score_delta > 0:
                result = await self.train_local_async(candidate, mode="light")
                if result.local_score > prior.score * 1.05:
                    receipt = await self.submit_async(candidate)
                    if receipt.accepted:
                        prior = candidate  # Update prior
        return prior
```

### 9. Security & Moat Protection

- Only **noisy priors** are distributed.
- The clean champion model is never exposed.
- High-value Landscape knowledge (raw causal graphs, detailed DML outputs) remains protected.
- The rigid validator pipeline (hidden stress + physics gates + progress-only rewards) acts as the primary filter against low-value strategies.
- All official scoring and emissions impact comes exclusively from validator-executed runs.

### 10. Phased Implementation

**Phase 0**:
- Black-box diagnostics with clear diagnostic tiers (reputation-gated)
- Noisy prior distribution + Estimation Mode (anchored to noisy priors)
- Miner Toolkit Docker image with local support and basic Python interface
- Light Training templates + local multi-fidelity evaluation
- Direct submission path (always available)
- Basic cost estimation for rented compute
- Trustless procedural data generation system (open generator + public seeding)

**Phase 1**:
- ModelingToolkit.jl integration for turning PySR symbolic constraints into structured loss terms
- Cloud rental integration (Targon + Chutes prioritized)
- Stronger strategic guidance generation from the Landscape Agent
- Initial scoped Abaqus ingestion utilities
- Enhanced trustless verification features (commit-reveal seeding, stronger generator validation)
- Air-Gapped Miner Toolkit for classified enclaves

**Phase 2+**:
- Cross-domain causal mapping via Double Machine Learning
- Advanced agent tooling and multi-asset emissions features
- Advanced gaming resistance mechanisms for the trustless verification system

---

## Scientific Motivation & Strategic Positioning

High-fidelity simulation remains the bottleneck in engineering design, optimization, digital twins, and real-time control. Traditional solvers scale poorly with design space size or real-time requirements. Pure data-driven ML surrogates are fast but frequently violate conservation laws, stability conditions, or boundary physics, rendering them unreliable for downstream engineering use.

The space for AI-powered physics simulation and Neural Operators is still **nascent**. There is a tremendous amount left to discover in *how* to best build, train, and use these models for real engineering problems.

Centralized teams explore this space linearly. Carbon is designed to explore it in parallel across thousands of strategies with strong selection pressure from hidden adversarial validation and compounding knowledge via the Landscape Agent.

**Core Thesis**: A properly aligned decentralized subnet can discover superior Neural Operator training methodologies faster and cheaper than centralized players, while providing trustless, verifiable robustness.

---

## Challenges by Phase (Specific Problem Definitions)

### 4.1 Phase 0: Foundational Single-Physics Challenges

| ID | Problem | Dimension | Key Physics | Reference / Notes |
|----|---------|-----------|-------------|-------------------|
| 1 | Poisson | 2D/3D | Elliptic, source-driven | Standard benchmark; variable source amplitude/location, coefficient fields |
| 2 | Darcy | 2D/3D | Elliptic, heterogeneous media | Variable permeability fields (smooth + discontinuous); tests conservation & maximum principle |
| 3 | Burgers | 2D | Hyperbolic, nonlinear advection + viscosity | Shock formation/capturing, conservation, long-time stability |
| 4 | Navier-Stokes (laminar) | 2D/3D | Incompressible flow | Reynolds-number variation in laminar regime; divergence-free constraint, energy stability |
| 5 | Heat | 2D | Parabolic, transient conduction | Time-dependent forcing, variable conductivity, long-term decay |
| 6 | Linear Elasticity | 2D | Vector mechanics, equilibrium | Material property variation (Young's modulus, Poisson ratio), boundary displacement |
| 7 | Thermo-Elasticity | 2D | Coupled thermal-mechanical | Thermal expansion, coupling strength, temperature loading; tests coupled conservation |

Each challenge includes public training/holdout splits and hidden stress configurations. Symbolic metadata is attached.

### 4.2 Phase 0.5: Defense-Relevant Benchmarks (Months 4-8)

Bridges academic PDEs to weapon-relevant physics. Each includes geometry, boundary conditions, flight envelope, and reference solutions for generator validation.

| ID | Benchmark | Physics | Key Physics | Reference |
|----|-----------|---------|-------------|-----------|
| 8 | NACA 0012 Transonic Flutter | 2D/3D Compressible NS | Shock-boundary layer interaction, flutter onset | NASA TP-2001-211214 |
| 9 | NASA CRM Wing-Body | 3D RANS | Transonic separation, buffet | AIAA DPW |
| 10 | HIFiRE-1 Scramjet Forebody | Reacting NS (5-species) | Hypersonic BL transition, finite-rate chemistry | AFRL HIFiRE |
| 11 | Turek/Hron FSI 3D | NS + Elasticity (preCICE) | Fluid-structure interaction, large displacement | FSI Benchmark |
| 12 | Store Separation (6-DOF) | NS + Rigid Body Dynamics | Moving boundaries, dynamic mesh, unsteady forces | AF SEEK EAGLE |
| 13 | Turbine Blade Heat Transfer | Conjugate Heat Transfer | Film cooling, CHT, rotating frame | NASA C3X |

Each includes public training/holdout splits, hidden stress configurations, and symbolic metadata. Generator validation against reference solvers (FEniCS/OpenFOAM/SU2) is mandatory before mainnet inclusion.

### Phase 0.5 Benchmarks — Detailed Specifications

**Per-Benchmark Generator Configuration (carbon/generators/phase05/):**

#### 8. NACA 0012 Transonic Flutter
```yaml
# carbon/generators/phase05/naca0012_transonic.yaml
geometry:
  source: "naca0012.stl"  # NACA 0012 airfoil, chord=1.0
  dimension: 2D  # extruded for 3D variant
  
pde: "compressible_navier_stokes"
turbulence_model: "spalart_allmaras"

flight_envelope:
  mach: [0.7, 0.9, 1.1, 1.2]  # transonic regime
  reynolds: [1e6, 5e6, 10e6]
  angle_of_attack: [-2.0, 0.0, 2.0, 4.0]
  
boundary_conditions:
  farfield: "riemann_invariant"
  wall: "adiabatic_no_slip"
  symmetry: "symmetry_plane"

generator_params:
  mach_distribution: "beta(2, 2) scaled to [0.7, 1.2]"
  reynolds_distribution: "log_uniform([1e6, 10e6])"
  aoa_distribution: "uniform([-2, 4])"
  mesh_perturbation: "gaussian(0, 0.001_chord)"
  turbulence_ic_perturbation: "log_normal(0, 0.3)"

physics_gates:
  - mass_conservation: {threshold: 1e-5, norm: "L2"}
  - energy_stability: {threshold: 1e-5, norm: "L2"}
  - boundary_satisfaction: {threshold: 1e-4, norm: "Linf"}
  - shock_capture: {threshold: 0.05, metric: "shock_position_error"}
  - rollout_stability: {steps: 5000, perturbation: 0.01}

reference_solver: "SU2 v7.5.0"
validation_cases: 50
validation_tolerance: "L2 < 2% vs SU2"
```

#### 9. NASA CRM Wing-Body
```yaml
# carbon/generators/phase05/crm_wingbody.yaml
geometry:
  source: "crm_wingbody.stl"  # NASA Common Research Model
  dimension: 3D
  
pde: "rans_3d"
turbulence_model: "komega_sst"

flight_envelope:
  mach: [0.85, 0.90, 0.95]
  reynolds: [5e6, 10e6, 20e6]
  angle_of_attack: [0.0, 2.0, 4.0, 6.0]

generator_params:
  mach_distribution: "uniform([0.85, 0.95])"
  reynolds_distribution: "log_uniform([5e6, 20e6])"
  aoa_distribution: "uniform([0, 6])"
  mesh_deformation: "free_form_deformation(control_points=20, magnitude=0.02)"

physics_gates:
  - mass_conservation: {threshold: 1e-5}
  - energy_stability: {threshold: 1e-5}
  - boundary_satisfaction: {threshold: 1e-4}
  - separation_capture: {threshold: 0.1, metric: "separation_point_error"}
  - rollout_stability: {steps: 2000}

reference_solver: "OpenFOAM v11 (simpleFoam + kOmegaSST)"
validation_cases: 30
validation_tolerance: "Cd < 3% error, Cl < 5% error vs OpenFOAM"
```

#### 10. HIFiRE-1 Scramjet Forebody
```yaml
# carbon/generators/phase05/hifire1.yaml
geometry:
  source: "hifire1_forebody.stl"  # 7° half-angle cone
  dimension: 2D axisymmetric / 3D
  
pde: "reacting_navier_stokes"
chemistry: "5_species_air"  # N2, O2, NO, N, O
turbulence_model: "none"  # laminar / transitional

flight_envelope:
  mach: [5.0, 6.0, 7.0, 8.0]
  reynolds: [1e5, 5e5, 1e6]  # based on nose radius
  wall_temperature: [300, 1000, 2000]  # K

generator_params:
  mach_distribution: "uniform([5, 8])"
  reynolds_distribution: "log_uniform([1e5, 1e6])"
  wall_temp_distribution: "uniform([300, 2000])"
  chemistry_perturbation: "arrhenius_noise(10%)"

physics_gates:
  - mass_conservation: {threshold: 1e-5, per_species: true}
  - energy_stability: {threshold: 1e-5}
  - species_conservation: {threshold: 1e-4, species: ["N", "O", "NO"]}
  - boundary_satisfaction: {threshold: 1e-4, catalytic_wall: true}
  - thermal_protection: {threshold: 50, metric: "heat_flux_error_K"}

reference_solver: "DPLR / US3D"
validation_cases: 20
validation_tolerance: "Heat flux < 10% vs DPLR, species < 15%"
```

#### 11. Turek/Hron FSI 3D
```yaml
# carbon/generators/phase05/fsi3d.yaml
geometry:
  fluid: "channel_with_cylinder.fluid.stl"
  solid: "cylinder.solid.stl"
  dimension: 3D
  
pde: "incompressible_navier_stokes + nonlinear_elasticity"
coupling: "preCICE"  # implicit coupling

physics:
  fluid: {rho: 1000, nu: 0.001}
  solid: {rho: 1000, E: 1.4e6, nu: 0.4}

generator_params:
  inlet_velocity: "uniform([0.5, 2.0]) m/s"
  solid_stiffness: "log_uniform([1e5, 1e7]) Pa"
  mesh_refinement: "adaptive(fluid: 2, solid: 3)"

physics_gates:
  - mass_conservation: {threshold: 1e-6}
  - momentum_conservation: {threshold: 1e-5}
  - energy_stability: {threshold: 1e-5}
  - interface_continuity: {threshold: 1e-4, metric: "velocity_jump"}
  - rollout_stability: {steps: 5000}

reference_solver: "preCICE + OpenFOAM + CalculiX"
validation_cases: 15
validation_tolerance: "Displacement < 5%, Drag < 3% vs reference"
```

#### 12. Store Separation (6-DOF)
```yaml
# carbon/generators/phase05/store_separation.yaml
geometry:
  aircraft: "f16_simplified.stl"
  store: "mk82_bomb.stl"
  dimension: 3D
  
pde: "compressible_navier_stokes + rigid_body_6dof"
coupling: "overset_mesh / dynamic_mesh"

generator_params:
  mach: "uniform([0.6, 0.95])"
  altitude: "uniform([1000, 10000]) ft"
  store_mass: "uniform([200, 1000]) kg"
  store_inertia: "scaled_from_mass"
  initial_position: "pylon_mounted"
  ejection_force: "uniform([500, 2000]) N"

physics_gates:
  - mass_conservation: {threshold: 1e-5}
  - energy_stability: {threshold: 1e-5}
  - trajectory_accuracy: {threshold: 0.05, metric: "position_error_m"}
  - rollout_stability: {steps: 10000}  # full separation

reference_solver: "OVERFLOW / FUN3D + 6DOF"
validation_cases: 10
validation_tolerance: "Trajectory < 0.5m RMS vs reference"
```

#### 13. Turbine Blade Heat Transfer (CHT)
```yaml
# carbon/generators/phase05/turbine_blade.yaml
geometry:
  source: "turbine_blade_with_cooling.stl"
  dimension: 3D
  
pde: "conjugate_heat_transfer"
physics:
  fluid: {rho: 1.2, cp: 1005, k: 0.026, mu: 1.8e-5}
  solid: {rho: 8000, cp: 500, k: 25}
  
turbulence: "komega_sst"
cooling: "film_cooling_rows_3"

generator_params:
  mainstream_mach: "uniform([0.3, 0.6])"
  mainstream_temp: "uniform([1500, 2000]) K"
  coolant_ratio: "uniform([0.5, 2.0])%"
  coolant_temp: "uniform([600, 900]) K"
  blowing_ratio: "uniform([0.5, 2.0])"

physics_gates:
  - mass_conservation: {threshold: 1e-5}
  - energy_conservation: {threshold: 1e-5}
  - adiabatic_effectiveness: {threshold: 0.05, metric: "eta_error"}
  - film_cooling_uniformity: {threshold: 0.1, metric: "lateral_uniformity"}
  - rollout_stability: {steps: 2000}

reference_solver: "ANSYS Fluent / OpenFOAM chtMultiRegionFoam"
validation_cases: 20
validation_tolerance: "Adiabatic effectiveness < 0.05 RMS error"
```

---

## Validation Strategy — Scientific Rigor & Competitive Edge

Multi-objective scoring (45/30/25), hard/soft physics gates, and hidden stress testing. The Trustless Verification and Data Generation System (see dedicated document) ensures that both stress and benchmark data are generated in a publicly seeded, auditable, and scientifically credible manner.

### Scoring Weights
- **Physics Fidelity (45%)**: Residuals, conservation laws, boundary conditions, stability.
- **Robustness (30%)**: Performance under hidden stress, long-term rollout, generalization.
- **Accuracy (25%)**: Benchmark/hold-out performance.

### ChallengeWinnerTracker (Emission Weighting)
Only strategies that set a new best *combined score* on a challenge receive meaningful weight.

- **Per-challenge leader tracking** with exponential decay on old performance:
  `weight = score × e^(-blocks_since_win / half_life)` (half_life = 30 days, tunable via governance).
- **Winner-heavy weighting** + participation dust for recent contributors.
- **Only genuine new best combined scores** drive strong rewards.
- Future phases add hybrid model with Breakthrough Bounties (record-setting improvements) and Decaying Top stipends; unclaimed allocations roll to treasury.

---

## Determinism & Reproducibility
Hierarchical seeding and Docker-based reproducibility harness ensure all training, stress testing, and scoring are reproducible and auditable. The trustless data generation system extends this reproducibility to the evaluation data itself.

**Seeding Hierarchy:**
```
master_seed = hash(challenge_id + block_hash + run_nonce)
  ├── data_seed = splitmix64(master_seed, 0)
  ├── stress_seed = splitmix64(master_seed, 1)
  ├── init_seed = splitmix64(master_seed, 2)
  ├── dropout_seed = splitmix64(master_seed, 3)
  └── shuffle_seed = splitmix64(master_seed, 4)
```

**Docker Determinism:**
```dockerfile
# Pinned base image
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:<pinned>

# Pinned Python packages via requirements-lock.txt
# PYTHONHASHSEED=0
# CUBLAS_WORKSPACE_CONFIG=:4096:8
# torch.use_deterministic_algorithms(True)
```

---

## Landscape Agent — Symbolic & Causal Compounding

Ingests results and Model Cards from production and high-quality test runs. Extracts symbolic features and applies causal analysis to discover effective training methodologies. ModelingToolkit.jl integration will be used to turn extracted symbolic constraints into structured, usable loss terms.

### Phasing
- **Phase 0**: PySR symbolic regression → conservation laws, symmetries, invariants → JSON serialization → ModelingToolkit.jl → structured loss terms.
- **Phase 1**: Double Machine Learning (EconML/Custom JAX) causal inference → strategy choices (loss weights, curriculum, architecture) → robustness outcomes → strategic guidance + specialist distillation.
- **Phase 2**: Cross-domain causal mapping → transferable insights across physics classes → universal priors.

**Bridge**: JSON serialization first (Python ↔ Julia), JuliaCall later for performance. Structured losses compiled to differentiable JAX loss terms for miner consumption.

### PySR Configuration (Phase 0)
```python
# carbon/landscape/pysr_config.py
PYSR_CONFIG = {
    "populations": 50,
    "population_size": 100,
    "ncycles_per_iteration": 500,
    "maxsize": 40,
    "maxdepth": 8,
    "binary_operators": ["+", "-", "*", "/", "^"],
    "unary_operators": ["sin", "cos", "exp", "log", "sqrt", "abs"],
    "constraints": {
        "pow": (-1, 1),  # no high powers
    },
    "complexity_of_operators": {"+": 1, "-": 1, "*": 2, "/": 2, "^": 3},
    "feature_names": [
        "loss_data_weight", "loss_physics_weight", "loss_boundary_weight",
        "lr_initial", "lr_decay_rate", "curriculum_phase", "backbone_depth",
        "backbone_width", "activation_type", "normalization_type",
        "physics_gate_margin", "residual_l2", "conservation_l2", "boundary_l2"
    ],
    "target_name": "robustness_score",
    "verbosity": 1,
}
```

### ModelingToolkit.jl Bridge (JSON → Loss Terms)
```julia
# carbon/landscape/bridge.jl
using ModelingToolkit, Symbolics, JSON3

function json_to_loss_term(json_expr::Dict) -> ModelingToolkit.Equation
    """Convert PySR JSON expression to MT differentiable loss term."""
    @variables x[1:n]  # strategy parameters
    @parameters p[1:m]  # physics state
    
    # Parse PySR expression tree
    expr = parse_pysr_json(json_expr)
    
    # Compile to differentiable function
    loss_fn = eval(build_function(expr, x, p))
    
    # Return as MT equation for composition
    return loss_fn
end

# Example output: structured loss term
# L_structured = λ₁ * (∇·u)² + λ₂ * (∂ρ/∂t + ∇·(ρu))² + λ₃ * (∂E/∂t + ∇·(u(E+p)))²
```

---

## Detailed Implementation Components

### Core Subnet

#### 1. Strategy JSON Schema (Miner Submission)
```json
{
  "$schema": "https://carbonsubnet.org/schemas/strategy/v1.0.json",
  "challenge_id": "navier-stokes-laminar-2d-v1",
  "backbone": "fno",
  "backbone_config": {
    "modes": 32,
    "width": 64,
    "depth": 4,
    "activation": "gelu",
    "lifting_dim": 128,
    "projection_dim": 128,
    "normalization": "instance_norm"
  },
  "training": {
    "epochs": 500,
    "batch_size": 32,
    "optimizer": "adamw",
    "learning_rate": 1e-3,
    "lr_schedule": "cosine_warm_restarts",
    "lr_warmup_epochs": 10,
    "weight_decay": 1e-4,
    "gradient_clip": 1.0,
    "mixed_precision": true,
    "compile": true
  },
  "loss": {
    "data_mse": 1.0,
    "physics_residual": 0.5,
    "boundary_mse": 0.3,
    "conservation_penalty": 0.2,
    "adaptive_reweighting": {
      "enabled": true,
      "bounds": {
        "physics_residual": [0.1, 2.0],
        "boundary_mse": [0.05, 1.0]
      },
      "update_frequency": 10
    }
  },
  "curriculum": [
    {"phase": 1, "reynolds": [100, 500], "epochs": 100},
    {"phase": 2, "reynolds": [500, 2000], "epochs": 200},
    {"phase": 3, "reynolds": [2000, 5000], "epochs": 200}
  ],
  "data": {
    "train_split": 0.8,
    "augmentation": ["rotation", "scaling", "noise"],
    "noise_level": 0.01
  },
  "checkpointing": {
    "save_every_n_epochs": 25,
    "keep_best_n": 5,
    "metric": "combined_score"
  }
}
```

#### 2. Model Card Schema (Validator Output)
```json
{
  "$schema": "https://carbonsubnet.org/schemas/model-card/v1.0.json",
  "model_card_version": "1.0",
  "carbon_provenance": {
    "subnet": "carbon",
    "generator_version": "v1.3.2",
    "challenge_id": "navier-stokes-laminar-2d-v1",
    "bittensor_block_height": 3840212,
    "validator_set": [
      {"hotkey": "5Grwva...", "stake": 12500, "consensus_weight": 0.987},
      {"hotkey": "5FHneW...", "stake": 8900, "consensus_weight": 0.982}
    ],
    "consensus_mechanism": "Yuma Consensus (Bittensor)",
    "reproducibility": {
      "docker_image": "carbon/validator:v1.3.2@sha256:<pinned>",
      "seed_derivation": "hash(challenge_id + block_hash + run_nonce)",
      "deterministic_training": true,
      "master_seed": "0xabc123..."
    }
  },
  "model_name": "NS-Laminar-2D-FNO-v4",
  "architecture": {
    "backbone": "FNO",
    "config": {"modes": 32, "width": 64, "depth": 4, "activation": "gelu"},
    "strategy_json_hash": "sha256-a1b2c3d4...",
    "parameter_count": 2400000
  },
  "training_dynamics": {
    "epochs_completed": 500,
    "final_losses": {"data_mse": 2.3e-4, "physics_residual": 1.1e-3, "boundary_mse": 4.2e-5},
    "loss_curves": {"data_mse": [...], "physics_residual": [...], "epochs": [1,2,...]},
    "lr_curve": [...],
    "adaptive_reweighting_history": [...]
  },
  "physics_gate_results": {
    "methodology": "Carbon Tier 2 Hidden Adversarial Evaluation",
    "generator_version": "v1.3.2",
    "stress_variants_tested": 247,
    "gates": [
      {
        "gate_id": "mass_conservation",
        "pde": "∂ρ/∂t + ∇·(ρu) = 0",
        "metric": "max_continuity_residual_L2",
        "threshold": 1.0e-6,
        "result": 4.12e-7,
        "status": "PASS",
        "worst_case_variant": "transonic_shock_042"
      },
      {
        "gate_id": "energy_stability",
        "pde": "ρ·De/Dt = -∇·q - p∇·u + Φ",
        "metric": "energy_dissipation_violation",
        "threshold": 1.0e-6,
        "result": 8.89e-7,
        "status": "PASS",
        "worst_case_variant": "high_mach_shear_118"
      },
      {
        "gate_id": "boundary_satisfaction",
        "pde": "u|_∂Ω = g_D, (σ·n)|_∂Ω = g_N",
        "metric": "max_boundary_residual_Linf",
        "threshold": 1.0e-5,
        "result": 2.34e-6,
        "status": "PASS"
      },
      {
        "gate_id": "rollout_stability",
        "protocol": "10,000 step autoregressive + 1% Gaussian perturbation",
        "metric": "blowup_detection",
        "threshold": "zero blowups",
        "result": "0 blowups",
        "status": "PASS"
      },
      {
        "gate_id": "uq_calibration",
        "method": "conformal_prediction",
        "metric": "coverage_probability",
        "threshold": "≥0.95 at 95% confidence",
        "result": 0.953,
        "status": "PASS"
      }
    ],
    "overall_score": {"physics_fidelity": 43, "robustness": 26, "accuracy": 23, "total": 92}
  },
  "failure_diagnostics": {
    "uncertainty_hotspots": [
      {"region": "leading_edge_shock", "condition": "Mach > 3.2", "degradation": "+15% residual"}
    ],
    "spectral_limits": {"max_resolved_wavenumber": 32, "aliasing_risk_above": 28},
    "operational_envelope": {
      "mach": [0.8, 3.5], "angle_of_attack": [-10, 20],
      "reynolds": [100, 5000], "thermal_gradient_limit_k_per_m": 1400
    }
  },
  "export_artifacts": {
    "onnx_model": {"path": "model.onnx", "opset": 18, "sha256": "sha256-e5f6g7h8..."},
    "weights_only": {"path": "model_weights.safetensors", "sha256": "sha256-i9j0k1l2..."},
    "strategy_json": {"path": "strategy.json", "sha256": "sha256-m3n4o5p6..."}
  }
}
```

#### 3. Physics Gates Implementation (JAX)
```python
# carbon/validator/physics_gates.py
import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
from dataclasses import dataclass
from typing import Callable, Dict, Any

@dataclass
class GateResult:
    gate_id: str
    threshold: float
    result: float
    status: str  # "PASS" | "FAIL"
    worst_case_variant: str = ""
    details: Dict[str, Any] = None

# --- Mass Conservation (Continuity) ---
@jit
def mass_conservation_residual(model_fn: Callable, coords: jnp.ndarray, params: Dict) -> jnp.ndarray:
    """∂ρ/∂t + ∇·(ρu) = 0"""
    # coords: (N, d+1) where last dim is time
    # model_fn outputs (rho, u_x, u_y, u_z)
    
    def continuity_eq(coord):
        rho, ux, uy, uz = model_fn(coord, params)
        # Time derivative of rho
        drho_dt = grad(lambda c: model_fn(c, params)[0])(coord)[-1]
        # Spatial divergence of rho*u
        def flux(c):
            rho, ux, uy, uz = model_fn(c, params)
            return jnp.array([rho*ux, rho*uy, rho*uz])
        div_flux = jnp.trace(jax.jacfwd(flux)(coord)[:, :3])  # spatial dims only
        return drho_dt + div_flux
    
    residuals = vmap(continuity_eq)(coords)
    return jnp.abs(residuals)

# --- Energy Stability ---
@jit
def energy_stability_residual(model_fn: Callable, coords: jnp.ndarray, params: Dict) -> jnp.ndarray:
    """ρ·De/Dt = -∇·q - p∇·u + Φ"""
    # Implementation depends on physics class
    # For compressible NS: total energy E = e + 0.5*|u|^2
    pass

# --- Boundary Satisfaction ---
@jit
def boundary_residual(model_fn: Callable, boundary_coords: jnp.ndarray, 
                      boundary_values: Dict, params: Dict) -> jnp.ndarray:
    """u|_∂Ω = g_D, (σ·n)|_∂Ω = g_N"""
    pred = vmap(model_fn)(boundary_coords, params)
    # Dirichlet
    dirichlet_error = jnp.abs(pred - boundary_values["dirichlet"])
    # Neumann (requires gradient)
    neumann_error = jnp.abs(grad(model_fn)(boundary_coords) - boundary_values["neumann"])
    return jnp.concatenate([dirichlet_error.flatten(), neumann_error.flatten()])

# --- Rollout Stability ---
@jit
def rollout_stability(model_fn: Callable, init_coords: jnp.ndarray, 
                      params: Dict, steps: int = 10000, perturb: float = 0.01) -> bool:
    """Autoregressive rollout with perturbation."""
    state = init_state
    for i in range(steps):
        state = model_fn(state, params)
        if i % 100 == 0:
            state = state + perturb * jax.random.normal(key, state.shape)
        if jnp.any(jnp.isnan(state)) or jnp.any(jnp.abs(state) > 1e6):
            return False
    return True

# --- UQ Calibration (Conformal Prediction) ---
def uq_calibration(model_fn: Callable, calibration_coords: jnp.ndarray,
                   calibration_targets: jnp.ndarray, params: Dict,
                   confidence: float = 0.95) -> float:
    """Split conformal prediction for coverage."""
    # Split calibration set
    n = len(calibration_coords)
    split_idx = n // 2
    
    # Train on first half, calibrate on second
    # Use conformal prediction to get prediction intervals
    # Return coverage probability
    pass

# --- Gate Runner ---
def run_all_gates(model_fn: Callable, challenge: str, params: Dict,
                  stress_data: Dict, generator_version: str) -> List[GateResult]:
    """Run all physics gates for a challenge."""
    gates = []
    
    # 1. Mass Conservation
    mass_residuals = mass_conservation_residual(model_fn, stress_data["coords"], params)
    max_mass_res = float(jnp.max(jnp.abs(mass_residuals)))
    gates.append(GateResult(
        gate_id="mass_conservation",
        threshold=1e-6,
        result=max_mass_res,
        status="PASS" if max_mass_res < 1e-6 else "FAIL",
        worst_case_variant=stress_data["worst_case"]["mass"]
    ))
    
    # 2. Energy Stability
    energy_residuals = energy_stability_residual(model_fn, stress_data["coords"], params)
    max_energy_res = float(jnp.max(jnp.abs(energy_residuals)))
    gates.append(GateResult(...))
    
    # 3. Boundary Satisfaction
    boundary_res = boundary_residual(model_fn, stress_data["boundary_coords"], 
                                     stress_data["boundary_values"], params)
    max_boundary_res = float(jnp.max(jnp.abs(boundary_res)))
    gates.append(GateResult(...))
    
    # 4. Rollout Stability
    rollout_ok = rollout_stability(model_fn, stress_data["init_coords"], params)
    gates.append(GateResult(...))
    
    # 5. UQ Calibration
    uq_coverage = uq_calibration(model_fn, stress_data["cal_coords"],
                                 stress_data["cal_targets"], params)
    gates.append(GateResult(...))
    
    return gates
```

#### 4. Procedural Stress Generator (JAX)
```python
# carbon/generators/base.py
import jax
import jax.numpy as jnp
from jax import random
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class GeneratorConfig:
    challenge_id: str
    physics_class: str
    dimension: int
    parameter_ranges: Dict[str, tuple]
    reference_solver: str
    validation_tolerance: str

class ProceduralGenerator(ABC):
    def __init__(self, config: GeneratorConfig):
        self.config = config
    
    @abstractmethod
    def generate_training_data(self, seed: int, n_samples: int) -> Dict:
        """Generate training data: {coords, fields, boundary_conditions}"""
        pass
    
    @abstractmethod
    def generate_stress_variants(self, seed: int, n_variants: int) -> Dict:
        """Generate hidden stress test variants."""
        pass
    
    @abstractmethod
    def generate_benchmark_data(self, seed: int, n_samples: int) -> Dict:
        """Generate held-out benchmark data."""
        pass
    
    def derive_seeds(self, master_seed: int) -> Dict[str, int]:
        """Hierarchical seed derivation."""
        keys = random.split(random.PRNGKey(master_seed), 5)
        return {
            "data": int(random.randint(keys[0], (), 0, 2**32)),
            "stress": int(random.randint(keys[1], (), 0, 2**32)),
            "init": int(random.randint(keys[2], (), 0, 2**32)),
            "dropout": int(random.randint(keys[3], (), 0, 2**32)),
            "shuffle": int(random.randint(keys[4], (), 0, 2**32)),
        }

# Example: Poisson Generator
class PoissonGenerator(ProceduralGenerator):
    def generate_training_data(self, seed: int, n_samples: int) -> Dict:
        key = random.PRNGKey(seed)
        # Sample coefficient field k(x) from log-normal
        # Sample source f(x) from GP
        # Solve -∇·(k∇u) = f using FEniCS (offline) or JAX-FEM (online)
        pass
```

#### 5. Backbone Registry (Dynamic Instantiation)
```python
# carbon/backbones/registry.py
from typing import Dict, Type, Any
import jax
import flax.linen as nn

BACKBONE_REGISTRY: Dict[str, Type[nn.Module]] = {}

def register_backbone(name: str):
    def decorator(cls: Type[nn.Module]):
        BACKBONE_REGISTRY[name] = cls
        return cls
    return decorator

def get_backbone(name: str, config: Dict) -> nn.Module:
    if name not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone: {name}. Available: {list(BACKBONE_REGISTRY.keys())}")
    return BACKBONE_REGISTRY[name](**config)

@register_backbone("fno")
class FNO(nn.Module):
    modes: int
    width: int
    depth: int
    activation: str = "gelu"
    lifting_dim: int = 128
    projection_dim: int = 128
    normalization: str = "instance_norm"
    
    @nn.compact
    def __call__(self, x):
        # x: (batch, *spatial, channels_in)
        # Lifting
        x = nn.Dense(self.lifting_dim)(x)
        x = getattr(nn, self.activation)(x)
        
        # Spectral layers
        for i in range(self.depth):
            x = SpectralConv(self.modes, self.width)(x)
            x = nn.Dense(self.width)(x)  # skip connection
            x = getattr(nn, self.activation)(x)
            if self.normalization == "instance_norm":
                x = nn.InstanceNorm()(x)
        
        # Projection
        x = nn.Dense(self.projection_dim)(x)
        x = getattr(nn, self.activation)(x)
        x = nn.Dense(self.projection_dim)(x)
        return x

@register_backbone("gino")
class GINO(nn.Module):
    # Graph-based Neural Operator
    pass

@register_backbone("wno")
class WNO(nn.Module):
    # Wavelet Neural Operator
    pass

@register_backbone("transolver")
class Transolver(nn.Module):
    # Attention-based Neural Operator
    pass

# Dynamic instantiation from strategy JSON
def instantiate_backbone(strategy: Dict) -> nn.Module:
    backbone_name = strategy["backbone"]
    config = strategy["backbone_config"]
    return get_backbone(backbone_name, config)
```

#### 6. CarbonScorer (Multi-Objective + Gates)
```python
# carbon/validator/scorer.py
from dataclasses import dataclass
import jax.numpy as jnp

@dataclass
class ScoreBreakdown:
    physics_fidelity: float  # 0-45
    robustness: float        # 0-30
    accuracy: float          # 0-25
    total: float             # 0-100
    gates_passed: bool
    gate_results: List[GateResult]

class CarbonScorer:
    WEIGHTS = {"physics_fidelity": 45, "robustness": 30, "accuracy": 25}
    
    def __init__(self, gate_thresholds: Dict = None):
        self.gate_thresholds = gate_thresholds or DEFAULT_GATE_THRESHOLDS
    
    def score(self, 
              model_fn: Callable,
              stress_data: Dict,
              benchmark_data: Dict,
              params: Dict,
              gate_results: List[GateResult]) -> ScoreBreakdown:
        
        # Hard gate check
        gates_passed = all(g.status == "PASS" for g in gate_results)
        if not gates_passed:
            return ScoreBreakdown(0, 0, 0, 0, False, gate_results)
        
        # Physics Fidelity (45%)
        pf_score = self._compute_physics_fidelity(gate_results)
        
        # Robustness (30%)
        rob_score = self._compute_robustness(stress_data, model_fn, params)
        
        # Accuracy (25%)
        acc_score = self._compute_accuracy(benchmark_data, model_fn, params)
        
        total = pf_score + rob_score + acc_score
        
        return ScoreBreakdown(
            physics_fidelity=pf_score,
            robustness=rob_score,
            accuracy=acc_score,
            total=total,
            gates_passed=True,
            gate_results=gate_results
        )
    
    def _compute_physics_fidelity(self, gate_results: List[GateResult]) -> float:
        """Score based on margin above gate thresholds."""
        margins = []
        for g in gate_results:
            if g.threshold > 0:
                margin = 1.0 - (g.result / g.threshold)
                margins.append(max(0.0, margin))
        avg_margin = sum(margins) / len(margins) if margins else 0.0
        return 45.0 * avg_margin
    
    def _compute_robustness(self, stress_data, model_fn, params) -> float:
        """Score based on performance across stress variants."""
        variant_scores = []
        for variant in stress_data["variants"]:
            pred = model_fn(variant["coords"], params)
            error = jnp.mean((pred - variant["target"])**2)
            variant_scores.append(float(1.0 / (1.0 + error)))
        avg_score = sum(variant_scores) / len(variant_scores)
        return 30.0 * avg_score
    
    def _compute_accuracy(self, benchmark_data, model_fn, params) -> float:
        pred = model_fn(benchmark_data["coords"], params)
        mse = float(jnp.mean((pred - benchmark_data["targets"])**2))
        # Normalize by target variance
        target_var = float(jnp.var(benchmark_data["targets"]))
        normalized_mse = mse / (target_var + 1e-8)
        accuracy = 1.0 / (1.0 + normalized_mse)
        return 25.0 * accuracy
```

#### 7. Validator Docker Image
```dockerfile
# carbon/validator/Dockerfile
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:abc123...

ENV PYTHONHASHSEED=0 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONUNBUFFERED=1 \
    CARBON_DETERMINISTIC=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/carbon
ENV PATH="/opt/carbon/bin:$PATH"

COPY requirements-validator.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-validator.txt

COPY carbon /opt/carbon/carbon
COPY docker/validator-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080 8081  # HTTP + MCP WebSocket
ENTRYPOINT ["/entrypoint.sh"]
CMD ["validator"]
```

**Validator Entrypoint:**
```bash
#!/bin/bash
# docker/validator-entrypoint.sh
set -euo pipefail

# Determinism
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TF_DETERMINISTIC_OPS=1
export TF_CUDNN_DETERMINISTIC=1

# GPU selection
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Start validator
exec python -m carbon.validator.main \
    --config /opt/carbon/config/validator.yaml \
    --hotkey ${VALIDATOR_HOTKEY} \
    --netuid ${NETUID:-1} \
    --subtensor.chain_endpoint ${SUBTENSOR_ENDPOINT:-wss://entrypoint-finney.opentensor.ai:443}
```

#### 8. Validator Training Pipeline
```python
# carbon/validator/training.py
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax.training import train_state
from typing import Callable, Dict, Any
import yaml

class TrainState(train_state.TrainState):
    epoch: int
    best_score: float
    rng: jax.Array

class ValidatorTrainer:
    def __init__(self, config: Dict):
        self.config = config
        self.checkpointer = ocp.StandardCheckpointer()
    
    def create_train_state(self, model_fn: Callable, params: Dict, 
                           strategy: Dict, rng: jax.Array) -> TrainState:
        """Initialize training state with optimizer from strategy."""
        tx = self._create_optimizer(strategy["training"])
        return TrainState.create(
            apply_fn=model_fn,
            params=params,
            tx=tx,
            epoch=0,
            best_score=-1.0,
            rng=rng
        )
    
    def _create_optimizer(self, training_config: Dict) -> optax.GradientTransformation:
        lr_schedule = self._create_lr_schedule(training_config)
        tx = optax.chain(
            optax.clip_by_global_norm(training_config.get("gradient_clip", 1.0)),
            optax.adamw(
                learning_rate=lr_schedule,
                weight_decay=training_config.get("weight_decay", 1e-4)
            )
        )
        return tx
    
    def _create_lr_schedule(self, training_config: Dict) -> optax.Schedule:
        schedule_type = training_config.get("lr_schedule", "cosine_warm_restarts")
        if schedule_type == "cosine_warm_restarts":
            return optax.cosine_decay_schedule(
                init_value=training_config["learning_rate"],
                decay_steps=training_config["epochs"],
                alpha=0.01
            )
        # ... other schedules
    
    def train(self, state: TrainState, train_loader, val_loader, 
              strategy: Dict, physics_gates_fn) -> TrainState:
        """Main training loop with checkpointing and adaptive loss reweighting."""
        
        for epoch in range(state.epoch, self.config["training"]["epochs"]):
            # Training step
            state, train_metrics = self._train_epoch(state, train_loader, strategy)
            
            # Validation + physics gates
            if epoch % 10 == 0 or epoch == self.config["training"]["epochs"] - 1:
                val_metrics = self._validate(state, val_loader)
                gate_results = self._run_physics_gates(state)
                
                # Adaptive loss reweighting
                state = self._adaptive_reweight(state, gate_results, strategy)
                
                # Checkpointing
                if self._should_checkpoint(epoch, val_metrics):
                    self._save_checkpoint(state, epoch)
                
                # Early stopping
                if self._should_stop(state):
                    break
        
        return state
    
    def _adaptive_reweight(self, state: TrainState, gate_results: List, 
                          strategy: Dict) -> TrainState:
        """Adaptive loss reweighting within bounds."""
        if not strategy["loss"].get("adaptive_reweighting", {}).get("enabled"):
            return state
        
        bounds = strategy["loss"]["adaptive_reweighting"]["bounds"]
        # Adjust loss weights based on gate margins
        # ... implementation
        return state
    
    def _save_checkpoint(self, state: TrainState, epoch: int):
        """Save checkpoint with Orbax."""
        ckpt = {
            "model": state.params,
            "optimizer": state.opt_state,
            "epoch": epoch,
            "best_score": state.best_score,
            "rng": state.rng
        }
        self.checkpointer.save(f"/checkpoints/epoch_{epoch}", ckpt)
```

#### 9. Online Residual Monitoring
```python
# carbon/validator/monitoring.py
class ResidualMonitor:
    """Callback for online physics residual monitoring during training."""
    
    def __init__(self, physics_gates_fn, update_frequency: int = 10):
        self.physics_gates_fn = physics_gates_fn
        self.update_frequency = update_frequency
        self.history = []
    
    def on_batch_end(self, state, batch, batch_idx):
        if batch_idx % self.update_frequency == 0:
            # Quick residual check on batch
            residuals = self._compute_batch_residuals(state, batch)
            self.history.append({"batch": batch_idx, "residuals": residuals})
    
    def on_epoch_end(self, state, epoch):
        if epoch % 10 == 0:
            # Full gate evaluation
            gate_results = self.physics_gates_fn(state.params)
            self.history.append({"epoch": epoch, "gates": gate_results})
            
            # Alert if gates degrading
            self._check_gate_health(gate_results)
```

#### 10. MCP Layer (Model Context Protocol)
```python
# carbon/mcp/protocol.py
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
import asyncio
import websockets

class MCPMessageType(Enum):
    # Client → Server
    GET_NOISY_PRIOR = "get_noisy_prior"
    SUBMIT_STRATEGY = "submit_strategy"
    GET_DIAGNOSTICS = "get_diagnostics"
    START_SESSION = "start_session"
    END_SESSION = "end_session"
    
    ESTIMATE = "estimate"
    TRAIN_LOCAL = "train_local"
    
    # Server → Client
    NOISY_PRIOR = "noisy_prior"
    SUBMISSION_RECEIPT = "submission_receipt"
    DIAGNOSTICS = "diagnostics"
    SESSION_STARTED = "session_started"
    STREAMING_UPDATE = "streaming_update"
    ERROR = "error"

@dataclass
class MCPMessage:
    type: MCPMessageType
    payload: Dict[str, Any]
    request_id: str = ""
    session_id: str = ""

class MCPClient:
    def __init__(self, endpoint: str = "wss://mcp.carbon.subnet:8081"):
        self.endpoint = endpoint
        self.ws = None
        self.session_id = None
        self.pending_requests = {}
    
    async def connect(self):
        self.ws = await websockets.connect(self.endpoint)
        asyncio.create_task(self._listen())
    
    async def _listen(self):
        async for message in self.ws:
            msg = MCPMessage(**json.loads(message))
            if msg.request_id in self.pending_requests:
                future = self.pending_requests.pop(msg.request_id)
                future.set_result(msg)
    
    async def call(self, method: str, **kwargs) -> Dict:
        request_id = str(uuid.uuid4())
        future = asyncio.Future()
        self.pending_requests[request_id] = future
        
        msg = MCPMessage(
            type=MCPMessageType(method.upper()),
            payload=kwargs,
            request_id=request_id,
            session_id=self.session_id
        )
        await self.ws.send(json.dumps(msg.__dict__))
        response = await asyncio.wait_for(future, timeout=30.0)
        return response.payload
    
    # Convenience methods
    async def get_noisy_prior(self, challenge: str, backbone: str) -> Strategy:
        return await self.call("get_noisy_prior", challenge=challenge, backbone=backbone)
    
    async def submit_strategy(self, strategy: Strategy) -> SubmissionReceipt:
        return await self.call("submit_strategy", strategy=strategy.dict())
    
    async def get_diagnostics(self, submission_id: str) -> Diagnostics:
        return await self.call("get_diagnostics", submission_id=submission_id)
    
    async def estimate(self, strategy: Strategy, prior: Strategy) -> EstimationResult:
        return await self.call("estimate", strategy=strategy.dict(), prior=prior.dict())
```

#### 11. generate_challenge() Factory
```python
# carbon/challenges/factory.py
from dataclasses import dataclass
from typing import Dict, Any, Optional
import yaml

@dataclass
class ChallengeSpec:
    challenge_id: str
    name: str
    physics_class: str
    dimension: str
    pde: str
    generator_config: Dict[str, Any]
    backbone_whitelist: list[str]
    gate_thresholds: Dict[str, float]
    scoring_weights: Dict[str, float] = None

def generate_challenge(spec: ChallengeSpec) -> Dict[str, Any]:
    """Generate complete challenge deployment package."""
    return {
        "challenge_id": spec.challenge_id,
        "metadata": {
            "name": spec.name,
            "physics_class": spec.physics_class,
            "dimension": spec.dimension,
            "pde": spec.pde,
        },
        "generator_config": spec.generator_config,
        "backbone_whitelist": spec.backbone_whitelist,
        "gate_thresholds": spec.gate_thresholds,
        "scoring_weights": spec.scoring_weights or {
            "physics_fidelity": 45,
            "robustness": 30,
            "accuracy": 25
        },
        "deployment": {
            "generator_version": "v1.3.2",
            "validator_image": "carbon/validator:v1.3.2",
            "min_validators": 5,
        }
    }

# CLI for sponsors
def create_sponsored_challenge(
    physics_class: str,
    geometry: str,  # STL/STEP file or generator params
    boundary_conditions: Dict,
    flight_envelope: Dict,
    tier: int = 2  # 2=Open, 3=IP-Licensed, 4=Private
) -> ChallengeSpec:
    """Factory for sponsored challenges."""
    # Auto-generator from geometry + BCs
    # Validate against reference solver
    # Return deployable challenge spec
    pass
```

#### 12. Reproducibility Harness
```python
# carbon/common/reproducibility.py
import jax
import jax.numpy as jnp
import hashlib
import json
import subprocess
from dataclasses import dataclass

@dataclass
class ReproducibilityReport:
    master_seed: int
    docker_image_hash: str
    git_commit: str
    python_hashseed: int
    cublas_config: str
    torch_deterministic: bool
    output_hash: str
    passed: bool

def set_global_determinism(seed: int = 42):
    """Set all random seeds and deterministic flags."""
    import os
    import random
    import numpy as np
    
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    random.seed(seed)
    np.random.seed(seed)
    jax.config.update("jax_default_prng_impl", "threefry")
    
    # JAX
    jax.config.update("jax_enable_x64", True)
    
    # PyTorch (if used)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    except ImportError:
        pass

def verify_reproducibility(run_fn, seed: int, n_runs: int = 3) -> ReproducibilityReport:
    """Run function n times, verify identical outputs."""
    set_global_determinism(seed)
    outputs = []
    for i in range(n_runs):
        set_global_determinism(seed)
        output = run_fn()
        outputs.append(hashlib.sha256(str(output).encode()).hexdigest())
    
    all_same = len(set(outputs)) == 1
    return ReproducibilityReport(
        master_seed=seed,
        docker_image_hash=get_docker_image_hash(),
        git_commit=get_git_commit(),
        python_hashseed=seed,
        cublas_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        torch_deterministic=os.environ.get("TORCH_DETERMINISTIC", "1") == "1",
        output_hash=outputs[0],
        passed=all_same
    )
```

#### 13. Genesis Contracts (Solidity Interfaces)
```solidity
// contracts/CarbonTreasury.sol
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IERC20 {
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

interface IDexRouter {
    function exactInputSingle(ExactInputSingleParams calldata params) external returns (uint256 amountOut);
    struct ExactInputSingleParams {
        address tokenIn;
        address tokenOut;
        uint24 fee;
        address recipient;
        uint256 deadline;
        uint256 amountIn;
        uint256 amountOutMinimum;
        uint160 sqrtPriceLimitX96;
    }
}

contract CarbonTreasury {
    IERC20 public immutable USDC;
    IERC20 public immutable ALPHA;
    IDexRouter public immutable DEX_ROUTER;
    address public immutable BOUNTY_POOL;
    address public immutable OPCO_MULTISIG;
    
    uint256 public immutable BUYBACK_BPS = 2500; // 25% of revenue
    uint256 public immutable TWAP_DURATION = 24 hours;
    
    event BuybackExecuted(uint256 usdcIn, uint256 alphaOut, uint256 blockNumber);
    
    constructor(
        address _usdc,
        address _alpha,
        address _dexRouter,
        address _bountyPool,
        address _opcoMultisig
    ) {
        USDC = IERC20(_usdc);
        ALPHA = IERC20(_alpha);
        DEX_ROUTER = IDexRouter(_dexRouter);
        BOUNTY_POOL = _bountyPool;
        OPCO_MULTISIG = _opcoMultisig;
    }
    
    function depositRevenue(uint256 amount) external {
        require(msg.sender == OPCO_MULTISIG, "Only OpCo");
        USDC.transferFrom(OPCO_MULTISIG, address(this), amount);
        
        uint256 buybackAmount = amount * BUYBACK_BPS / 10000;
        _executeTwapBuyback(buybackAmount);
    }
    
    function _executeTwapBuyback(uint256 usdcAmount) internal {
        // Split into N chunks over TWAP_DURATION
        uint256 chunks = 24; // hourly
        uint256 chunkSize = usdcAmount / chunks;
        
        for (uint256 i = 0; i < chunks; i++) {
            // Execute via DEX with TWAP logic
            // In practice: use Uniswap V3 TWAP oracle + limit orders
            // Simplified here:
            IDexRouter.ExactInputSingleParams memory params = IDexRouter.ExactInputSingleParams({
                tokenIn: address(USDC),
                tokenOut: address(ALPHA),
                fee: 3000, // 0.3%
                recipient: BOUNTY_POOL,
                deadline: block.timestamp + 1 hours,
                amountIn: chunkSize,
                amountOutMinimum: 0, // Protected by TWAP oracle
                sqrtPriceLimitX96: 0
            });
            DEX_ROUTER.exactInputSingle(params);
        }
        
        emit BuybackExecuted(usdcAmount, 0, block.number); // alphaOut tracked off-chain
    }
}
```

```solidity
// contracts/BountyPool.sol
contract BountyPool {
    IERC20 public immutable ALPHA;
    address public immutable CHALLENGE_WINNER_TRACKER; // Off-chain oracle
    
    mapping(bytes32 => uint256) public challengeBounties; // challenge_id -> alpha amount
    
    event BountyDeposited(bytes32 indexed challengeId, uint256 amount);
    event BountyClaimed(bytes32 indexed challengeId, address indexed miner, uint256 amount);
    
    constructor(address _alpha, address _tracker) {
        ALPHA = IERC20(_alpha);
        CHALLENGE_WINNER_TRACKER = _tracker;
    }
    
    function depositBounty(bytes32 challengeId, uint256 amount) external {
        require(msg.sender == address(CarbonTreasury), "Only Treasury");
        challengeBounties[challengeId] += amount;
        emit BountyDeposited(challengeId, amount);
    }
    
    function claimBounty(bytes32 challengeId, uint256 minerScore, bytes calldata proof) external {
        // Verify miner won challenge via ChallengeWinnerTracker (off-chain oracle)
        // Simplified: check merkle proof against off-chain root
        require(_verifyWinner(challengeId, msg.sender, minerScore, proof), "Invalid claim");
        
        uint256 amount = challengeBounties[challengeId];
        require(amount > 0, "No bounty");
        challengeBounties[challengeId] = 0;
        
        ALPHA.transfer(msg.sender, amount);
        emit BountyClaimed(challengeId, msg.sender, amount);
    }
}
```

```solidity
// contracts/VerificationRegistry.sol
contract VerificationRegistry {
    struct ModelProof {
        bytes32 modelCardHash;
        string generatorVersion;
        bytes32 operationalEnvelopeHash;
        bytes32 validatorSetHash;
        uint256 blockHeight;
        uint256 timestamp;
    }
    
    mapping(bytes32 => ModelProof) public models; // model_id -> proof
    
    event ModelRegistered(bytes32 indexed modelId, ModelProof proof);
    
    function registerModel(
        bytes32 modelId,
        bytes32 modelCardHash,
        string calldata generatorVersion,
        bytes32 operationalEnvelopeHash,
        bytes32 validatorSetHash
    ) external {
        models[modelId] = ModelProof({
            modelCardHash: modelCardHash,
            generatorVersion: generatorVersion,
            operationalEnvelopeHash: operationalEnvelopeHash,
            validatorSetHash: validatorSetHash,
            blockHeight: block.number,
            timestamp: block.timestamp
        });
        emit ModelRegistered(modelId, models[modelId]);
    }
    
    function getModelProof(bytes32 modelId) external view returns (ModelProof memory) {
        return models[modelId];
    }
    
    function verifyModelCard(bytes32 modelId, bytes32 modelCardHash) external view returns (bool) {
        return models[modelId].modelCardHash == modelCardHash;
    }
}
```

```solidity
// contracts/VerificationGas.sol
contract VerificationGas {
    IERC20 public immutable ALPHA;
    AggregatorV3Interface public immutable PRICE_FEED; // Chainlink α/USD
    
    uint256 public constant USD_PRICE_PER_QUERY = 0.001 * 1e18; // $0.001 in 18 decimals
    
    mapping(address => uint256) public prepaidBalance; // Partners stake α
    mapping(address => uint256) public tier; // 0=none, 1=10k/day, 2=100k/day, 3=unlimited
    
    uint256[] public TIER_STAKES = [0, 100_000 * 1e18, 500_000 * 1e18, 2_000_000 * 1e18];
    uint256[] public TIER_QUOTAS = [0, 10_000, 100_000, type(uint256).max];
    
    event Staked(address indexed partner, uint256 tier);
    event QueryExecuted(address indexed caller, bytes32 modelId, uint256 alphaCost);
    
    constructor(address _alpha, address _priceFeed) {
        ALPHA = IERC20(_alpha);
        PRICE_FEED = AggregatorV3Interface(_priceFeed);
    }
    
    function stakeForTier(uint256 tier) external {
        require(tier <= 3, "Invalid tier");
        uint256 amount = TIER_STAKES[tier];
        ALPHA.transferFrom(msg.sender, address(this), amount);
        prepaidBalance[msg.sender] += amount;
        tier[msg.sender] = tier;
        emit Staked(msg.sender, tier);
    }
    
    function query(bytes32 modelId) external payable returns (VerificationData memory) {
        uint256 gasCost = getGasInAlpha();
        
        if (prepaidBalance[msg.sender] >= gasCost) {
            prepaidBalance[msg.sender] -= gasCost;
        } else {
            require(msg.value >= gasCost, "Insufficient gas");
            // Refund excess
            if (msg.value > gasCost) {
                payable(msg.sender).transfer(msg.value - gasCost);
            }
        }
        
        emit QueryExecuted(msg.sender, modelId, gasCost);
        return _getVerificationData(modelId);
    }
    
    function getGasInAlpha() public view returns (uint256) {
        (, int256 price, , , ) = PRICE_FEED.latestRoundData();
        require(price > 0, "Stale price");
        return (USD_PRICE_PER_QUERY * 1e18) / uint256(price);
    }
}
```

```solidity
// contracts/PartnerStaking.sol
contract PartnerStaking {
    IERC20 public immutable ALPHA;
    
    enum Tier { NONE, TIER_1, TIER_2, TIER_3 }
    
    struct PartnerInfo {
        Tier tier;
        uint256 stakeAmount;
        uint256 stakedAt;
        bool slashed;
    }
    
    mapping(address => PartnerInfo) public partners;
    
    uint256 public constant TIER_1_STAKE = 100_000 * 1e18;
    uint256 public constant TIER_2_STAKE = 500_000 * 1e18;
    uint256 public constant TIER_3_STAKE = 2_000_000 * 1e18;
    
    uint256 public constant UNBONDING_PERIOD = 30 days;
    
    event PartnerStaked(address indexed partner, Tier tier);
    event PartnerUnstaked(address indexed partner);
    event PartnerSlashed(address indexed partner, uint256 amount);
    
    constructor(address _alpha) {
        ALPHA = IERC20(_alpha);
    }
    
    function stake(Tier tier) external {
        uint256 amount = _tierToAmount(tier);
        ALPHA.transferFrom(msg.sender, address(this), amount);
        partners[msg.sender] = PartnerInfo({
            tier: tier,
            stakeAmount: amount,
            stakedAt: block.timestamp,
            slashed: false
        });
        emit PartnerStaked(msg.sender, tier);
    }
    
    function unstake() external {
        PartnerInfo storage info = partners[msg.sender];
        require(info.tier != Tier.NONE, "Not staked");
        require(block.timestamp >= info.stakedAt + UNBONDING_PERIOD, "Unbonding period");
        require(!info.slashed, "Slashed");
        
        ALPHA.transfer(msg.sender, info.stakeAmount);
        delete partners[msg.sender];
        emit PartnerUnstaked(msg.sender);
    }
    
    function getTier(address partner) external view returns (Tier) {
        return partners[partner].tier;
    }
    
    function slash(address partner, uint256 amount) external {
        // Only governance
        PartnerInfo storage info = partners[partner];
        info.slashed = true;
        ALPHA.transfer(governanceTreasury, amount);
        emit PartnerSlashed(partner, amount);
    }
}
```

#### 14. Landscape Agent Pipeline
```python
# carbon/landscape/pipeline.py
class LandscapeAgent:
    def __init__(self, config: Dict):
        self.pysr_config = config.get("pysr", PYSR_CONFIG)
        self.dml_config = config.get("dml", DML_CONFIG)
        self.mt_bridge = ModelingToolkitBridge()
        self.specialist_bank = SpecialistBank()
        self.prior_engine = PriorEngine()
    
    def ingest_model_card(self, model_card: ModelCard):
        """Process new model card, update knowledge base."""
        features = self._extract_features(model_card)
        self.pysr_dataset.append(features, model_card.gate_results)
        self.dml_dataset.append(features, model_card.robustness_score)
        
        # Periodic batch processing
        if len(self.pysr_dataset) % 100 == 0:
            self._run_pysr()
        if len(self.dml_dataset) % 500 == 0:
            self._run_dml()
    
    def _run_pysr(self):
        """Run PySR symbolic regression on accumulated data."""
        equations = pysr_regress(self.pysr_dataset, self.pysr_config)
        for eq in equations:
            # Convert to structured loss term
            loss_term = self.mt_bridge.json_to_loss_term(eq.json)
            self.specialist_bank.add_loss_term(loss_term)
        
        # Update noisy priors
        self.prior_engine.update_from_symbolic(equations)
    
    def _run_dml(self):
        """Run Double ML causal inference."""
        # Treatment: strategy choices (discretized)
        # Outcome: robustness_score
        # Confounders: physics_class, data_seed, backbone
        causal_effects = double_ml(self.dml_dataset, self.dml_config)
        
        # Generate strategic guidance
        guidance = self._generate_guidance(causal_effects)
        self.prior_engine.update_from_causal(guidance)
    
    def get_noisy_prior(self, challenge: str, backbone: str) -> Strategy:
        """Get current best prior for challenge+backbone."""
        base = self.prior_engine.get_base_prior(challenge, backbone)
        return self._add_noise(base, noise_scale=0.1)
```

#### 15. PySR Configuration (Phase 0)
```python
# carbon/landscape/pysr_config.py
PYSR_CONFIG = {
    "populations": 50,
    "population_size": 100,
    "ncycles_per_iteration": 500,
    "maxsize": 40,
    "maxdepth": 8,
    "binary_operators": ["+", "-", "*", "/", "^"],
    "unary_operators": ["sin", "cos", "exp", "log", "sqrt", "abs"],
    "constraints": {"pow": (-1, 1)},
    "complexity_of_operators": {"+": 1, "-": 1, "*": 2, "/": 2, "^": 3},
    "feature_names": [
        "loss_data_weight", "loss_physics_weight", "loss_boundary_weight",
        "lr_initial", "lr_decay_rate", "curriculum_phase", "backbone_depth",
        "backbone_width", "activation_type", "normalization_type",
        "physics_gate_margin", "residual_l2", "conservation_l2", "boundary_l2"
    ],
    "target_name": "robustness_score",
    "verbosity": 1,
    "batch_size": 1000,
    "early_stop_condition": "stop_if_no_improvement(50)",
}
```

#### 16. ModelingToolkit.jl Bridge
```julia
# carbon/landscape/bridge.jl
module CarbonMTBridge
using ModelingToolkit, Symbolics, JSON3, StructTypes

function json_to_loss_term(json_expr::Dict) 
    @variables t x y z
    @parameters p[1:20]  # strategy params
    
    # Parse PySR expression tree
    expr = parse_pysr_json(json_expr)
    
    # Compile to differentiable function
    loss_fn = eval(build_function(expr, [p...], [t, x, y, z]))
    
    return loss_fn
end

function parse_pysr_json(json::Dict)
    # Recursively parse PySR expression tree to Symbolics expression
    # Handles: +, -, *, /, ^, sin, cos, exp, log, sqrt, abs
    # Returns Symbolics expression
end

function loss_terms_to_jax(loss_fns::Vector) -> String
    """Generate JAX code for compiled loss terms."""
    # Generate: 
    # def structured_loss(params, physics_state):
    #     term1 = λ₁ * (div(u))^2
    #     term2 = λ₂ * (dρ/dt + div(ρu))^2
    #     return sum(terms)
end
end
```

---

## Phased Roadmap (Build-Level)

**Phase 0 (Months 0-4)**:
- Black-box diagnostics with clear diagnostic tiers (reputation-gated)
- Noisy prior distribution + Estimation Mode (anchored to noisy priors)
- Miner Toolkit Docker image with local support and basic Python interface
- Light Training templates + local multi-fidelity evaluation
- Direct submission path (always available)
- Basic cost estimation for rented compute
- Trustless procedural data generation system (open generator + public seeding)
- **7 Phase 0 PDE generators + FEniCS validation harness**
- **3 Backbones (FNO, GINO, WNO) in JAX**
- **Physics Gates 1-5 implemented + calibrated**
- **Model Card Generator + ONNX Export + Verification Registry write**
- **5 Genesis Contracts deployed (testnet → mainnet)**
- **5 Validators live (10× H100)**
- **Model Zoo API v1 (7 specialists) + 3 Pilot Subscribers**

**Phase 0.5 (Months 4-8)**:
- **6 Defense Benchmark Generators** (NACA 0012, CRM, HIFiRE-1, FSI 3D, Store Separation, Turbine CHT)
- **Generator validation vs. reference solvers (SU2/OpenFOAM/preCICE) published**
- **Phase 0.5 challenges live on subnet**
- **Model Zoo: 15 specialists (7 academic + 8 defense-relevant)**

**Phase 1 (Months 6-12)**:
- ModelingToolkit.jl integration for structured losses from PySR
- Stronger strategic guidance generation from Landscape Agent (PySR → loss terms)
- Initial scoped Abaqus ingestion utilities (ODB → mesh + fields)
- Cloud rental integration (Targon + Chutes prioritized)
- Enhanced trustless verification features (commit-reveal seeding, stronger generator validation)
- **First Sponsored Challenge LOIs (Tier 2/3)**
- **Prime Teaming Agreement signed**
- **SBIR Phase I submitted via Prime**
- **Verification Gas pricing enabled (first partner integration)**
- **Air-Gapped Miner Toolkit v1**

**Phase 2 (Months 12-24)**:
- Cross-domain causal mapping via Double Machine Learning
- Advanced agent tooling and multi-asset emissions features
- Advanced gaming resistance mechanisms for trustless verification
- Verified multi-physics benchmarks (FSI, CHT, Thermo-Elasticity) with preCICE
- **Tier 4 Private/On-Prem deployments**
- **SBIR Phase II awarded**

**Phase 3 (Months 24-36)**:
- 3D multi-physics (FSI, CHT, Thermo-elasticity with turbulence)
- 3D-specific gates (vorticity, boundary layers, turbulence spectra)
- Curriculum progression from 2D specialists
- Advanced confidential computing integration
- **$10M+ ARR target**

---

## Scientific Defensibility & Competitive Differentiation

All extensions are designed to be scientifically grounded, reproducible, and auditable while enabling faster discovery of superior Neural Operator training methodologies. The Trustless Verification and Data Generation System is a core part of this defensibility, replacing authority-based trust with verifiable, publicly inspectable mechanisms.

---

*This specification is written to be scientifically rigorous and buildable. Reference the implementation in `neurons/` and supporting design documents for concrete code.*
