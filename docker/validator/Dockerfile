# hydrogen/validator:pino-v0.1 (Phase 0 MVP)
# Pinned, reproducible environment for validators

FROM nvcr.io/nvidia/pytorch:24.09-py3

LABEL maintainer="Hydrogen contributors <jbequ5>"
LABEL description="Hydrogen validator image with PhysicsNeMo + deterministic harness (Phase 0)"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates julia=1.10.* \
    && rm -rf /var/lib/apt/lists/*

RUN julia -e 'using Pkg; Pkg.add(["ModelingToolkit", "Symbolics", "DataDrivenDiffEq"]); Pkg.precompile()'

RUN pip install --no-cache-dir \
    "physicsnemo" \
    "neuraloperator>=0.3.0" \
    "bittensor>=6.0" \
    "torch>=2.3" \
    "numpy" "scipy" "pandas" "pyyaml" "tqdm" "juliacall" "python-dotenv"

WORKDIR /workspace
COPY . /workspace/hydrogen

ENTRYPOINT ["python", "-m", "hydrogen.validator.entrypoint"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('CUDA available:', torch.cuda.is_available())" || exit 1
