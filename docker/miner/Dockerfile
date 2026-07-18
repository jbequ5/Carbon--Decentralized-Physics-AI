# Docker image for running Hydrogen miners and agentic miners

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml .
RUN pip install --upgrade pip
RUN pip install -e .

# Copy the rest of the code
COPY . .

# Default command runs the example agentic miner
# Users can override this
CMD ["python", "examples/run_agentic_miner.py"]
