# Hydrogen Miner Docker Environment

This directory contains everything needed to run miners and agentic miners easily.

## Quick Start

```bash
# Build and run the default agentic miner
cd ..
docker compose up miner
```

## Configuration

Set these environment variables (in `.env` or directly):

```bash
export HYDROGEN_HOTKEY=your_hotkey_here
export HYDROGEN_WALLET=your_wallet_name
export HYDROGEN_API_KEY=optional_api_key_for_mcp
```

## Running Your Own Agent

```bash
docker compose run miner python /app/your_custom_agent.py
```

Or mount a local script:

```bash
docker run -v $(pwd)/my_agent.py:/app/my_agent.py hydrogen-miner python my_agent.py
```

## Available Modes (inside container)

- `python examples/run_agentic_miner.py` — Built-in example agent loop
- `python -m hydrogen.miner.mcp_server` — Start MCP-style tool server
- Custom Python scripts using `AgenticMiner`

## Features

- Pre-installed Hydrogen SDK + AgenticMiner
- MCP-style tool interface ready for LLM agents
- Automatic guidance on startup
- Easy configuration via environment variables

## Next Steps

See the main `STRATEGY.md` for how to write good strategies and use priors.
