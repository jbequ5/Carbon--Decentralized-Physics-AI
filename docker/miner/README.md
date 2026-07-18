# Hydrogen Miner Docker Environment

This provides a clean, user-friendly way to run miners and agentic miners.

## Quick Start

```bash
# 1. Copy the example env file
cp docker/miner/.env.example .env

# 2. Edit .env with your hotkey
nano .env

# 3. Run the miner
cd ..
docker compose up miner
```

## Available Modes

| Mode                        | Command                                      | Description |
|-----------------------------|----------------------------------------------|-------------|
| Default agentic miner       | `python examples/run_agentic_miner.py`       | Runs the built-in example loop |
| MCP-style tool server       | `python -m hydrogen.miner.mcp_server`        | Exposes tools for LLM agents |
| Custom agent                | `python your_agent.py`                       | Run your own script |
| Interactive shell           | `bash`                                       | Explore inside the container |

## Configuration

You can configure via:

- Environment variables (recommended)
- `.env` file (copied from `.env.example`)

Key variables:

- `HYDROGEN_HOTKEY` — Your Bittensor hotkey
- `HYDROGEN_WALLET` — Wallet name
- `HYDROGEN_API_KEY` — Optional key for MCP server

## Running Custom Agents

```bash
# Mount and run your script
docker compose run miner python /app/my_agent.py

# Or with a volume
 docker run -v $(pwd)/my_agent.py:/app/my_agent.py hydrogen-miner python my_agent.py
```

## Tips for Best Experience

- Always start strategies from `get_priors` when possible
- Use `validate_strategy` before submitting
- Check `STRATEGY.md` for guidance on writing good strategies
- The container is pre-configured with the full SDK and `AgenticMiner`

## Advanced

You can also start the container with a custom command:

```bash
docker compose run miner bash
```
