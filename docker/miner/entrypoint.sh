#!/bin/bash
set -e

echo "=============================================="
echo "   Hydrogen Miner / Agent Environment"
echo "=============================================="
echo ""
echo "Environment ready."
echo ""

echo "Available commands:"
echo "  python examples/run_agentic_miner.py     # Run example agentic miner"
echo "  python -m hydrogen.miner.agent           # Use AgenticMiner programmatically"
echo "  hydrogen-miner --help                    # CLI (if installed)"
echo ""

echo "Environment variables you can set:"
echo "  HYDROGEN_HOTKEY     - Your Bittensor hotkey"
echo "  HYDROGEN_WALLET     - Your wallet name"
echo "  HYDROGEN_API_KEY    - Optional API key for MCP server"
echo ""

echo "To run your own agent, mount your script and override the command."
echo "Example:"
echo "  docker run -v ​$(pwd)/my_agent.py:/app/my_agent.py hydrogen-miner python my_agent.py"
echo ""

echo "Starting container..."
echo "=============================================="
echo ""

exec "$@"
