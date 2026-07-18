#!/bin/bash
set -e

echo "=============================================="
echo "   Hydrogen Miner / Agentic Environment"
echo "=============================================="
echo ""

# Show helpful info
echo "Container started successfully."
echo ""

echo "Environment variables detected:"
echo "  HYDROGEN_HOTKEY     = ${HYDROGEN_HOTKEY:-<not set>}"
echo "  HYDROGEN_WALLET     = ${HYDROGEN_WALLET:-<not set>}"
echo "  HYDROGEN_API_KEY    = ${HYDROGEN_API_KEY:-<not set>}"
echo ""

# Intelligent guidance
echo "Quick start options:"
echo "  1. Run the built-in agentic miner example:"
echo "     python examples/run_agentic_miner.py"
echo ""
echo "  2. Use the AgenticMiner class in your own script"
echo ""
echo "  3. Start the MCP-style server:"
echo "     python -m hydrogen.miner.mcp_server"
echo ""
echo "Tips:"
echo "  - Mount your own agent script with -v"
echo "  - Override the command to run custom logic"
echo "  - Use environment variables for configuration"
echo ""
echo "For full documentation see: STRATEGY.md and examples/"
echo "=============================================="
echo ""

# Execute the command passed to the container
exec "$@"
