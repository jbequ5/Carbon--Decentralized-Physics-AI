#!/bin/bash
set -e

echo "=============================================="
echo "   Hydrogen Miner Environment (v0.2)"
echo "=============================================="
echo ""

# Detect configuration
echo "Configuration:"
echo "  Hotkey : ${HYDROGEN_HOTKEY:-<not set>}"
echo "  Wallet : ${HYDROGEN_WALLET:-<not set>}"
echo "  API Key: ${HYDROGEN_API_KEY:-<not set>}"
echo ""

# Intelligent guidance
echo "Quick Commands:"
echo "  Run default agentic miner:"
echo "    python examples/run_agentic_miner.py"
echo ""
echo "  Start MCP-style tool server:"
echo "    python -m hydrogen.miner.mcp_server"
echo ""
echo "  Run your own agent script:"
echo "    python your_agent.py"
echo ""
echo "  Interactive shell:"
echo "    bash"
echo ""
echo "Tips:"
echo "  - Use environment variables or mount a .env file"
echo "  - Mount custom scripts with -v"
echo "  - See STRATEGY.md for how to write good strategies"
echo "  - Use 'get_priors' to start from Landscape recommendations"
echo ""
echo "=============================================="
echo ""

# Execute whatever command was passed
exec "$@"
