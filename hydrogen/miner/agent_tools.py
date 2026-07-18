"""MCP-style Tool Interface for AgenticMiner.

This module exposes the core mining actions as clean, well-documented
callable tools. It is designed to be easily wrapped by MCP servers,
LangChain tools, or any agent framework.

Agents (especially LLMs) can discover and call these tools reliably.
"""

from typing import Dict, Any

from hydrogen.miner.agent import AgenticMiner


class HydrogenMiningTools:
    """
    Structured tool interface for agentic mining.

    Each method is designed to be exposed as a tool with clear
    input/output schemas.
    """

    def __init__(self, miner: AgenticMiner):
        self.miner = miner

    async def list_challenges(self) -> Dict[str, Any]:
        """List all currently active challenges."""
        challenges = await self.miner.get_challenges()
        return {"challenges": challenges}

    async def get_priors(self, challenge_id: str) -> Dict[str, Any]:
        """
        Get symbolically and causally recommended priors for a challenge.
        Use this to bootstrap good strategies.
        """
        priors = await self.miner.get_priors(challenge_id)
        return priors

    async def propose_strategy(
        self, challenge_id: str, base_strategy: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Propose a new strategy for a challenge.

        If no base_strategy is provided, starts from Landscape priors
        and applies light mutation.
        """
        strategy = await self.miner.propose_strategy(
            challenge_id=challenge_id, base_strategy=base_strategy
        )
        return strategy

    async def validate_strategy(
        self, strategy: Dict[str, Any], challenge_id: str, quick: bool = True
    ) -> Dict[str, Any]:
        """
        Locally validate a strategy before submitting.

        Returns estimated performance and physics gate diagnostics.
        Use this to iterate cheaply.
        """
        result = await self.miner.validate_locally(
            strategy=strategy, challenge_id=challenge_id, quick=quick
        )
        return result

    async def submit_strategy(
        self, strategy: Dict[str, Any], challenge_id: str
    ) -> Dict[str, Any]:
        """
        Submit a strategy to the network.

        Only do this after validating locally and being confident
        in the estimated score.
        """
        result = await self.miner.submit(strategy=strategy, challenge_id=challenge_id)
        return result

    async def get_my_recent_results(self, limit: int = 10) -> Dict[str, Any]:
        """Get recent submission results for learning/iteration."""
        results = await self.miner.get_recent_performance(limit=limit)
        return {"recent_results": results}


# Example of how an LLM agent might use these tools
async def example_llm_agent_usage(tools: HydrogenMiningTools):
    challenges = await tools.list_challenges()
    print("Available challenges:", challenges)

    for challenge_id in challenges.get("challenges", []):
        priors = await tools.get_priors(challenge_id)
        print(f"Priors for {challenge_id}:", priors)

        strategy = await tools.propose_strategy(challenge_id)
        validation = await tools.validate_strategy(strategy, challenge_id)

        print(f"Estimated score for proposed strategy: {validation.get('estimated_score')}")

        if validation.get("estimated_score", 0) > 0.07:
            result = await tools.submit_strategy(strategy, challenge_id)
            print("Submitted!", result)
