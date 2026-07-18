"""Lightweight MCP-compatible Server for Hydrogen Mining Tools.

This provides a simple server that exposes the Hydrogen mining tools
in a structured way that can be consumed by MCP clients or LLM agents.

It uses FastAPI + Pydantic for clean schemas and auto-generated docs.
"""

import uvicorn
from fastapi import FastAPI

from hydrogen.miner.agent import AgenticMiner
from hydrogen.miner.agent_tools import HydrogenMiningTools


app = FastAPI(
    title="Hydrogen Mining Tools",
    description="MCP-style tools for autonomous mining in the Hydrogen subnet.",
    version="0.1.0",
)

# In production, initialize the real client here
# from hydrogen_miner import HydrogenClient
# client = HydrogenClient(...)
# miner = AgenticMiner(client)

miner = None  # Placeholder - replace with real initialization
tools = None


def init_tools(miner_instance: AgenticMiner):
    global miner, tools
    miner = miner_instance
    tools = HydrogenMiningTools(miner)


@app.get("/tools/list_challenges", response_model=HydrogenMiningTools.__annotations__['list_challenges'].__annotations__['return'])
async def list_challenges():
    return await tools.list_challenges()


@app.post("/tools/get_priors")
async def get_priors(input_data: HydrogenMiningTools.__annotations__['get_priors'].__annotations__['input']):
    return await tools.get_priors(**input_data.dict())


@app.post("/tools/propose_strategy")
async def propose_strategy(input_data: HydrogenMiningTools.__annotations__['propose_strategy'].__annotations__['input']):
    return await tools.propose_strategy(**input_data.dict())


@app.post("/tools/validate_strategy")
async def validate_strategy(input_data: HydrogenMiningTools.__annotations__['validate_strategy'].__annotations__['input']):
    return await tools.validate_strategy(**input_data.dict())


@app.post("/tools/submit_strategy")
async def submit_strategy(input_data: HydrogenMiningTools.__annotations__['submit_strategy'].__annotations__['input']):
    return await tools.submit_strategy(**input_data.dict())


@app.post("/tools/get_my_recent_results")
async def get_my_recent_results(input_data: HydrogenMiningTools.__annotations__['get_my_recent_results'].__annotations__['input']):
    return await tools.get_my_recent_results(**input_data.dict())


if __name__ == "__main__":
    # Example: uvicorn hydrogen.miner.mcp_server:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
