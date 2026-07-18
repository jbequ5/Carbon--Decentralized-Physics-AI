"""Improved MCP-compatible Server for Hydrogen Mining Tools.

Features:
- Proper error handling
- Request validation
- Optional API key authentication
- Health check
- Clean structured responses
"""

import os

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import JSONResponse

from hydrogen.miner.agent import AgenticMiner
from hydrogen.miner.agent_tools import (
    HydrogenMiningTools,
    ListChallengesOutput,
    GetPriorsInput,
    GetPriorsOutput,
    ProposeStrategyInput,
    ProposeStrategyOutput,
    ValidateStrategyInput,
    ValidateStrategyOutput,
    SubmitStrategyInput,
    SubmitStrategyOutput,
    GetRecentResultsInput,
    GetRecentResultsOutput,
)


app = FastAPI(
    title="Hydrogen Mining Tools (MCP-style)",
    description="Structured tools for autonomous/agentic mining in Hydrogen.",
    version="0.2.0",
)

# ============================================================
# Configuration
# ============================================================

API_KEY = os.getenv("HYDROGEN_API_KEY")  # Optional API key


def verify_api_key(x_api_key: str = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


def get_tools() -> HydrogenMiningTools:
    if tools is None:
        raise HTTPException(status_code=503, detail="Tools not initialized")
    return tools


miner = None
tools = None


def init_tools(miner_instance: AgenticMiner):
    global miner, tools
    miner = miner_instance
    tools = HydrogenMiningTools(miner)


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "Hydrogen Mining Tools"}


# ============================================================
# Tools
# ============================================================

@app.get("/tools/list_challenges", response_model=ListChallengesOutput)
async def list_challenges(auth: bool = Depends(verify_api_key)):
    try:
        return await tools.list_challenges()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/get_priors", response_model=GetPriorsOutput)
async def get_priors(
    input_data: GetPriorsInput, auth: bool = Depends(verify_api_key)
):
    try:
        return await tools.get_priors(**input_data.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/propose_strategy", response_model=ProposeStrategyOutput)
async def propose_strategy(
    input_data: ProposeStrategyInput, auth: bool = Depends(verify_api_key)
):
    try:
        return await tools.propose_strategy(**input_data.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/validate_strategy", response_model=ValidateStrategyOutput)
async def validate_strategy(
    input_data: ValidateStrategyInput, auth: bool = Depends(verify_api_key)
):
    try:
        return await tools.validate_strategy(**input_data.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/submit_strategy", response_model=SubmitStrategyOutput)
async def submit_strategy(
    input_data: SubmitStrategyInput, auth: bool = Depends(verify_api_key)
):
    try:
        return await tools.submit_strategy(**input_data.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/get_my_recent_results", response_model=GetRecentResultsOutput)
async def get_my_recent_results(
    input_data: GetRecentResultsInput, auth: bool = Depends(verify_api_key)
):
    try:
        return await tools.get_my_recent_results(**input_data.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
