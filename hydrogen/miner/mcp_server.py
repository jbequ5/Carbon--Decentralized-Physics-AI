"""MCP-style Server with Basic Session Support.

This version adds simple session management so agents can maintain
context across multiple tool calls.
"""

import os
import uuid
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel

from hydrogen.miner.agent import AgenticMiner
from hydrogen.miner.client import MockHydrogenClient


app = FastAPI(
    title="Hydrogen Mining MCP Server (with Sessions)",
    description="MCP-style tools with basic session support for agentic mining.",
    version="0.4.0",
)

# ============================================================
# In-Memory Session Store (for demo purposes)
# ============================================================

sessions: Dict[str, Dict[str, Any]] = {}


def create_session() -> str:
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "miner": AgenticMiner(MockHydrogenClient()),
        "context": {},
    }
    return session_id


def get_session_miner(session_id: str) -> AgenticMiner:
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]["miner"]


def verify_api_key(x_api_key: str = Header(None)):
    api_key = os.getenv("HYDROGEN_API_KEY")
    if api_key and x_api_key != api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


# ============================================================
# Session Management
# ============================================================

class CreateSessionResponse(BaseModel):
    session_id: str


@app.post("/sessions", response_model=CreateSessionResponse)
async def create_new_session(auth: bool = Depends(verify_api_key)):
    session_id = create_session()
    return CreateSessionResponse(session_id=session_id)


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
async def health():
    return {"status": "ok", "active_sessions": len(sessions)}


# ============================================================
# Tool Endpoints (with optional session support)
# ============================================================

@app.get("/tools/list_challenges")
async def list_challenges(
    session_id: str = None, auth: bool = Depends(verify_api_key)
):
    miner = get_session_miner(session_id) if session_id else AgenticMiner(MockHydrogenClient())
    return await miner.get_challenges()


@app.post("/tools/get_priors")
async def get_priors(payload: dict, session_id: str = None, auth: bool = Depends(verify_api_key)):
    challenge_id = payload.get("challenge_id")
    if not challenge_id:
        raise HTTPException(status_code=400, detail="challenge_id is required")

    miner = get_session_miner(session_id) if session_id else AgenticMiner(MockHydrogenClient())
    return await miner.get_priors(challenge_id)


@app.post("/tools/propose_strategy")
async def propose_strategy(payload: dict, session_id: str = None, auth: bool = Depends(verify_api_key)):
    challenge_id = payload.get("challenge_id")
    base = payload.get("base_strategy")

    miner = get_session_miner(session_id) if session_id else AgenticMiner(MockHydrogenClient())
    return await miner.propose_strategy(challenge_id, base_strategy=base)


@app.post("/tools/validate_strategy")
async def validate_strategy(payload: dict, session_id: str = None, auth: bool = Depends(verify_api_key)):
    strategy = payload.get("strategy")
    challenge_id = payload.get("challenge_id")
    quick = payload.get("quick", True)

    miner = get_session_miner(session_id) if session_id else AgenticMiner(MockHydrogenClient())
    return await miner.validate_locally(strategy, challenge_id, quick=quick)


@app.post("/tools/submit_strategy")
async def submit_strategy(payload: dict, session_id: str = None, auth: bool = Depends(verify_api_key)):
    strategy = payload.get("strategy")
    challenge_id = payload.get("challenge_id")

    miner = get_session_miner(session_id) if session_id else AgenticMiner(MockHydrogenClient())
    return await miner.submit(strategy, challenge_id)


@app.post("/tools/get_recent_results")
async def get_recent_results(payload: dict, session_id: str = None, auth: bool = Depends(verify_api_key)):
    limit = payload.get("limit", 10)

    miner = get_session_miner(session_id) if session_id else AgenticMiner(MockHydrogenClient())
    return await miner.get_recent_performance(limit=limit)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
