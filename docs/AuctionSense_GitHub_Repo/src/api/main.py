"""
AuctionSense — FastAPI Application
Production-ready REST API for the AuctionSense pipeline.
"""

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import BriefingRequest, BriefingResponse, HealthResponse
from src.agents.strategist import run_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: warm up models."""
    print("AuctionSense API starting up — warming models...")
    # Models are loaded lazily on first request in production
    # Pre-load here for low-latency first-request behaviour
    yield
    print("AuctionSense API shutting down.")


app = FastAPI(
    title="AuctionSense API",
    description="Agentic AI for Seller-Side Pre-Auction Intelligence",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(status="healthy", version="1.0.0")


@app.post("/briefing", response_model=BriefingResponse, tags=["Core"])
async def generate_briefing(request: BriefingRequest):
    """
    Generate a Pre-Auction Briefing for the provided lot.

    Runs the full four-agent LangGraph pipeline:
    Market Intel → Reserve Price → Buyer Participation → Lot Config → Strategist

    Returns the complete structured briefing with reserve price recommendation,
    participation risk assessment, lot configuration advice, and executive summary.
    """
    start = time.time()
    try:
        lot_dict = request.lot.model_dump()
        result = run_pipeline(lot_dict, operator_sector=request.operator_sector)
        elapsed = round(time.time() - start, 1)

        return BriefingResponse(
            lot_category=request.lot.lot_category,
            pre_auction_briefing=result["pre_auction_briefing"],
            reserve_result=result["reserve_result"],
            participation_result=result["participation_result"],
            lot_config_result=result["lot_config_result"],
            compliance_flags=result["compliance_flags"],
            pipeline_errors=result["pipeline_errors"],
            latency_seconds=elapsed,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@app.get("/", tags=["System"])
async def root():
    return {
        "service": "AuctionSense",
        "version": "1.0.0",
        "paper": "https://zenodo.org/record/XXXXXXX",
        "github": "https://github.com/debray2523/auctionsense",
        "docs": "/docs",
    }
