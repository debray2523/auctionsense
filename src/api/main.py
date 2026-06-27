"""
AuctionSense — FastAPI Application
Custom JSON encoder handles numpy types from XGBoost/SHAP output.
"""

import time
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from src.api.schemas import BriefingRequest, BriefingResponse, HealthResponse
from src.agents.strategist import run_pipeline


def numpy_safe(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {k: numpy_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [numpy_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    # Pydantic enum → string
    if hasattr(obj, "value"):
        return obj.value
    return obj


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("AuctionSense API starting up — warming models...")
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


@app.post("/briefing", tags=["Core"])
async def generate_briefing(request: BriefingRequest):
    """
    Generate a Pre-Auction Briefing for the provided lot.
    Returns JSON with all agent outputs, numpy types safely serialised.
    """
    start = time.time()
    try:
        # Convert enum fields to plain strings before passing to pipeline
        lot_dict = {}
        for k, v in request.lot.model_dump().items():
            lot_dict[k] = v.value if hasattr(v, "value") else v

        result = run_pipeline(lot_dict, operator_sector=request.operator_sector.value
                              if hasattr(request.operator_sector, "value")
                              else str(request.operator_sector))
        elapsed = round(time.time() - start, 1)

        # Safely convert all numpy types before serialisation
        safe_result = numpy_safe(result)

        response_data = {
            "lot_category":          lot_dict.get("lot_category", ""),
            "pre_auction_briefing":  safe_result.get("pre_auction_briefing", ""),
            "reserve_result":        safe_result.get("reserve_result", {}),
            "participation_result":  safe_result.get("participation_result", {}),
            "lot_config_result":     safe_result.get("lot_config_result", {}),
            "compliance_flags":      safe_result.get("compliance_flags", []),
            "pipeline_errors":       safe_result.get("pipeline_errors", []),
            "latency_seconds":       elapsed,
        }
        return JSONResponse(content=response_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@app.get("/", tags=["System"])
async def root():
    return {
        "service": "AuctionSense",
        "version": "1.0.0",
        "paper":   "https://zenodo.org/record/XXXXXXX",
        "github":  "https://github.com/debray2523/auctionsense",
        "docs":    "/docs",
    }
