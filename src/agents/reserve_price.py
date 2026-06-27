"""
AuctionSense — Reserve Price Agent (fixed)
Injects default values for features not in API request.
Converts Pydantic enums to strings before model inference.
"""

import os
import pickle
from pathlib import Path
from typing import Any

MODEL_PATH = Path("models/xgboost_reserve.pkl")

# Defaults for features not sent by the API — calibrated to training data means
FEATURE_DEFAULTS = {
    "historical_avg_premium":        0.04,
    "historical_participation_rate":  8.2,
    "secondary_commodity_idx":        0.0,   # will be overridden by market_context
    "fx_rate":                       83.5,   # will be overridden by market_context
}


def _to_str(val: Any) -> str:
    """Convert Pydantic enum or any object to plain string."""
    if hasattr(val, "value"):
        return val.value
    return str(val)


def _build_rationale(predicted_price, reserve_lower, reserve_upper,
                     shap_top5, market_context, lot_input) -> str:
    trend_pct = market_context.get("commodity_30d_trend", 0) * 100
    grade     = _to_str(lot_input.get("lot_grade", "B"))
    category  = _to_str(lot_input.get("lot_category", "")).replace("_", " ")

    if shap_top5:
        top_feature = shap_top5[0]["feature"].replace("_", " ")
        direction   = "upward" if shap_top5[0]["shap_value"] > 0 else "downward"
    else:
        top_feature, direction = "commodity spot price", "upward"

    rationale = (
        f"The predicted clearing price of ₹{predicted_price:,.0f}/MT is primarily driven by "
        f"{top_feature} ({direction} pressure). "
        f"The 30-day commodity trend of {trend_pct:+.1f}% and Grade {grade} {category} material "
        f"are the key price factors. "
        f"The recommended reserve band of ₹{reserve_lower:,.0f}–₹{reserve_upper:,.0f} "
        f"provides a 3–7% buffer below the predicted clearing price to attract competitive bidding."
    )

    # Try LLM enhancement
    azure_key  = os.getenv("AZURE_OPENAI_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not (azure_key or openai_key):
        return rationale

    try:
        from langchain_openai import AzureChatOpenAI, ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage

        if azure_key and os.getenv("AZURE_OPENAI_ENDPOINT"):
            llm = AzureChatOpenAI(
                azure_endpoint   = os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
                api_version      = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
                api_key          = azure_key, temperature=0.2)
        else:
            llm = ChatOpenAI(model="gpt-4o", api_key=openai_key, temperature=0.2)

        shap_str = ", ".join(
            f"{s['feature']} ({s['shap_value']:+.1f})" for s in shap_top5[:3]
        ) if shap_top5 else "N/A"

        resp = llm.invoke([
            SystemMessage(content="You explain AI auction pricing recommendations clearly."),
            HumanMessage(content=(
                f"Write a 3-sentence plain-English explanation for a non-technical auction manager.\n"
                f"Predicted clearing price: ₹{predicted_price:,.0f}/MT\n"
                f"Reserve band: ₹{reserve_lower:,.0f}–₹{reserve_upper:,.0f}\n"
                f"Lot: {category}, Grade {grade}, Qty {lot_input.get('lot_quantity_mt',0):.0f} MT\n"
                f"Market trend (30d): {trend_pct:+.1f}%\n"
                f"Top SHAP drivers: {shap_str}\n\n"
                f"Sentence 1: Predicted price and main market driver.\n"
                f"Sentence 2: Most important lot-specific factor.\n"
                f"Sentence 3: Reserve band and why it is set there."
            ))
        ])
        return resp.content.strip()
    except Exception:
        return rationale


def run_reserve_price_agent(lot_input: dict[str, Any],
                             market_context: dict[str, Any]) -> dict[str, Any]:
    if not MODEL_PATH.exists():
        return {
            "predicted_clearing_price": 0.0,
            "reserve_lower":   0.0,
            "reserve_upper":   0.0,
            "confidence_margin": 0.10,
            "shap_top5":       [],
            "rationale_text":  "Model not trained. Run: python -m src.models.xgboost_reserve --train",
            "model_version":   "not_loaded",
        }

    with open(MODEL_PATH, "rb") as f:
        model_obj = pickle.load(f)

    # Build enriched feature dict:
    # 1. Start with defaults for missing features
    # 2. Overlay lot_input (converting enums to strings)
    # 3. Overlay market_context values
    enriched = dict(FEATURE_DEFAULTS)

    for k, v in lot_input.items():
        enriched[k] = _to_str(v) if hasattr(v, "value") else v

    # Market context overrides
    enriched["commodity_spot_price"]    = market_context.get("commodity_spot_price", 28000.0)
    enriched["commodity_30d_trend"]     = market_context.get("commodity_30d_trend", 0.02)
    enriched["secondary_commodity_idx"] = market_context.get("secondary_commodity_idx",
                                           enriched.get("commodity_spot_price", 28000.0) * 0.95)
    enriched["fx_rate"]                 = market_context.get("fx_rate", 83.5)
    enriched["market_sentiment_score"]  = market_context.get("market_sentiment_score", 0.55)

    # Ensure season_quarter has a default
    if "season_quarter" not in enriched or not enriched["season_quarter"]:
        enriched["season_quarter"] = "Q1"

    result = model_obj.predict(enriched, market_context)

    result["rationale_text"] = _build_rationale(
        predicted_price = result["predicted_clearing_price"],
        reserve_lower   = result["reserve_lower"],
        reserve_upper   = result["reserve_upper"],
        shap_top5       = result.get("shap_top5", []),
        market_context  = market_context,
        lot_input       = enriched,
    )
    return result
