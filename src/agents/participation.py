"""
AuctionSense — Buyer Participation Agent
Forecasts buyer participation risk and generates outreach recommendations.
"""

import os
import pickle
from pathlib import Path
from typing import Any

MODEL_PATH = Path("models/lgbm_participation.pkl")

# Synthetic buyer profiles for demo when no CRM connected
DEMO_BUYER_PROFILES = [
    {"buyer_id": f"BUYER-{i:03d}",
     "buyer_category_affinity":  round(0.1 + (i % 10) * 0.08, 2),
     "buyer_region_match":       int(i % 3 == 0),
     "buyer_recency_days":       (i * 7) % 120,
     "buyer_win_rate_90d":       round(0.05 + (i % 8) * 0.06, 2),
     "buyer_credit_active":      int(i % 2 == 0),
     "buyer_lot_size_affinity":  ["small","medium","large"][i % 3],
     "buyer_category_recency":   (i * 5) % 60,
    }
    for i in range(1, 31)   # 30 simulated buyers
]


def _generate_outreach_template(lot_input: dict, top_buyers: list,
                                  part_result: dict) -> str:
    """Generate outreach message — LLM-enhanced if key available."""
    lot_category = lot_input.get("lot_category","").replace("_"," ")
    qty          = lot_input.get("lot_quantity_mt", 0)
    grade        = lot_input.get("lot_grade","B")
    region       = lot_input.get("lot_location_region","").title()
    days         = lot_input.get("days_to_auction", 3)
    fmt          = lot_input.get("auction_format","english_ascending").replace("_"," ")

    # Template fallback
    template = (
        f"Dear Valued Buyer,\n\n"
        f"We have an upcoming auction for {qty:.0f} MT of Grade {grade} {lot_category} "
        f"located in the {region} region, scheduled in {days} day(s).\n\n"
        f"Auction format: {fmt.title()}. "
        f"Based on your buying profile and category interest, this lot matches your procurement needs.\n\n"
        f"Please review the lot details on the platform and register your intent to bid at the earliest.\n\n"
        f"Best regards,\nAuction Operations Team"
    )

    azure_key  = os.getenv("AZURE_OPENAI_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not (azure_key or openai_key):
        return template

    try:
        from langchain_openai import AzureChatOpenAI, ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage

        if azure_key and os.getenv("AZURE_OPENAI_ENDPOINT"):
            llm = AzureChatOpenAI(
                azure_endpoint   = os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
                api_version      = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
                temperature      = 0.3,
            )
        else:
            llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

        prompt = (
            f"Write a concise professional outreach message (80-100 words) to a B2B buyer "
            f"about an upcoming industrial auction. Be specific, not promotional.\n\n"
            f"Lot: {qty:.0f} MT Grade {grade} {lot_category}, {region} region\n"
            f"Auction: {days} day(s) from today, {fmt} format\n"
            f"Participation risk: {part_result.get('participation_risk','HIGH')} "
            f"(expected {part_result.get('expected_bidder_count',2):.0f} bidders)\n\n"
            f"Write the outreach message now:"
        )
        resp     = llm.invoke([
            SystemMessage(content="You write professional B2B industrial auction outreach messages."),
            HumanMessage(content=prompt)
        ])
        return resp.content.strip()
    except Exception:
        return template


def run_participation_agent(lot_input: dict[str, Any]) -> dict[str, Any]:
    """
    Participation Agent entry point.

    Args:
        lot_input: Lot features from the API request

    Returns:
        Participation forecast with risk classification and outreach recommendations.
    """
    lot_category = lot_input.get("lot_category", "ferrous_scrap")

    if not MODEL_PATH.exists():
        return {
            "expected_bidder_count": 0,
            "participation_risk":    "UNKNOWN",
            "top_10_buyer_ids":      [],
            "outreach_message_template": "Participation model not trained. Run: python -m src.models.lgbm_participation --train",
            "confidence":            "LOW",
        }

    with open(MODEL_PATH, "rb") as f:
        model_obj = pickle.load(f)

    # Use registered buyer count to scale demo profiles
    registered = lot_input.get("registered_buyer_count", 30)
    n_profiles  = min(registered, len(DEMO_BUYER_PROFILES))
    profiles    = DEMO_BUYER_PROFILES[:n_profiles]

    result = model_obj.predict_lot(lot_category, profiles)

    # Generate outreach template if HIGH risk
    outreach = ""
    if result.get("participation_risk") == "HIGH":
        outreach = _generate_outreach_template(lot_input, profiles, result)

    result["outreach_message_template"] = outreach
    return result
