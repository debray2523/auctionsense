"""
AuctionSense — Lot Configuration Agent
Recommends optimal lot split/bundle strategy using few-shot LLM reasoning.
Falls back to rule-based recommendations when LLM unavailable.
"""

import os
from typing import Any


# Historical analogues (embedded — no pgvector required for demo)
# In production these come from pgvector similarity search
HISTORICAL_ANALOGUES = [
    {
        "lot_category": "ferrous_scrap", "quantity_mt": 500, "grade": "B",
        "region": "east", "config": "split_5",
        "outcome": "Split into 5 lots of 100MT each. 12 bidders across lots. +18% vs single lot estimate.",
        "lesson": "Large ferrous lots attract more bidders when split — specialist buyers prefer manageable quantities."
    },
    {
        "lot_category": "ferrous_scrap", "quantity_mt": 80, "grade": "A",
        "region": "west", "config": "as-is",
        "outcome": "Sold as single lot. 7 bidders. Premium Grade A attracted bulk buyers willing to pay premium.",
        "lesson": "High-grade small lots sell well as-is — grade premium outweighs size discount."
    },
    {
        "lot_category": "nonferrous_scrap", "quantity_mt": 200, "grade": "B",
        "region": "north", "config": "split_2",
        "outcome": "Split copper and aluminium separately. 9 bidders total. +22% vs bundle estimate.",
        "lesson": "Mixed non-ferrous lots should be split by metal type — specialist buyers for each metal."
    },
    {
        "lot_category": "idle_industrial_asset", "quantity_mt": 50, "grade": "C",
        "region": "south", "config": "bundle",
        "outcome": "Bundled with complementary machinery. Single large buyer paid premium for complete set.",
        "lesson": "Complementary idle assets bundle well — buyers prefer turnkey operational packages."
    },
    {
        "lot_category": "coal", "quantity_mt": 1000, "grade": "A",
        "region": "east", "config": "split_4",
        "outcome": "Split into 4 rakes of 250MT. 8 bidders per rake. Statutory compliance easier per rake.",
        "lesson": "Large coal lots split at rake-size units (250MT) for compliance and logistics."
    },
    {
        "lot_category": "agricultural", "quantity_mt": 300, "grade": "A",
        "region": "west", "config": "as-is",
        "outcome": "Single lot. 15 bidders. Peak season demand meant large buyers competed aggressively.",
        "lesson": "Agricultural commodities in season sell better as single large lots — demand depth is high."
    },
]


def _find_analogues(lot_input: dict, n: int = 3) -> list[dict]:
    """
    Simple similarity search over hardcoded analogues.
    In production: pgvector cosine similarity over embeddings.
    """
    category = lot_input.get("lot_category", "")
    grade    = lot_input.get("lot_grade", "B")
    qty      = lot_input.get("lot_quantity_mt", 100)

    scored = []
    for a in HISTORICAL_ANALOGUES:
        score = 0
        if a["lot_category"] == category:  score += 3
        if a["grade"] == grade:             score += 2
        if abs(a["quantity_mt"] - qty) < 200: score += 1
        scored.append((score, a))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [a for _, a in scored[:n]]


def _rule_based_recommendation(lot_input: dict,
                                 market_context: dict) -> dict[str, Any]:
    """Deterministic fallback when LLM unavailable."""
    qty      = lot_input.get("lot_quantity_mt", 100)
    category = lot_input.get("lot_category", "")
    grade    = lot_input.get("lot_grade", "B")
    buyers   = lot_input.get("registered_buyer_count", 20)
    trend    = market_context.get("commodity_30d_trend", 0)

    if qty > 400 and buyers > 30 and category in ["ferrous_scrap","nonferrous_scrap","coal"]:
        n_splits  = max(2, min(5, int(qty / 100)))
        config    = f"split_{n_splits}"
        rationale = (f"Lot size of {qty:.0f} MT is large for this category. "
                     f"With {buyers} registered buyers, splitting into {n_splits} lots "
                     f"of ~{qty/n_splits:.0f} MT each will attract more specialist bidders "
                     f"and increase competitive tension. Estimated uplift: +12–18%.")
        uplift    = 15
        confidence= "MEDIUM"
    elif qty < 50 or grade == "A":
        config    = "as-is"
        rationale = (f"{'High-grade (A)' if grade=='A' else 'Small'} lot of {qty:.0f} MT. "
                     f"Premium buyers prefer to acquire in single lots for quality assurance. "
                     f"Splitting would reduce premium and fragment the buyer pool.")
        uplift    = 0
        confidence= "HIGH"
    elif category == "idle_industrial_asset":
        config    = "bundle"
        rationale = (f"Idle industrial assets have higher value as complementary packages. "
                     f"Bundling with similar assets on the platform attracts operations buyers "
                     f"seeking complete solutions rather than individual components.")
        uplift    = 8
        confidence= "MEDIUM"
    else:
        config    = "as-is"
        rationale = (f"Lot size {qty:.0f} MT with {buyers} registered buyers is within "
                     f"optimal range for single-lot auction. No configuration change recommended.")
        uplift    = 0
        confidence= "HIGH"

    return {
        "recommended_config":                config,
        "estimated_realisation_uplift_pct":  uplift,
        "confidence_level":                  confidence,
        "rationale_text":                    rationale,
        "analogues_used":                    [],
        "reasoning_method":                  "rule_based",
    }


def run_lot_config_agent(lot_input: dict[str, Any],
                          market_context: dict[str, Any]) -> dict[str, Any]:
    """
    Lot Configuration Agent entry point.

    Args:
        lot_input:      Lot features from the API request
        market_context: Output from Market Intel Agent

    Returns:
        Configuration recommendation with rationale.
    """
    analogues = _find_analogues(lot_input, n=3)

    # Try LLM reasoning if key available
    azure_key  = os.getenv("AZURE_OPENAI_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    if not (azure_key or openai_key):
        result = _rule_based_recommendation(lot_input, market_context)
        result["analogues_used"] = [a["outcome"] for a in analogues]
        return result

    try:
        from langchain_openai import AzureChatOpenAI, ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
        import json

        if azure_key and os.getenv("AZURE_OPENAI_ENDPOINT"):
            llm = AzureChatOpenAI(
                azure_endpoint   = os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
                api_version      = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
                temperature      = 0.2,
            )
        else:
            llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

        analogue_str = "\n".join([
            f"Case {i+1}: {a['lot_category']}, {a['quantity_mt']}MT, Grade {a['grade']}, "
            f"Region {a['region']} → {a['config']} → {a['outcome']}\n  Lesson: {a['lesson']}"
            for i, a in enumerate(analogues)
        ])

        prompt = (
            f"You are an expert industrial auction strategist advising on lot configuration.\n\n"
            f"LOT TO CONFIGURE:\n"
            f"  Category:   {lot_input.get('lot_category','')}\n"
            f"  Quantity:   {lot_input.get('lot_quantity_mt',0):.0f} MT\n"
            f"  Grade:      {lot_input.get('lot_grade','B')}\n"
            f"  Region:     {lot_input.get('lot_location_region','')}\n"
            f"  Registered buyers: {lot_input.get('registered_buyer_count',0)}\n"
            f"  Days to auction:   {lot_input.get('days_to_auction',3)}\n\n"
            f"MARKET CONTEXT:\n"
            f"  Commodity trend (30d): {market_context.get('commodity_30d_trend',0)*100:+.1f}%\n"
            f"  Market sentiment: {market_context.get('market_sentiment_score',0.5)}/1.0\n\n"
            f"MOST RELEVANT HISTORICAL ANALOGUES:\n{analogue_str}\n\n"
            f"Based on the analogues, provide your recommendation as JSON only:\n"
            f'{{"recommended_config": "as-is|split_N|bundle", '
            f'"estimated_realisation_uplift_pct": <number>, '
            f'"confidence_level": "HIGH|MEDIUM|LOW", '
            f'"rationale_text": "<3 sentences citing the most relevant analogue>"}}'
        )

        resp = llm.invoke([
            SystemMessage(content="You are an expert industrial auction strategist. Respond only with JSON."),
            HumanMessage(content=prompt)
        ])

        content = resp.content.strip()
        # Strip markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()

        result              = json.loads(content)
        result["analogues_used"]   = [a["outcome"] for a in analogues]
        result["reasoning_method"] = "llm_few_shot"
        return result

    except Exception as e:
        result = _rule_based_recommendation(lot_input, market_context)
        result["analogues_used"]   = [a["outcome"] for a in analogues]
        result["reasoning_method"] = f"rule_based_fallback (llm_error: {str(e)[:80]})"
        return result
