"""
AuctionSense — Auction Strategist Agent (LangGraph Orchestrator)
Coordinates all four agents via LangGraph StateGraph and produces the Pre-Auction Briefing.
"""

from __future__ import annotations

import json
import os
from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.graph import END, StateGraph

from src.agents.market_intel import run_market_intel_agent
from src.agents.reserve_price import run_reserve_price_agent
from src.agents.participation import run_participation_agent
from src.agents.lot_config import run_lot_config_agent


# ── LLM initialisation ───────────────────────────────────────────────
def _get_llm():
    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        return AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
            temperature=0.2,
        )
    return ChatOpenAI(model="gpt-4o", temperature=0.2)


LLM = _get_llm()


# ── LangGraph State ───────────────────────────────────────────────────
class AuctionSenseState(TypedDict):
    # Input
    lot_input: dict[str, Any]
    operator_sector: str

    # Agent outputs (populated sequentially)
    market_context: dict[str, Any]
    reserve_result: dict[str, Any]
    participation_result: dict[str, Any]
    lot_config_result: dict[str, Any]

    # Orchestrator
    outreach_template: str
    compliance_flags: list[str]
    pre_auction_briefing: str
    pipeline_errors: list[str]


# ── Node functions ────────────────────────────────────────────────────
def node_market_intel(state: AuctionSenseState) -> dict:
    try:
        ctx = run_market_intel_agent(state["lot_input"])
        return {"market_context": ctx}
    except Exception as e:
        return {"market_context": {}, "pipeline_errors": [f"MarketIntelAgent: {e}"]}


def node_reserve_price(state: AuctionSenseState) -> dict:
    try:
        result = run_reserve_price_agent(state["lot_input"], state["market_context"])
        return {"reserve_result": result}
    except Exception as e:
        return {"reserve_result": {}, "pipeline_errors": [f"ReservePriceAgent: {e}"]}


def node_participation(state: AuctionSenseState) -> dict:
    try:
        result = run_participation_agent(state["lot_input"])
        return {"participation_result": result}
    except Exception as e:
        return {"participation_result": {"participation_risk": "UNKNOWN"},
                "pipeline_errors": [f"ParticipationAgent: {e}"]}


def node_outreach(state: AuctionSenseState) -> dict:
    """Generate outreach template when participation risk is HIGH."""
    part = state["participation_result"]
    lot  = state["lot_input"]
    top_buyers = part.get("top_10_buyer_ids", [])[:5]

    prompt = f"""You are an auction operations specialist.
Write a concise, professional outreach message (max 120 words) to inform high-affinity buyers
about an upcoming lot. Be specific. Do not include placeholders like [Name].

Lot details:
- Category: {lot.get('lot_category', 'N/A')}
- Quantity: {lot.get('lot_quantity_mt', 'N/A')} MT
- Grade: {lot.get('lot_grade', 'N/A')}
- Location: {lot.get('lot_location_region', 'N/A')} region
- Auction date: {lot.get('days_to_auction', 'N/A')} days from today
- Format: {lot.get('auction_format', 'English ascending')}

Write the outreach message now:"""

    resp = LLM.invoke([SystemMessage(content="You write targeted auction outreach messages."),
                       HumanMessage(content=prompt)])
    return {"outreach_template": resp.content.strip()}


def node_lot_config(state: AuctionSenseState) -> dict:
    try:
        result = run_lot_config_agent(state["lot_input"], state["market_context"])
        return {"lot_config_result": result}
    except Exception as e:
        return {"lot_config_result": {"recommended_config": "as-is", "confidence_level": "LOW",
                                       "rationale_text": f"Config agent unavailable: {e}"},
                "pipeline_errors": [f"LotConfigAgent: {e}"]}


def node_compliance_check(state: AuctionSenseState) -> dict:
    """Deterministic rules engine — no LLM."""
    flags = []
    reserve = state["reserve_result"]
    lot = state["lot_input"]
    sector = state.get("operator_sector", "commercial_scrap")

    predicted = reserve.get("predicted_clearing_price", 0)
    r_lower   = reserve.get("reserve_lower", 0)

    # Configurable rules (in production, load from configs/compliance_rules.yaml)
    RULES = {
        "government_liquidation": {"floor_pct": 0.50, "approval_threshold": 5_000_000},
        "commercial_scrap":       {"floor_pct": 0.85, "approval_threshold": None},
    }
    rule = RULES.get(sector, RULES["commercial_scrap"])

    if predicted > 0:
        floor = predicted * rule["floor_pct"]
        if r_lower < floor:
            flags.append(
                f"COMPLIANCE: Recommended reserve ({r_lower:,.0f}) is below the "
                f"{rule['floor_pct']:.0%} floor ({floor:,.0f}) for sector '{sector}'. "
                f"Adjust reserve before proceeding."
            )
    threshold = rule.get("approval_threshold")
    if threshold and predicted > threshold:
        flags.append(
            f"APPROVAL REQUIRED: Predicted clearing price ({predicted:,.0f}) exceeds "
            f"the {threshold:,.0f} approval threshold for sector '{sector}'. "
            f"Obtain committee approval before setting reserve."
        )
    return {"compliance_flags": flags}


def node_strategist(state: AuctionSenseState) -> dict:
    """Synthesise all agent outputs into the Pre-Auction Briefing."""
    lot    = state["lot_input"]
    mkt    = state["market_context"]
    rsv    = state["reserve_result"]
    part   = state["participation_result"]
    cfg    = state["lot_config_result"]
    outreach = state.get("outreach_template", "")
    flags  = state.get("compliance_flags", [])
    errors = state.get("pipeline_errors", [])

    summary_prompt = f"""You are an expert auction operations analyst writing a Pre-Auction Briefing.
Write a clear, professional 200-word executive summary for an auction manager.
Use plain language. Be specific. No hedging. Include numbers.

Data:
LOT: {json.dumps(lot, indent=2)}
MARKET: commodity_spot={mkt.get('commodity_spot_price','N/A')}, trend={mkt.get('commodity_30d_trend','N/A')}, sentiment={mkt.get('market_sentiment_score','N/A')}
RESERVE: predicted={rsv.get('predicted_clearing_price','N/A')}, band=[{rsv.get('reserve_lower','N/A')} – {rsv.get('reserve_upper','N/A')}]
PARTICIPATION: expected_bidders={part.get('expected_bidder_count','N/A')}, risk={part.get('participation_risk','N/A')}
LOT CONFIG: recommendation={cfg.get('recommended_config','N/A')}, confidence={cfg.get('confidence_level','N/A')}, est_uplift={cfg.get('estimated_realisation_uplift_pct','N/A')}%
COMPLIANCE: {'; '.join(flags) if flags else 'No flags'}

Write the executive summary:"""

    resp = LLM.invoke([SystemMessage(content="You write expert Pre-Auction Briefings."),
                       HumanMessage(content=summary_prompt)])

    # Assemble full structured briefing
    briefing_sections = [
        "=" * 60,
        "PRE-AUCTION BRIEFING — AuctionSense",
        "=" * 60,
        "",
        f"LOT: {lot.get('lot_category','N/A').replace('_',' ').title()} | "
        f"{lot.get('lot_quantity_mt','N/A')} MT | Grade {lot.get('lot_grade','N/A')} | "
        f"{lot.get('lot_location_region','N/A').title()} region",
        "",
        "─" * 40,
        "1. RESERVE PRICE RECOMMENDATION",
        "─" * 40,
        f"  Predicted Clearing Price : {rsv.get('predicted_clearing_price', 'N/A'):,.0f}",
        f"  Recommended Reserve Band : {rsv.get('reserve_lower', 'N/A'):,.0f} – {rsv.get('reserve_upper', 'N/A'):,.0f}",
        f"  Confidence Margin        : ±{rsv.get('confidence_margin', 0)*100:.1f}%",
        "",
        "  Rationale:",
        f"  {rsv.get('rationale_text', 'N/A')}",
        "",
        "─" * 40,
        "2. BUYER PARTICIPATION FORECAST",
        "─" * 40,
        f"  Expected Bidder Count : {part.get('expected_bidder_count', 'N/A')}",
        f"  Participation Risk    : {part.get('participation_risk', 'N/A')}",
        f"  Forecast Confidence   : {part.get('confidence', 'N/A')}",
    ]

    if part.get("participation_risk") == "HIGH":
        briefing_sections += [
            "",
            "  ⚠ THIN-PARTICIPATION ALERT",
            "  Top buyer targets for outreach:",
        ]
        for i, bid in enumerate(part.get("top_10_buyer_ids", [])[:5], 1):
            briefing_sections.append(f"    {i}. {bid}")
        if outreach:
            briefing_sections += ["", "  Suggested outreach message:", f"  {outreach}"]

    briefing_sections += [
        "",
        "─" * 40,
        "3. LOT CONFIGURATION",
        "─" * 40,
        f"  Recommendation    : {cfg.get('recommended_config', 'N/A')}",
        f"  Confidence        : {cfg.get('confidence_level', 'N/A')}",
        f"  Estimated Uplift  : {cfg.get('estimated_realisation_uplift_pct', 'N/A')}%",
        "",
        "  Rationale:",
        f"  {cfg.get('rationale_text', 'N/A')}",
        "",
        "─" * 40,
        "4. MARKET CONTEXT",
        "─" * 40,
        f"  Commodity Spot Price  : {mkt.get('commodity_spot_price', 'N/A'):,.0f}",
        f"  30-Day Trend          : {mkt.get('commodity_30d_trend', 0)*100:+.1f}%",
        f"  Market Sentiment      : {mkt.get('market_sentiment_score', 'N/A')} / 1.0",
        f"  Data Staleness        : {mkt.get('data_staleness_hours', 0)}h",
        "",
        f"  Market Summary: {mkt.get('news_summary', 'N/A')}",
    ]

    if flags:
        briefing_sections += [
            "",
            "─" * 40,
            "5. COMPLIANCE FLAGS",
            "─" * 40,
        ]
        for flag in flags:
            briefing_sections.append(f"  ⚠ {flag}")

    briefing_sections += [
        "",
        "─" * 40,
        "6. EXECUTIVE SUMMARY",
        "─" * 40,
        resp.content.strip(),
        "",
        "─" * 40,
        "  Generated by AuctionSense | github.com/debray2523/auctionsense",
        "  All recommendations require operator review before action.",
        "=" * 60,
    ]

    if errors:
        briefing_sections += ["", "PIPELINE WARNINGS:", *[f"  {e}" for e in errors]]

    return {"pre_auction_briefing": "\n".join(briefing_sections)}


# ── Conditional edges ────────────────────────────────────────────────
def route_participation(state: AuctionSenseState) -> str:
    risk = state["participation_result"].get("participation_risk", "LOW")
    return "outreach" if risk == "HIGH" else "lot_config"


# ── Build the graph ──────────────────────────────────────────────────
def build_graph() -> StateGraph:
    graph = StateGraph(AuctionSenseState)

    graph.add_node("market_intel",     node_market_intel)
    graph.add_node("reserve_price",    node_reserve_price)
    graph.add_node("participation",    node_participation)
    graph.add_node("outreach",         node_outreach)
    graph.add_node("lot_config",       node_lot_config)
    graph.add_node("compliance_check", node_compliance_check)
    graph.add_node("strategist",       node_strategist)

    graph.set_entry_point("market_intel")
    graph.add_edge("market_intel",   "reserve_price")
    graph.add_edge("reserve_price",  "participation")
    graph.add_conditional_edges("participation", route_participation,
                                 {"outreach": "outreach", "lot_config": "lot_config"})
    graph.add_edge("outreach",       "lot_config")
    graph.add_edge("lot_config",     "compliance_check")
    graph.add_edge("compliance_check","strategist")
    graph.add_edge("strategist",     END)

    return graph.compile()


# ── Public entry point ───────────────────────────────────────────────
def run_pipeline(lot_input: dict, operator_sector: str = "commercial_scrap") -> dict:
    """
    Run the full AuctionSense pipeline for a given lot.

    Args:
        lot_input: Dict of lot features (see schemas.py for full schema)
        operator_sector: Sector for compliance rule selection

    Returns:
        Final LangGraph state dict containing pre_auction_briefing and all agent outputs
    """
    app = build_graph()
    initial_state: AuctionSenseState = {
        "lot_input": lot_input,
        "operator_sector": operator_sector,
        "market_context": {},
        "reserve_result": {},
        "participation_result": {},
        "lot_config_result": {},
        "outreach_template": "",
        "compliance_flags": [],
        "pre_auction_briefing": "",
        "pipeline_errors": [],
    }
    result = app.invoke(initial_state)
    return result
