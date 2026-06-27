"""
AuctionSense — Auction Strategist Agent (LangGraph Orchestrator)
Coordinates all four agents via LangGraph StateGraph and produces the Pre-Auction Briefing.
LLM is initialised lazily — server starts without API keys; falls back to templates.
"""

from __future__ import annotations
import json
import os
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.agents.market_intel    import run_market_intel_agent
from src.agents.reserve_price   import run_reserve_price_agent
from src.agents.participation   import run_participation_agent
from src.agents.lot_config      import run_lot_config_agent


# ── Lazy LLM — initialised on first call, not at import time ─────────
_LLM = None

def _get_llm():
    global _LLM
    if _LLM is not None:
        return _LLM

    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_key      = os.getenv("AZURE_OPENAI_API_KEY", "")
    openai_key     = os.getenv("OPENAI_API_KEY", "")

    if not (azure_key or openai_key):
        return None

    try:
        if azure_endpoint and azure_key:
            from langchain_openai import AzureChatOpenAI
            _LLM = AzureChatOpenAI(
                azure_endpoint   = azure_endpoint,
                azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
                api_version      = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
                api_key          = azure_key,
                temperature      = 0.2,
            )
        else:
            from langchain_openai import ChatOpenAI
            _LLM = ChatOpenAI(model="gpt-4o", api_key=openai_key, temperature=0.2)
        return _LLM
    except Exception:
        return None


def _llm_invoke(system: str, user: str, fallback: str) -> str:
    """Call LLM with fallback string if unavailable."""
    from langchain_core.messages import HumanMessage, SystemMessage
    llm = _get_llm()
    if llm is None:
        return fallback
    try:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return resp.content.strip()
    except Exception:
        return fallback


# ── LangGraph State ───────────────────────────────────────────────────
class AuctionSenseState(TypedDict):
    lot_input:            dict[str, Any]
    operator_sector:      str
    market_context:       dict[str, Any]
    reserve_result:       dict[str, Any]
    participation_result: dict[str, Any]
    lot_config_result:    dict[str, Any]
    outreach_template:    str
    compliance_flags:     list[str]
    pre_auction_briefing: str
    pipeline_errors:      list[str]


# ── Node functions ────────────────────────────────────────────────────
def node_market_intel(state: AuctionSenseState) -> dict:
    try:
        return {"market_context": run_market_intel_agent(state["lot_input"])}
    except Exception as e:
        return {"market_context": {}, "pipeline_errors": [f"MarketIntelAgent: {e}"]}


def node_reserve_price(state: AuctionSenseState) -> dict:
    try:
        return {"reserve_result": run_reserve_price_agent(state["lot_input"], state["market_context"])}
    except Exception as e:
        return {"reserve_result": {}, "pipeline_errors": [f"ReservePriceAgent: {e}"]}


def node_participation(state: AuctionSenseState) -> dict:
    try:
        return {"participation_result": run_participation_agent(state["lot_input"])}
    except Exception as e:
        return {"participation_result": {"participation_risk": "UNKNOWN"},
                "pipeline_errors": [f"ParticipationAgent: {e}"]}


def node_outreach(state: AuctionSenseState) -> dict:
    """Generate outreach template when participation risk is HIGH."""
    lot = state["lot_input"]
    cat = lot.get("lot_category", "N/A").replace("_", " ")
    qty = lot.get("lot_quantity_mt", "N/A")
    grd = lot.get("lot_grade", "N/A")
    reg = lot.get("lot_location_region", "N/A").title()
    day = lot.get("days_to_auction", "N/A")
    fmt = lot.get("auction_format", "English ascending").replace("_", " ").title()

    fallback = (
        f"Dear Valued Buyer,\n\n"
        f"We have an upcoming auction for {qty} MT of Grade {grd} {cat} "
        f"in the {reg} region, scheduled in {day} day(s).\n\n"
        f"Auction format: {fmt}. Based on your buying profile, this lot matches your interest.\n\n"
        f"Please review the lot details and register your intent to bid.\n\n"
        f"Best regards,\nAuction Operations Team"
    )

    result = _llm_invoke(
        system   = "You write concise professional B2B industrial auction outreach messages.",
        user     = (f"Write a 80-100 word outreach message about:\n"
                    f"Lot: {qty} MT Grade {grd} {cat}, {reg} region\n"
                    f"Auction: {day} day(s) from today, {fmt} format\n"
                    f"Be specific, professional, not promotional."),
        fallback = fallback
    )
    return {"outreach_template": result}


def node_lot_config(state: AuctionSenseState) -> dict:
    try:
        return {"lot_config_result": run_lot_config_agent(state["lot_input"], state["market_context"])}
    except Exception as e:
        return {"lot_config_result": {"recommended_config": "as-is", "confidence_level": "LOW",
                                       "rationale_text": f"Config agent unavailable: {e}"},
                "pipeline_errors": [f"LotConfigAgent: {e}"]}


def node_compliance_check(state: AuctionSenseState) -> dict:
    """Deterministic rules — no LLM."""
    flags     = []
    reserve   = state["reserve_result"]
    sector    = state.get("operator_sector", "commercial_scrap")
    predicted = reserve.get("predicted_clearing_price", 0)
    r_lower   = reserve.get("reserve_lower", 0)

    RULES = {
        "government_liquidation": {"floor_pct": 0.50, "approval_threshold": 5_000_000},
        "commercial_scrap":       {"floor_pct": 0.85, "approval_threshold": None},
    }
    rule = RULES.get(sector, RULES["commercial_scrap"])

    if predicted > 0:
        floor = predicted * rule["floor_pct"]
        if r_lower < floor:
            flags.append(
                f"COMPLIANCE: Reserve ({r_lower:,.0f}) below {rule['floor_pct']:.0%} "
                f"floor ({floor:,.0f}) for sector '{sector}'."
            )
    threshold = rule.get("approval_threshold")
    if threshold and predicted > threshold:
        flags.append(
            f"APPROVAL REQUIRED: Predicted price ({predicted:,.0f}) exceeds "
            f"approval threshold ({threshold:,.0f}) for sector '{sector}'."
        )
    return {"compliance_flags": flags}


def node_strategist(state: AuctionSenseState) -> dict:
    """Synthesise all outputs into the Pre-Auction Briefing."""
    lot      = state["lot_input"]
    mkt      = state["market_context"]
    rsv      = state["reserve_result"]
    part     = state["participation_result"]
    cfg      = state["lot_config_result"]
    outreach = state.get("outreach_template", "")
    flags    = state.get("compliance_flags", [])
    errors   = state.get("pipeline_errors", [])

    # ── Template briefing (always generated, no LLM needed) ──────────
    pred   = rsv.get("predicted_clearing_price", 0)
    r_lo   = rsv.get("reserve_lower", 0)
    r_hi   = rsv.get("reserve_upper", 0)
    margin = rsv.get("confidence_margin", 0)
    trend  = mkt.get("commodity_30d_trend", 0)
    risk   = part.get("participation_risk", "N/A")
    count  = part.get("expected_bidder_count", "N/A")

    lines = [
        "=" * 60,
        "PRE-AUCTION BRIEFING — AuctionSense v1.0",
        "=" * 60,
        "",
        f"LOT : {lot.get('lot_category','N/A').replace('_',' ').title()} | "
        f"{lot.get('lot_quantity_mt','N/A')} MT | Grade {lot.get('lot_grade','N/A')} | "
        f"{lot.get('lot_location_region','N/A').title()} region",
        "",
        "─" * 40,
        "1. RESERVE PRICE RECOMMENDATION",
        "─" * 40,
        f"  Predicted Clearing Price : ₹{pred:,.0f} / MT",
        f"  Recommended Reserve Band : ₹{r_lo:,.0f} – ₹{r_hi:,.0f}",
        f"  Confidence Margin        : ±{margin*100:.1f}%",
        "",
        "  Rationale:",
        f"  {rsv.get('rationale_text', 'N/A')}",
        "",
        "─" * 40,
        "2. BUYER PARTICIPATION FORECAST",
        "─" * 40,
        f"  Expected Bidder Count : {count}",
        f"  Participation Risk    : {risk}",
        f"  Forecast Confidence   : {part.get('confidence', 'N/A')}",
    ]

    if risk == "HIGH":
        lines += ["", "  ⚠  THIN-PARTICIPATION ALERT — Targeted outreach recommended",
                  "  Top buyer targets:"]
        for i, b in enumerate(part.get("top_10_buyer_ids", [])[:5], 1):
            lines.append(f"    {i}. {b}")
        if outreach:
            lines += ["", "  Outreach message template:", f"  {outreach}"]

    lines += [
        "",
        "─" * 40,
        "3. LOT CONFIGURATION",
        "─" * 40,
        f"  Recommendation   : {cfg.get('recommended_config', 'N/A')}",
        f"  Confidence       : {cfg.get('confidence_level', 'N/A')}",
        f"  Estimated Uplift : {cfg.get('estimated_realisation_uplift_pct', 'N/A')}%",
        "",
        "  Rationale:",
        f"  {cfg.get('rationale_text', 'N/A')}",
        "",
        "─" * 40,
        "4. MARKET CONTEXT",
        "─" * 40,
        f"  Spot Price   : ₹{mkt.get('commodity_spot_price', 0):,.0f}",
        f"  30-Day Trend : {trend*100:+.1f}%",
        f"  Sentiment    : {mkt.get('market_sentiment_score', 'N/A')} / 1.0",
        f"  Staleness    : {mkt.get('data_staleness_hours', 0)}h",
        "",
        f"  {mkt.get('news_summary', '')}",
    ]

    if flags:
        lines += ["", "─" * 40, "5. COMPLIANCE FLAGS", "─" * 40]
        lines += [f"  ⚠  {f}" for f in flags]

    # ── LLM executive summary (optional enhancement) ──────────────────
    summary_fallback = (
        f"This {lot.get('lot_category','').replace('_',' ')} lot of "
        f"{lot.get('lot_quantity_mt',0):.0f} MT (Grade {lot.get('lot_grade','B')}) "
        f"is predicted to clear at ₹{pred:,.0f}/MT. "
        f"Recommended reserve band: ₹{r_lo:,.0f}–₹{r_hi:,.0f}. "
        f"Buyer participation risk is {risk} with {count} expected bidders. "
        f"Lot configuration: {cfg.get('recommended_config','as-is')} "
        f"(est. uplift {cfg.get('estimated_realisation_uplift_pct',0)}%). "
        f"Commodity 30-day trend: {trend*100:+.1f}%. "
        f"{'No compliance flags.' if not flags else 'COMPLIANCE FLAG — review before proceeding.'}"
    )

    exec_summary = _llm_invoke(
        system   = "You write expert Pre-Auction Briefings for industrial auction managers.",
        user     = (
            f"Write a 150-word executive summary for an auction manager. Be specific with numbers.\n\n"
            f"LOT: {lot.get('lot_category','')} {lot.get('lot_quantity_mt',0):.0f}MT "
            f"Grade {lot.get('lot_grade','B')} {lot.get('lot_location_region','').title()}\n"
            f"RESERVE: predicted=₹{pred:,.0f}, band=₹{r_lo:,.0f}–₹{r_hi:,.0f}\n"
            f"PARTICIPATION: {count} expected bidders, risk={risk}\n"
            f"LOT CONFIG: {cfg.get('recommended_config','as-is')}, "
            f"uplift={cfg.get('estimated_realisation_uplift_pct',0)}%\n"
            f"MARKET: trend={trend*100:+.1f}%, sentiment={mkt.get('market_sentiment_score',0.5)}\n"
            f"COMPLIANCE: {'No flags' if not flags else '; '.join(flags)}"
        ),
        fallback = summary_fallback
    )

    lines += [
        "",
        "─" * 40,
        "6. EXECUTIVE SUMMARY",
        "─" * 40,
        exec_summary,
        "",
        "─" * 40,
        "  Generated by AuctionSense v1.0 | github.com/debray2523/auctionsense",
        "  All recommendations require operator review before action.",
        "=" * 60,
    ]

    if errors:
        lines += ["", "PIPELINE WARNINGS:"] + [f"  {e}" for e in errors]

    return {"pre_auction_briefing": "\n".join(lines)}


# ── Conditional routing ───────────────────────────────────────────────
def route_participation(state: AuctionSenseState) -> str:
    return "outreach" if state["participation_result"].get("participation_risk") == "HIGH" else "lot_config"


# ── Build the LangGraph ───────────────────────────────────────────────
def build_graph():
    g = StateGraph(AuctionSenseState)
    g.add_node("market_intel",     node_market_intel)
    g.add_node("reserve_price",    node_reserve_price)
    g.add_node("participation",    node_participation)
    g.add_node("outreach",         node_outreach)
    g.add_node("lot_config",       node_lot_config)
    g.add_node("compliance_check", node_compliance_check)
    g.add_node("strategist",       node_strategist)

    g.set_entry_point("market_intel")
    g.add_edge("market_intel",    "reserve_price")
    g.add_edge("reserve_price",   "participation")
    g.add_conditional_edges("participation", route_participation,
                             {"outreach": "outreach", "lot_config": "lot_config"})
    g.add_edge("outreach",        "lot_config")
    g.add_edge("lot_config",      "compliance_check")
    g.add_edge("compliance_check","strategist")
    g.add_edge("strategist",      END)
    return g.compile()


# ── Public entry point ────────────────────────────────────────────────
def run_pipeline(lot_input: dict, operator_sector: str = "commercial_scrap") -> dict:
    app = build_graph()
    initial: AuctionSenseState = {
        "lot_input":            lot_input,
        "operator_sector":      operator_sector,
        "market_context":       {},
        "reserve_result":       {},
        "participation_result": {},
        "lot_config_result":    {},
        "outreach_template":    "",
        "compliance_flags":     [],
        "pre_auction_briefing": "",
        "pipeline_errors":      [],
    }
    return app.invoke(initial)
