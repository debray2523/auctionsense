"""
AuctionSense — Streamlit Demo Dashboard
Interactive Pre-Auction Briefing generator.

Run: streamlit run src/ui/streamlit_app.py
"""

import json
import os
import time

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AuctionSense",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Header ────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='color:#1B3A6B; margin-bottom:4px;'>🏭 AuctionSense</h1>
<p style='color:#6B7280; font-size:16px; margin-top:0;'>
Agentic AI for Seller-Side Pre-Auction Intelligence
</p>
""", unsafe_allow_html=True)
st.divider()

# ── Sidebar — Lot Input ───────────────────────────────────────────────
with st.sidebar:
    st.header("Lot Configuration")

    lot_category = st.selectbox("Lot Category", [
        "ferrous_scrap", "nonferrous_scrap", "idle_industrial_asset",
        "coal", "agricultural"
    ], format_func=lambda x: x.replace("_", " ").title())

    lot_quantity = st.number_input("Quantity (MT)", min_value=1.0, max_value=5000.0,
                                    value=150.0, step=10.0)

    lot_grade = st.selectbox("Grade", ["A", "B", "C", "D"],
                              index=1, help="A=Best quality, D=Lowest quality")

    lot_region = st.selectbox("Location Region", ["east", "west", "north", "south"],
                               format_func=str.title)

    days_to_auction = st.slider("Days to Auction", min_value=1, max_value=14, value=3)

    auction_format = st.selectbox("Auction Format", [
        "english_ascending", "sealed_bid", "dutch_descending"
    ], format_func=lambda x: x.replace("_", " ").title())

    registered_buyers = st.number_input("Registered Buyers in Category",
                                         min_value=0, max_value=500, value=47)

    quarter = st.selectbox("Season Quarter", ["Q1", "Q2", "Q3", "Q4"])

    st.divider()
    st.subheader("Operator Settings")
    operator_sector = st.selectbox("Operator Sector", [
        "commercial_scrap", "government_liquidation", "agricultural_commodity",
        "coal_resources", "real_estate"
    ], format_func=lambda x: x.replace("_", " ").title())

    run_btn = st.button("🚀 Generate Pre-Auction Briefing", type="primary", use_container_width=True)

# ── Main — Results ────────────────────────────────────────────────────
if run_btn:
    payload = {
        "lot": {
            "lot_category":           lot_category,
            "lot_quantity_mt":        lot_quantity,
            "lot_grade":              lot_grade,
            "lot_location_region":    lot_region,
            "days_to_auction":        days_to_auction,
            "auction_format":         auction_format,
            "registered_buyer_count": registered_buyers,
            "season_quarter":         quarter,
        },
        "operator_sector": operator_sector,
    }

    with st.spinner("Running AuctionSense pipeline (Market Intel → Reserve Price → Participation → Lot Config → Briefing)..."):
        t0 = time.time()
        try:
            resp = requests.post(f"{API_BASE}/briefing", json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            elapsed = round(time.time() - t0, 1)
        except Exception as e:
            st.error(f"API error: {e}")
            st.stop()

    # ── KPI Row ──────────────────────────────────────────────────────
    rsv  = data.get("reserve_result", {})
    part = data.get("participation_result", {})
    cfg  = data.get("lot_config_result", {})

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        pred = rsv.get("predicted_clearing_price", "—")
        st.metric("Predicted Clearing Price",
                  f"₹{pred:,.0f}" if isinstance(pred, (int,float)) else pred)
    with col2:
        lo = rsv.get("reserve_lower", "—")
        hi = rsv.get("reserve_upper", "—")
        band = f"₹{lo:,.0f} – ₹{hi:,.0f}" if isinstance(lo, (int,float)) else "—"
        st.metric("Recommended Reserve Band", band)
    with col3:
        risk  = part.get("participation_risk", "—")
        count = part.get("expected_bidder_count", "—")
        delta_color = "inverse" if risk == "HIGH" else "normal"
        st.metric("Participation Risk", risk,
                  delta=f"{count} expected bidders" if isinstance(count, (int,float)) else None,
                  delta_color=delta_color)
    with col4:
        st.metric("Pipeline Latency", f"{elapsed}s")

    st.divider()

    # ── Tabs ─────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Full Briefing", "💰 Reserve Price", "👥 Participation", "⚙️ Lot Config", "🔍 Debug"
    ])

    with tab1:
        st.text(data.get("pre_auction_briefing", "No briefing generated"))
        flags = data.get("compliance_flags", [])
        if flags:
            st.warning("**Compliance Flags**")
            for f in flags:
                st.warning(f"⚠ {f}")

    with tab2:
        st.subheader("Reserve Price Agent Output")
        st.json(rsv)
        if rsv.get("shap_top5"):
            st.subheader("Top 5 SHAP Feature Contributions")
            import pandas as pd, plotly.express as px
            shap_df = pd.DataFrame(rsv["shap_top5"])
            fig = px.bar(shap_df, x="shap_value", y="feature", orientation="h",
                         color="shap_value", color_continuous_scale="RdYlGn",
                         title="SHAP Feature Contributions to Reserve Price Prediction")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Buyer Participation Agent Output")
        st.json(part)
        risk = part.get("participation_risk", "LOW")
        if risk == "HIGH":
            st.error("⚠ THIN-PARTICIPATION RISK DETECTED")
            st.info("The system has generated targeted outreach recommendations. "
                    "See the Full Briefing tab for the outreach message template.")
        elif risk == "MEDIUM":
            st.warning("⚡ Moderate participation risk. Monitor closely.")
        else:
            st.success("✅ Healthy participation expected.")

    with tab4:
        st.subheader("Lot Configuration Agent Output")
        st.json(cfg)
        rec = cfg.get("recommended_config", "as-is")
        uplift = cfg.get("estimated_realisation_uplift_pct", 0)
        confidence = cfg.get("confidence_level", "MEDIUM")
        conf_colors = {"HIGH": "✅", "MEDIUM": "⚡", "LOW": "⚠"}
        st.info(f"{conf_colors.get(confidence,'⚡')} **Recommendation:** {rec}  |  "
                f"**Estimated uplift:** +{uplift}%  |  **Confidence:** {confidence}")
        rationale = cfg.get("rationale_text", "")
        if rationale:
            st.markdown(f"**Rationale:** {rationale}")

    with tab5:
        st.subheader("Pipeline Debug")
        errors = data.get("pipeline_errors", [])
        if errors:
            for e in errors:
                st.error(e)
        else:
            st.success("No pipeline errors")
        st.json({"full_api_response": data})

else:
    st.info("👈 Configure your lot in the sidebar and click **Generate Pre-Auction Briefing** to run AuctionSense.")
    st.markdown("""
    ### What AuctionSense Does

    AuctionSense runs a four-agent AI pipeline to produce a comprehensive pre-auction briefing:

    | Agent | What it does |
    |---|---|
    | 🌐 **Market Intel** | Fetches live commodity prices and market news |
    | 💰 **Reserve Price** | Predicts clearing price with SHAP explainability |
    | 👥 **Participation** | Forecasts bidder count and flags thin-participation risk |
    | ⚙️ **Lot Configuration** | Recommends split, bundle, or as-is strategy |
    | 🧠 **Strategist** | Synthesises everything into a one-page briefing |

    **Paper:** [Zenodo DOI pending] | **GitHub:** [github.com/debray2523/auctionsense](https://github.com/debray2523/auctionsense)
    """)
