"""
AuctionSense — Market Intel Agent
Fetches live commodity prices and market context for the lot being evaluated.
Falls back gracefully when external APIs are unavailable.
"""

import os
import datetime
from typing import Any

# Optional imports — fall back gracefully if not installed
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# Commodity ticker mapping by lot category
COMMODITY_TICKERS = {
    "ferrous_scrap":         "STEEL.L",   # London Steel
    "nonferrous_scrap":      "HG=F",      # Copper futures (LME proxy)
    "idle_industrial_asset": "STEEL.L",
    "coal":                  "MTFSG.SI",  # Coal futures proxy
    "agricultural":          "ZW=F",      # Wheat futures
}

# Fallback prices (₹/MT) when API unavailable — calibrated to synthetic data
FALLBACK_PRICES = {
    "ferrous_scrap":         28000.0,
    "nonferrous_scrap":      95000.0,
    "idle_industrial_asset": 150000.0,
    "coal":                  12000.0,
    "agricultural":          45000.0,
}


def _fetch_commodity_price(lot_category: str) -> tuple[float, float, bool]:
    """
    Returns (spot_price, trend_30d, is_live).
    Falls back to calibrated defaults if yfinance unavailable or API fails.
    """
    if not YFINANCE_AVAILABLE:
        base = FALLBACK_PRICES.get(lot_category, 28000.0)
        return base, 0.02, False

    ticker_sym = COMMODITY_TICKERS.get(lot_category, "HG=F")
    try:
        ticker = yf.Ticker(ticker_sym)
        hist   = ticker.history(period="35d")
        if hist.empty or len(hist) < 5:
            raise ValueError("Insufficient data")

        latest_price = float(hist["Close"].iloc[-1])
        price_30d    = float(hist["Close"].iloc[0])

        # Convert to ₹/MT using approximate INR scaling
        INR_SCALE = {
            "ferrous_scrap":         1.0,
            "nonferrous_scrap":      750.0,   # USD/lb → ₹/MT approx
            "idle_industrial_asset": 1.0,
            "coal":                  80.0,    # USD/MT → ₹/MT
            "agricultural":          80.0,
        }
        scale        = INR_SCALE.get(lot_category, 1.0)
        spot_inr     = latest_price * scale
        trend_30d    = (latest_price - price_30d) / price_30d if price_30d > 0 else 0.0
        return spot_inr, trend_30d, True

    except Exception:
        base = FALLBACK_PRICES.get(lot_category, 28000.0)
        return base, 0.02, False


def _fetch_fx_rate() -> float:
    """Fetch USD/INR rate, fallback to 83.5."""
    if not YFINANCE_AVAILABLE:
        return 83.5
    try:
        ticker = yf.Ticker("INR=X")
        hist   = ticker.history(period="2d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return 83.5


def _market_sentiment(trend_30d: float) -> float:
    """
    Simple sentiment score 0-1 based on price trend.
    0.3 = bearish, 0.5 = neutral, 0.7+ = bullish
    """
    if trend_30d > 0.10:   return 0.80
    if trend_30d > 0.05:   return 0.65
    if trend_30d > 0.00:   return 0.55
    if trend_30d > -0.05:  return 0.45
    if trend_30d > -0.10:  return 0.35
    return 0.25


def run_market_intel_agent(lot_input: dict[str, Any]) -> dict[str, Any]:
    """
    Market Intel Agent entry point.

    Args:
        lot_input: dict containing at minimum 'lot_category'

    Returns:
        Market context dict consumed by downstream agents.
    """
    lot_category = lot_input.get("lot_category", "ferrous_scrap")

    spot_price, trend_30d, is_live = _fetch_commodity_price(lot_category)
    fx_rate                        = _fetch_fx_rate()
    sentiment                      = _market_sentiment(trend_30d)

    # Secondary commodity index (simplified)
    secondary_idx = spot_price * 0.95  # approximate related commodity

    # Staleness
    staleness_hours = 0 if is_live else 24
    staleness_flag  = not is_live

    # Brief market summary
    trend_pct    = trend_30d * 100
    direction    = "upward" if trend_30d > 0 else "downward"
    intensity    = "strongly" if abs(trend_30d) > 0.08 else "moderately" if abs(trend_30d) > 0.03 else "slightly"
    category_str = lot_category.replace("_", " ")

    news_summary = (
        f"The {category_str} market has moved {intensity} {direction} over the past 30 days "
        f"({trend_pct:+.1f}%). "
        f"Current reference price: ₹{spot_price:,.0f}/MT. "
        f"USD/INR exchange rate: {fx_rate:.1f}. "
        f"Market sentiment score: {sentiment:.2f}/1.0 ({'bullish' if sentiment > 0.6 else 'bearish' if sentiment < 0.4 else 'neutral'}). "
        f"{'Live market data.' if is_live else 'Note: Using calibrated reference prices — live market API unavailable.'}"
    )

    return {
        "commodity_spot_price":    round(spot_price, 2),
        "commodity_30d_trend":     round(trend_30d, 4),
        "secondary_commodity_idx": round(secondary_idx, 2),
        "fx_rate":                 round(fx_rate, 2),
        "market_sentiment_score":  round(sentiment, 2),
        "news_summary":            news_summary,
        "data_staleness_hours":    staleness_hours,
        "staleness_flag":          staleness_flag,
        "lot_category":            lot_category,
        "timestamp":               datetime.datetime.utcnow().isoformat(),
    }
