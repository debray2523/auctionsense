"""
AuctionSense — Synthetic Data Generator
Generates calibrated B2B industrial auction lot and buyer-lot datasets.

Usage:
    python -m src.data.synthetic_generator --lots 12000 --buyers 60000 --seed 42
"""

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────
CATEGORIES = {
    "ferrous_scrap":       {"weight": 0.35, "base_price": 28000, "price_std": 4000,  "base_premium": 0.04},
    "idle_industrial_asset":{"weight": 0.25, "base_price": 150000,"price_std": 60000, "base_premium": -0.05},
    "nonferrous_scrap":    {"weight": 0.20, "base_price": 95000,  "price_std": 15000, "base_premium": 0.03},
    "coal":                {"weight": 0.12, "base_price": 12000,  "price_std": 2000,  "base_premium": 0.02},
    "agricultural":        {"weight": 0.08, "base_price": 45000,  "price_std": 8000,  "base_premium": 0.01},
}

GRADES = {"A": 1.12, "B": 1.00, "C": 0.88, "D": 0.74}
REGIONS = ["east", "west", "north", "south"]
FORMATS = ["english_ascending", "sealed_bid", "dutch_descending"]
QUARTERS = ["Q1", "Q2", "Q3", "Q4"]

QUARTER_FACTORS = {"Q1": 1.05, "Q2": 0.98, "Q3": 0.95, "Q4": 1.08}
FORMAT_FACTORS  = {"english_ascending": 1.04, "sealed_bid": 0.98, "dutch_descending": 0.96}


def _commodity_spot(category: str, trend: float, rng: np.random.Generator) -> float:
    cfg = CATEGORIES[category]
    noise = rng.normal(0, cfg["price_std"] * 0.3)
    trend_impact = cfg["base_price"] * trend * 0.5
    return max(cfg["base_price"] * 0.6, cfg["base_price"] + trend_impact + noise)


def _clearing_price(row: dict, rng: np.random.Generator) -> float:
    cfg = CATEGORIES[row["lot_category"]]
    base = row["commodity_spot_price"] * GRADES[row["lot_grade"]]
    premium = cfg["base_premium"] + row["commodity_30d_trend"] * 0.3
    qty_discount = -0.02 if row["lot_quantity_mt"] > 500 else (0.01 if row["lot_quantity_mt"] < 50 else 0.0)
    buyer_effect = 0.01 * min(row["registered_buyer_count"] / 20, 3.0)
    season = QUARTER_FACTORS[row["season_quarter"]] - 1.0
    fmt = FORMAT_FACTORS[row["auction_format"]] - 1.0
    noise = rng.normal(0, 0.04)
    total_factor = 1.0 + premium + qty_discount + buyer_effect + season + fmt + noise
    return round(base * max(0.5, total_factor), 2)


def generate_lots(n: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    cats = list(CATEGORIES.keys())
    cat_weights = [CATEGORIES[c]["weight"] for c in cats]

    for _ in range(n):
        cat = rng.choice(cats, p=cat_weights)
        grade = rng.choice(list(GRADES.keys()), p=[0.15, 0.45, 0.30, 0.10])
        region = rng.choice(REGIONS)
        quarter = rng.choice(QUARTERS)
        fmt = rng.choice(FORMATS, p=[0.65, 0.25, 0.10])
        qty = float(np.round(rng.lognormal(mean=4.5, sigma=1.1), 1))
        qty = min(max(qty, 5.0), 2000.0)
        days = int(rng.integers(1, 15))
        trend = float(rng.normal(0, 0.08))
        trend = max(-0.25, min(0.25, trend))
        spot = _commodity_spot(cat, trend, rng)
        fx = float(rng.normal(83.5, 2.5))
        buyers = int(rng.integers(5, 120))
        hist_premium = float(rng.normal(CATEGORIES[cat]["base_premium"], 0.03))
        hist_part = float(rng.normal(8, 3))
        hist_part = max(2.0, hist_part)

        row = {
            "lot_category": cat,
            "lot_quantity_mt": qty,
            "lot_grade": grade,
            "lot_location_region": region,
            "days_to_auction": days,
            "commodity_spot_price": round(spot, 2),
            "commodity_30d_trend": round(trend, 4),
            "secondary_commodity_idx": round(spot * rng.uniform(0.8, 1.2), 2),
            "fx_rate": round(fx, 2),
            "registered_buyer_count": buyers,
            "historical_avg_premium": round(hist_premium, 4),
            "historical_participation_rate": round(hist_part, 1),
            "season_quarter": quarter,
            "auction_format": fmt,
        }
        row["clearing_price"] = _clearing_price(row, rng)
        rows.append(row)

    return pd.DataFrame(rows)


def generate_buyer_lot_records(lots_df: pd.DataFrame, n_buyers: int,
                                rng: np.random.Generator) -> pd.DataFrame:
    records = []
    # mean 5 buyer profiles per lot
    for lot_idx, lot in lots_df.iterrows():
        n = max(2, int(rng.normal(5, 2)))
        for _ in range(n):
            affinity = float(rng.beta(2, 5))
            region_match = bool(rng.random() < 0.4)
            recency = int(rng.integers(0, 180))
            win_rate = float(rng.beta(2, 6))
            credit = bool(rng.random() < 0.55)
            size_pref = rng.choice(["small", "medium", "large"], p=[0.35, 0.45, 0.20])
            cat_recency = int(rng.integers(0, 90))

            # Participation probability model (ground truth used to generate labels)
            log_odds = (
                2.0 * affinity
                + 0.8 * int(region_match)
                - 0.01 * recency
                + 1.2 * win_rate
                + 0.4 * int(credit)
                - 0.015 * cat_recency
                + float(rng.normal(0, 0.5))
            )
            prob = 1.0 / (1.0 + np.exp(-log_odds))
            participated = int(rng.random() < prob)

            records.append({
                "lot_idx": lot_idx,
                "lot_category": lot["lot_category"],
                "buyer_category_affinity": round(affinity, 3),
                "buyer_region_match": int(region_match),
                "buyer_recency_days": recency,
                "buyer_win_rate_90d": round(win_rate, 3),
                "buyer_credit_active": int(credit),
                "buyer_lot_size_affinity": size_pref,
                "buyer_category_recency": cat_recency,
                "participated_binary": participated,
            })
            if len(records) >= n_buyers:
                break
        if len(records) >= n_buyers:
            break

    return pd.DataFrame(records[:n_buyers])


def main():
    parser = argparse.ArgumentParser(description="Generate AuctionSense synthetic datasets")
    parser.add_argument("--lots",   type=int, default=12000, help="Number of lot records")
    parser.add_argument("--buyers", type=int, default=60000, help="Number of buyer-lot records")
    parser.add_argument("--seed",   type=int, default=42,    help="Random seed")
    parser.add_argument("--output", type=str, default="data", help="Output directory")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    Path(args.output).mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.lots} lot records...")
    lots_df = generate_lots(args.lots, rng)
    lots_path = Path(args.output) / "lots.parquet"
    lots_df.to_parquet(lots_path, index=False)
    print(f"  Saved {len(lots_df)} lots → {lots_path}")
    print(f"  Category distribution:\n{lots_df['lot_category'].value_counts().to_string()}")
    print(f"  Clearing price stats: mean={lots_df['clearing_price'].mean():.0f} "
          f"std={lots_df['clearing_price'].std():.0f}")

    print(f"\nGenerating {args.buyers} buyer-lot records...")
    buyer_df = generate_buyer_lot_records(lots_df, args.buyers, rng)
    buyer_path = Path(args.output) / "buyer_lot.parquet"
    buyer_df.to_parquet(buyer_path, index=False)
    pos_rate = buyer_df["participated_binary"].mean()
    print(f"  Saved {len(buyer_df)} buyer-lot records → {buyer_path}")
    print(f"  Participation rate: {pos_rate:.1%} positive / {1-pos_rate:.1%} negative")

    # Dataset summary JSON
    summary = {
        "lot_records": len(lots_df),
        "buyer_lot_records": len(buyer_df),
        "seed": args.seed,
        "lot_category_distribution": lots_df["lot_category"].value_counts().to_dict(),
        "clearing_price_mean": float(lots_df["clearing_price"].mean()),
        "clearing_price_std":  float(lots_df["clearing_price"].std()),
        "participation_rate":  float(pos_rate),
    }
    with open(Path(args.output) / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDataset summary saved → {Path(args.output) / 'dataset_summary.json'}")
    print("Data generation complete.")


if __name__ == "__main__":
    main()
