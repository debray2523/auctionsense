"""
AuctionSense — Test Suite
Run: pytest tests/ -v
"""

import json
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

# ── Data generator tests ──────────────────────────────────────────────
class TestSyntheticGenerator:

    def test_lot_generation_count(self):
        from src.data.synthetic_generator import generate_lots
        rng = np.random.default_rng(42)
        df = generate_lots(100, rng)
        assert len(df) == 100

    def test_lot_schema_columns(self):
        from src.data.synthetic_generator import generate_lots
        rng = np.random.default_rng(42)
        df = generate_lots(50, rng)
        required = ["lot_category", "lot_quantity_mt", "lot_grade", "clearing_price"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_clearing_price_positive(self):
        from src.data.synthetic_generator import generate_lots
        rng = np.random.default_rng(42)
        df = generate_lots(200, rng)
        assert (df["clearing_price"] > 0).all()

    def test_category_distribution(self):
        from src.data.synthetic_generator import generate_lots, CATEGORIES
        rng = np.random.default_rng(42)
        df = generate_lots(5000, rng)
        cats = df["lot_category"].unique()
        for cat in CATEGORIES:
            assert cat in cats, f"Category {cat} missing from 5000-lot dataset"

    def test_buyer_lot_generation(self):
        from src.data.synthetic_generator import generate_lots, generate_buyer_lot_records
        rng = np.random.default_rng(42)
        lots = generate_lots(100, rng)
        buyers = generate_buyer_lot_records(lots, 400, rng)
        assert len(buyers) <= 400
        assert "participated_binary" in buyers.columns
        assert buyers["participated_binary"].isin([0, 1]).all()

    def test_participation_rate_reasonable(self):
        from src.data.synthetic_generator import generate_lots, generate_buyer_lot_records
        rng = np.random.default_rng(42)
        lots = generate_lots(200, rng)
        buyers = generate_buyer_lot_records(lots, 2000, rng)
        rate = buyers["participated_binary"].mean()
        assert 0.15 <= rate <= 0.55, f"Participation rate {rate:.2f} out of expected range"


# ── Reserve price model tests ────────────────────────────────────────
class TestReservePriceModel:

    def test_model_initialises(self):
        from src.models.xgboost_reserve import ReservePriceModel
        model = ReservePriceModel()
        assert model.model is None
        assert model.encoders == {}

    def test_predict_returns_schema(self):
        from src.models.xgboost_reserve import ReservePriceModel
        model = ReservePriceModel()
        # Mock the trained model
        mock_xgb = MagicMock()
        mock_xgb.predict.return_value = np.array([29000.0])
        model.model = mock_xgb
        model.encoders = {
            col: MagicMock(**{"transform.return_value": np.array([0])})
            for col in ["lot_category", "lot_grade", "lot_location_region",
                        "season_quarter", "auction_format"]
        }
        model.feature_names = ["lot_category", "lot_grade", "lot_location_region",
                                "season_quarter", "auction_format", "lot_quantity_mt",
                                "days_to_auction", "commodity_spot_price",
                                "commodity_30d_trend", "secondary_commodity_idx",
                                "fx_rate", "registered_buyer_count",
                                "historical_avg_premium", "historical_participation_rate",
                                "market_sentiment_score"]
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = np.zeros((1, 15))
        model.explainer = mock_explainer

        sample_lot = {
            "lot_category": "ferrous_scrap", "lot_quantity_mt": 150.0,
            "lot_grade": "B", "lot_location_region": "east",
            "days_to_auction": 3, "season_quarter": "Q1",
            "auction_format": "english_ascending",
            "registered_buyer_count": 45, "historical_avg_premium": 0.04,
            "historical_participation_rate": 8.2,
        }
        sample_ctx = {
            "commodity_spot_price": 28500.0, "commodity_30d_trend": 0.05,
            "secondary_commodity_idx": 27000.0, "fx_rate": 83.2,
            "market_sentiment_score": 0.6,
        }

        result = model.predict(sample_lot, sample_ctx)
        assert "predicted_clearing_price" in result
        assert "reserve_lower" in result
        assert "reserve_upper" in result
        assert "shap_top5" in result
        assert result["reserve_lower"] < result["predicted_clearing_price"]
        assert result["reserve_upper"] < result["predicted_clearing_price"]


# ── Participation model tests ────────────────────────────────────────
class TestParticipationModel:

    def test_model_initialises(self):
        from src.models.lgbm_participation import ParticipationModel
        model = ParticipationModel()
        assert model.model is None
        assert model.operating_threshold == 0.50

    def test_predict_lot_empty_buyers(self):
        from src.models.lgbm_participation import ParticipationModel
        model = ParticipationModel()
        result = model.predict_lot("ferrous_scrap", [])
        assert result["participation_risk"] == "HIGH"
        assert result["expected_bidder_count"] == 0

    def test_participation_risk_classification(self):
        from src.models.lgbm_participation import ParticipationModel, THIN_PARTICIPATION_THRESHOLD
        model = ParticipationModel()
        mock_lgb = MagicMock()

        # Test HIGH risk (< 4 expected bidders)
        mock_lgb.predict_proba.return_value = np.array([[0.9, 0.1], [0.8, 0.2]])
        model.model = mock_lgb
        model.encoders = {
            "buyer_lot_size_affinity": MagicMock(**{"transform.return_value": np.array([0, 0])}),
            "lot_category": MagicMock(**{"transform.return_value": np.array([0, 0])}),
        }
        model.feature_names = ["buyer_category_affinity", "buyer_region_match",
                                "buyer_recency_days", "buyer_win_rate_90d",
                                "buyer_credit_active", "buyer_category_recency",
                                "buyer_lot_size_affinity", "lot_category"]
        buyers = [
            {"buyer_id": "B1", "buyer_category_affinity": 0.1, "buyer_region_match": 0,
             "buyer_recency_days": 100, "buyer_win_rate_90d": 0.05, "buyer_credit_active": 0,
             "buyer_lot_size_affinity": "medium", "buyer_category_recency": 60},
            {"buyer_id": "B2", "buyer_category_affinity": 0.1, "buyer_region_match": 0,
             "buyer_recency_days": 100, "buyer_win_rate_90d": 0.05, "buyer_credit_active": 0,
             "buyer_lot_size_affinity": "medium", "buyer_category_recency": 60},
        ]
        result = model.predict_lot("ferrous_scrap", buyers)
        assert "expected_bidder_count" in result
        assert "participation_risk" in result
        assert result["participation_risk"] in ["HIGH", "MEDIUM", "LOW"]


# ── API schema tests ──────────────────────────────────────────────────
class TestAPISchemas:

    def test_lot_input_valid(self):
        from src.api.schemas import LotInput
        lot = LotInput(
            lot_category="ferrous_scrap",
            lot_quantity_mt=150.0,
            lot_grade="B",
            lot_location_region="east",
            days_to_auction=3,
            registered_buyer_count=47,
        )
        assert lot.lot_category == "ferrous_scrap"
        assert lot.lot_grade == "B"

    def test_lot_input_invalid_quantity(self):
        from src.api.schemas import LotInput
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            LotInput(
                lot_category="ferrous_scrap",
                lot_quantity_mt=-10.0,  # Invalid: must be > 0
                lot_grade="B",
                lot_location_region="east",
                days_to_auction=3,
                registered_buyer_count=47,
            )

    def test_lot_input_invalid_days(self):
        from src.api.schemas import LotInput
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            LotInput(
                lot_category="ferrous_scrap",
                lot_quantity_mt=100.0,
                lot_grade="A",
                lot_location_region="west",
                days_to_auction=50,  # Invalid: max is 30
                registered_buyer_count=20,
            )


# ── Integration smoke test ───────────────────────────────────────────
class TestIntegrationSmoke:
    """Quick smoke tests — run with mocked LLM to avoid API calls in CI."""

    def test_pipeline_state_structure(self):
        """Verify the LangGraph state TypedDict has all required keys."""
        from src.agents.strategist import AuctionSenseState
        required_keys = [
            "lot_input", "operator_sector", "market_context",
            "reserve_result", "participation_result", "lot_config_result",
            "pre_auction_briefing", "compliance_flags", "pipeline_errors"
        ]
        hints = AuctionSenseState.__annotations__
        for key in required_keys:
            assert key in hints, f"Missing key in AuctionSenseState: {key}"
