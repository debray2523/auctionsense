"""
AuctionSense — Reserve Price Agent: XGBoost Model
Trains and serves the clearing price prediction model.

Usage:
    python -m src.models.xgboost_reserve --train
    python -m src.models.xgboost_reserve --evaluate
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = Path("models/xgboost_reserve.pkl")
ENCODERS_PATH = Path("models/label_encoders.pkl")
DATA_PATH = Path("data/lots.parquet")

CATEGORICAL_FEATURES = ["lot_category", "lot_grade", "lot_location_region",
                         "season_quarter", "auction_format"]
NUMERIC_FEATURES = [
    "lot_quantity_mt", "days_to_auction", "commodity_spot_price",
    "commodity_30d_trend", "secondary_commodity_idx", "fx_rate",
    "registered_buyer_count", "historical_avg_premium", "historical_participation_rate",
]
TARGET = "clearing_price"

PARAM_GRID = {
    "n_estimators":    [200, 400, 600],
    "max_depth":       [4, 6, 8],
    "learning_rate":   [0.05, 0.10, 0.20],
    "subsample":       [0.7, 0.9],
    "colsample_bytree":[0.7, 0.9],
}
BEST_PARAMS = {  # Pre-tuned defaults — override with grid search
    "n_estimators": 400, "max_depth": 6, "learning_rate": 0.10,
    "subsample": 0.9, "colsample_bytree": 0.9, "random_state": 42,
    "objective": "reg:squarederror", "eval_metric": "rmse",
}


class ReservePriceModel:
    def __init__(self):
        self.model: Optional[xgb.XGBRegressor] = None
        self.encoders: dict[str, LabelEncoder] = {}
        self.feature_names: list[str] = []
        self.explainer: Optional[shap.TreeExplainer] = None

    def _encode(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        df = df.copy()
        for col in CATEGORICAL_FEATURES:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
            else:
                le = self.encoders[col]
                df[col] = le.transform(df[col].astype(str))
        return df

    def _feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]

    def train(self, data_path: Path = DATA_PATH, grid_search: bool = False) -> dict:
        df = pd.read_parquet(data_path)
        df["lot_quantity_mt"] = np.log1p(df["lot_quantity_mt"])  # log transform

        df_enc = self._encode(df, fit=True)
        X = self._feature_matrix(df_enc)
        y = df[TARGET]
        self.feature_names = list(X.columns)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=df["lot_category"]
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.176, random_state=42  # 0.176 * 0.85 ≈ 0.15
        )

        if grid_search:
            base = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
            gs = GridSearchCV(base, PARAM_GRID, cv=5, scoring="neg_mean_absolute_percentage_error",
                              n_jobs=-1, verbose=2)
            gs.fit(X_train, y_train)
            params = gs.best_params_
            params.update({"objective": "reg:squarederror", "random_state": 42})
        else:
            params = BEST_PARAMS

        self.model = xgb.XGBRegressor(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)

        # Evaluate
        y_pred = self.model.predict(X_test)
        metrics = {
            "mape": float(mean_absolute_percentage_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "r2":   float(r2_score(y_test, y_pred)),
            "n_test": len(y_test),
            "params": params,
        }

        # Save
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved → {MODEL_PATH}")
        print(f"Results: MAPE={metrics['mape']:.3f} RMSE={metrics['rmse']:.1f} R²={metrics['r2']:.3f}")
        return metrics

    @classmethod
    def load(cls) -> "ReservePriceModel":
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)

    def predict(self, lot: dict, market_context: dict) -> dict:
        """
        Predict clearing price and return reserve price band with SHAP explanation.

        Args:
            lot: dict with lot features (lot_category, lot_quantity_mt, lot_grade, etc.)
            market_context: dict from Market Intel Agent (commodity_spot_price, etc.)

        Returns:
            dict with predicted_clearing_price, reserve_lower, reserve_upper,
                  shap_top5, confidence_margin
        """
        row = {**lot, **market_context}
        df = pd.DataFrame([row])
        df["lot_quantity_mt"] = np.log1p(df["lot_quantity_mt"])
        df_enc = self._encode(df, fit=False)
        X = self._feature_matrix(df_enc)

        pred = float(self.model.predict(X)[0])
        shap_vals = self.explainer.shap_values(X)[0]

        # Top 5 SHAP features
        shap_pairs = sorted(
            zip(self.feature_names, shap_vals),
            key=lambda x: abs(x[1]), reverse=True
        )[:5]
        shap_top5 = [{"feature": f, "shap_value": round(v, 2)} for f, v in shap_pairs]

        # Confidence margin — approximated from category-level validation MAPE
        confidence_margin = 0.07  # 7% default; calibrate from val set per category in production

        return {
            "predicted_clearing_price": round(pred, 2),
            "reserve_lower":  round(pred * (1 - confidence_margin), 2),
            "reserve_upper":  round(pred * 0.97, 2),
            "confidence_margin": confidence_margin,
            "shap_top5": shap_top5,
            "model_version": "xgboost_v1",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",    action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--grid-search", action="store_true")
    args = parser.parse_args()

    if args.train:
        model = ReservePriceModel()
        metrics = model.train(grid_search=args.grid_search)
        print(json.dumps(metrics, indent=2))
    elif args.evaluate:
        model = ReservePriceModel.load()
        # Quick sanity check prediction
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
        }
        result = model.predict(sample_lot, sample_ctx)
        print("Sample prediction:")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
