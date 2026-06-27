"""
AuctionSense — Buyer Participation Agent: LightGBM Model (v2 — fixed)
Key fix: removed class_weight="balanced" (data is ~47/53, not imbalanced),
added proper feature engineering, tuned num_leaves and min_child_samples.
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = Path("models/lgbm_participation.pkl")
DATA_PATH  = Path("data/buyer_lot.parquet")

CATEGORICAL_FEATURES = ["buyer_lot_size_affinity", "lot_category"]
NUMERIC_FEATURES = [
    "buyer_category_affinity", "buyer_region_match", "buyer_recency_days",
    "buyer_win_rate_90d", "buyer_credit_active", "buyer_category_recency",
]
TARGET = "participated_binary"

THIN_PARTICIPATION_THRESHOLD = 4

# FIX: Removed class_weight="balanced" — data is near-balanced (47/53).
# class_weight="balanced" was causing the model to predict almost everything
# as positive (recall=0.97 but precision=0.50 — essentially random).
# Added regularisation (reg_alpha, reg_lambda, min_child_samples) to prevent
# trivial splits and force meaningful feature discrimination.
BEST_PARAMS = {
    "n_estimators":     500,
    "max_depth":        5,
    "learning_rate":    0.05,
    "num_leaves":       31,
    "min_child_samples":50,      # prevents trivial leaf splits
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,     # L1 regularisation
    "reg_lambda":       1.0,     # L2 regularisation
    "random_state":     42,
    "objective":        "binary",
    "metric":           "auc",
    "verbose":         -1,       # suppress warnings
}


class ParticipationModel:
    def __init__(self):
        self.model: Optional[lgb.LGBMClassifier] = None
        self.encoders: dict[str, LabelEncoder] = {}
        self.feature_names: list[str] = []
        self.operating_threshold: float = 0.50

    def _encode(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        df = df.copy()
        for col in CATEGORICAL_FEATURES:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
            else:
                le = self.encoders[col]
                known = set(le.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in known else le.classes_[0]
                )
                df[col] = le.transform(df[col])
        return df

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features that improve discrimination."""
        df = df.copy()
        # Engagement score: combines recency, affinity, win rate
        df["engagement_score"] = (
            df["buyer_category_affinity"] * 0.4
            + (1 - df["buyer_recency_days"].clip(0, 180) / 180) * 0.3
            + df["buyer_win_rate_90d"] * 0.2
            + df["buyer_credit_active"] * 0.1
        )
        # Category freshness: inverse of category recency
        df["category_freshness"] = 1 - (
            df["buyer_category_recency"].clip(0, 90) / 90
        )
        # Active buyer flag: recency < 30 days AND credit active
        df["active_buyer"] = (
            (df["buyer_recency_days"] < 30) & (df["buyer_credit_active"] == 1)
        ).astype(int)
        return df

    def _feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        engineered = ["engagement_score", "category_freshness", "active_buyer"]
        cols = [c for c in CATEGORICAL_FEATURES + NUMERIC_FEATURES + engineered
                if c in df.columns]
        return df[cols]

    def train(self, data_path: Path = DATA_PATH) -> dict:
        df = pd.read_parquet(data_path)

        print(f"  Dataset: {len(df)} records")
        print(f"  Participation rate: {df[TARGET].mean():.1%}")

        df_feat = self._add_features(df)
        df_enc  = self._encode(df_feat, fit=True)
        X = self._feature_matrix(df_enc)
        y = df[TARGET]
        self.feature_names = list(X.columns)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )

        self.model = lgb.LGBMClassifier(**BEST_PARAMS)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=100)
            ]
        )

        y_prob = self.model.predict_proba(X_test)[:, 1]
        auc    = roc_auc_score(y_test, y_prob)

        # Find threshold maximising F1 (balanced between precision and recall)
        best_thresh, best_f1 = 0.50, 0.0
        for thresh in np.arange(0.30, 0.70, 0.01):
            y_pred = (y_prob >= thresh).astype(int)
            f1 = f1_score(y_test, y_pred)
            if f1 > best_f1:
                best_f1, best_thresh = f1, thresh
        self.operating_threshold = best_thresh

        y_pred = (y_prob >= self.operating_threshold).astype(int)
        metrics = {
            "auc_roc":   float(auc),
            "precision": float(precision_score(y_test, y_pred)),
            "recall":    float(recall_score(y_test, y_pred)),
            "f1":        float(f1_score(y_test, y_pred)),
            "threshold": float(self.operating_threshold),
            "n_test":    int(len(y_test)),
            "pos_rate":  float(y_test.mean()),
        }

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self, f)

        print(f"Model saved → {MODEL_PATH}")
        print(f"AUC={metrics['auc_roc']:.3f}  "
              f"F1={metrics['f1']:.3f}  "
              f"Recall={metrics['recall']:.3f}  "
              f"Precision={metrics['precision']:.3f}  "
              f"Threshold={metrics['threshold']:.2f}")
        return metrics

    @classmethod
    def load(cls) -> "ParticipationModel":
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)

    def predict_lot(self, lot_category: str, buyer_profiles: list[dict]) -> dict:
        if not buyer_profiles:
            return {
                "expected_bidder_count": 0,
                "participation_risk": "HIGH",
                "top_10_buyer_ids": [],
                "confidence": "LOW",
            }

        records = [{**bp, "lot_category": lot_category} for bp in buyer_profiles]
        df = pd.DataFrame(records)
        df = self._add_features(df)
        df_enc = self._encode(df, fit=False)
        X = self._feature_matrix(df_enc)
        probs = self.model.predict_proba(X)[:, 1]

        expected_count = float(probs.sum())
        risk = (
            "HIGH"   if expected_count < THIN_PARTICIPATION_THRESHOLD else
            "MEDIUM" if expected_count < 6 else
            "LOW"
        )

        ranked  = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
        top_ids = [buyer_profiles[i].get("buyer_id", f"buyer_{i}")
                   for i, _ in ranked[:10]]

        return {
            "expected_bidder_count":    round(expected_count, 1),
            "participation_risk":       risk,
            "top_10_buyer_ids":         top_ids,
            "top_10_participation_probs": [round(float(probs[i]), 3)
                                           for i, _ in ranked[:10]],
            "confidence": "HIGH" if len(buyer_profiles) > 20 else "MEDIUM",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",    action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    if args.train:
        model = ParticipationModel()
        metrics = model.train()
        print(json.dumps(metrics, indent=2))
    elif args.evaluate:
        model = ParticipationModel.load()
        sample_buyers = [
            {"buyer_id": "B001", "buyer_category_affinity": 0.8,
             "buyer_region_match": 1, "buyer_recency_days": 5,
             "buyer_win_rate_90d": 0.4, "buyer_credit_active": 1,
             "buyer_lot_size_affinity": "medium", "buyer_category_recency": 3},
            {"buyer_id": "B002", "buyer_category_affinity": 0.2,
             "buyer_region_match": 0, "buyer_recency_days": 120,
             "buyer_win_rate_90d": 0.1, "buyer_credit_active": 0,
             "buyer_lot_size_affinity": "large", "buyer_category_recency": 60},
        ]
        result = model.predict_lot("ferrous_scrap", sample_buyers)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
