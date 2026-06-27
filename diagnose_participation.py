"""
Diagnose why LightGBM AUC is stuck at 0.71 on the participation dataset.
Runs feature importance, correlation analysis, and a baseline check.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

TARGET = "participated_binary"
CATEGORICAL = ["buyer_lot_size_affinity", "lot_category"]
NUMERIC = [
    "buyer_category_affinity", "buyer_region_match", "buyer_recency_days",
    "buyer_win_rate_90d", "buyer_credit_active", "buyer_category_recency",
]

df = pd.read_parquet("data/buyer_lot.parquet")
print(f"Records: {len(df)}  Pos rate: {df[TARGET].mean():.3f}")
print()

# 1. Per-feature AUC (how much signal does each feature carry alone?)
print("=== Per-feature AUC (single-feature models) ===")
for col in NUMERIC:
    feat = df[col].values.reshape(-1,1)
    auc = roc_auc_score(df[TARGET], feat)
    auc = max(auc, 1-auc)  # flip if inverted
    print(f"  {col:<35} AUC={auc:.4f}")

print()

# 2. Correlation of each numeric feature with target
print("=== Point-biserial correlation with target ===")
for col in NUMERIC:
    corr = df[col].corr(df[TARGET])
    print(f"  {col:<35} r={corr:.4f}")

print()

# 3. Check if the ground-truth participation probability is deterministic
# (i.e., does the label perfectly follow the logistic function in the generator?)
print("=== Top/bottom 5% engagement: do they actually differ in participation? ===")
df["eng"] = (
    df["buyer_category_affinity"] * 0.4
    + (1 - df["buyer_recency_days"].clip(0,180)/180) * 0.3
    + df["buyer_win_rate_90d"] * 0.2
    + df["buyer_credit_active"] * 0.1
)
top5  = df[df["eng"] >= df["eng"].quantile(0.95)][TARGET].mean()
bot5  = df[df["eng"] <= df["eng"].quantile(0.05)][TARGET].mean()
print(f"  Top 5% engagement → participation rate: {top5:.3f}")
print(f"  Bot 5% engagement → participation rate: {bot5:.3f}")
print(f"  Spread: {top5-bot5:.3f}  (if <0.15, data is too noisy to learn from)")

print()

# 4. What AUC does a simple logistic regression achieve?
df_enc = df.copy()
for col in CATEGORICAL:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))
X = df_enc[CATEGORICAL + NUMERIC + ["eng"]]
y = df_enc[TARGET]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15,
                                            random_state=42, stratify=y)
lr = LogisticRegression(max_iter=1000)
lr.fit(X_tr, y_tr)
lr_auc = roc_auc_score(y_te, lr.predict_proba(X_te)[:,1])
print(f"=== Logistic Regression AUC: {lr_auc:.4f} ===")

# 5. Dummy baseline
dum = DummyClassifier(strategy="prior")
dum.fit(X_tr, y_tr)
dum_auc = roc_auc_score(y_te, dum.predict_proba(X_te)[:,1])
print(f"=== Dummy (prior) AUC:        {dum_auc:.4f} ===")

print()
print("=== Noise analysis: std of participation within same engagement decile ===")
df["eng_decile"] = pd.qcut(df["eng"], 10, labels=False)
noise = df.groupby("eng_decile")[TARGET].agg(["mean","std","count"])
print(noise.to_string())
