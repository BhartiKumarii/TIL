"""
Random Forests & XGBoost
==========================
Predicting California housing prices.
Covers:
  • RandomForestRegressor — ensemble of trees, bagging + feature subsampling
  • XGBRegressor          — gradient boosting
  • Hyperparameter intuition (n_estimators, max_depth, learning_rate)
  • SHAP-style feature importance comparison
  • OOB (out-of-bag) error for RF
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor
    XGBOOST_AVAILABLE = False
    print("  [info] xgboost not installed — using sklearn GradientBoostingRegressor instead.")

# ── 1. Data ───────────────────────────────────────────────────────────────────
data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target
feature_names = list(X.columns)

print("=" * 55)
print("  DATASET: California Housing")
print("=" * 55)
print(f"  Samples  : {X.shape[0]:,}")
print(f"  Features : {X.shape[1]}  →  {feature_names}")
print(f"  Target   : Median house value (100k $), range [{y.min():.2f}, {y.max():.2f}]")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 2. Random Forest ──────────────────────────────────────────────────────────
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,          # grow full trees
    min_samples_leaf=5,
    max_features=0.5,        # 50% features per split (key RF trick)
    oob_score=True,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)

rf_pred   = rf.predict(X_test)
rf_rmse   = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2     = r2_score(y_test, rf_pred)
rf_oob    = rf.oob_score_

# ── 3. XGBoost (or GBM) ───────────────────────────────────────────────────────
if XGBOOST_AVAILABLE:
    xgb = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )
else:
    xgb = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )

xgb.fit(X_train, y_train)

xgb_pred  = xgb.predict(X_test)
xgb_rmse  = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_r2    = r2_score(y_test, xgb_pred)

# ── 4. Print results ──────────────────────────────────────────────────────────
model_name = "XGBoost" if XGBOOST_AVAILABLE else "GradientBoosting"
print("\n" + "=" * 55)
print("  RESULTS COMPARISON")
print("=" * 55)
print(f"  {'Model':<22}  {'RMSE':>8}  {'R²':>8}  Notes")
print("-" * 55)
print(f"  {'Random Forest':<22}  {rf_rmse:>8.4f}  {rf_r2:>8.4f}  OOB R²={rf_oob:.4f}")
print(f"  {model_name:<22}  {xgb_rmse:>8.4f}  {xgb_r2:>8.4f}")
print("=" * 55)

# ── 5. OOB learning curve (RF) ────────────────────────────────────────────────
oob_scores = []
estimator_counts = [10, 25, 50, 100, 150, 200]
for n in estimator_counts:
    m = RandomForestRegressor(n_estimators=n, max_features=0.5,
                               oob_score=True, random_state=42, n_jobs=-1)
    m.fit(X_train, y_train)
    oob_scores.append(m.oob_score_)

# ── 6. Feature importances ────────────────────────────────────────────────────
rf_imp  = pd.Series(rf.feature_importances_,  index=feature_names)
xgb_imp = pd.Series(xgb.feature_importances_, index=feature_names)

# ── 7. Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Random Forest & XGBoost — California Housing", fontsize=13, fontweight="bold")

# OOB curve
ax = axes[0, 0]
ax.plot(estimator_counts, oob_scores, "o-", color="#4C72B0")
ax.set_xlabel("Number of trees (n_estimators)")
ax.set_ylabel("OOB R² score")
ax.set_title("RF Out-of-Bag Score vs. n_estimators\n(free validation — no test set needed)")

# Feature importance comparison
ax = axes[0, 1]
x_pos = np.arange(len(feature_names))
width = 0.4
ax.barh(x_pos + width/2, rf_imp[feature_names],  width, label="Random Forest", color="#4C72B0")
ax.barh(x_pos - width/2, xgb_imp[feature_names], width, label=model_name,      color="#DD8452")
ax.set_yticks(x_pos)
ax.set_yticklabels(feature_names)
ax.set_xlabel("Importance")
ax.set_title("Feature Importance Comparison")
ax.legend()

# Actual vs predicted — RF
ax = axes[1, 0]
ax.scatter(y_test, rf_pred, alpha=0.2, s=8, color="#4C72B0")
lim = (0, y.max())
ax.plot(lim, lim, "r--", lw=1.5)
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.set_title(f"Random Forest — Actual vs Predicted\nR²={rf_r2:.4f}  RMSE={rf_rmse:.4f}")

# Actual vs predicted — XGB
ax = axes[1, 1]
ax.scatter(y_test, xgb_pred, alpha=0.2, s=8, color="#DD8452")
ax.plot(lim, lim, "r--", lw=1.5)
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.set_title(f"{model_name} — Actual vs Predicted\nR²={xgb_r2:.4f}  RMSE={xgb_rmse:.4f}")

plt.tight_layout()
plt.savefig("04_rf_xgboost.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  Plot saved → 04_rf_xgboost.png")
