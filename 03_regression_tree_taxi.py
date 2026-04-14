"""
Regression Trees — Taxi Tip Prediction
========================================
Predicts the tip amount for NYC taxi rides.
Covers:
  • Decision Tree Regressor (shallow vs deep — bias/variance tradeoff)
  • Feature importance
  • Visualising a shallow tree
  • Comparing train vs test error across tree depths
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ── 1. Generate realistic taxi-like data ─────────────────────────────────────
np.random.seed(7)
n = 2000

trip_distance   = np.random.exponential(scale=3, size=n).clip(0.5, 30)
fare_amount     = 2.5 + 2.0 * trip_distance + np.random.normal(0, 1, n)
passenger_count = np.random.choice([1, 2, 3, 4], size=n, p=[0.55, 0.25, 0.12, 0.08])
hour            = np.random.randint(0, 24, n)
is_rush_hour    = ((hour >= 7) & (hour <= 9) | (hour >= 17) & (hour <= 19)).astype(int)
payment_credit  = np.random.choice([0, 1], size=n, p=[0.3, 0.7])

# Tip: credit card users tip ~18-20%; cash users ~5%
tip_pct         = np.where(payment_credit == 1,
                            np.random.normal(0.18, 0.05, n),
                            np.random.normal(0.05, 0.04, n)).clip(0, 0.5)
tip_amount      = (fare_amount * tip_pct + np.random.normal(0, 0.3, n)).clip(0)

df = pd.DataFrame({
    "trip_distance":   trip_distance,
    "fare_amount":     fare_amount,
    "passenger_count": passenger_count,
    "hour":            hour,
    "is_rush_hour":    is_rush_hour,
    "payment_credit":  payment_credit,
    "tip_amount":      tip_amount,
})

print(df.head())
print(f"\nDataset shape: {df.shape}")
print(df.describe().round(2))

features = ["trip_distance", "fare_amount", "passenger_count",
            "hour", "is_rush_hour", "payment_credit"]
X = df[features].values
y = df["tip_amount"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 2. Compare depths ─────────────────────────────────────────────────────────
depths      = list(range(1, 16))
train_rmse  = []
test_rmse   = []

for d in depths:
    m = DecisionTreeRegressor(max_depth=d, random_state=42)
    m.fit(X_train, y_train)
    train_rmse.append(np.sqrt(mean_squared_error(y_train, m.predict(X_train))))
    test_rmse.append(np.sqrt( mean_squared_error(y_test,  m.predict(X_test))))

best_depth = depths[np.argmin(test_rmse)]
print(f"\n  Best depth (lowest test RMSE): {best_depth}")

# ── 3. Final model ────────────────────────────────────────────────────────────
best_model = DecisionTreeRegressor(max_depth=best_depth, random_state=42)
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
r2     = r2_score(y_test, y_pred)

print("\n" + "=" * 50)
print("   REGRESSION TREE — TAXI TIP RESULTS")
print("=" * 50)
print(f"  Best max_depth : {best_depth}")
print(f"  Test RMSE      : ${rmse:.4f}")
print(f"  Test R²        : {r2:.4f}")
print("=" * 50)

importances = best_model.feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values(ascending=True)
print("\n  Feature importances:")
print(feat_imp.sort_values(ascending=False).to_string())

# ── 4. Visualise ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Regression Trees — Taxi Tip Prediction", fontsize=13, fontweight="bold")

# Depth vs RMSE
ax = axes[0]
ax.plot(depths, train_rmse, "o-", color="#4C72B0", label="Train RMSE")
ax.plot(depths, test_rmse,  "s-", color="#C44E52", label="Test RMSE")
ax.axvline(best_depth, color="gray", ls="--", label=f"Best depth={best_depth}")
ax.set_xlabel("max_depth")
ax.set_ylabel("RMSE ($)")
ax.set_title("Bias–Variance Tradeoff")
ax.legend()

# Feature importance
ax = axes[1]
feat_imp.plot(kind="barh", ax=ax, color="#55A868")
ax.set_title("Feature Importance")
ax.set_xlabel("Importance")

# Predicted vs Actual
ax = axes[2]
ax.scatter(y_test, y_pred, alpha=0.3, color="#DD8452", s=10)
lim = max(y_test.max(), y_pred.max())
ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect")
ax.set_xlabel("Actual Tip ($)")
ax.set_ylabel("Predicted Tip ($)")
ax.set_title(f"Actual vs Predicted\nR² = {r2:.3f}")
ax.legend()

plt.tight_layout()
plt.savefig("03_regression_tree_taxi.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  Plot saved → 03_regression_tree_taxi.png")

# ── 5. Visualise a shallow tree (depth=3) ────────────────────────────────────
shallow = DecisionTreeRegressor(max_depth=3, random_state=42).fit(X_train, y_train)
fig2, ax2 = plt.subplots(figsize=(18, 7))
plot_tree(shallow, feature_names=features, filled=True, rounded=True,
          impurity=False, ax=ax2, fontsize=9)
ax2.set_title("Decision Tree (max_depth=3) — for interpretability", fontsize=12)
plt.tight_layout()
plt.savefig("03b_tree_structure.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Plot saved → 03b_tree_structure.png")
