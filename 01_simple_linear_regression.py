"""
Simple Linear Regression
========================
Predicting house prices based on size (sq ft).
Covers: data generation, train/test split, model fitting,
        evaluation metrics (MSE, R²), and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ── 1. Generate synthetic data ────────────────────────────────────────────────
np.random.seed(42)
n = 200

size_sqft = np.random.uniform(500, 3500, n)
price = 150 * size_sqft + 50_000 + np.random.normal(0, 30_000, n)   # true β=150

X = size_sqft.reshape(-1, 1)
y = price

# ── 2. Train / Test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 3. Fit model ──────────────────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)

coef   = model.coef_[0]
intercept = model.intercept_

print("=" * 55)
print("        SIMPLE LINEAR REGRESSION — RESULTS")
print("=" * 55)
print(f"  Learned coefficient (slope) : {coef:,.2f}  (true: 150)")
print(f"  Learned intercept           : {intercept:,.2f}")
print(f"  Equation : price = {coef:.2f} × size + {intercept:.2f}")

# ── 4. Evaluate ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)

mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print()
print(f"  MSE   : {mse:,.0f}")
print(f"  RMSE  : ${rmse:,.0f}")
print(f"  R²    : {r2:.4f}  (1.0 = perfect)")
print("=" * 55)

# ── 5. Visualise ──────────────────────────────────────────────────────────────
x_line = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
y_line = model.predict(x_line)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Simple Linear Regression — House Price vs. Size",
             fontsize=14, fontweight="bold")

# Scatter + regression line
ax = axes[0]
ax.scatter(X_train, y_train, alpha=0.4, color="#4C72B0", label="Train")
ax.scatter(X_test,  y_test,  alpha=0.6, color="#DD8452", label="Test")
ax.plot(x_line, y_line, color="#C44E52", lw=2.5,
        label=f"Fit: price = {coef:.1f}×size + {intercept:.0f}")
ax.set_xlabel("Size (sq ft)")
ax.set_ylabel("Price ($)")
ax.set_title("Data & Regression Line")
ax.legend()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1e3:.0f}k"))

# Residual plot
ax = axes[1]
residuals = y_test - y_pred
ax.scatter(y_pred, residuals, alpha=0.6, color="#55A868")
ax.axhline(0, color="red", linestyle="--", lw=1.5)
ax.set_xlabel("Predicted Price ($)")
ax.set_ylabel("Residual ($)")
ax.set_title("Residual Plot\n(random scatter = good fit)")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1e3:.0f}k"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1e3:.0f}k"))

plt.tight_layout()
plt.savefig("01_simple_linear_regression.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  Plot saved → 01_simple_linear_regression.png")
