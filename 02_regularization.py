"""
Regularization in Linear Regression
=====================================
OLS  vs  Ridge (L2)  vs  Lasso (L1)

Shows:
  • Why OLS overfits on noisy / high-dim data
  • How Ridge shrinks coefficients uniformly
  • How Lasso drives weak coefficients to exactly zero (feature selection)
  • Cross-validated alpha selection
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ── 1. Synthetic dataset with many features, most irrelevant ──────────────────
np.random.seed(0)
n_samples   = 150
n_features  = 30          # total features
n_informative = 5         # only 5 truly matter

X = np.random.randn(n_samples, n_features)
true_coefs = np.zeros(n_features)
true_coefs[:n_informative] = [3, -2.5, 4, -1.8, 2.2]   # strong signal
y = X @ true_coefs + np.random.randn(n_samples) * 1.5   # noise σ=1.5

# ── 2. Scale + split ──────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)

# ── 3. Fit models ─────────────────────────────────────────────────────────────
ols   = LinearRegression().fit(X_train, y_train)

ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 100), cv=5)
ridge_cv.fit(X_train, y_train)
ridge = Ridge(alpha=ridge_cv.alpha_).fit(X_train, y_train)

lasso_cv = LassoCV(alphas=np.logspace(-3, 1, 100), cv=5, max_iter=10_000)
lasso_cv.fit(X_train, y_train)
lasso = Lasso(alpha=lasso_cv.alpha_, max_iter=10_000).fit(X_train, y_train)

# ── 4. Metrics ────────────────────────────────────────────────────────────────
def rmse(m, Xtr, ytr, Xte, yte):
    tr = np.sqrt(mean_squared_error(ytr, m.predict(Xtr)))
    te = np.sqrt(mean_squared_error(yte, m.predict(Xte)))
    return tr, te

ols_tr,   ols_te   = rmse(ols,   X_train, y_train, X_test, y_test)
ridge_tr, ridge_te = rmse(ridge, X_train, y_train, X_test, y_test)
lasso_tr, lasso_te = rmse(lasso, X_train, y_train, X_test, y_test)

nonzero_lasso = np.sum(np.abs(lasso.coef_) > 1e-4)

print("=" * 60)
print("     REGULARIZATION COMPARISON — OLS vs RIDGE vs LASSO")
print("=" * 60)
print(f"  {'Model':<10}  {'Train RMSE':>12}  {'Test RMSE':>12}  {'Non-zero coefs':>15}")
print("-" * 60)
print(f"  {'OLS':<10}  {ols_tr:>12.4f}  {ols_te:>12.4f}  {n_features:>15}")
print(f"  {'Ridge':<10}  {ridge_tr:>12.4f}  {ridge_te:>12.4f}  {n_features:>15}  (α={ridge_cv.alpha_:.4f})")
print(f"  {'Lasso':<10}  {lasso_tr:>12.4f}  {lasso_te:>12.4f}  {nonzero_lasso:>15}  (α={lasso_cv.alpha_:.4f})")
print("=" * 60)
print(f"\n  True informative features : {n_informative}")
print(f"  Lasso kept               : {nonzero_lasso} features")

# ── 5. Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Regularization: OLS vs Ridge vs Lasso", fontsize=14, fontweight="bold")

feature_ids = np.arange(n_features)
colors = {"OLS": "#C44E52", "Ridge": "#4C72B0", "Lasso": "#55A868"}

for ax, (name, coefs) in zip(axes, [
    ("OLS",   ols.coef_),
    ("Ridge", ridge.coef_),
    ("Lasso", lasso.coef_),
]):
    ax.bar(feature_ids[:n_informative],  coefs[:n_informative],
           color=colors[name], label="Informative")
    ax.bar(feature_ids[n_informative:],  coefs[n_informative:],
           color="lightgray", label="Noise features")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"{name}  (Test RMSE={rmse(locals()[name.lower()], X_train, y_train, X_test, y_test)[1]:.3f})")
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Coefficient value")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("02_regularization.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  Plot saved → 02_regularization.png")

# ── 6. Regularisation path for Lasso ─────────────────────────────────────────
alphas = np.logspace(-2, 1, 200)
coef_path = []
for a in alphas:
    m = Lasso(alpha=a, max_iter=10_000).fit(X_train, y_train)
    coef_path.append(m.coef_.copy())
coef_path = np.array(coef_path)

fig, ax = plt.subplots(figsize=(10, 5))
for i in range(n_features):
    lw = 2.0 if i < n_informative else 0.5
    ls = "-"  if i < n_informative else "--"
    ax.plot(np.log10(alphas), coef_path[:, i], lw=lw, ls=ls)

ax.axvline(np.log10(lasso_cv.alpha_), color="red", ls=":", lw=2,
           label=f"CV-chosen α = {lasso_cv.alpha_:.4f}")
ax.set_xlabel("log₁₀(α)")
ax.set_ylabel("Coefficient value")
ax.set_title("Lasso Regularization Path\n(solid = informative features, dashed = noise)")
ax.legend()
plt.tight_layout()
plt.savefig("02b_lasso_path.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Plot saved → 02b_lasso_path.png")
