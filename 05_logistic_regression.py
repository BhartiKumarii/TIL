"""
Logistic Regression
====================
Binary classification: Predict whether a bank customer will churn.
Covers:
  • Sigmoid function intuition
  • Model fitting, probability outputs
  • Confusion matrix, precision, recall, F1, ROC-AUC
  • Decision threshold tuning
  • Coefficient interpretation (log-odds → odds ratios)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve, ConfusionMatrixDisplay
)

# ── 1. Synthetic churn dataset ────────────────────────────────────────────────
np.random.seed(42)
n = 1500

age           = np.random.randint(18, 70, n)
balance       = np.random.exponential(25_000, n)
num_products  = np.random.choice([1, 2, 3, 4], n, p=[0.5, 0.35, 0.1, 0.05])
is_active     = np.random.choice([0, 1], n, p=[0.35, 0.65])
credit_score  = np.random.randint(300, 850, n)
tenure_years  = np.random.randint(0, 10, n)

# Log-odds linear combination (true model)
log_odds = (
    -2.0
    + 0.02  * (age - 40)
    - 0.000008 * balance
    + 0.5   * (num_products - 2)
    - 1.2   * is_active
    - 0.003 * (credit_score - 600)
    - 0.05  * tenure_years
)
prob_churn = 1 / (1 + np.exp(-log_odds))
churn = (np.random.rand(n) < prob_churn).astype(int)

df = pd.DataFrame({
    "age": age, "balance": balance, "num_products": num_products,
    "is_active": is_active, "credit_score": credit_score,
    "tenure_years": tenure_years, "churn": churn,
})

print("=" * 50)
print("   DATASET: Bank Customer Churn")
print("=" * 50)
print(f"  Total customers : {n}")
print(f"  Churned         : {churn.sum()} ({100*churn.mean():.1f}%)")

features = ["age", "balance", "num_products", "is_active", "credit_score", "tenure_years"]
X = df[features].values
y = df["churn"].values

# ── 2. Preprocess + split ─────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ── 3. Fit ────────────────────────────────────────────────────────────────────
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred      = model.predict(X_test)
y_prob      = model.predict_proba(X_test)[:, 1]

# ── 4. Metrics ────────────────────────────────────────────────────────────────
auc = roc_auc_score(y_test, y_prob)
print("\n" + "=" * 50)
print("   LOGISTIC REGRESSION — RESULTS")
print("=" * 50)
print(f"  ROC-AUC : {auc:.4f}")
print()
print(classification_report(y_test, y_pred, target_names=["Stay", "Churn"]))

# Coefficient → odds ratio
coefs = model.coef_[0]
odds_ratios = np.exp(coefs)
print("  Feature coefficients (scaled) → Odds Ratios:")
for feat, c, o in sorted(zip(features, coefs, odds_ratios), key=lambda x: abs(x[1]), reverse=True):
    direction = "↑ churn risk" if c > 0 else "↓ churn risk"
    print(f"    {feat:<18}  coef={c:+.3f}  OR={o:.3f}  {direction}")

# ── 5. Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("Logistic Regression — Bank Churn Prediction", fontsize=13, fontweight="bold")

# Sigmoid function
ax = axes[0, 0]
z  = np.linspace(-6, 6, 300)
ax.plot(z, 1/(1+np.exp(-z)), color="#4C72B0", lw=2.5)
ax.axhline(0.5, color="gray", ls="--", lw=1)
ax.axvline(0,   color="gray", ls="--", lw=1)
ax.fill_between(z, 0.5, 1/(1+np.exp(-z)),
                where=(z>0), alpha=0.15, color="#C44E52", label="Predict churn")
ax.fill_between(z, 0, 1/(1+np.exp(-z)),
                where=(z<0), alpha=0.15, color="#4C72B0", label="Predict stay")
ax.set_title("Sigmoid (logistic) Function")
ax.set_xlabel("log-odds  z = wᵀx + b")
ax.set_ylabel("P(churn = 1)")
ax.legend()

# Confusion matrix
ax = axes[0, 1]
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Stay", "Churn"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix")

# ROC curve
ax = axes[1, 0]
fpr, tpr, _ = roc_curve(y_test, y_prob)
ax.plot(fpr, tpr, color="#4C72B0", lw=2, label=f"Logistic Reg (AUC={auc:.3f})")
ax.plot([0,1],[0,1],"--", color="gray", label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()

# Probability distribution by true class
ax = axes[1, 1]
ax.hist(y_prob[y_test==0], bins=30, alpha=0.6, color="#4C72B0", label="True: Stay")
ax.hist(y_prob[y_test==1], bins=30, alpha=0.6, color="#C44E52", label="True: Churn")
ax.axvline(0.5, color="black", ls="--", lw=1.5, label="Threshold=0.5")
ax.set_xlabel("Predicted Probability of Churn")
ax.set_ylabel("Count")
ax.set_title("Predicted Probability Distribution")
ax.legend()

plt.tight_layout()
plt.savefig("05_logistic_regression.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  Plot saved → 05_logistic_regression.png")
