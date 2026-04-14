"""
Decision Trees — Classification
==================================
Classifying mushrooms: poisonous vs edible.
Covers:
  • Gini impurity vs entropy (information gain) as split criteria
  • Pruning via max_depth, min_samples_leaf, ccp_alpha (cost-complexity)
  • Visualising the tree structure
  • Feature importance
  • The bias–variance tradeoff across depths
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay, roc_auc_score)
from sklearn.preprocessing import LabelEncoder

# ── 1. Synthetic "Mushroom" dataset ───────────────────────────────────────────
np.random.seed(99)
X_raw, y = make_classification(
    n_samples=2000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    n_classes=2,
    weights=[0.55, 0.45],   # slightly imbalanced
    random_state=99,
)
feature_names = [
    "cap_diameter", "gill_color_enc", "stalk_height",
    "ring_number", "spore_print_enc", "odor_enc",
    "bruises", "habitat_enc", "season_enc", "cap_shape_enc",
]
class_names = ["Edible", "Poisonous"]

print("=" * 55)
print("  DATASET: Mushroom Classification (synthetic)")
print("=" * 55)
print(f"  Samples  : {X_raw.shape[0]}")
print(f"  Features : {X_raw.shape[1]}")
print(f"  Classes  : {class_names}")
print(f"  Poisonous: {y.sum()} ({100*y.mean():.1f}%)")

X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y
)

# ── 2. Gini vs Entropy ────────────────────────────────────────────────────────
for criterion in ("gini", "entropy"):
    dt = DecisionTreeClassifier(criterion=criterion, max_depth=5, random_state=42)
    cv = cross_val_score(dt, X_train, y_train, cv=5, scoring="roc_auc").mean()
    print(f"  {criterion:<10}  CV ROC-AUC = {cv:.4f}")

# ── 3. Depth sweep ────────────────────────────────────────────────────────────
depths       = list(range(1, 21))
train_accs   = []
test_accs    = []

for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    train_accs.append(dt.score(X_train, y_train))
    test_accs.append( dt.score(X_test,  y_test))

best_depth = depths[np.argmax(test_accs)]
print(f"\n  Best test accuracy at depth={best_depth}: {max(test_accs):.4f}")

# ── 4. Cost-complexity pruning (ccp_alpha) ────────────────────────────────────
full_tree = DecisionTreeClassifier(random_state=42)
full_tree.fit(X_train, y_train)

path    = full_tree.cost_complexity_pruning_path(X_train, y_train)
alphas  = path.ccp_alphas[:-1]
ccp_acc = []
for a in alphas:
    dt = DecisionTreeClassifier(ccp_alpha=a, random_state=42)
    dt.fit(X_train, y_train)
    ccp_acc.append(dt.score(X_test, y_test))

best_alpha = alphas[np.argmax(ccp_acc)]
print(f"  Best ccp_alpha: {best_alpha:.6f}  (test acc={max(ccp_acc):.4f})")

# ── 5. Final model ────────────────────────────────────────────────────────────
final_dt = DecisionTreeClassifier(
    max_depth=best_depth,
    min_samples_leaf=10,
    criterion="gini",
    random_state=42,
)
final_dt.fit(X_train, y_train)
y_pred = final_dt.predict(X_test)
y_prob = final_dt.predict_proba(X_test)[:, 1]
auc    = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 55)
print("   DECISION TREE (FINAL) — RESULTS")
print("=" * 55)
print(f"  Depth    : {best_depth}")
print(f"  Leaves   : {final_dt.get_n_leaves()}")
print(f"  Nodes    : {final_dt.tree_.node_count}")
print(f"  ROC-AUC  : {auc:.4f}")
print()
print(classification_report(y_test, y_pred, target_names=class_names))

# ── 6. Text representation of shallow version ─────────────────────────────────
shallow = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train)
print("\n  Shallow tree (depth=3) — text rules:")
print(export_text(shallow, feature_names=feature_names))

# ── 7. Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle("Decision Trees — Mushroom Classification", fontsize=13, fontweight="bold")

# Depth vs accuracy
ax = axes[0, 0]
ax.plot(depths, train_accs, "o-", color="#4C72B0", label="Train")
ax.plot(depths, test_accs,  "s-", color="#C44E52", label="Test")
ax.axvline(best_depth, color="gray", ls="--", label=f"Best depth={best_depth}")
ax.set_xlabel("max_depth")
ax.set_ylabel("Accuracy")
ax.set_title("Bias–Variance Tradeoff")
ax.legend()

# ccp_alpha pruning
ax = axes[0, 1]
ax.plot(alphas, ccp_acc, "o-", color="#55A868", markersize=3)
ax.axvline(best_alpha, color="red", ls="--", lw=1.5, label=f"Best α={best_alpha:.5f}")
ax.set_xlabel("ccp_alpha (pruning strength)")
ax.set_ylabel("Test Accuracy")
ax.set_title("Cost-Complexity Pruning")
ax.legend()

# Feature importance
ax = axes[0, 2]
importances = pd.Series(final_dt.feature_importances_, index=feature_names).sort_values()
importances.plot(kind="barh", ax=ax, color="#4C72B0")
ax.set_title("Feature Importance")
ax.set_xlabel("Gini Importance")

# Confusion matrix
ax = axes[1, 0]
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, colorbar=False)
ax.set_title("Confusion Matrix")

# Tree visualisation (depth=3 for readability)
ax = axes[1, 1]
plot_tree(shallow, feature_names=feature_names, class_names=class_names,
          filled=True, rounded=True, impurity=True, ax=ax, fontsize=7)
ax.set_title("Shallow Tree Structure (depth=3)")

# Gini impurity formula illustration
ax = axes[1, 2]
p = np.linspace(0, 1, 300)
gini = 2 * p * (1 - p)
entropy = -np.where(p>0, p*np.log2(np.clip(p,1e-9,1)), 0) \
          -np.where((1-p)>0, (1-p)*np.log2(np.clip(1-p,1e-9,1)), 0)
entropy_norm = entropy / entropy.max()   # normalise to same scale

ax.plot(p, gini,         color="#4C72B0", lw=2, label="Gini impurity")
ax.plot(p, entropy_norm, color="#C44E52", lw=2, ls="--", label="Entropy (normalised)")
ax.axvline(0.5, color="gray", ls=":", lw=1)
ax.set_xlabel("p (fraction of class 1)")
ax.set_ylabel("Impurity measure")
ax.set_title("Gini vs Entropy")
ax.legend()

plt.tight_layout()
plt.savefig("07_decision_trees.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  Plot saved → 07_decision_trees.png")
