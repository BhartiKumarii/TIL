"""
K-Nearest Neighbors (KNN) Classification
==========================================
Classifying iris flowers (3 classes).
Covers:
  • How KNN works (distance-based majority vote)
  • Effect of k on decision boundary (bias/variance)
  • Choosing optimal k via cross-validation
  • Decision boundary visualization
  • Scaling matters for KNN!
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ── 1. Data ───────────────────────────────────────────────────────────────────
iris = load_iris()
X_full = iris.data        # 4 features
y      = iris.target
names  = iris.target_names

print("=" * 50)
print("   DATASET: Iris Flowers")
print("=" * 50)
print(f"  Samples  : {X_full.shape[0]}")
print(f"  Features : {list(iris.feature_names)}")
print(f"  Classes  : {list(names)}")

# ── 2. Scale (CRITICAL for KNN) ───────────────────────────────────────────────
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_full)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ── 3. Cross-validate to choose best k ───────────────────────────────────────
k_range  = range(1, 31)
cv_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    cv  = cross_val_score(knn, X_train, y_train, cv=5, scoring="accuracy")
    cv_scores.append(cv.mean())

best_k  = k_range[np.argmax(cv_scores)]
best_cv = max(cv_scores)

print(f"\n  Best k (5-fold CV): {best_k}  (CV accuracy = {best_cv:.4f})")

# ── 4. Final model ────────────────────────────────────────────────────────────
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
test_acc = (y_pred == y_test).mean()

print("\n" + "=" * 50)
print("   KNN CLASSIFICATION — RESULTS")
print("=" * 50)
print(f"  Best k   : {best_k}")
print(f"  Test Acc : {test_acc:.4f} ({100*test_acc:.1f}%)")
print()
print(classification_report(y_test, y_pred, target_names=names))

# ── 5. Decision boundary (using 2 most important features for 2D viz) ─────────
# Use petal length (idx 2) and petal width (idx 3) — most discriminative
X2_train = X_train[:, 2:4]
X2_test  = X_test[:,  2:4]

knn_2d = KNeighborsClassifier(n_neighbors=best_k)
knn_2d.fit(X2_train, y_train)

def plot_decision_boundary(ax, model, X, y, title, k):
    h = 0.02
    x_min, x_max = X[:, 0].min()-0.5, X[:, 0].max()+0.5
    y_min, y_max = X[:, 1].min()-0.5, X[:, 1].max()+0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                          np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cmap_bg   = ListedColormap(["#AEC6CF", "#B5E8B0", "#F4B8A0"])
    cmap_pts  = ListedColormap(["#1F77B4", "#2CA02C", "#D62728"])

    ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_bg)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_pts,
                         edgecolors="k", linewidths=0.4, s=40, zorder=3)
    ax.set_title(title)
    ax.set_xlabel("Petal Length (scaled)")
    ax.set_ylabel("Petal Width (scaled)")

# ── 6. Plots ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.suptitle("KNN Classification — Iris Flowers", fontsize=13, fontweight="bold")

# Cross-val score vs k
ax1 = fig.add_subplot(2, 3, 1)
ax1.plot(list(k_range), cv_scores, "o-", color="#4C72B0")
ax1.axvline(best_k, color="#C44E52", ls="--", label=f"Best k={best_k}")
ax1.set_xlabel("k (number of neighbors)")
ax1.set_ylabel("CV Accuracy")
ax1.set_title("Choosing k via Cross-Validation")
ax1.legend()

# Confusion matrix
ax2 = fig.add_subplot(2, 3, 2)
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=names).plot(ax=ax2, colorbar=False)
ax2.set_title(f"Confusion Matrix (k={best_k})")

# Decision boundaries for k=1, 5, best_k
boundary_ks = [1, 5, best_k] if best_k not in [1, 5] else [1, 5, 15]
axes_bd = [fig.add_subplot(2, 3, i) for i in [4, 5, 6]]
for ax, k in zip(axes_bd, boundary_ks):
    m = KNeighborsClassifier(n_neighbors=k).fit(X2_train, y_train)
    acc = m.score(X2_test, y_test)
    plot_decision_boundary(ax, m, X2_train, y_train,
                           f"k={k}  (test acc={acc:.2f})", k)

# Legend patch
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=n)
                   for c, n in zip(["#1F77B4","#2CA02C","#D62728"], names)]
fig.legend(handles=legend_elements, loc="upper right", fontsize=9)

# Hide empty subplot
fig.add_subplot(2, 3, 3).axis("off")

plt.tight_layout()
plt.savefig("06_knn_classification.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  Plot saved → 06_knn_classification.png")
