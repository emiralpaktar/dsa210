"""
DSA210 - Machine Learning Models
Predicts TargetSegment membership (high-potential & under-engaged customers)
using classification and clustering methods.
Run after data_preprocessing.py.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance

# ── Paths ─────────────────────────────────────────────────────────────────────
PROC_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
FIG_DIR  = os.path.join(os.path.dirname(__file__), "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

df = pd.read_csv(os.path.join(PROC_DIR, "customers_processed.csv"))

# ── Feature selection ─────────────────────────────────────────────────────────
FEATURES = [
    "EduTier", "Income", "RelativeIncome", "Age",
    "TotalSpend", "Frequency", "Recency",
    "CPI_at_enrollment", "HasChild"
]
TARGET = "TargetSegment"

df_ml = df[FEATURES + [TARGET]].dropna().copy()
X = df_ml[FEATURES]
y = df_ml[TARGET]

print(f"Dataset: {len(df_ml)} rows | Target class balance: {y.mean()*100:.1f}% positive")

# ── Train / test split (stratified) ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Helper: evaluate a classifier ────────────────────────────────────────────
def evaluate(name, model, X_tr, y_tr, X_te, y_te, scaled=True):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    auc    = roc_auc_score(y_te, y_prob)
    cv     = cross_val_score(model, X_tr, y_tr, cv=StratifiedKFold(5),
                             scoring="roc_auc", n_jobs=-1)
    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    print(classification_report(y_te, y_pred, target_names=["Non-Target","TargetSegment"]))
    print(f"  ROC-AUC (test)  : {auc:.4f}")
    print(f"  ROC-AUC (5-fold CV): {cv.mean():.4f} ± {cv.std():.4f}")
    return model, y_pred, y_prob, auc, cv.mean()

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Logistic Regression (baseline)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n═══ CLASSIFICATION MODELS ═══")
lr, lr_pred, lr_prob, lr_auc, lr_cv = evaluate(
    "Logistic Regression (baseline)",
    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    X_train_sc, y_train, X_test_sc, y_test
)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Decision Tree
# ═══════════════════════════════════════════════════════════════════════════════
dt, dt_pred, dt_prob, dt_auc, dt_cv = evaluate(
    "Decision Tree",
    DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
    X_train, y_train, X_test, y_test, scaled=False
)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Random Forest
# ═══════════════════════════════════════════════════════════════════════════════
rf, rf_pred, rf_prob, rf_auc, rf_cv = evaluate(
    "Random Forest",
    RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced",
                           random_state=42, n_jobs=-1),
    X_train, y_train, X_test, y_test, scaled=False
)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Gradient Boosting
# ═══════════════════════════════════════════════════════════════════════════════
gb, gb_pred, gb_prob, gb_auc, gb_cv = evaluate(
    "Gradient Boosting",
    GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                               random_state=42),
    X_train, y_train, X_test, y_test, scaled=False
)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Model comparison plot
# ═══════════════════════════════════════════════════════════════════════════════
model_names = ["Logistic\nRegression", "Decision\nTree", "Random\nForest", "Gradient\nBoosting"]
auc_scores  = [lr_auc, dt_auc, rf_auc, gb_auc]
cv_scores   = [lr_cv,  dt_cv,  rf_cv,  gb_cv]

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(model_names))
w = 0.35
b1 = ax.bar(x - w/2, auc_scores, w, label="Test AUC",   color=sns.color_palette("muted")[0], edgecolor="white")
b2 = ax.bar(x + w/2, cv_scores,  w, label="CV AUC (5-fold)", color=sns.color_palette("muted")[1], edgecolor="white")
ax.bar_label(b1, fmt="%.3f", padding=3, fontsize=9)
ax.bar_label(b2, fmt="%.3f", padding=3, fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(model_names)
ax.set_ylim(0, 1.0)
ax.axhline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.6, label="Random baseline")
ax.set(title="Classification Model Comparison — ROC-AUC",
       xlabel="Model", ylabel="ROC-AUC")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "13_model_comparison_auc.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\n  Saved → 13_model_comparison_auc.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. ROC curves (all models)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5))
for name, prob, auc_val in [
    ("Logistic Regression", lr_prob, lr_auc),
    ("Decision Tree",       dt_prob, dt_auc),
    ("Random Forest",       rf_prob, rf_auc),
    ("Gradient Boosting",   gb_prob, gb_auc),
]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc_val:.3f})")

ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
ax.set(title="ROC Curves — All Models",
       xlabel="False Positive Rate", ylabel="True Positive Rate")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "14_roc_curves.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved → 14_roc_curves.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. Confusion matrix — best model (Random Forest)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_test, rf_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Non-Target", "TargetSegment"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix — Random Forest")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "15_confusion_matrix_rf.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved → 15_confusion_matrix_rf.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. Feature importance — Random Forest
# ═══════════════════════════════════════════════════════════════════════════════
importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(7, 5))
importances.plot(kind="barh", ax=ax, color=sns.color_palette("muted")[2], edgecolor="white")
ax.set(title="Feature Importances — Random Forest",
       xlabel="Importance (Gini)", ylabel="Feature")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "16_feature_importance_rf.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved → 16_feature_importance_rf.png")
print("\nFeature importances (Random Forest):")
print(importances.sort_values(ascending=False).round(4))

# ═══════════════════════════════════════════════════════════════════════════════
# 9. K-Means Clustering (unsupervised: segment discovery)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n═══ CLUSTERING (K-MEANS) ═══")

CLUSTER_FEATURES = ["TotalSpend", "CampaignScore", "Income", "RelativeIncome",
                    "EduTier", "Frequency", "CPI_at_enrollment", "HasChild"]
df_cluster = df[CLUSTER_FEATURES + ["Education", "TargetSegment"]].dropna().copy()
X_cl = StandardScaler().fit_transform(df_cluster[CLUSTER_FEATURES])

# Elbow method
inertias = []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_cl)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(list(K_range), inertias, "o-", color=sns.color_palette("muted")[0], lw=2)
ax.set(title="K-Means Elbow Curve", xlabel="Number of Clusters (k)", ylabel="Inertia")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "17_kmeans_elbow.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved → 17_kmeans_elbow.png")

# Fit with k=4 (elbow typically appears here)
k_opt = 4
km = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
df_cluster["Cluster"] = km.fit_predict(X_cl)

cluster_profile = df_cluster.groupby("Cluster")[CLUSTER_FEATURES].mean().round(2)
print(f"\nCluster profiles (k={k_opt}):")
print(cluster_profile.to_string())

cluster_target = df_cluster.groupby("Cluster")["TargetSegment"].mean() * 100
print("\nTargetSegment rate per cluster (%):")
print(cluster_target.round(1))

# ═══════════════════════════════════════════════════════════════════════════════
# 10. PCA + Cluster scatter
# ═══════════════════════════════════════════════════════════════════════════════
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_cl)
var_explained = pca.explained_variance_ratio_ * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: colour by cluster
sc1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1],
                      c=df_cluster["Cluster"], cmap="tab10", alpha=0.5, s=15)
axes[0].set(title=f"PCA — K-Means Clusters (k={k_opt})",
            xlabel=f"PC1 ({var_explained[0]:.1f}% var)",
            ylabel=f"PC2 ({var_explained[1]:.1f}% var)")
plt.colorbar(sc1, ax=axes[0], label="Cluster")

# Right: colour by TargetSegment
colors = df_cluster["TargetSegment"].map({0: "#6aaee0", 1: "#e07b6a"})
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.5, s=15)
from matplotlib.patches import Patch
legend_els = [Patch(facecolor="#e07b6a", label="TargetSegment"),
              Patch(facecolor="#6aaee0", label="Non-Target")]
axes[1].legend(handles=legend_els, fontsize=9)
axes[1].set(title="PCA — TargetSegment Highlighted",
            xlabel=f"PC1 ({var_explained[0]:.1f}% var)",
            ylabel=f"PC2 ({var_explained[1]:.1f}% var)")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "18_pca_clusters.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved → 18_pca_clusters.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 11. Cluster profile heatmap
# ═══════════════════════════════════════════════════════════════════════════════
profile_norm = (cluster_profile - cluster_profile.mean()) / cluster_profile.std()

fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(profile_norm, annot=cluster_profile.values, fmt=".1f",
            cmap="RdBu_r", center=0, linewidths=0.5, ax=ax, annot_kws={"size": 8})
ax.set_title(f"K-Means Cluster Profiles (k={k_opt}) — Normalised")
ax.set_ylabel("Cluster")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "19_cluster_profile_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved → 19_cluster_profile_heatmap.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n═══ ML SUMMARY ═══")
print(f"\n  Best classifier (test AUC): Random Forest  →  AUC = {rf_auc:.4f}")
print(f"  CV AUC (5-fold): {rf_cv:.4f}")
print(f"\n  Key features (top 3): {', '.join(importances.sort_values(ascending=False).index[:3])}")
print(f"\n  Cluster with highest TargetSegment rate: Cluster {cluster_target.idxmax()}"
      f"  ({cluster_target.max():.1f}%)")
print("\n✓ ML models complete. All figures saved to reports/figures/")
