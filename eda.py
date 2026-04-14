"""
DSA210 - Exploratory Data Analysis (EDA)
Produces visualisations saved to reports/figures/.
Run after data_preprocessing.py.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Setup ─────────────────────────────────────────────────────────────────────
PROC_DIR   = os.path.join(os.path.dirname(__file__), "data", "processed")
FIG_DIR    = os.path.join(os.path.dirname(__file__), "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE = sns.color_palette("muted")

df = pd.read_csv(os.path.join(PROC_DIR, "customers_processed.csv"))

EDU_ORDER = ["Basic", "2n Cycle", "Graduation", "Master", "PhD"]

# ── Helper ────────────────────────────────────────────────────────────────────
def save(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Dataset overview
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── 1. Dataset overview ──")
print(df[["Age", "Income", "TotalSpend", "CampaignScore", "Recency",
          "Frequency", "RelativeIncome", "CPI_at_enrollment"]].describe().round(2))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Education distribution
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── 2. Education distribution ──")
edu_counts = df["Education"].value_counts().reindex(EDU_ORDER)
print(edu_counts)

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(edu_counts.index, edu_counts.values, color=PALETTE[:5], edgecolor="white")
ax.bar_label(bars, padding=3, fontsize=10)
ax.set(title="Customer Count by Education Level", xlabel="Education", ylabel="Count")
save("01_education_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Total spend by education
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── 3. Total spend by education ──")
spend_edu = df.groupby("Education")["TotalSpend"].agg(["mean", "median"]).reindex(EDU_ORDER)
print(spend_edu.round(0))

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(EDU_ORDER))
w = 0.35
ax.bar(x - w/2, spend_edu["mean"],   w, label="Mean",   color=PALETTE[0], edgecolor="white")
ax.bar(x + w/2, spend_edu["median"], w, label="Median", color=PALETTE[1], edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(EDU_ORDER)
ax.set(title="Total Spend by Education Level", xlabel="Education", ylabel="Spend ($)")
ax.legend()
save("02_spend_by_education.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Campaign engagement rate by education
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── 4. Campaign engagement rate by education ──")
eng_rate = df.groupby("Education")["CampaignEngaged"].mean().reindex(EDU_ORDER) * 100
print(eng_rate.round(1))

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(eng_rate.index, eng_rate.values, color=PALETTE[2], edgecolor="white")
ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=10)
ax.set(title="Campaign Engagement Rate by Education Level",
       xlabel="Education", ylabel="% Responded to ≥1 Campaign")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
save("03_campaign_engagement_by_education.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. The PhD paradox: spend vs. campaign engagement (scatter)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── 5. PhD paradox scatter ──")
fig, ax = plt.subplots(figsize=(8, 5))
for edu, grp in df.groupby("Education"):
    ax.scatter(grp["TotalSpend"], grp["CampaignScore"],
               alpha=0.35, s=18, label=edu)
ax.set(title="Total Spend vs. Campaign Score by Education",
       xlabel="Total Spend ($)", ylabel="Campaign Score (0–6)")
ax.legend(title="Education", fontsize=9)
save("04_spend_vs_campaign_scatter.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Income distribution by education (box)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── 6. Income by education ──")
fig, ax = plt.subplots(figsize=(8, 5))
df_edu = df[df["Education"].isin(EDU_ORDER)].copy()
df_edu["Education"] = pd.Categorical(df_edu["Education"], categories=EDU_ORDER, ordered=True)
sns.boxplot(data=df_edu, x="Education", y="Income", order=EDU_ORDER,
            palette="muted", ax=ax, flierprops={"markersize": 3})
ax.set(title="Income Distribution by Education Level",
       xlabel="Education", ylabel="Income ($)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
save("05_income_by_education.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Relative income vs. campaign score (PhD focus)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── 7. Relative income vs campaign score (PhD) ──")
phd = df[df["Education"] == "PhD"].copy()
fig, ax = plt.subplots(figsize=(7, 4))
sns.boxplot(data=phd, x="CampaignScore", y="RelativeIncome",
            palette="Blues", ax=ax)
ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="Peer median")
ax.set(title="PhD Customers: Relative Income vs. Campaign Score",
       xlabel="Campaign Score (0–6)", ylabel="Income / Education Peer Median")
ax.legend()
save("06_phd_relative_income_vs_campaign.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. CPI at enrollment vs. campaign engagement
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── 8. CPI at enrollment vs. campaign engagement ──")
fig, ax = plt.subplots(figsize=(8, 5))
for engaged, grp in df.groupby("CampaignEngaged"):
    label = "Engaged (≥1 campaign)" if engaged else "Not Engaged"
    ax.hist(grp["CPI_at_enrollment"].dropna(), bins=20, alpha=0.6, label=label)
ax.set(title="CPI at Enrollment: Engaged vs. Not Engaged Customers",
       xlabel="CPI at Enrollment", ylabel="Count")
ax.legend()
save("07_cpi_vs_engagement.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Target segment breakdown
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── 9. Target segment (high-potential & under-engaged) ──")
seg = df.groupby("Education")["TargetSegment"].agg(["sum", "count"])
seg["pct"] = seg["sum"] / seg["count"] * 100
seg = seg.reindex(EDU_ORDER)
print(seg.round(1))

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(seg.index, seg["pct"], color=PALETTE[3], edgecolor="white")
ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=10)
ax.set(title="High-Potential & Under-Engaged Customers by Education",
       xlabel="Education", ylabel="% of Group")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
save("08_target_segment_by_education.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Correlation heatmap (key features)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── 10. Correlation heatmap ──")
corr_cols = ["Age", "Income", "RelativeIncome", "EduTier", "TotalSpend",
             "CampaignScore", "Recency", "Frequency", "CPI_at_enrollment",
             "HasChild", "TargetSegment"]
corr = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, linewidths=0.5, ax=ax, annot_kws={"size": 8})
ax.set_title("Correlation Heatmap — Key Features")
save("09_correlation_heatmap.png")

print("\n✓ EDA complete. All figures saved to reports/figures/")
