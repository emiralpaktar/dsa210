"""
DSA210 - Hypothesis Tests
Tests whether education level, income, and CPI at enrollment
explain campaign disengagement among high-spending customers.
Run after data_preprocessing.py.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── Setup ─────────────────────────────────────────────────────────────────────
PROC_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
df = pd.read_csv(os.path.join(PROC_DIR, "customers_processed.csv"))

EDU_ORDER  = ["Basic", "2n Cycle", "Graduation", "Master", "PhD"]
LOW_EDU    = ["Basic", "2n Cycle"]
HIGH_EDU   = ["Graduation", "Master", "PhD"]

def separator(title):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print('═'*60)

def result(reject, p, alpha=0.05):
    decision = "REJECT H₀" if reject else "FAIL TO REJECT H₀"
    sig = "✓ Significant" if reject else "✗ Not significant"
    print(f"  p-value  : {p:.4f}  (α = {alpha})")
    print(f"  Decision : {decision}  →  {sig}")


# ═══════════════════════════════════════════════════════════════════════════════
# H1 — Does education level affect campaign engagement rate?
#      Chi-square test of independence: Education × CampaignEngaged
# ═══════════════════════════════════════════════════════════════════════════════
separator("H1 — Education level vs. campaign engagement")
print("  H₀: Campaign engagement is independent of education level")
print("  H₁: Campaign engagement differs across education levels")

contingency = pd.crosstab(df["Education"], df["CampaignEngaged"])
print("\n  Contingency table (Not Engaged | Engaged):")
print(contingency.reindex(EDU_ORDER))

chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)
print(f"\n  χ²({dof}) = {chi2:.3f}")
result(p_chi2 < 0.05, p_chi2)

# Effect size: Cramér's V
n = contingency.sum().sum()
cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
print(f"  Cramér's V : {cramers_v:.3f}  ({'small' if cramers_v < 0.1 else 'medium' if cramers_v < 0.3 else 'large'} effect)")


# ═══════════════════════════════════════════════════════════════════════════════
# H2 — Do PhD customers spend more than non-PhD customers?
#      Mann-Whitney U test (non-parametric, spend is skewed)
# ═══════════════════════════════════════════════════════════════════════════════
separator("H2 — PhD customers vs. others: total spend")
print("  H₀: Median total spend is equal for PhD and non-PhD customers")
print("  H₁: PhD customers have higher median total spend")

phd_spend     = df[df["Education"] == "PhD"]["TotalSpend"]
non_phd_spend = df[df["Education"] != "PhD"]["TotalSpend"]

print(f"\n  PhD     — n={len(phd_spend):,}, median=${phd_spend.median():.0f}, mean=${phd_spend.mean():.0f}")
print(f"  Non-PhD — n={len(non_phd_spend):,}, median=${non_phd_spend.median():.0f}, mean=${non_phd_spend.mean():.0f}")

u_stat, p_mw = stats.mannwhitneyu(phd_spend, non_phd_spend, alternative="greater")
print(f"\n  Mann-Whitney U = {u_stat:.0f}")
result(p_mw < 0.05, p_mw)

# Effect size: rank-biserial correlation
n1, n2 = len(phd_spend), len(non_phd_spend)
r_rb = 1 - (2 * u_stat) / (n1 * n2)
print(f"  Rank-biserial r : {r_rb:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# H3 — Among PhD customers, do campaign-engaged and non-engaged differ in spend?
#      Mann-Whitney U test
# ═══════════════════════════════════════════════════════════════════════════════
separator("H3 — PhD: engaged vs. non-engaged customers: total spend")
print("  H₀: Among PhD customers, spend is equal regardless of campaign engagement")
print("  H₁: Non-engaged PhD customers spend more (campaign-immune high spenders)")

phd = df[df["Education"] == "PhD"]
phd_engaged     = phd[phd["CampaignEngaged"] == 1]["TotalSpend"]
phd_not_engaged = phd[phd["CampaignEngaged"] == 0]["TotalSpend"]

print(f"\n  Engaged     — n={len(phd_engaged):,}, median=${phd_engaged.median():.0f}")
print(f"  Not Engaged — n={len(phd_not_engaged):,}, median=${phd_not_engaged.median():.0f}")

u_stat3, p_mw3 = stats.mannwhitneyu(phd_not_engaged, phd_engaged, alternative="greater")
print(f"\n  Mann-Whitney U = {u_stat3:.0f}")
result(p_mw3 < 0.05, p_mw3)


# ═══════════════════════════════════════════════════════════════════════════════
# H4 — Does relative income differ between campaign-engaged and non-engaged
#      customers within the PhD group?
#      Mann-Whitney U test
# ═══════════════════════════════════════════════════════════════════════════════
separator("H4 — PhD: relative income vs. campaign engagement")
print("  H₀: Relative income is equal for engaged and non-engaged PhD customers")
print("  H₁: Non-engaged PhD customers have higher relative income")

phd_eng_ri  = phd[phd["CampaignEngaged"] == 1]["RelativeIncome"]
phd_neng_ri = phd[phd["CampaignEngaged"] == 0]["RelativeIncome"]

print(f"\n  Engaged     — median relative income = {phd_eng_ri.median():.3f}")
print(f"  Not Engaged — median relative income = {phd_neng_ri.median():.3f}")

u_stat4, p_mw4 = stats.mannwhitneyu(phd_neng_ri, phd_eng_ri, alternative="greater")
print(f"\n  Mann-Whitney U = {u_stat4:.0f}")
result(p_mw4 < 0.05, p_mw4)


# ═══════════════════════════════════════════════════════════════════════════════
# H5 — Does CPI at enrollment differ between campaign-engaged and
#      non-engaged customers (full dataset)?
#      Mann-Whitney U test
# ═══════════════════════════════════════════════════════════════════════════════
separator("H5 — CPI at enrollment vs. campaign engagement")
print("  H₀: CPI at enrollment is equal for engaged and non-engaged customers")
print("  H₁: CPI at enrollment differs between the two groups")

cpi_eng  = df[df["CampaignEngaged"] == 1]["CPI_at_enrollment"].dropna()
cpi_neng = df[df["CampaignEngaged"] == 0]["CPI_at_enrollment"].dropna()

print(f"\n  Engaged     — n={len(cpi_eng):,}, median CPI = {cpi_eng.median():.2f}")
print(f"  Not Engaged — n={len(cpi_neng):,}, median CPI = {cpi_neng.median():.2f}")

u_stat5, p_mw5 = stats.mannwhitneyu(cpi_eng, cpi_neng, alternative="two-sided")
print(f"\n  Mann-Whitney U = {u_stat5:.0f}")
result(p_mw5 < 0.05, p_mw5)


# ═══════════════════════════════════════════════════════════════════════════════
# H6 — Kruskal-Wallis: does campaign score differ across education groups?
#      Non-parametric one-way ANOVA equivalent
# ═══════════════════════════════════════════════════════════════════════════════
separator("H6 — Campaign score across all education groups (Kruskal-Wallis)")
print("  H₀: Campaign score distribution is the same across all education levels")
print("  H₁: At least one education group differs in campaign score")

groups = [df[df["Education"] == edu]["CampaignScore"].values for edu in EDU_ORDER]
for edu, grp in zip(EDU_ORDER, groups):
    print(f"  {edu:<12} — median={np.median(grp):.2f}, mean={np.mean(grp):.2f}")

h_stat, p_kw = stats.kruskal(*groups)
print(f"\n  Kruskal-Wallis H = {h_stat:.3f}")
result(p_kw < 0.05, p_kw)

# Post-hoc pairwise Mann-Whitney with Bonferroni correction
print("\n  Post-hoc pairwise Mann-Whitney (Bonferroni corrected):")
n_pairs = len(EDU_ORDER) * (len(EDU_ORDER) - 1) // 2
alpha_bonf = 0.05 / n_pairs
print(f"  Bonferroni α = {alpha_bonf:.4f}  ({n_pairs} comparisons)\n")

for i in range(len(EDU_ORDER)):
    for j in range(i + 1, len(EDU_ORDER)):
        e1, e2 = EDU_ORDER[i], EDU_ORDER[j]
        g1 = df[df["Education"] == e1]["CampaignScore"]
        g2 = df[df["Education"] == e2]["CampaignScore"]
        u, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        sig = "✓" if p < alpha_bonf else " "
        print(f"  {sig} {e1:<12} vs {e2:<12} — p={p:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# H7 — Does TargetSegment rate differ across education groups?
#      Chi-square test of independence: Education × TargetSegment
# ═══════════════════════════════════════════════════════════════════════════════
separator("H7 — TargetSegment rate across education groups (χ²)")
print("  H₀: TargetSegment membership is independent of education level")
print("  H₁: TargetSegment rate differs across education levels")
print("  (Motivation: Basic=24.1%, 2n Cycle=12.4%, Graduation=11.5%,")
print("               Master=11.4%, PhD=10.9% — is this gap significant?)")

contingency7 = pd.crosstab(df["Education"], df["TargetSegment"])
print("\n  Contingency table (Not Target | Target):")
print(contingency7.reindex(EDU_ORDER))

chi2_7, p_chi2_7, dof7, _ = stats.chi2_contingency(contingency7)
print(f"\n  χ²({dof7}) = {chi2_7:.3f}")
result(p_chi2_7 < 0.05, p_chi2_7)

n7 = contingency7.sum().sum()
cramers_v7 = np.sqrt(chi2_7 / (n7 * (min(contingency7.shape) - 1)))
print(f"  Cramér's V : {cramers_v7:.3f}  ({'small' if cramers_v7 < 0.1 else 'medium' if cramers_v7 < 0.3 else 'large'} effect)")


# ═══════════════════════════════════════════════════════════════════════════════
# H8 — Do lower-education high-spenders engage less with campaigns than
#      higher-education high-spenders?
#      Mann-Whitney U test on CampaignScore among high-spending customers only
# ═══════════════════════════════════════════════════════════════════════════════
separator("H8 — Low vs. high education: campaign engagement among high spenders")
print("  H₀: Among high-spending customers, campaign score is equal across")
print("      low-education (Basic, 2n Cycle) and high-education groups")
print("  H₁: High-spending low-education customers have lower campaign scores")
print("      (i.e., engagement does not scale uniformly with education)")

high_spenders = df[df["HighPotential"] == 1]
low_edu_cs  = high_spenders[high_spenders["Education"].isin(LOW_EDU)]["CampaignScore"]
high_edu_cs = high_spenders[high_spenders["Education"].isin(HIGH_EDU)]["CampaignScore"]

print(f"\n  Low education  (Basic, 2n Cycle) — n={len(low_edu_cs):,}, median={low_edu_cs.median():.2f}")
print(f"  High education (Grad, Master, PhD) — n={len(high_edu_cs):,}, median={high_edu_cs.median():.2f}")

u_stat8, p_mw8 = stats.mannwhitneyu(low_edu_cs, high_edu_cs, alternative="less")
print(f"\n  Mann-Whitney U = {u_stat8:.0f}")
result(p_mw8 < 0.05, p_mw8)

n8a, n8b = len(low_edu_cs), len(high_edu_cs)
r_rb8 = 1 - (2 * u_stat8) / (n8a * n8b)
print(f"  Rank-biserial r : {r_rb8:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# H9 — Is having children associated with TargetSegment membership?
#      Chi-square test of independence: HasChild × TargetSegment
# ═══════════════════════════════════════════════════════════════════════════════
separator("H9 — HasChild vs. TargetSegment membership (χ²)")
print("  H₀: TargetSegment membership is independent of having children")
print("  H₁: Customers with children are more/less likely to be in TargetSegment")
print("  (Motivation: HasChild correlates -0.52 with TotalSpend in heatmap)")

contingency9 = pd.crosstab(df["HasChild"], df["TargetSegment"])
contingency9.index = ["No Children", "Has Children"]
contingency9.columns = ["Not Target", "Target"]
print("\n  Contingency table:")
print(contingency9)

chi2_9, p_chi2_9, dof9, _ = stats.chi2_contingency(contingency9)
print(f"\n  χ²({dof9}) = {chi2_9:.3f}")
result(p_chi2_9 < 0.05, p_chi2_9)

n9 = contingency9.sum().sum()
cramers_v9 = np.sqrt(chi2_9 / (n9 * (min(contingency9.shape) - 1)))
print(f"  Cramér's V : {cramers_v9:.3f}  ({'small' if cramers_v9 < 0.1 else 'medium' if cramers_v9 < 0.3 else 'large'} effect)")

# Observed rates
for label, row in contingency9.iterrows():
    rate = row["Target"] / row.sum() * 100
    print(f"  {label:<15} — TargetSegment rate: {rate:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# H10 — Does total spend differ across all education groups?
#       Kruskal-Wallis + post-hoc (non-parametric one-way ANOVA on spend)
# ═══════════════════════════════════════════════════════════════════════════════
separator("H10 — Total spend across all education groups (Kruskal-Wallis)")
print("  H₀: Total spend distribution is the same across all education levels")
print("  H₁: At least one education group differs in total spend")

spend_groups = [df[df["Education"] == edu]["TotalSpend"].values for edu in EDU_ORDER]
for edu, grp in zip(EDU_ORDER, spend_groups):
    print(f"  {edu:<12} — median=${np.median(grp):.0f}, mean=${np.mean(grp):.0f}")

h_stat10, p_kw10 = stats.kruskal(*spend_groups)
print(f"\n  Kruskal-Wallis H = {h_stat10:.3f}")
result(p_kw10 < 0.05, p_kw10)

# Post-hoc pairwise Mann-Whitney with Bonferroni correction
print("\n  Post-hoc pairwise Mann-Whitney (Bonferroni corrected):")
n_pairs10 = len(EDU_ORDER) * (len(EDU_ORDER) - 1) // 2
alpha_bonf10 = 0.05 / n_pairs10
print(f"  Bonferroni α = {alpha_bonf10:.4f}  ({n_pairs10} comparisons)\n")

for i in range(len(EDU_ORDER)):
    for j in range(i + 1, len(EDU_ORDER)):
        e1, e2 = EDU_ORDER[i], EDU_ORDER[j]
        g1 = df[df["Education"] == e1]["TotalSpend"]
        g2 = df[df["Education"] == e2]["TotalSpend"]
        u, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        sig = "✓" if p < alpha_bonf10 else " "
        print(f"  {sig} {e1:<12} vs {e2:<12} — p={p:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
separator("SUMMARY")
results_summary = [
    ("H1",  "Education × Campaign engagement (χ²)",            p_chi2),
    ("H2",  "PhD vs. non-PhD total spend (MW)",                 p_mw),
    ("H3",  "PhD engaged vs. non-engaged spend (MW)",           p_mw3),
    ("H4",  "PhD engaged vs. non-engaged relative income (MW)", p_mw4),
    ("H5",  "CPI at enrollment × engagement (MW)",              p_mw5),
    ("H6",  "Campaign score across education (KW)",             p_kw),
    ("H7",  "TargetSegment rate across education (χ²)",         p_chi2_7),
    ("H8",  "Low vs. high edu campaign score — high spenders",  p_mw8),
    ("H9",  "HasChild × TargetSegment membership (χ²)",         p_chi2_9),
    ("H10", "Total spend across education groups (KW)",         p_kw10),
]
print(f"\n  {'Test':<6} {'Description':<48} {'p-value':<10} {'Significant?'}")
print("  " + "-"*75)
for test, desc, p in results_summary:
    sig = "Yes ✓" if p < 0.05 else "No  ✗"
    print(f"  {test:<6} {desc:<48} {p:<10.4f} {sig}")

print("\n✓ Hypothesis tests complete.")
