"""
DSA210 - Data Preprocessing
Loads raw data, merges CPI, engineers features, saves cleaned dataset.
"""

import pandas as pd
import numpy as np
import os

# ── Paths ────────────────────────────────────────────────────────────────────
RAW_DIR  = os.path.join(os.path.dirname(__file__), "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

# ── 1. Load raw data ──────────────────────────────────────────────────────────
print("Loading raw data...")
df  = pd.read_csv(os.path.join(RAW_DIR, "marketing_campaign.csv"), sep="\t")
cpi = pd.read_csv(os.path.join(RAW_DIR, "US_inflation_rates.csv"))

print(f"  Customers  : {len(df):,} rows, {len(df.columns)} columns")
print(f"  CPI records: {len(cpi):,} rows")

# ── 2. Parse dates ────────────────────────────────────────────────────────────
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
cpi["date"]       = pd.to_datetime(cpi["date"])

# Create a month-level key for CPI merge
df["enroll_month"] = df["Dt_Customer"].dt.to_period("M")
cpi["enroll_month"] = cpi["date"].dt.to_period("M")
cpi = cpi.rename(columns={"value": "CPI_at_enrollment"})

# ── 3. Merge CPI ──────────────────────────────────────────────────────────────
df = df.merge(
    cpi[["enroll_month", "CPI_at_enrollment"]],
    on="enroll_month",
    how="left"
)
print(f"  CPI merge nulls: {df['CPI_at_enrollment'].isna().sum()}")

# ── 4. Handle missing values ──────────────────────────────────────────────────
# Income: fill 24 missing with median within education group
df["Income"] = df.groupby("Education")["Income"].transform(
    lambda x: x.fillna(x.median())
)
print(f"  Missing Income after fill: {df['Income'].isna().sum()}")

# ── 5. Feature engineering ────────────────────────────────────────────────────
# Age
df["Age"] = 2024 - df["Year_Birth"]

# Total spend across all product categories
spend_cols = ["MntWines", "MntFruits", "MntMeatProducts",
              "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
df["TotalSpend"] = df[spend_cols].sum(axis=1)

# Campaign Engagement Score (0–6)
campaign_cols = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
                 "AcceptedCmp4", "AcceptedCmp5", "Response"]
df["CampaignScore"] = df[campaign_cols].sum(axis=1)
df["CampaignEngaged"] = (df["CampaignScore"] > 0).astype(int)  # binary flag

# RFM metrics
#   Recency  : already in dataset (days since last purchase)
#   Frequency: total number of purchases across all channels
purchase_cols = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
df["Frequency"] = df[purchase_cols].sum(axis=1)
#   Monetary : TotalSpend (already computed)

# Has children flag
df["HasChild"] = ((df["Kidhome"] + df["Teenhome"]) > 0).astype(int)

# Education tier (ordinal)
edu_order = {"Basic": 0, "2n Cycle": 1, "Graduation": 2, "Master": 3, "PhD": 4}
df["EduTier"] = df["Education"].map(edu_order)

# Relative income: customer income vs median income of same education group
df["EduMedianIncome"] = df.groupby("Education")["Income"].transform("median")
df["RelativeIncome"]  = df["Income"] / df["EduMedianIncome"]

# High-potential definition:
#   TotalSpend >= 75th percentile within education group
#   AND CampaignScore == 0  (never responded to any campaign)
spend_q75 = df.groupby("Education")["TotalSpend"].transform(
    lambda x: x.quantile(0.75)
)
df["HighPotential"]    = (df["TotalSpend"] >= spend_q75).astype(int)
df["UnderEngaged"]     = (df["CampaignScore"] == 0).astype(int)
df["TargetSegment"]    = ((df["HighPotential"] == 1) & (df["UnderEngaged"] == 1)).astype(int)

# ── 6. Clean up anomalous values ─────────────────────────────────────────────
# Remove clearly wrong birth years (age > 100)
before = len(df)
df = df[df["Age"] <= 100].copy()
print(f"  Removed {before - len(df)} rows with Age > 100")

# ── 7. Save processed dataset ─────────────────────────────────────────────────
out_path = os.path.join(PROC_DIR, "customers_processed.csv")
df.drop(columns=["enroll_month"]).to_csv(out_path, index=False)
print(f"\nSaved processed data → {out_path}")
print(f"Final shape: {df.shape}")
print(f"\nTarget segment (high-potential & under-engaged): {df['TargetSegment'].sum()} customers")
print(df[["Education", "TargetSegment"]].groupby("Education")["TargetSegment"].sum().sort_values(ascending=False))
