# ============================================================
# STEP 1 — DATA PREPROCESSING
# Cleans both datasets and saves them as cleaned CSVs
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

os.makedirs("data", exist_ok=True)

# ─────────────────────────────────────────
# 1A. LOAD DATASETS
# ─────────────────────────────────────────
heart_df  = pd.read_csv("data/heart_disease_uci.csv")
diabetes_df = pd.read_csv("data/diabetes.csv")

print("=" * 50)
print("RAW SHAPES")
print(f"  Heart Disease : {heart_df.shape}")
print(f"  Diabetes      : {diabetes_df.shape}")
print("=" * 50)


# ─────────────────────────────────────────
# 1B. CLEAN HEART DISEASE DATASET
# ─────────────────────────────────────────
print("\n[HEART DISEASE] Cleaning...")

# Drop columns with too many missing values (>50%)
heart_df.drop(columns=["id", "dataset", "ca", "thal", "slope"], inplace=True)

# Convert bool-like columns
heart_df["fbs"]   = heart_df["fbs"].map({True: 1, False: 0, "True": 1, "False": 0})
heart_df["exang"] = heart_df["exang"].map({True: 1, False: 0, "True": 1, "False": 0})

# Encode string categoricals
le = LabelEncoder()
for col in ["sex", "cp", "restecg"]:
    heart_df[col] = le.fit_transform(heart_df[col].astype(str))

# Fill remaining numeric missing values with median
for col in heart_df.select_dtypes(include=[np.number]).columns:
    heart_df[col] = heart_df[col].fillna(heart_df[col].median())

# Convert target: 0 = no disease, 1 = disease (num > 0)
heart_df["target"] = (heart_df["num"] > 0).astype(int)
heart_df.drop(columns=["num"], inplace=True)

# Add disease label column
heart_df["disease_type"] = "heart_disease"

print(f"  Cleaned shape : {heart_df.shape}")
print(f"  Missing values: {heart_df.isnull().sum().sum()}")
print(f"  Target dist   : {heart_df['target'].value_counts().to_dict()}")


# ─────────────────────────────────────────
# 1C. CLEAN DIABETES DATASET
# ─────────────────────────────────────────
print("\n[DIABETES] Cleaning...")

# Columns where 0 is biologically impossible — treat as missing
zero_invalid = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in zero_invalid:
    diabetes_df[col] = diabetes_df[col].replace(0, np.nan)

# Fill missing with median
for col in diabetes_df.select_dtypes(include=[np.number]).columns:
    diabetes_df[col] = diabetes_df[col].fillna(diabetes_df[col].median())

# Rename target column for consistency
diabetes_df.rename(columns={"Outcome": "target"}, inplace=True)

# Add disease label column
diabetes_df["disease_type"] = "diabetes"

print(f"  Cleaned shape : {diabetes_df.shape}")
print(f"  Missing values: {diabetes_df.isnull().sum().sum()}")
print(f"  Target dist   : {diabetes_df['target'].value_counts().to_dict()}")


# ─────────────────────────────────────────
# 1D. SAVE CLEANED DATASETS
# ─────────────────────────────────────────
heart_df.to_csv("data/heart_cleaned.csv", index=False)
diabetes_df.to_csv("data/diabetes_cleaned.csv", index=False)

print("\n[DONE] Cleaned files saved:")
print("  → data/heart_cleaned.csv")
print("  → data/diabetes_cleaned.csv")
print("=" * 50)