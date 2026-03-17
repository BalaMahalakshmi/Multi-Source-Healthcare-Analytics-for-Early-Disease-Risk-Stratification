# ============================================================
# STEP 2 — FEATURE ENGINEERING
# Creates new meaningful health indicators from cleaned data
# ============================================================

import pandas as pd
import numpy as np

# ─────────────────────────────────────────
# LOAD CLEANED DATA
# ─────────────────────────────────────────
heart_df    = pd.read_csv("data/heart_cleaned.csv")
diabetes_df = pd.read_csv("data/diabetes_cleaned.csv")


# ─────────────────────────────────────────
# 2A. HEART DISEASE — NEW FEATURES
# ─────────────────────────────────────────
print("[HEART DISEASE] Engineering features...")

# Age group buckets
def age_group(age):
    if age < 40:   return 0   # Young
    elif age < 55: return 1   # Middle-aged
    elif age < 65: return 2   # Senior
    else:          return 3   # Elderly

heart_df["age_group"] = heart_df["age"].apply(age_group)

# Blood pressure category
def bp_category(bp):
    if bp < 120:   return 0   # Normal
    elif bp < 130: return 1   # Elevated
    elif bp < 140: return 2   # Stage 1 hypertension
    else:          return 3   # Stage 2 hypertension

heart_df["bp_category"] = heart_df["trestbps"].apply(bp_category)

# Cholesterol risk flag (>200 mg/dL is borderline high)
heart_df["high_cholesterol"] = (heart_df["chol"] > 200).astype(int)

# Max heart rate risk (low max HR is a risk indicator)
heart_df["low_max_hr"] = (heart_df["thalch"] < 120).astype(int)

# Combined risk score (simple sum of risk flags)
heart_df["risk_factor_count"] = (
    heart_df["fbs"].fillna(0).astype(int) +
    heart_df["exang"].fillna(0).astype(int) +
    heart_df["high_cholesterol"] +
    heart_df["low_max_hr"]
)

print(f"  New features added: age_group, bp_category, high_cholesterol, low_max_hr, risk_factor_count")
print(f"  Final shape: {heart_df.shape}")


# ─────────────────────────────────────────
# 2B. DIABETES — NEW FEATURES
# ─────────────────────────────────────────
print("\n[DIABETES] Engineering features...")

# BMI category (WHO standard)
def bmi_category(bmi):
    if bmi < 18.5: return 0   # Underweight
    elif bmi < 25: return 1   # Normal
    elif bmi < 30: return 2   # Overweight
    else:          return 3   # Obese

diabetes_df["bmi_category"] = diabetes_df["BMI"].apply(bmi_category)

# Glucose risk level
def glucose_risk(g):
    if g < 100:  return 0   # Normal
    elif g < 126: return 1  # Pre-diabetic
    else:        return 2   # Diabetic range

diabetes_df["glucose_risk"] = diabetes_df["Glucose"].apply(glucose_risk)

# Age group
diabetes_df["age_group"] = diabetes_df["Age"].apply(age_group)

# Insulin resistance indicator
diabetes_df["insulin_resistance"] = (
    (diabetes_df["Insulin"] > 100) & (diabetes_df["Glucose"] > 120)
).astype(int)

# Risk factor count
diabetes_df["risk_factor_count"] = (
    (diabetes_df["Glucose"] > 126).astype(int) +
    (diabetes_df["BMI"] > 30).astype(int) +
    (diabetes_df["BloodPressure"] > 80).astype(int) +
    diabetes_df["insulin_resistance"]
)

print(f"  New features added: bmi_category, glucose_risk, age_group, insulin_resistance, risk_factor_count")
print(f"  Final shape: {diabetes_df.shape}")


# ─────────────────────────────────────────
# 2C. SAVE
# ─────────────────────────────────────────
heart_df.to_csv("data/heart_features.csv", index=False)
diabetes_df.to_csv("data/diabetes_features.csv", index=False)

print("\n[DONE] Feature-engineered files saved:")
print("  → data/heart_features.csv")
print("  → data/diabetes_features.csv")