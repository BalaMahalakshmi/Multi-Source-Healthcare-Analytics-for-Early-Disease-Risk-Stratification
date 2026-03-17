# ============================================================
# STEP 4 — RISK STRATIFICATION + RECOMMENDATIONS
# Given patient data → returns risk score, category, advice
# ============================================================

import pickle
import numpy as np
import pandas as pd


def load_model(disease: str):
    """Load model, scaler, and feature list for a given disease."""
    tag = disease.replace(" ", "_").lower()
    model    = pickle.load(open(f"models/{tag}_model.pkl", "rb"))
    scaler   = pickle.load(open(f"models/{tag}_scaler.pkl", "rb"))
    features = pickle.load(open(f"models/{tag}_features.pkl", "rb"))
    return model, scaler, features


def stratify_risk(probability: float) -> tuple[str, str]:
    """Map probability to risk category and colour label."""
    if probability < 0.30:
        return "Low Risk", "🟢"
    elif probability < 0.60:
        return "Medium Risk", "🟡"
    else:
        return "High Risk", "🔴"


def get_recommendations(disease: str, risk_category: str, patient: dict) -> list[str]:
    """Return actionable recommendations based on disease + risk level."""
    recs = []

    if disease == "heart_disease":
        if risk_category == "High Risk":
            recs += [
                "🚨 Immediate cardiology consultation recommended.",
                "📋 Schedule ECG and stress test within 1 week.",
                "💊 Review current medications with cardiologist.",
                "🏥 Consider cardiac monitoring.",
            ]
        elif risk_category == "Medium Risk":
            recs += [
                "📅 Schedule routine cardiac check-up within 1 month.",
                "🥗 Follow a low-sodium, low-cholesterol diet.",
                "🚶 Aim for 30 min moderate exercise 5 days/week.",
            ]
        else:
            recs += [
                "✅ Continue healthy lifestyle habits.",
                "📅 Annual cardiac screening recommended.",
            ]
        if patient.get("chol", 0) > 200:
            recs.append("⚠️  Cholesterol elevated — reduce saturated fats.")
        if patient.get("trestbps", 0) > 140:
            recs.append("⚠️  Blood pressure high — consult doctor about management.")

    elif disease == "diabetes":
        if risk_category == "High Risk":
            recs += [
                "🚨 Immediate HbA1c and fasting glucose test recommended.",
                "👨‍⚕️ Endocrinologist consultation within 1 week.",
                "💉 Monitor blood glucose twice daily.",
                "🥗 Start diabetic-friendly meal plan immediately.",
            ]
        elif risk_category == "Medium Risk":
            recs += [
                "📋 Schedule fasting blood glucose test.",
                "🥗 Reduce sugar and refined carbohydrate intake.",
                "🏃 Increase physical activity to 150 min/week.",
            ]
        else:
            recs += [
                "✅ Blood sugar levels appear normal.",
                "📅 Annual glucose screening recommended.",
            ]
        if patient.get("BMI", 0) > 30:
            recs.append("⚠️  BMI indicates obesity — weight management advised.")
        if patient.get("Glucose", 0) > 126:
            recs.append("⚠️  Glucose in diabetic range — urgent medical review.")

    return recs


def predict_patient(disease: str, patient_data: dict) -> dict:
    """
    Full pipeline: patient dict → risk score → category → recommendations.

    Parameters
    ----------
    disease      : "heart_disease" or "diabetes"
    patient_data : dict with feature values matching that disease

    Returns
    -------
    dict with keys: probability, risk_category, icon, recommendations
    """
    model, scaler, features = load_model(disease)

    # Build feature vector (fill missing with 0)
    row = [patient_data.get(f, 0) for f in features]
    X   = scaler.transform([row])

    prob          = model.predict_proba(X)[0][1]
    risk_cat, icon = stratify_risk(prob)
    recs          = get_recommendations(disease, risk_cat, patient_data)

    return {
        "disease"        : disease,
        "probability"    : round(float(prob), 4),
        "risk_score_pct" : round(float(prob) * 100, 1),
        "risk_category"  : risk_cat,
        "icon"           : icon,
        "recommendations": recs,
    }


# ─────────────────────────────────────────
# QUICK TEST (run this file directly)
# ─────────────────────────────────────────
if __name__ == "__main__":

    # --- Sample heart disease patient ---
    heart_patient = {
        "age": 55, "sex": 1, "cp": 2, "trestbps": 148,
        "chol": 250, "fbs": 1, "restecg": 1, "thalch": 110,
        "exang": 1, "oldpeak": 2.5,
        "age_group": 2, "bp_category": 2, "high_cholesterol": 1,
        "low_max_hr": 1, "risk_factor_count": 4
    }

    # --- Sample diabetes patient ---
    diabetes_patient = {
        "Pregnancies": 3, "Glucose": 155, "BloodPressure": 85,
        "SkinThickness": 32, "Insulin": 120, "BMI": 34.2,
        "DiabetesPedigreeFunction": 0.72, "Age": 45,
        "bmi_category": 3, "glucose_risk": 2, "age_group": 1,
        "insulin_resistance": 1, "risk_factor_count": 4
    }

    for disease, patient in [("heart_disease", heart_patient),
                              ("diabetes", diabetes_patient)]:
        result = predict_patient(disease, patient)
        print(f"\n{'='*50}")
        print(f"  Disease   : {result['disease'].replace('_',' ').title()}")
        print(f"  Risk Score: {result['risk_score_pct']}%")
        print(f"  Category  : {result['icon']} {result['risk_category']}")
        print(f"  Recommendations:")
        for r in result["recommendations"]:
            print(f"    {r}")