# ============================================================
# STEP 8 — BATCH PREDICTION
# Runs predictions on ALL patients in both datasets
# Saves results to: data/batch_predictions.csv
# ============================================================

import pandas as pd
import numpy as np
import pickle
from datetime import datetime

def run_batch_predictions():
    results = []

    for tag, df_path, disease_label in [
        ("heart_disease", "data/heart_features.csv",    "Heart Disease"),
        ("diabetes",      "data/diabetes_features.csv", "Diabetes"),
    ]:
        print(f"\n[BATCH] Processing {disease_label}...")

        model    = pickle.load(open(f"models/{tag}_model.pkl",    "rb"))
        scaler   = pickle.load(open(f"models/{tag}_scaler.pkl",   "rb"))
        features = pickle.load(open(f"models/{tag}_features.pkl", "rb"))

        df = pd.read_csv(df_path)
        X  = scaler.transform(df[features].fillna(0))

        probs     = model.predict_proba(X)[:, 1]
        preds     = model.predict(X)

        for i, (prob, pred) in enumerate(zip(probs, preds)):
            if prob < 0.30:   cat = "Low Risk"
            elif prob < 0.60: cat = "Medium Risk"
            else:             cat = "High Risk"

            results.append({
                "patient_index" : i + 1,
                "disease"       : disease_label,
                "risk_score_pct": round(prob * 100, 1),
                "risk_category" : cat,
                "predicted_label": int(pred),
                "actual_label"  : int(df["target"].iloc[i]),
                "correct"       : int(pred) == int(df["target"].iloc[i]),
            })

        total  = len(df)
        high   = sum(1 for r in results if r["disease"] == disease_label and r["risk_category"] == "High Risk")
        medium = sum(1 for r in results if r["disease"] == disease_label and r["risk_category"] == "Medium Risk")
        low    = sum(1 for r in results if r["disease"] == disease_label and r["risk_category"] == "Low Risk")

        print(f"  Total patients : {total}")
        print(f"  🔴 High Risk   : {high}  ({high/total*100:.1f}%)")
        print(f"  🟡 Medium Risk : {medium}  ({medium/total*100:.1f}%)")
        print(f"  🟢 Low Risk    : {low}  ({low/total*100:.1f}%)")

    batch_df = pd.DataFrame(results)
    batch_df.to_csv("data/batch_predictions.csv", index=False)

    print(f"\n[BATCH] Total predictions: {len(batch_df)}")
    print(f"[BATCH] Overall accuracy : {batch_df['correct'].mean()*100:.2f}%")
    print("[BATCH] Saved → data/batch_predictions.csv")
    return batch_df


if __name__ == "__main__":
    df = run_batch_predictions()
    print("\nSample output:")
    print(df.head(10).to_string(index=False))