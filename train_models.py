# ============================================================
# STEP 3 — MODEL TRAINING
# Trains ML models for both diseases, saves best model + scaler
# ============================================================

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix)

os.makedirs("models", exist_ok=True)

# ─────────────────────────────────────────
# HELPER: Train & evaluate multiple models
# ─────────────────────────────────────────
def train_models(X_train, X_test, y_train, y_test, disease_name):
    print(f"\n{'='*55}")
    print(f"  TRAINING MODELS FOR: {disease_name.upper()}")
    print(f"{'='*55}")
    print(f"  Train size : {X_train.shape[0]} samples")
    print(f"  Test size  : {X_test.shape[0]} samples")
    print(f"  Features   : {X_train.shape[1]}")

    models = {
        "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting"   : GradientBoostingClassifier(n_estimators=100, random_state=42),
        "SVM"                 : SVC(probability=True, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        auc  = roc_auc_score(y_test, y_prob)
        cv   = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean()

        results[name] = {
            "model": model, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "auc": auc, "cv_accuracy": cv
        }

        print(f"\n  [{name}]")
        print(f"    Accuracy  : {acc:.4f}  |  CV Accuracy: {cv:.4f}")
        print(f"    Precision : {prec:.4f}  |  Recall    : {rec:.4f}")
        print(f"    F1 Score  : {f1:.4f}  |  AUC-ROC   : {auc:.4f}")

    # Pick best model by F1 score
    best_name = max(results, key=lambda k: results[k]["f1"])
    best = results[best_name]
    print(f"\n  ✓ BEST MODEL → {best_name}  (F1={best['f1']:.4f})")
    print(f"\n  Full Classification Report ({best_name}):")
    print(classification_report(y_test, best["model"].predict(X_test),
                                target_names=["No Disease", "Disease"]))

    return best_name, best["model"], results


# ─────────────────────────────────────────
# HELPER: Save model + scaler + feature names
# ─────────────────────────────────────────
def save_artifacts(model, scaler, feature_cols, all_results, disease_name):
    tag = disease_name.replace(" ", "_").lower()
    pickle.dump(model,        open(f"models/{tag}_model.pkl", "wb"))
    pickle.dump(scaler,       open(f"models/{tag}_scaler.pkl", "wb"))
    pickle.dump(feature_cols, open(f"models/{tag}_features.pkl", "wb"))

    # Save metrics summary as CSV
    rows = []
    for mname, r in all_results.items():
        rows.append({
            "model": mname,
            "accuracy": round(r["accuracy"],4),
            "precision": round(r["precision"],4),
            "recall": round(r["recall"],4),
            "f1": round(r["f1"],4),
            "auc": round(r["auc"],4),
            "cv_accuracy": round(r["cv_accuracy"],4),
        })
    pd.DataFrame(rows).to_csv(f"models/{tag}_metrics.csv", index=False)
    print(f"\n  Artifacts saved → models/{tag}_*.pkl / models/{tag}_metrics.csv")


# ─────────────────────────────────────────
# TRAIN HEART DISEASE MODEL
# ─────────────────────────────────────────
heart_df = pd.read_csv("data/heart_features.csv")

HEART_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalch", "exang", "oldpeak",
    "age_group", "bp_category", "high_cholesterol", "low_max_hr", "risk_factor_count"
]

X_h = heart_df[HEART_FEATURES].fillna(0)
y_h = heart_df["target"]

scaler_h = StandardScaler()
X_h_scaled = scaler_h.fit_transform(X_h)

X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_h_scaled, y_h, test_size=0.2, random_state=42, stratify=y_h
)

best_name_h, best_model_h, results_h = train_models(
    X_train_h, X_test_h, y_train_h, y_test_h, "Heart Disease"
)
save_artifacts(best_model_h, scaler_h, HEART_FEATURES, results_h, "heart_disease")


# ─────────────────────────────────────────
# TRAIN DIABETES MODEL
# ─────────────────────────────────────────
diabetes_df = pd.read_csv("data/diabetes_features.csv")

DIABETES_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
    "bmi_category", "glucose_risk", "age_group", "insulin_resistance", "risk_factor_count"
]

X_d = diabetes_df[DIABETES_FEATURES].fillna(0)
y_d = diabetes_df["target"]

scaler_d = StandardScaler()
X_d_scaled = scaler_d.fit_transform(X_d)

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_d_scaled, y_d, test_size=0.2, random_state=42, stratify=y_d
)

best_name_d, best_model_d, results_d = train_models(
    X_train_d, X_test_d, y_train_d, y_test_d, "Diabetes"
)
save_artifacts(best_model_d, scaler_d, DIABETES_FEATURES, results_d, "diabetes")

print("\n" + "="*55)
print("  ALL MODELS TRAINED AND SAVED SUCCESSFULLY")
print("="*55)