# ============================================================
# STEP 7 — DATABASE MODULE
# Creates SQLite database, stores patients + predictions
# Run once to initialize: python step7_database.py
# ============================================================

import sqlite3
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
import os

DB_PATH = "data/healthcare.db"


# ─────────────────────────────────────────
# CREATE TABLES
# ─────────────────────────────────────────
def init_database():
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    # Patients table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            patient_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            name         TEXT    NOT NULL,
            age          INTEGER,
            sex          TEXT,
            created_at   TEXT    DEFAULT (datetime('now'))
        )
    """)

    # Predictions table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            pred_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id     INTEGER,
            disease        TEXT,
            risk_score     REAL,
            risk_category  TEXT,
            model_used     TEXT,
            predicted_at   TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
        )
    """)

    # Patient vitals table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS vitals (
            vital_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id    INTEGER,
            glucose       REAL,
            blood_pressure REAL,
            cholesterol   REAL,
            bmi           REAL,
            heart_rate    REAL,
            recorded_at   TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
        )
    """)

    conn.commit()
    conn.close()
    print(f"[DB] Database initialized at {DB_PATH}")


# ─────────────────────────────────────────
# INSERT / QUERY HELPERS
# ─────────────────────────────────────────
def add_patient(name, age, sex):
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute(
        "INSERT INTO patients (name, age, sex) VALUES (?, ?, ?)",
        (name, age, sex)
    )
    pid = cur.lastrowid
    conn.commit()
    conn.close()
    return pid


def save_prediction(patient_id, disease, risk_score, risk_category, model_used="auto"):
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO predictions (patient_id, disease, risk_score, risk_category, model_used)
        VALUES (?, ?, ?, ?, ?)
    """, (patient_id, disease, round(risk_score, 4), risk_category, model_used))
    conn.commit()
    conn.close()


def save_vitals(patient_id, glucose=None, blood_pressure=None,
                cholesterol=None, bmi=None, heart_rate=None):
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO vitals (patient_id, glucose, blood_pressure, cholesterol, bmi, heart_rate)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (patient_id, glucose, blood_pressure, cholesterol, bmi, heart_rate))
    conn.commit()
    conn.close()


def get_all_predictions():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT p.name, p.age, p.sex,
               pr.disease, pr.risk_score, pr.risk_category,
               pr.model_used, pr.predicted_at
        FROM predictions pr
        JOIN patients p ON p.patient_id = pr.patient_id
        ORDER BY pr.predicted_at DESC
    """, conn)
    conn.close()
    return df


def get_patient_history(patient_id):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT disease, risk_score, risk_category, predicted_at
        FROM predictions
        WHERE patient_id = ?
        ORDER BY predicted_at ASC
    """, conn, params=(patient_id,))
    conn.close()
    return df


def get_stats():
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    stats = {}
    stats["total_patients"]    = cur.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    stats["total_predictions"] = cur.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    stats["high_risk_count"]   = cur.execute(
        "SELECT COUNT(*) FROM predictions WHERE risk_category='High Risk'"
    ).fetchone()[0]
    stats["medium_risk_count"] = cur.execute(
        "SELECT COUNT(*) FROM predictions WHERE risk_category='Medium Risk'"
    ).fetchone()[0]
    stats["low_risk_count"]    = cur.execute(
        "SELECT COUNT(*) FROM predictions WHERE risk_category='Low Risk'"
    ).fetchone()[0]
    conn.close()
    return stats


# ─────────────────────────────────────────
# SEED: Load existing dataset patients into DB
# ─────────────────────────────────────────
def seed_from_datasets():
    """Reads both CSVs, runs predictions, saves top 20 patients per disease."""
    print("[DB] Seeding database from datasets...")

    for tag, df_path, name_prefix in [
        ("heart_disease", "data/heart_features.csv",    "HRT"),
        ("diabetes",      "data/diabetes_features.csv", "DIA"),
    ]:
        df      = pd.read_csv(df_path).head(20)
        model   = pickle.load(open(f"models/{tag}_model.pkl",    "rb"))
        scaler  = pickle.load(open(f"models/{tag}_scaler.pkl",   "rb"))
        features = pickle.load(open(f"models/{tag}_features.pkl","rb"))

        disease_label = tag.replace("_", " ").title()

        for i, row in df.iterrows():
            # Create a fake patient name for demo
            sex_raw = row.get("sex", row.get("Sex", 0))
            sex_str = "Male" if str(sex_raw) in ["1", "Male", "1.0"] else "Female"
            age_col = "age" if "age" in row else "Age"
            age_val = int(row.get(age_col, 40))

            pid = add_patient(
                name=f"Patient-{name_prefix}-{i+1:03d}",
                age=age_val,
                sex=sex_str
            )

            X    = scaler.transform([row[features].fillna(0).values])
            prob = model.predict_proba(X)[0][1]

            if prob < 0.30:   cat = "Low Risk"
            elif prob < 0.60: cat = "Medium Risk"
            else:             cat = "High Risk"

            save_prediction(pid, disease_label, prob, cat, model_used=type(model).__name__)

            # Save vitals if available
            save_vitals(
                pid,
                glucose        = row.get("Glucose",     row.get("chol",      None)),
                blood_pressure = row.get("BloodPressure",row.get("trestbps",  None)),
                cholesterol    = row.get("chol",         None),
                bmi            = row.get("BMI",          None),
                heart_rate     = row.get("thalch",       None),
            )

        print(f"  ✓ Seeded 20 patients for {disease_label}")

    stats = get_stats()
    print(f"\n[DB] Database stats:")
    print(f"  Total patients    : {stats['total_patients']}")
    print(f"  Total predictions : {stats['total_predictions']}")
    print(f"  High Risk         : {stats['high_risk_count']}")
    print(f"  Medium Risk       : {stats['medium_risk_count']}")
    print(f"  Low Risk          : {stats['low_risk_count']}")


# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────
if __name__ == "__main__":
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("[DB] Existing database removed.")
    init_database()
    seed_from_datasets()
    print("\n[DB] Sample prediction history:")
    print(get_all_predictions().head(10).to_string(index=False))