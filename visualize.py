# ============================================================
# STEP 5 — VISUALIZATION
# Generates all charts used in the dashboard
# Saves them as PNG files in the charts/ folder
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

os.makedirs("charts", exist_ok=True)

# ── colour palette ─────────────────────────────────────────
COLORS = {
    "low":    "#2ecc71",
    "medium": "#f39c12",
    "high":   "#e74c3c",
    "blue":   "#3498db",
    "purple": "#9b59b6",
    "bg":     "#f8f9fa",
}
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams.update({"figure.facecolor": COLORS["bg"], "axes.facecolor": COLORS["bg"]})


def load_artifacts(tag):
    model    = pickle.load(open(f"models/{tag}_model.pkl", "rb"))
    scaler   = pickle.load(open(f"models/{tag}_scaler.pkl", "rb"))
    features = pickle.load(open(f"models/{tag}_features.pkl", "rb"))
    return model, scaler, features


# ─────────────────────────────────────────────────────────
# CHART 1 — Risk distribution pie chart
# ─────────────────────────────────────────────────────────
def chart_risk_distribution(df_h, df_d):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Risk Category Distribution", fontsize=15, fontweight="bold", y=1.02)

    for ax, (label, df, tag) in zip(axes, [
        ("Heart Disease", df_h, "heart"),
        ("Diabetes",      df_d, "diabetes"),
    ]):
        model, scaler, features = load_artifacts(f"{tag}_disease" if tag == "heart" else tag)
        X = df[features].fillna(0)
        probs = model.predict_proba(scaler.transform(X))[:, 1]

        low    = (probs < 0.30).sum()
        medium = ((probs >= 0.30) & (probs < 0.60)).sum()
        high   = (probs >= 0.60).sum()

        sizes  = [low, medium, high]
        labels = [f"Low\n({low})", f"Medium\n({medium})", f"High\n({high})"]
        clrs   = [COLORS["low"], COLORS["medium"], COLORS["high"]]
        explode = [0.03, 0.03, 0.07]

        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=clrs, autopct="%1.1f%%",
            startangle=140, explode=explode,
            textprops={"fontsize": 11}
        )
        for at in autotexts:
            at.set_fontsize(10)
            at.set_fontweight("bold")
        ax.set_title(label, fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig("charts/risk_distribution.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("  ✓ charts/risk_distribution.png")


# ─────────────────────────────────────────────────────────
# CHART 2 — Feature importance (Random Forest)
# ─────────────────────────────────────────────────────────
def chart_feature_importance(df, features, tag, title):
    from sklearn.ensemble import RandomForestClassifier
    X = df[features].fillna(0)
    y = df["target"]
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    importance_df = pd.DataFrame({
        "Feature":    features,
        "Importance": rf.feature_importances_
    }).sort_values("Importance", ascending=True).tail(12)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = sns.color_palette("Blues_r", len(importance_df))
    bars = ax.barh(importance_df["Feature"], importance_df["Importance"], color=colors)
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title(f"Top Feature Importance — {title}", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, importance_df["Importance"]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.set_facecolor(COLORS["bg"])
    fig.patch.set_facecolor(COLORS["bg"])
    plt.tight_layout()
    plt.savefig(f"charts/feature_importance_{tag}.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  ✓ charts/feature_importance_{tag}.png")


# ─────────────────────────────────────────────────────────
# CHART 3 — Model comparison bar chart
# ─────────────────────────────────────────────────────────
def chart_model_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Performance Comparison", fontsize=15, fontweight="bold")

    for ax, tag in zip(axes, ["heart_disease", "diabetes"]):
        try:
            metrics_df = pd.read_csv(f"models/{tag}_metrics.csv")
        except FileNotFoundError:
            continue
        metrics_df = metrics_df.set_index("model")
        cols   = ["accuracy", "precision", "recall", "f1", "auc"]
        x      = np.arange(len(metrics_df))
        width  = 0.15
        clrs   = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]

        for i, (col, color) in enumerate(zip(cols, clrs)):
            ax.bar(x + i * width, metrics_df[col], width, label=col.upper(), color=color, alpha=0.85)

        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(metrics_df.index, rotation=15, ha="right", fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.set_title(tag.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_facecolor(COLORS["bg"])

    fig.patch.set_facecolor(COLORS["bg"])
    plt.tight_layout()
    plt.savefig("charts/model_comparison.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("  ✓ charts/model_comparison.png")


# ─────────────────────────────────────────────────────────
# CHART 4 — ROC Curve
# ─────────────────────────────────────────────────────────
def chart_roc_curves(df_h, df_d):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("ROC Curves", fontsize=15, fontweight="bold")

    for ax, (tag, df, title) in zip(axes, [
        ("heart_disease", df_h, "Heart Disease"),
        ("diabetes",      df_d, "Diabetes"),
    ]):
        model, scaler, features = load_artifacts(tag)
        X = scaler.transform(df[features].fillna(0))
        y = df["target"]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        y_prob = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc     = auc(fpr, tpr)

        ax.plot(fpr, tpr, color=COLORS["blue"], lw=2.5,
                label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5, label="Random")
        ax.fill_between(fpr, tpr, alpha=0.08, color=COLORS["blue"])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=11)
        ax.set_facecolor(COLORS["bg"])

    fig.patch.set_facecolor(COLORS["bg"])
    plt.tight_layout()
    plt.savefig("charts/roc_curves.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("  ✓ charts/roc_curves.png")


# ─────────────────────────────────────────────────────────
# CHART 5 — Confusion Matrix
# ─────────────────────────────────────────────────────────
def chart_confusion_matrices(df_h, df_d):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Confusion Matrices (Test Set)", fontsize=15, fontweight="bold")

    for ax, (tag, df, title) in zip(axes, [
        ("heart_disease", df_h, "Heart Disease"),
        ("diabetes",      df_d, "Diabetes"),
    ]):
        model, scaler, features = load_artifacts(tag)
        X = scaler.transform(df[features].fillna(0))
        y = df["target"]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        y_pred = model.predict(X_test)
        cm     = confusion_matrix(y_test, y_pred)

        sns.heatmap(cm, annot=True, fmt="d", ax=ax,
                    cmap="Blues", linewidths=0.5,
                    xticklabels=["No Disease", "Disease"],
                    yticklabels=["No Disease", "Disease"])
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")

    fig.patch.set_facecolor(COLORS["bg"])
    plt.tight_layout()
    plt.savefig("charts/confusion_matrices.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("  ✓ charts/confusion_matrices.png")


# ─────────────────────────────────────────────────────────
# CHART 6 — Health indicator distributions
# ─────────────────────────────────────────────────────────
def chart_health_distributions(df_h, df_d):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Key Health Indicator Distributions", fontsize=15, fontweight="bold")

    plots = [
        (axes[0][0], df_h, "age",       "Heart Disease",  "Age",         "#3498db"),
        (axes[0][1], df_h, "chol",      "Heart Disease",  "Cholesterol", "#e74c3c"),
        (axes[0][2], df_h, "trestbps",  "Heart Disease",  "Blood Pressure", "#f39c12"),
        (axes[1][0], df_d, "Age",       "Diabetes",       "Age",         "#3498db"),
        (axes[1][1], df_d, "Glucose",   "Diabetes",       "Glucose",     "#e74c3c"),
        (axes[1][2], df_d, "BMI",       "Diabetes",       "BMI",         "#2ecc71"),
    ]

    for ax, df, col, source, xlabel, color in plots:
        no_disease = df[df["target"] == 0][col].dropna()
        disease    = df[df["target"] == 1][col].dropna()
        ax.hist(no_disease, bins=25, alpha=0.65, color=COLORS["low"],   label="No Disease", density=True)
        ax.hist(disease,    bins=25, alpha=0.65, color=COLORS["high"],  label="Disease",    density=True)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"{source} — {xlabel}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_facecolor(COLORS["bg"])

    fig.patch.set_facecolor(COLORS["bg"])
    plt.tight_layout()
    plt.savefig("charts/health_distributions.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("  ✓ charts/health_distributions.png")


# ─────────────────────────────────────────────────────────
# RUN ALL CHARTS
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating all charts...")

    heart_df    = pd.read_csv("data/heart_features.csv")
    diabetes_df = pd.read_csv("data/diabetes_features.csv")

    HEART_FEATURES = [
        "age","sex","cp","trestbps","chol","fbs","restecg",
        "thalch","exang","oldpeak",
        "age_group","bp_category","high_cholesterol","low_max_hr","risk_factor_count"
    ]
    DIABETES_FEATURES = [
        "Pregnancies","Glucose","BloodPressure","SkinThickness",
        "Insulin","BMI","DiabetesPedigreeFunction","Age",
        "bmi_category","glucose_risk","age_group","insulin_resistance","risk_factor_count"
    ]

    chart_risk_distribution(heart_df, diabetes_df)
    chart_feature_importance(heart_df,    HEART_FEATURES,    "heart",    "Heart Disease")
    chart_feature_importance(diabetes_df, DIABETES_FEATURES, "diabetes", "Diabetes")
    chart_model_comparison()
    chart_roc_curves(heart_df, diabetes_df)
    chart_confusion_matrices(heart_df, diabetes_df)
    chart_health_distributions(heart_df, diabetes_df)

    print("\n✅ All charts saved to charts/ folder")