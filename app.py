# ============================================================
# STREAMLIT DASHBOARD — FULL VERSION WITH LOGIN (app.py)
# Run: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle, os, sys, sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime 

sys.path.insert(0, ".")

# Safe optional imports
try:
    from reportlab.lib.pagesizes import A4
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False

st.set_page_config(page_title="Healthcare Risk Analytics", page_icon="🏥",
                   layout="wide", initial_sidebar_state="expanded")

# ══════════════════════════════════════════════════════════════
# USER ACCOUNTS
# Format: "username": ("password", "role", "full_name")
# ══════════════════════════════════════════════════════════════
USERS = {
    "admin":   ("admin123",  "admin",  "Administrator"),
    "doctor1": ("doctor123", "doctor", "Dr. Priya Sharma"),
    "doctor2": ("doc456",    "doctor", "Dr. Ramesh Kumar"),
    "nurse1":  ("nurse123",  "nurse",  "Nurse Anjali"),
}

ROLE_PERMISSIONS = {
    "admin":  ["🏠  Overview","🫀  Heart Disease","🩸  Diabetes",
               "📊  Model Performance","📈  Data Insights",
               "🗄️  Patient Database","📦  Batch Predictions","📄  Generate Report"],
    "doctor": ["🏠  Overview","🫀  Heart Disease","🩸  Diabetes",
               "📈  Data Insights","🗄️  Patient Database","📄  Generate Report"],
    "nurse":  ["🏠  Overview","🫀  Heart Disease","🩸  Diabetes",
               "🗄️  Patient Database"],
}

ROLE_COLORS = {
    "admin":  "#8e44ad",
    "doctor": "#2980b9",
    "nurse":  "#27ae60",
}

ROLE_ICONS = {
    "admin":  "🛡️",
    "doctor": "👨‍⚕️",
    "nurse":  "👩‍⚕️",
}

# ══════════════════════════════════════════════════════════════
# LOGIN PAGE — Clean, Secure, No hints shown
# ══════════════════════════════════════════════════════════════
def show_login():
    st.markdown("""
    <style>
    [data-testid="stSidebar"]        { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }
    [data-testid="stHeader"]         { background: transparent !important; }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%) !important;
        min-height: 100vh;
    }
    .block-container { padding-top: 2rem !important; }

    .login-card {
        background: white;
        max-width: 420px;
        margin: 40px auto 0;
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 25px 60px rgba(0,0,0,0.5);
    }
    .login-top {
        background: linear-gradient(135deg, #1a3a5c 0%, #2980b9 100%);
        padding: 42px 30px 32px;
        text-align: center;
    }
    .login-top .hosp-icon { font-size: 58px; line-height: 1; display: block; }
    .login-top h1 {
        color: white; font-size: 24px;
        margin: 14px 0 6px; font-weight: 700;
    }
    .login-top .sub { color: #d6eaf8; font-size: 13px; margin: 0; }
    .secure-tag {
        display: inline-block; margin-top: 16px;
        background: rgba(255,255,255,0.15);
        color: #d6eaf8; font-size: 11px;
        padding: 5px 16px; border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.3);
        letter-spacing: 1.5px; text-transform: uppercase;
    }
    .login-body { padding: 34px 38px 38px; background: white; }
    .lbl {
        font-size: 11px; font-weight: 700; color: #95a5a6;
        letter-spacing: 1.2px; text-transform: uppercase;
        margin-bottom: 4px; margin-top: 18px;
    }
    .lbl:first-child { margin-top: 0; }
    .footer-note {
        text-align: center; margin-top: 10px;
        color: #bdc3c7; font-size: 12px;
        display: flex; align-items: center;
        justify-content: center; gap: 6px;
    }
    .page-footer {
        text-align: center;
        color: rgba(255,255,255,0.3);
        font-size: 11px; margin-top: 24px;
        letter-spacing: 0.5px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Card
    st.markdown("""
    <div class="login-card">
      <div class="login-top">
        <span class="hosp-icon">🏥</span>
        <h1>Healthcare Analytics</h1>
        <p class="sub">Disease Risk Stratification System</p>
        <span class="secure-tag">🔐 &nbsp; Secure Portal</span>
      </div>
      <div class="login-body">
    """, unsafe_allow_html=True)

    # Failed attempt tracking
    if "fail_count" not in st.session_state:
        st.session_state["fail_count"] = 0

    st.markdown('<div class="lbl">Username</div>', unsafe_allow_html=True)
    username = st.text_input("u", placeholder="Enter your username",
                              key="login_user", label_visibility="collapsed")

    st.markdown('<div class="lbl">Password</div>', unsafe_allow_html=True)
    password = st.text_input("p", placeholder="Enter your password",
                              type="password", key="login_pass",
                              label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    login_btn = st.button("Login  →", use_container_width=True, type="primary")

    if login_btn:
        u = username.strip()
        p = password
        if not u or not p:
            st.error("⚠️  Please enter both username and password.")
        elif u in USERS and USERS[u][0] == p:
            st.session_state["logged_in"]  = True
            st.session_state["username"]   = u
            st.session_state["role"]       = USERS[u][1]
            st.session_state["full_name"]  = USERS[u][2]
            st.session_state["login_time"] = datetime.now().strftime("%d %b %Y, %I:%M %p")
            st.session_state["fail_count"] = 0
            st.rerun()
        else:
            st.session_state["fail_count"] += 1
            c = st.session_state["fail_count"]
            if c >= 3:
                st.error(f"❌  Too many failed attempts ({c}). Please contact your administrator.")
            else:
                st.error("❌  Invalid username or password. Please try again.")

    st.markdown("""
      <div class="footer-note">🔒 &nbsp; Authorised Personnel Only</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="page-footer">
        Powered by Machine Learning &nbsp;•&nbsp;
        UCI Heart Disease + Pima Diabetes
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# GATE — Show login if not authenticated
# ══════════════════════════════════════════════════════════════
if not st.session_state.get("logged_in"):
    show_login()
    st.stop()

# ── From here, user is authenticated ─────────────────────────
current_role  = st.session_state.get("role", "nurse")
current_user  = st.session_state.get("full_name", "User")
current_uname = st.session_state.get("username", "")
login_time    = st.session_state.get("login_time", "")
allowed_pages = ROLE_PERMISSIONS.get(current_role, [])
role_color    = ROLE_COLORS.get(current_role, "#555")
role_icon     = ROLE_ICONS.get(current_role, "👤")

# ── Global CSS ─────────────────────────────────────────────
st.markdown("""
<style>
.risk-low    {background:#d4edda;color:#155724;padding:14px 20px;border-radius:12px;
              font-size:22px;font-weight:bold;text-align:center;border-left:6px solid #28a745;}
.risk-medium {background:#fff3cd;color:#856404;padding:14px 20px;border-radius:12px;
              font-size:22px;font-weight:bold;text-align:center;border-left:6px solid #ffc107;}
.risk-high   {background:#f8d7da;color:#721c24;padding:14px 20px;border-radius:12px;
              font-size:22px;font-weight:bold;text-align:center;border-left:6px solid #dc3545;}
.access-denied {background:#f8d7da;color:#721c24;padding:30px;border-radius:12px;
                text-align:center;font-size:18px;border:2px dashed #dc3545;}
.role-pill {display:inline-block;padding:3px 12px;border-radius:20px;
            font-size:12px;font-weight:bold;color:white;margin-bottom:6px;}
.user-card {background:rgba(255,255,255,0.07);border-radius:10px;padding:12px 14px;
            margin-bottom:10px;border:1px solid rgba(255,255,255,0.1);}
</style>
""", unsafe_allow_html=True)

# ── Access control ─────────────────────────────────────────
def access_denied():
    st.markdown(
        f'<div class="access-denied">🚫 <b>Access Denied</b><br><br>'
        f'Your role <b>({current_role.upper()})</b> does not have permission for this page.<br>'
        f'Please contact the administrator.</div>',
        unsafe_allow_html=True
    )
    st.stop()

# ── Helpers ────────────────────────────────────────────────
@st.cache_resource
def load_model(tag):
    model    = pickle.load(open(f"models/{tag}_model.pkl",    "rb"))
    scaler   = pickle.load(open(f"models/{tag}_scaler.pkl",   "rb"))
    features = pickle.load(open(f"models/{tag}_features.pkl", "rb"))
    return model, scaler, features

@st.cache_data
def load_datasets():
    return pd.read_csv("data/heart_features.csv"), pd.read_csv("data/diabetes_features.csv")

def stratify(prob):
    if prob < 0.30:   return "🟢 Low Risk",   "risk-low",    "#27ae60"
    elif prob < 0.60: return "🟡 Medium Risk", "risk-medium", "#f39c12"
    else:             return "🔴 High Risk",   "risk-high",   "#e74c3c"

def predict_score(tag, patient_dict):
    model, scaler, features = load_model(tag)
    X = scaler.transform([[patient_dict.get(f, 0) for f in features]])
    return float(model.predict_proba(X)[0][1])

def get_recs(disease, risk_cat, patient):
    recs = []
    if disease == "heart":
        if "High" in risk_cat:
            recs += ["🚨 Immediate cardiology consultation.",
                     "📋 Schedule ECG and stress test within 1 week.",
                     "💊 Review medications with cardiologist.",
                     "🏥 Consider cardiac monitoring."]
        elif "Medium" in risk_cat:
            recs += ["📅 Cardiac check-up within 1 month.",
                     "🥗 Low-sodium, low-cholesterol diet.",
                     "🚶 30 min exercise, 5 days/week."]
        else:
            recs += ["✅ Continue healthy habits.", "📅 Annual cardiac screening."]
        if patient.get("chol", 0) > 200:
            recs.append("⚠️  Cholesterol elevated — reduce saturated fats.")
        if patient.get("trestbps", 0) > 140:
            recs.append("⚠️  High blood pressure — consult doctor.")
    else:
        if "High" in risk_cat:
            recs += ["🚨 Immediate HbA1c and fasting glucose test.",
                     "👨‍⚕️ Endocrinologist consultation within 1 week.",
                     "💉 Monitor blood glucose twice daily.",
                     "🥗 Start diabetic-friendly meal plan."]
        elif "Medium" in risk_cat:
            recs += ["📋 Schedule fasting blood glucose test.",
                     "🥗 Reduce sugar and refined carbohydrates.",
                     "🏃 150 min physical activity per week."]
        else:
            recs += ["✅ Blood sugar appears normal.", "📅 Annual glucose screening."]
        if patient.get("BMI", 0) > 30:
            recs.append("⚠️  BMI indicates obesity — weight management advised.")
        if patient.get("Glucose", 0) > 126:
            recs.append("⚠️  Glucose in diabetic range — urgent review.")
    return recs

# ── SIDEBAR ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Healthcare Analytics")
    st.markdown("---")
    st.markdown(
        f'<div class="user-card">'
        f'<div style="font-size:13px;opacity:0.7;margin-bottom:4px;">Logged in as</div>'
        f'<div style="font-weight:bold;font-size:15px;">{role_icon} {current_user}</div>'
        f'<span class="role-pill" style="background:{role_color};">{current_role.upper()}</span>'
        f'<div style="font-size:11px;opacity:0.6;margin-top:6px;">Since {login_time}</div>'
        f'</div>',
        unsafe_allow_html=True
    )
    page = st.radio("Navigate", allowed_pages)
    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    st.caption("UCI Heart Disease + Pima Diabetes\nPowered by Scikit-learn & Streamlit")


# ════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════
if "Overview" in page:
    st.title("🏥 Multi-Source Healthcare Analytics")
    st.markdown("### Early Disease Risk Stratification System")
    st.markdown("---")
    heart_df, diabetes_df = load_datasets()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Records",       f"{len(heart_df)+len(diabetes_df):,}")
    c2.metric("Heart Disease Cases", f"{len(heart_df):,}")
    c3.metric("Diabetes Cases",      f"{len(diabetes_df):,}")
    c4.metric("ML Models Trained",   "4 per disease")
    st.markdown("---")
    st.markdown("### 📊 Live Risk Distribution")
    col1, col2 = st.columns(2)
    for col, (tag, df, title) in zip([col1,col2],[
        ("heart_disease", heart_df, "Heart Disease"),
        ("diabetes",      diabetes_df, "Diabetes"),
    ]):
        try:
            model, scaler, features = load_model(tag)
            probs = model.predict_proba(scaler.transform(df[features].fillna(0)))[:, 1]
            low = int((probs<0.3).sum()); med = int(((probs>=0.3)&(probs<0.6)).sum()); hi = int((probs>=0.6).sum())
            fig, ax = plt.subplots(figsize=(5,4))
            ax.pie([low,med,hi],
                   labels=[f"Low\n({low})",f"Medium\n({med})",f"High\n({hi})"],
                   colors=["#27ae60","#f39c12","#e74c3c"],
                   autopct="%1.1f%%", startangle=140, explode=[0.02,0.02,0.06],
                   textprops={"fontsize":10})
            ax.set_title(title, fontsize=13, fontweight="bold")
            fig.patch.set_facecolor("#f0f4f8")
            with col: st.pyplot(fig)
            plt.close()
        except Exception as e:
            with col: st.warning(f"Train models first.\n{e}")
    st.markdown("---")
    st.markdown("### 📌 Navigation Guide")
    st.markdown("""
| Page | What You Can Do |
|------|----------------|
| 🫀 Heart Disease | Enter patient vitals → get heart risk score |
| 🩸 Diabetes | Enter patient vitals → get diabetes risk score |
| 📊 Model Performance | View accuracy, F1, AUC for all 4 ML models |
| 📈 Data Insights | Charts, feature importance, distributions |
| 🗄️ Patient Database | Browse stored predictions, add new patients |
| 📦 Batch Predictions | Run predictions on all 1,688 patients at once |
| 📄 Generate Report | Download a PDF risk report for any patient |
""")


# ════════════════════════════════════════════════════════════
# PAGE 2 — HEART DISEASE
# ════════════════════════════════════════════════════════════
elif "Heart Disease" in page:
    st.title("🫀 Heart Disease Risk Predictor")
    st.markdown("---")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.subheader("Demographics")
        age     = st.slider("Age", 20, 80, 55)
        sex_v   = 1 if "Male" in st.selectbox("Sex", ["Male (1)","Female (0)"]) else 0
    with c2:
        st.subheader("Clinical Values")
        trestbps = st.slider("Resting BP (mm Hg)", 90, 200, 130)
        chol     = st.slider("Cholesterol (mg/dL)", 100, 600, 240)
        thalch   = st.slider("Max Heart Rate", 60, 210, 150)
        oldpeak  = st.slider("ST Depression", 0.0, 6.0, 1.0, 0.1)
    with c3:
        st.subheader("Symptoms & Tests")
        cp_v      = {"Typical Angina":0,"Atypical":1,"Non-Anginal":2,"Asymptomatic":3}[
                     st.selectbox("Chest Pain Type", ["Typical Angina","Atypical","Non-Anginal","Asymptomatic"])]
        fbs_v     = 1 if "Yes" in st.selectbox("Fasting BS >120", ["No (0)","Yes (1)"]) else 0
        restecg_v = {"Normal":0,"ST-T Abnormality":1,"LV Hypertrophy":2}[
                     st.selectbox("Resting ECG", ["Normal","ST-T Abnormality","LV Hypertrophy"])]
        exang_v   = 1 if "Yes" in st.selectbox("Exercise Angina", ["No (0)","Yes (1)"]) else 0

    ag = 0 if age<40 else (1 if age<55 else (2 if age<65 else 3))
    bp = 0 if trestbps<120 else (1 if trestbps<130 else (2 if trestbps<140 else 3))
    hc = int(chol>200); lh = int(thalch<120)
    patient = {"age":age,"sex":sex_v,"cp":cp_v,"trestbps":trestbps,"chol":chol,
               "fbs":fbs_v,"restecg":restecg_v,"thalch":thalch,"exang":exang_v,
               "oldpeak":oldpeak,"age_group":ag,"bp_category":bp,
               "high_cholesterol":hc,"low_max_hr":lh,"risk_factor_count":fbs_v+exang_v+hc+lh}
    st.markdown("---")
    if st.button("🔍 Predict Heart Disease Risk", use_container_width=True, type="primary"):
        try:
            prob  = predict_score("heart_disease", patient)
            label, css, color = stratify(prob)
            _, cm, _ = st.columns([1,2,1])
            with cm:
                st.markdown(f'<div class="{css}">{label}<br><span style="font-size:34px">{prob*100:.1f}%</span></div>',
                            unsafe_allow_html=True)
            st.markdown("---")
            ca, cb = st.columns(2)
            with ca:
                st.subheader("Risk Gauge")
                st.progress(int(prob*100))
                st.markdown(f"**Score: {prob*100:.1f}%** — {label}")
                st.caption("🟢 Low < 30%  |  🟡 Medium 30-60%  |  🔴 High > 60%")
            with cb:
                st.subheader("📋 Recommendations")
                recs = get_recs("heart", label, patient)
                for r in recs: st.markdown(r)
            st.session_state["heart_result"] = {
                "risk_score_pct": round(prob*100,1), "risk_category": label.split(" ",1)[1],
                "recommendations": recs
            }
        except Exception as e:
            st.error(f"Error: {e}\nRun step3_train_models.py first.")


# ════════════════════════════════════════════════════════════
# PAGE 3 — DIABETES
# ════════════════════════════════════════════════════════════
elif "Diabetes" in page:
    st.title("🩸 Diabetes Risk Predictor")
    st.markdown("---")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.subheader("Demographics")
        age         = st.slider("Age", 18, 90, 40)
        pregnancies = st.slider("Pregnancies", 0, 17, 2)
        dpf         = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5, 0.01)
    with c2:
        st.subheader("Blood Values")
        glucose = st.slider("Glucose (mg/dL)", 50, 250, 120)
        insulin = st.slider("Insulin (mu U/ml)", 0, 900, 80)
        bp      = st.slider("Blood Pressure (mm Hg)", 40, 130, 70)
    with c3:
        st.subheader("Body Measurements")
        bmi  = st.slider("BMI", 10.0, 70.0, 28.0, 0.1)
        skin = st.slider("Skin Thickness (mm)", 0, 100, 25)

    bc = 0 if bmi<18.5 else (1 if bmi<25 else (2 if bmi<30 else 3))
    gr = 0 if glucose<100 else (1 if glucose<126 else 2)
    ag = 0 if age<40 else (1 if age<55 else (2 if age<65 else 3))
    ir = int(insulin>100 and glucose>120)
    patient = {"Pregnancies":pregnancies,"Glucose":glucose,"BloodPressure":bp,
               "SkinThickness":skin,"Insulin":insulin,"BMI":bmi,
               "DiabetesPedigreeFunction":dpf,"Age":age,
               "bmi_category":bc,"glucose_risk":gr,"age_group":ag,
               "insulin_resistance":ir,
               "risk_factor_count":int(glucose>126)+int(bmi>30)+int(bp>80)+ir}
    st.markdown("---")
    if st.button("🔍 Predict Diabetes Risk", use_container_width=True, type="primary"):
        try:
            prob  = predict_score("diabetes", patient)
            label, css, color = stratify(prob)
            _, cm, _ = st.columns([1,2,1])
            with cm:
                st.markdown(f'<div class="{css}">{label}<br><span style="font-size:34px">{prob*100:.1f}%</span></div>',
                            unsafe_allow_html=True)
            st.markdown("---")
            ca, cb = st.columns(2)
            with ca:
                st.subheader("Risk Gauge")
                st.progress(int(prob*100))
                st.markdown(f"**Score: {prob*100:.1f}%** — {label}")
                st.caption("🟢 Low < 30%  |  🟡 Medium 30-60%  |  🔴 High > 60%")
            with cb:
                st.subheader("📋 Recommendations")
                recs = get_recs("diabetes", label, patient)
                for r in recs: st.markdown(r)
            st.session_state["diabetes_result"] = {
                "risk_score_pct": round(prob*100,1), "risk_category": label.split(" ",1)[1],
                "recommendations": recs
            }
        except Exception as e:
            st.error(f"Error: {e}\nRun step3_train_models.py first.")


# ════════════════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════
elif "Model Performance" in page:
    if "📊  Model Performance" not in allowed_pages: access_denied()
    st.title("📊 Model Performance Dashboard")
    st.markdown("---")
    for tag, title in [("heart_disease","🫀 Heart Disease"),("diabetes","🩸 Diabetes")]:
        st.subheader(title)
        try:
            mdf = pd.read_csv(f"models/{tag}_metrics.csv")
            st.dataframe(mdf.style.highlight_max(
                subset=["accuracy","precision","recall","f1","auc"], color="#d4edda", axis=0),
                use_container_width=True)
        except: st.warning("Run step3_train_models.py first.")
        st.markdown("---")
    for path, cap in [
        ("charts/model_comparison.png",  "Model Comparison"),
        ("charts/roc_curves.png",        "ROC Curves"),
        ("charts/confusion_matrices.png","Confusion Matrices"),
    ]:
        if os.path.exists(path): st.image(path, caption=cap, use_container_width=True)
        else: st.info(f"Run step5_visualizations.py to generate: {cap}")


# ════════════════════════════════════════════════════════════
# PAGE 5 — DATA INSIGHTS
# ════════════════════════════════════════════════════════════
elif "Data Insights" in page:
    st.title("📈 Data Insights & Distributions")
    st.markdown("---")
    for path, cap in [
        ("charts/risk_distribution.png",         "Risk Distribution"),
        ("charts/health_distributions.png",       "Health Indicator Distributions"),
        ("charts/feature_importance_heart.png",   "Heart Disease — Feature Importance"),
        ("charts/feature_importance_diabetes.png","Diabetes — Feature Importance"),
    ]:
        if os.path.exists(path):
            st.image(path, caption=cap, use_container_width=True); st.markdown("---")
        else:
            st.info(f"Run step5_visualizations.py to generate: {cap}")
    st.subheader("🔎 Raw Data Explorer")
    h, d = load_datasets()
    t1,t2 = st.tabs(["Heart Disease","Diabetes"])
    with t1: st.dataframe(h.head(50), use_container_width=True); st.caption(f"{len(h)} records")
    with t2: st.dataframe(d.head(50), use_container_width=True); st.caption(f"{len(d)} records")


# ════════════════════════════════════════════════════════════
# PAGE 6 — PATIENT DATABASE
# ════════════════════════════════════════════════════════════
elif "Database" in page:
    st.title("🗄️ Patient Database")
    st.markdown("---")
    DB_PATH = "data/healthcare.db"
    if not os.path.exists(DB_PATH):
        st.warning("Database not found. Run `python step7_database.py` first.")
        if st.button("Initialize Database Now"):
            import subprocess
            r = subprocess.run(["python","step7_database.py"], capture_output=True, text=True)
            st.code(r.stdout); st.rerun()
    else:
        conn = sqlite3.connect(DB_PATH)
        total_p    = pd.read_sql("SELECT COUNT(*) as c FROM patients", conn).iloc[0,0]
        total_pred = pd.read_sql("SELECT COUNT(*) as c FROM predictions", conn).iloc[0,0]
        high_c     = pd.read_sql("SELECT COUNT(*) as c FROM predictions WHERE risk_category='High Risk'", conn).iloc[0,0]
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Patients Stored",   int(total_p))
        c2.metric("Total Predictions", int(total_pred))
        c3.metric("High Risk",         int(high_c))
        c4.metric("Database Status",   "✅ Active")
        st.markdown("---")
        st.subheader("All Patient Predictions")
        df = pd.read_sql_query("""
            SELECT p.name, p.age, p.sex, pr.disease, pr.risk_score,
                   pr.risk_category, pr.predicted_at
            FROM predictions pr JOIN patients p ON p.patient_id = pr.patient_id
            ORDER BY pr.predicted_at DESC
        """, conn)
        def color_risk(val):
            if val == "High Risk":   return "background-color:#f8d7da"
            if val == "Medium Risk": return "background-color:#fff3cd"
            if val == "Low Risk":    return "background-color:#d4edda"
            return ""
        if not df.empty:
            st.dataframe(df.style.applymap(color_risk, subset=["risk_category"]),
                         use_container_width=True, height=350)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Export Database as CSV", csv, "patient_predictions.csv", "text/csv")
        else:
            st.info("No records yet.")
        st.markdown("---")
        st.subheader("Risk Distribution in Database")
        risk_df = pd.read_sql("""
            SELECT disease, risk_category, COUNT(*) as count
            FROM predictions GROUP BY disease, risk_category
        """, conn)
        if not risk_df.empty:
            fig, axes = plt.subplots(1,2, figsize=(12,4))
            for ax, disease in zip(axes, risk_df["disease"].unique()):
                sub  = risk_df[risk_df["disease"]==disease]
                clrs = {"Low Risk":"#27ae60","Medium Risk":"#f39c12","High Risk":"#e74c3c"}
                ax.bar(sub["risk_category"], sub["count"],
                       color=[clrs.get(c,"#3498db") for c in sub["risk_category"]])
                ax.set_title(disease, fontsize=12, fontweight="bold")
                ax.set_ylabel("Patients"); ax.set_facecolor("#f8f9fa")
                for i,v in enumerate(sub["count"]): ax.text(i, v+0.1, str(v), ha="center", fontweight="bold")
            fig.patch.set_facecolor("#f8f9fa"); plt.tight_layout(); st.pyplot(fig); plt.close()
        conn.close()
        st.markdown("---")
        st.subheader("➕ Add New Patient")
        with st.form("add_patient"):
            nc1,nc2,nc3 = st.columns(3)
            with nc1: p_name = st.text_input("Name","John Doe")
            with nc2: p_age  = st.number_input("Age",18,100,40)
            with nc3: p_sex  = st.selectbox("Sex",["Male","Female"])
            p_disease = st.selectbox("Disease",["Heart Disease","Diabetes"])
            p_risk    = st.selectbox("Risk Category",["Low Risk","Medium Risk","High Risk"])
            p_score   = st.slider("Risk Score %", 0.0, 100.0, 50.0)
            if st.form_submit_button("Save Patient"):
                from database import add_patient, save_prediction
                pid = add_patient(p_name, int(p_age), p_sex)
                save_prediction(pid, p_disease, p_score/100, p_risk, "Manual Entry")
                st.success(f"✅ Saved '{p_name}' (ID: {pid})"); st.rerun()


# ════════════════════════════════════════════════════════════
# PAGE 7 — BATCH PREDICTIONS
# ════════════════════════════════════════════════════════════
elif "Batch" in page:
    if "📦  Batch Predictions" not in allowed_pages: access_denied()
    st.title("📦 Batch Predictions")
    st.markdown("Run predictions on all patients in both datasets simultaneously.")
    st.markdown("---")
    if st.button("▶️ Run Batch Predictions", type="primary", use_container_width=True):
        with st.spinner("Running predictions on 1,688 patients..."):
            from batch_prediction import run_batch_predictions
            df = run_batch_predictions()
            st.session_state["batch_df"] = df
        st.success(f"✅ Done! {len(df):,} predictions completed.")
    if os.path.exists("data/batch_predictions.csv"):
        batch_df = pd.read_csv("data/batch_predictions.csv")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total",       f"{len(batch_df):,}")
        c2.metric("High Risk",   int((batch_df["risk_category"]=="High Risk").sum()))
        c3.metric("Medium Risk", int((batch_df["risk_category"]=="Medium Risk").sum()))
        c4.metric("Accuracy",    f"{batch_df['correct'].mean()*100:.1f}%")
        st.markdown("---")
        col1,col2 = st.columns(2)
        with col1: dis_f = st.selectbox("Disease",["All","Heart Disease","Diabetes"])
        with col2: ris_f = st.selectbox("Risk",   ["All","Low Risk","Medium Risk","High Risk"])
        view = batch_df.copy()
        if dis_f != "All": view = view[view["disease"]==dis_f]
        if ris_f != "All": view = view[view["risk_category"]==ris_f]
        def color_risk2(val):
            if val=="High Risk":   return "background-color:#f8d7da"
            if val=="Medium Risk": return "background-color:#fff3cd"
            if val=="Low Risk":    return "background-color:#d4edda"
            return ""
        st.dataframe(view.style.applymap(color_risk2, subset=["risk_category"]),
                     use_container_width=True, height=380)
        st.download_button("⬇️ Download CSV", view.to_csv(index=False).encode(),
                           "batch_predictions.csv", "text/csv", use_container_width=True)
        st.markdown("---")
        fig, axes = plt.subplots(1,2, figsize=(12,4))
        for ax, dis in zip(axes, ["Heart Disease","Diabetes"]):
            sub = batch_df[batch_df["disease"]==dis]
            for cat, color in [("Low Risk","#27ae60"),("Medium Risk","#f39c12"),("High Risk","#e74c3c")]:
                ax.hist(sub[sub["risk_category"]==cat]["risk_score_pct"],
                        bins=20, alpha=0.7, color=color, label=cat)
            ax.set_xlabel("Risk Score (%)"); ax.set_ylabel("Count")
            ax.set_title(dis, fontsize=12, fontweight="bold")
            ax.legend(fontsize=9); ax.set_facecolor("#f8f9fa")
        fig.patch.set_facecolor("#f8f9fa"); plt.tight_layout(); st.pyplot(fig); plt.close()
    else:
        st.info("Click the button above to run batch predictions first.")


# ════════════════════════════════════════════════════════════
# PAGE 8 — GENERATE REPORT
# ════════════════════════════════════════════════════════════
elif "Report" in page:
    if "📄  Generate Report" not in allowed_pages: access_denied()
    st.title("📄 Generate Patient Risk Report")
    st.markdown("---")
    c1,c2 = st.columns(2)
    with c1:
        p_name   = st.text_input("Patient Name",   "Rajesh Kumar")
        p_age    = st.number_input("Age", 18, 100, 45)
        p_sex    = st.selectbox("Sex", ["Male","Female"])
        p_id     = st.text_input("Patient ID",     "PAT-2024-001")
    with c2:
        p_doctor = st.text_input("Referring Doctor","Dr. Priya Sharma")
    st.markdown("---")
    st.subheader("Prediction Scores")
    st.caption("Run Heart Disease / Diabetes predictor first, or enter manually below.")
    hr  = st.session_state.get("heart_result",   {"risk_score_pct":55.0,"risk_category":"Medium Risk"})
    dia = st.session_state.get("diabetes_result",{"risk_score_pct":40.0,"risk_category":"Medium Risk"})
    rc1,rc2 = st.columns(2)
    with rc1:
        st.markdown("**Heart Disease**")
        h_score = st.slider("Heart Risk %", 0.0, 100.0, float(hr["risk_score_pct"]))
        cats    = ["Low Risk","Medium Risk","High Risk"]
        h_cat   = st.selectbox("Heart Category", cats,
                               index=cats.index(hr["risk_category"]) if hr["risk_category"] in cats else 1)
    with rc2:
        st.markdown("**Diabetes**")
        d_score = st.slider("Diabetes Risk %", 0.0, 100.0, float(dia["risk_score_pct"]))
        d_cat   = st.selectbox("Diabetes Category", cats,
                               index=cats.index(dia["risk_category"]) if dia["risk_category"] in cats else 1)
    st.markdown("---")

    # Generate PDF
    if st.button("📄 Generate PDF Report", type="primary", use_container_width=True):
        try:
            from generate_report import generate_pdf_report, REPORTLAB_AVAILABLE
            if not REPORTLAB_AVAILABLE:
                st.error("Install ReportLab first: pip install reportlab")
            else:
                patient_info = {"name":p_name,"age":p_age,"sex":p_sex,
                                "id":p_id,"doctor":p_doctor,
                                "date":datetime.now().strftime("%d %B %Y, %I:%M %p")}
                hr_full  = {"risk_score_pct":h_score,"risk_category":h_cat,
                            "recommendations":get_recs("heart",   h_cat,{})}
                dia_full = {"risk_score_pct":d_score,"risk_category":d_cat,
                            "recommendations":get_recs("diabetes",d_cat,{})}
                os.makedirs("reports", exist_ok=True)
                out = f"reports/{p_name.replace(' ','_')}_report.pdf"
                generate_pdf_report(patient_info, hr_full, dia_full, out)
                with open(out,"rb") as f: pdf_bytes = f.read()
                st.success(f"✅ Report generated for {p_name}!")
                st.download_button("⬇️ Download PDF Report", pdf_bytes,
                                   f"{p_name.replace(' ','_')}_report.pdf",
                                   "application/pdf", use_container_width=True)
                st.session_state["report_path"]  = out
                st.session_state["report_hr"]    = hr_full
                st.session_state["report_dia"]   = dia_full
                st.session_state["report_name"]  = p_name
                st.session_state["report_ready"] = True
                st.markdown("---")
                st.subheader("Report Preview")
                pa,pb = st.columns(2)
                with pa:
                    icon = "🔴" if "High" in h_cat else ("🟡" if "Medium" in h_cat else "🟢")
                    st.markdown(f"**Heart Disease** {icon} {h_cat} — {h_score:.1f}%")
                    for r in hr_full["recommendations"]: st.markdown(r)
                with pb:
                    icon = "🔴" if "High" in d_cat else ("🟡" if "Medium" in d_cat else "🟢")
                    st.markdown(f"**Diabetes** {icon} {d_cat} — {d_score:.1f}%")
                    for r in dia_full["recommendations"]: st.markdown(r)
        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Make sure reportlab is installed: pip install reportlab")

    # ── Email via Gmail SMTP ──────────────────────────────
    st.markdown("---")
    st.subheader("📧 Email Report to Patient")

    with st.expander("⚙️ How to get Gmail App Password — Click to read", expanded=False):
        st.markdown("""
        **You need a Gmail App Password (16 chars) — NOT your normal Gmail password**

        **Steps (takes 2 minutes):**
        1. Go to 👉 [myaccount.google.com](https://myaccount.google.com)
        2. Click **Security** on the left side
        3. Turn ON **2-Step Verification** (if not already on)
        4. In the search bar at top, type **App Passwords** → click it
        5. Under "Select App" choose **Mail**
        6. Under "Select Device" choose **Windows Computer**
        7. Click **Generate**
        8. Copy the **16-character password** shown (like: `abcd efgh ijkl mnop`)
        9. Paste it in the **App Password** field below

        ✅ That's it! No other setup needed.
        """)

    with st.form("gmail_form"):
        st.markdown("#### Your Gmail (Sender)")
        gc1, gc2 = st.columns(2)
        with gc1:
            sender_email = st.text_input(
                "Your Gmail Address",
                placeholder="hospital@gmail.com"
            )
        with gc2:
            sender_password = st.text_input(
                "Gmail App Password (16 chars)",
                type="password",
                placeholder="xxxx xxxx xxxx xxxx"
            )

        st.markdown("#### Patient Details")
        pc1, pc2 = st.columns(2)
        with pc1:
            patient_email = st.text_input(
                "Patient Email Address",
                placeholder="patient@gmail.com"
            )
        with pc2:
            email_doctor = st.text_input(
                "Doctor Name",
                value=p_doctor if p_doctor else "Healthcare Team"
            )

        send_btn = st.form_submit_button(
            "📨 Send Report to Patient",
            use_container_width=True,
            type="primary"
        )

        if send_btn:
            if not sender_email or not sender_password:
                st.error("❌ Please enter your Gmail address and App Password.")
            elif "@" not in sender_email:
                st.error("❌ Invalid sender Gmail address.")
            elif not patient_email or "@" not in patient_email:
                st.error("❌ Please enter a valid patient email address.")
            elif not st.session_state.get("report_ready"):
                st.error("❌ Please click Generate PDF Report first before sending email.")
            else:
                with st.spinner(f"Sending report to {patient_email}..."):
                    try:
                        from send_email import send_report_email
                        result = send_report_email(
                            sender_email    = sender_email,
                            sender_password = sender_password,
                            patient_email   = patient_email,
                            patient_name    = st.session_state.get("report_name", p_name),
                            pdf_path        = st.session_state.get("report_path", ""),
                            heart_result    = st.session_state.get("report_hr", {}),
                            diabetes_result = st.session_state.get("report_dia", {}),
                            doctor_name     = email_doctor,
                        )
                        if result["success"]:
                            st.success(f"✅ {result['message']}")
                            st.balloons()
                            st.info(f"📬 Report delivered to **{patient_email}** with PDF attached!")
                        else:
                            st.error(result["message"])
                    except Exception as e:
                        st.error(f"Error sending email: {e}")

    st.markdown("---")
    st.info("""
    💡 **Workflow — Do these in order:**
    1. Fill patient name, age, doctor details at the top
    2. Set risk scores using the sliders
    3. Click **Generate PDF Report**
    4. Enter your Gmail + App Password + patient email
    5. Click **Send Report to Patient**
    """)