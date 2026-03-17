# ============================================================
# STEP 9 — PDF REPORT GENERATOR
# Generates a patient risk report as a PDF
# Run: python step9_generate_report.py
# Requires: pip install reportlab
# ============================================================

import os
import pickle
import numpy as np
from datetime import datetime

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, HRFlowable, Image)
    from reportlab.graphics.shapes import Drawing, Rect, String, Line
    from reportlab.graphics import renderPDF
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


def generate_pdf_report(patient_info: dict, heart_result: dict, diabetes_result: dict,
                         output_path: str = "reports/patient_report.pdf"):
    """
    Generate a professional PDF risk report for a patient.

    Parameters
    ----------
    patient_info  : dict — name, age, sex, date
    heart_result  : dict — from step4_risk_stratification.predict_patient()
    diabetes_result: dict — from step4_risk_stratification.predict_patient()
    output_path   : where to save the PDF
    """
    if not REPORTLAB_AVAILABLE:
        print("ReportLab not installed. Run: pip install reportlab")
        return

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    doc    = SimpleDocTemplate(output_path, pagesize=A4,
                                topMargin=2*cm, bottomMargin=2*cm,
                                leftMargin=2*cm, rightMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []

    # ── colour palette ─────────────────────────────────────
    BLUE   = colors.HexColor("#2980b9")
    DARK   = colors.HexColor("#2c3e50")
    GREEN  = colors.HexColor("#27ae60")
    ORANGE = colors.HexColor("#f39c12")
    RED    = colors.HexColor("#e74c3c")
    GRAY   = colors.HexColor("#ecf0f1")
    WHITE  = colors.white

    def risk_color(cat):
        if "Low"    in cat: return GREEN
        if "Medium" in cat: return ORANGE
        return RED

    # ── custom styles ───────────────────────────────────────
    title_style = ParagraphStyle("title", parent=styles["Title"],
                                  fontSize=22, textColor=DARK, spaceAfter=4,
                                  fontName="Helvetica-Bold")
    sub_style   = ParagraphStyle("sub", parent=styles["Normal"],
                                  fontSize=11, textColor=BLUE, spaceAfter=2)
    head_style  = ParagraphStyle("head", parent=styles["Heading2"],
                                  fontSize=13, textColor=DARK,
                                  fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6)
    body_style  = ParagraphStyle("body", parent=styles["Normal"],
                                  fontSize=10, textColor=DARK, spaceAfter=4)
    rec_style   = ParagraphStyle("rec", parent=styles["Normal"],
                                  fontSize=10, textColor=DARK,
                                  leftIndent=12, spaceAfter=3)

    # ── HEADER ─────────────────────────────────────────────
    story.append(Paragraph("🏥 Healthcare Risk Analytics", title_style))
    story.append(Paragraph("Multi-Source Disease Risk Assessment Report", sub_style))
    story.append(HRFlowable(width="100%", thickness=2, color=BLUE, spaceAfter=12))

    # ── PATIENT INFO TABLE ─────────────────────────────────
    story.append(Paragraph("Patient Information", head_style))
    report_date = patient_info.get("date", datetime.now().strftime("%d %B %Y, %I:%M %p"))
    info_data = [
        ["Patient Name",  patient_info.get("name", "N/A"),
         "Report Date",   report_date],
        ["Age",           str(patient_info.get("age", "N/A")),
         "Sex",           patient_info.get("sex", "N/A")],
        ["Patient ID",    patient_info.get("id", "AUTO-001"),
         "Referring Doctor", patient_info.get("doctor", "N/A")],
    ]
    info_table = Table(info_data, colWidths=[3.5*cm, 6*cm, 3.5*cm, 5.5*cm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (0, -1), GRAY),
        ("BACKGROUND",  (2, 0), (2, -1), GRAY),
        ("FONTNAME",    (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",    (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 10),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, GRAY]),
        ("PADDING",     (0, 0), (-1, -1), 6),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 14))

    # ── RISK SUMMARY TABLE ─────────────────────────────────
    story.append(Paragraph("Risk Assessment Summary", head_style))

    def pct_bar(score):
        """Simple text-based progress bar."""
        filled = int(score / 5)
        return "█" * filled + "░" * (20 - filled) + f"  {score:.1f}%"

    summary_data = [
        ["Disease", "Risk Score", "Risk Category", "Indicator"],
        [
            "Heart Disease",
            f"{heart_result['risk_score_pct']}%",
            heart_result["risk_category"],
            pct_bar(heart_result["risk_score_pct"]),
        ],
        [
            "Diabetes",
            f"{diabetes_result['risk_score_pct']}%",
            diabetes_result["risk_category"],
            pct_bar(diabetes_result["risk_score_pct"]),
        ],
    ]

    summary_table = Table(summary_data, colWidths=[4*cm, 3*cm, 4*cm, 7.5*cm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), BLUE),
        ("TEXTCOLOR",   (0, 0), (-1, 0), WHITE),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 10),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("ALIGN",       (3, 1), (3, -1), "LEFT"),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
        ("PADDING",     (0, 0), (-1, -1), 7),
        ("FONTNAME",    (0, 1), (0, -1), "Helvetica-Bold"),
        # Risk category cell colors
        ("BACKGROUND",  (2, 1), (2, 1), risk_color(heart_result["risk_category"])),
        ("BACKGROUND",  (2, 2), (2, 2), risk_color(diabetes_result["risk_category"])),
        ("TEXTCOLOR",   (2, 1), (2, -1), WHITE),
        ("FONTNAME",    (2, 1), (2, -1), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, GRAY]),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 14))

    # ── HEART DISEASE DETAIL ───────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#bdc3c7")))
    story.append(Paragraph("Heart Disease — Detailed Assessment", head_style))
    story.append(Paragraph(
        f"<b>Risk Score:</b> {heart_result['risk_score_pct']}%  &nbsp;&nbsp; "
        f"<b>Category:</b> {heart_result['risk_category']}", body_style
    ))
    story.append(Paragraph("<b>Clinical Recommendations:</b>", body_style))
    for rec in heart_result.get("recommendations", []):
        clean = rec.replace("🚨","[URGENT]").replace("📋","[NOTE]").replace("💊","[MED]") \
                   .replace("🏥","[HOSPITAL]").replace("🥗","[DIET]").replace("🚶","[EXERCISE]") \
                   .replace("✅","[OK]").replace("📅","[SCHEDULE]").replace("⚠️","[WARNING]") \
                   .replace("👨‍⚕️","[DOCTOR]").replace("💉","[MONITOR]")
        story.append(Paragraph(f"• {clean}", rec_style))

    story.append(Spacer(1, 10))

    # ── DIABETES DETAIL ────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#bdc3c7")))
    story.append(Paragraph("Diabetes — Detailed Assessment", head_style))
    story.append(Paragraph(
        f"<b>Risk Score:</b> {diabetes_result['risk_score_pct']}%  &nbsp;&nbsp; "
        f"<b>Category:</b> {diabetes_result['risk_category']}", body_style
    ))
    story.append(Paragraph("<b>Clinical Recommendations:</b>", body_style))
    for rec in diabetes_result.get("recommendations", []):
        clean = rec.replace("🚨","[URGENT]").replace("📋","[NOTE]").replace("💊","[MED]") \
                   .replace("🏥","[HOSPITAL]").replace("🥗","[DIET]").replace("🚶","[EXERCISE]") \
                   .replace("✅","[OK]").replace("📅","[SCHEDULE]").replace("⚠️","[WARNING]") \
                   .replace("👨‍⚕️","[DOCTOR]").replace("💉","[MONITOR]")
        story.append(Paragraph(f"• {clean}", rec_style))

    story.append(Spacer(1, 14))

    # ── RISK SCALE LEGEND ──────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#bdc3c7")))
    story.append(Paragraph("Risk Scale Reference", head_style))
    legend_data = [
        ["Category",    "Score Range", "Recommended Action"],
        ["Low Risk",    "0% — 30%",    "Annual routine check-up"],
        ["Medium Risk", "30% — 60%",   "Schedule tests within 1 month"],
        ["High Risk",   "60% — 100%",  "Immediate medical attention required"],
    ]
    legend_table = Table(legend_data, colWidths=[4*cm, 4*cm, 10.5*cm])
    legend_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), DARK),
        ("TEXTCOLOR",   (0, 0), (-1, 0), WHITE),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 10),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
        ("PADDING",     (0, 0), (-1, -1), 7),
        ("BACKGROUND",  (0, 1), (0, 1), GREEN),
        ("BACKGROUND",  (0, 2), (0, 2), ORANGE),
        ("BACKGROUND",  (0, 3), (0, 3), RED),
        ("TEXTCOLOR",   (0, 1), (0, -1), WHITE),
        ("FONTNAME",    (0, 1), (0, -1), "Helvetica-Bold"),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("ALIGN",       (2, 1), (2, -1), "LEFT"),
    ]))
    story.append(legend_table)
    story.append(Spacer(1, 16))

    # ── FOOTER ─────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=2, color=BLUE))
    story.append(Spacer(1, 6))
    footer_text = (
        "<font size='8' color='#7f8c8d'>"
        "This report is generated by the Multi-Source Healthcare Analytics System. "
        "It is intended to assist healthcare professionals and does NOT replace clinical diagnosis. "
        "All predictions are based on machine learning models trained on UCI Heart Disease "
        "and Pima Indians Diabetes datasets. "
        f"Generated: {datetime.now().strftime('%d %B %Y at %I:%M %p')}"
        "</font>"
    )
    story.append(Paragraph(footer_text, styles["Normal"]))

    doc.build(story)
    print(f"[PDF] Report saved → {output_path}")
    return output_path


# ─────────────────────────────────────────
# TEST RUN WITH SAMPLE PATIENT
# ─────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from risk import predict_patient

    # Sample patient
    patient_info = {
        "name"   : "Rajesh Kumar",
        "age"    : 55,
        "sex"    : "Male",
        "id"     : "PAT-2024-001",
        "doctor" : "Dr. Priya Sharma",
        "date"   : datetime.now().strftime("%d %B %Y, %I:%M %p"),
    }

    heart_patient = {
        "age": 55, "sex": 1, "cp": 2, "trestbps": 148,
        "chol": 260, "fbs": 1, "restecg": 1, "thalch": 108,
        "exang": 1, "oldpeak": 2.8,
        "age_group": 2, "bp_category": 2, "high_cholesterol": 1,
        "low_max_hr": 1, "risk_factor_count": 4
    }

    diabetes_patient = {
        "Pregnancies": 3, "Glucose": 162, "BloodPressure": 88,
        "SkinThickness": 35, "Insulin": 130, "BMI": 35.8,
        "DiabetesPedigreeFunction": 0.82, "Age": 55,
        "bmi_category": 3, "glucose_risk": 2, "age_group": 2,
        "insulin_resistance": 1, "risk_factor_count": 4
    }

    heart_result    = predict_patient("heart_disease", heart_patient)
    diabetes_result = predict_patient("diabetes",      diabetes_patient)

    os.makedirs("reports", exist_ok=True)
    generate_pdf_report(patient_info, heart_result, diabetes_result,
                        output_path="reports/sample_report.pdf")