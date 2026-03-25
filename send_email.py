# ============================================================
# STEP 10 — EMAIL PDF REPORT TO PATIENT
# Uses Gmail SMTP to send the PDF as an attachment
# ============================================================

import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text       import MIMEText
from email.mime.base       import MIMEBase
from email                 import encoders
from datetime              import datetime


def send_report_email(
    sender_email:    str,
    sender_password: str,
    patient_email:   str,
    patient_name:    str,
    pdf_path:        str,
    heart_result:    dict,
    diabetes_result: dict,
) -> dict:
    """
    Send the PDF risk report to the patient via Gmail SMTP.

    Parameters
    ----------
    sender_email    : your Gmail address (e.g. hospital@gmail.com)
    sender_password : Gmail App Password (16-char, NOT your login password)
    patient_email   : patient's email address
    patient_name    : patient's full name
    pdf_path        : path to the generated PDF file
    heart_result    : dict with risk_score_pct and risk_category
    diabetes_result : dict with risk_score_pct and risk_category

    Returns
    -------
    dict with keys: success (bool), message (str)
    """

    # ── Validate PDF exists ────────────────────────────────
    if not os.path.exists(pdf_path):
        return {"success": False, "message": f"PDF not found at: {pdf_path}"}

    try:
        # ── Build email ────────────────────────────────────
        msg = MIMEMultipart("mixed")
        msg["From"]    = sender_email
        msg["To"]      = patient_email
        msg["Subject"] = f"🏥 Your Health Risk Assessment Report — {patient_name}"

        # ── Risk icons for email body ──────────────────────
        def risk_icon(cat):
            if "Low"    in cat: return "🟢"
            if "Medium" in cat: return "🟡"
            return "🔴"

        h_icon = risk_icon(heart_result.get("risk_category", ""))
        d_icon = risk_icon(diabetes_result.get("risk_category", ""))
        h_score = heart_result.get("risk_score_pct", 0)
        d_score = diabetes_result.get("risk_score_pct", 0)
        h_cat   = heart_result.get("risk_category", "N/A")
        d_cat   = diabetes_result.get("risk_category", "N/A")
        date_str = datetime.now().strftime("%d %B %Y")

        # ── HTML email body ────────────────────────────────
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; background:#f4f6f9; padding:0; margin:0;">

        <div style="max-width:600px; margin:30px auto; background:white;
                    border-radius:12px; overflow:hidden;
                    box-shadow:0 4px 20px rgba(0,0,0,0.1);">

            <!-- HEADER -->
            <div style="background:#2980b9; padding:30px; text-align:center;">
                <h1 style="color:white; margin:0; font-size:24px;">🏥 Healthcare Risk Analytics</h1>
                <p style="color:#d6eaf8; margin:8px 0 0;">Your Personal Health Risk Assessment Report</p>
            </div>

            <!-- GREETING -->
            <div style="padding:30px 35px 10px;">
                <p style="font-size:16px; color:#2c3e50;">Dear <b>{patient_name}</b>,</p>
                <p style="color:#555; line-height:1.7;">
                    Please find attached your personalised health risk assessment report
                    generated on <b>{date_str}</b>. The report contains your disease risk
                    scores, risk categories, and clinical recommendations based on your
                    health data.
                </p>
            </div>

            <!-- RISK SUMMARY BOX -->
            <div style="margin:10px 35px 20px; border-radius:10px;
                        border:1px solid #e0e0e0; overflow:hidden;">

                <div style="background:#2c3e50; padding:12px 20px;">
                    <h3 style="color:white; margin:0; font-size:15px;">📊 Risk Summary</h3>
                </div>

                <!-- Heart Disease Row -->
                <div style="display:flex; padding:16px 20px; border-bottom:1px solid #f0f0f0;
                            align-items:center; background:#fafafa;">
                    <div style="flex:1;">
                        <b style="color:#2c3e50;">🫀 Heart Disease</b>
                    </div>
                    <div style="flex:1; text-align:center;">
                        <span style="font-size:22px; font-weight:bold; color:#2980b9;">
                            {h_score:.1f}%
                        </span>
                    </div>
                    <div style="flex:1; text-align:right;">
                        <span style="
                            padding:5px 14px; border-radius:20px; font-weight:bold;
                            font-size:13px; color:white;
                            background:{'#27ae60' if 'Low' in h_cat else ('#f39c12' if 'Medium' in h_cat else '#e74c3c')};">
                            {h_icon} {h_cat}
                        </span>
                    </div>
                </div>

                <!-- Diabetes Row -->
                <div style="display:flex; padding:16px 20px; align-items:center;">
                    <div style="flex:1;">
                        <b style="color:#2c3e50;">🩸 Diabetes</b>
                    </div>
                    <div style="flex:1; text-align:center;">
                        <span style="font-size:22px; font-weight:bold; color:#2980b9;">
                            {d_score:.1f}%
                        </span>
                    </div>
                    <div style="flex:1; text-align:right;">
                        <span style="
                            padding:5px 14px; border-radius:20px; font-weight:bold;
                            font-size:13px; color:white;
                            background:{'#27ae60' if 'Low' in d_cat else ('#f39c12' if 'Medium' in d_cat else '#e74c3c')};">
                            {d_icon} {d_cat}
                        </span>
                    </div>
                </div>
            </div>

            <!-- RISK SCALE -->
            <div style="margin:0 35px 20px; padding:16px; background:#f8f9fa;
                        border-radius:8px; border-left:4px solid #2980b9;">
                <p style="margin:0 0 8px; font-weight:bold; color:#2c3e50;">Risk Scale Reference:</p>
                <p style="margin:4px 0; color:#555; font-size:13px;">
                    🟢 <b>Low Risk (0–30%)</b> — Annual routine check-up recommended
                </p>
                <p style="margin:4px 0; color:#555; font-size:13px;">
                    🟡 <b>Medium Risk (30–60%)</b> — Schedule tests within 1 month
                </p>
                <p style="margin:4px 0; color:#555; font-size:13px;">
                    🔴 <b>High Risk (60–100%)</b> — Immediate medical attention required
                </p>
            </div>

            <!-- NOTE -->
            <div style="margin:0 35px 25px; padding:14px 18px; background:#fff3cd;
                        border-radius:8px; border-left:4px solid #f39c12;">
                <p style="margin:0; color:#856404; font-size:13px;">
                    <b>⚠️ Important:</b> This report is generated by an AI-assisted analytics
                    system and is intended to <b>support</b>, not replace, professional medical
                    diagnosis. Please consult your doctor for proper evaluation.
                </p>
            </div>

            <!-- ATTACHMENT NOTE -->
            <div style="margin:0 35px 25px; color:#555; font-size:14px; line-height:1.6;">
                <p>📎 Your full detailed report is attached as a <b>PDF file</b>.
                It includes clinical recommendations specific to your risk level.</p>
                <p>If you have any questions about your results, please contact
                your healthcare provider.</p>
                <p>Stay healthy! 💙</p>
            </div>

            <!-- FOOTER -->
            <div style="background:#2c3e50; padding:20px 35px; text-align:center;">
                <p style="color:#95a5a6; font-size:12px; margin:0;">
                    This email was sent by the Healthcare Risk Analytics System.
                    Report generated on {date_str}.
                </p>
                <p style="color:#7f8c8d; font-size:11px; margin:6px 0 0;">
                    Powered by Machine Learning • UCI Heart Disease + Pima Diabetes Datasets
                </p>
            </div>

        </div>
        </body>
        </html>
        """

        # Attach HTML body
        msg.attach(MIMEText(html_body, "html"))

        # ── Attach PDF ─────────────────────────────────────
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()

        pdf_part = MIMEBase("application", "octet-stream")
        pdf_part.set_payload(pdf_data)
        encoders.encode_base64(pdf_part)
        pdf_filename = os.path.basename(pdf_path)
        pdf_part.add_header("Content-Disposition",
                            f'attachment; filename="{pdf_filename}"')
        msg.attach(pdf_part)

        # ── Send via Gmail SMTP ────────────────────────────
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, patient_email, msg.as_string())

        return {
            "success": True,
            "message": f"✅ Report successfully sent to {patient_email}"
        }

    except smtplib.SMTPAuthenticationError:
        return {
            "success": False,
            "message": (
                "❌ Gmail authentication failed.\n"
                "Make sure you are using an App Password, not your Gmail login password.\n"
                "Go to: Google Account → Security → 2-Step Verification → App Passwords"
            )
        }
    except smtplib.SMTPException as e:
        return {"success": False, "message": f"❌ SMTP error: {str(e)}"}
    except Exception as e:
        return {"success": False, "message": f"❌ Error: {str(e)}"}


# ─────────────────────────────────────────
# TEST (run directly)
# ─────────────────────────────────────────
if __name__ == "__main__":
    result = send_report_email(
        sender_email    = "your_hospital_email@gmail.com",
        sender_password = "your_16_char_app_password",
        patient_email   = "patient@example.com",
        patient_name    = "Rajesh Kumar",
        pdf_path        = "reports/sample_report.pdf",
        heart_result    = {"risk_score_pct": 79.2, "risk_category": "High Risk"},
        diabetes_result = {"risk_score_pct": 89.7, "risk_category": "High Risk"},
    )
    print(result["message"])