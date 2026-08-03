"""CAVYAA 1.0 — Healing Journey Web App."""

from __future__ import annotations

import random
import os
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "cavyaa-dev-key-change-in-prod")

# ---------------------------------------------------------------------------
# Mock data — real backend will replace this
# ---------------------------------------------------------------------------

PATIENTS = [
    {
        "id": "P001",
        "name": "Olivia Chen",
        "age": 34,
        "cancer_type": "Lung Adenocarcinoma",
        "weeks_post_op": 24,
        "wellness_score": 83,
        "risk_level": "low",
        "trend": "improving",
        "draws": 8,
        "recurrence_prob": 0.12,
        "last_draw": "2026-07-28",
        "notes": "Responding well to monitoring protocol.",
        "metrics": {
            "roc_auc": 0.91, "sensitivity": 0.88, "specificity": 0.93,
            "fragment_signal": 0.24, "protein_signal": 0.31, "inflammation": 0.08,
        },
    },
    {
        "id": "P002",
        "name": "Marcus Johnson",
        "age": 51,
        "cancer_type": "Glioblastoma",
        "weeks_post_op": 12,
        "wellness_score": 61,
        "risk_level": "medium",
        "trend": "stable",
        "draws": 5,
        "recurrence_prob": 0.34,
        "last_draw": "2026-07-30",
        "notes": "Sequencer batch effect noted; follow up at week 16.",
        "metrics": {
            "roc_auc": 0.87, "sensitivity": 0.82, "specificity": 0.91,
            "fragment_signal": 0.58, "protein_signal": 0.62, "inflammation": 0.22,
        },
    },
    {
        "id": "P003",
        "name": "Sofia Reyes",
        "age": 28,
        "cancer_type": "Osteosarcoma",
        "weeks_post_op": 36,
        "wellness_score": 91,
        "risk_level": "low",
        "trend": "improving",
        "draws": 12,
        "recurrence_prob": 0.08,
        "last_draw": "2026-08-01",
        "notes": "Excellent post-op recovery trajectory.",
        "metrics": {
            "roc_auc": 0.94, "sensitivity": 0.91, "specificity": 0.96,
            "fragment_signal": 0.11, "protein_signal": 0.14, "inflammation": 0.04,
        },
    },
    {
        "id": "P004",
        "name": "David Kim",
        "age": 62,
        "cancer_type": "Neuroblastoma",
        "weeks_post_op": 8,
        "wellness_score": 47,
        "risk_level": "high",
        "trend": "declining",
        "draws": 3,
        "recurrence_prob": 0.61,
        "last_draw": "2026-07-25",
        "notes": "Elevated fragment signal. Schedule additional draw.",
        "metrics": {
            "roc_auc": 0.83, "sensitivity": 0.79, "specificity": 0.87,
            "fragment_signal": 0.82, "protein_signal": 0.74, "inflammation": 0.41,
        },
    },
    {
        "id": "P005",
        "name": "Amara Osei",
        "age": 44,
        "cancer_type": "Melanoma",
        "weeks_post_op": 52,
        "wellness_score": 78,
        "risk_level": "low",
        "trend": "stable",
        "draws": 18,
        "recurrence_prob": 0.19,
        "last_draw": "2026-08-02",
        "notes": "Long-term monitoring; signal decay consistent with recovery.",
        "metrics": {
            "roc_auc": 0.89, "sensitivity": 0.85, "specificity": 0.92,
            "fragment_signal": 0.33, "protein_signal": 0.28, "inflammation": 0.11,
        },
    },
]

DOCTOR_CREDENTIALS = {"dr.chen@cavyaa.io": "demo1234", "doctor": "demo1234"}
PATIENT_CREDENTIALS = {
    "olivia": "P001", "marcus": "P002",
    "sofia": "P003", "david": "P004", "amara": "P005",
}

def _patient_journey_messages(score: int) -> list[str]:
    if score >= 80:
        return [
            "Your body is healing beautifully — keep going.",
            "Every draw tells a story of resilience.",
            "Your signal is calm and steady today.",
        ]
    elif score >= 60:
        return [
            "Progress takes time — you're right on track.",
            "Focus on rest and let your body do its work.",
            "Steady signals. Steady healing.",
        ]
    else:
        return [
            "Every step forward matters, no matter how small.",
            "Your care team is watching over you closely.",
            "Healing isn't linear — and that's okay.",
        ]

def _week_days() -> list[dict]:
    today = datetime.now()
    start = today - timedelta(days=today.weekday())
    days = []
    for i in range(7):
        d = start + timedelta(days=i)
        days.append({
            "label": d.strftime("%a")[0],
            "active": d.date() == today.date(),
            "past": d.date() < today.date(),
        })
    return days

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/favicon.ico")
def favicon():
    return send_from_directory("static", "favicon.svg", mimetype="image/svg+xml")

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/select/<role>")
def select_role(role):
    if role not in ("patient", "doctor"):
        return redirect(url_for("landing"))
    return render_template("login.html", role=role)

@app.route("/login", methods=["POST"])
def login():
    role = request.form.get("role")
    username = request.form.get("username", "").strip().lower()
    password = request.form.get("password", "").strip()

    if role == "doctor":
        if DOCTOR_CREDENTIALS.get(username) == password:
            session["role"] = "doctor"
            session["username"] = username
            return redirect(url_for("doctor_dashboard"))
        return render_template("login.html", role="doctor", error="Invalid credentials. Try doctor / demo1234")

    elif role == "patient":
        patient_id = PATIENT_CREDENTIALS.get(username)
        if patient_id and password == "hello":
            session["role"] = "patient"
            session["patient_id"] = patient_id
            session["username"] = username
            return redirect(url_for("patient_dashboard"))
        return render_template("login.html", role="patient",
                               error="Invalid credentials. Try olivia / hello")

    return redirect(url_for("landing"))

@app.route("/patient")
def patient_dashboard():
    if session.get("role") != "patient":
        return redirect(url_for("landing"))
    patient = next((p for p in PATIENTS if p["id"] == session["patient_id"]), PATIENTS[0])
    messages = _patient_journey_messages(patient["wellness_score"])
    week = _week_days()
    return render_template("patient_dashboard.html", patient=patient,
                           messages=messages, week=week,
                           today=datetime.now().strftime("%A, %d %b %Y"))

@app.route("/doctor")
def doctor_dashboard():
    if session.get("role") != "doctor":
        return redirect(url_for("landing"))
    high = sum(1 for p in PATIENTS if p["risk_level"] == "high")
    medium = sum(1 for p in PATIENTS if p["risk_level"] == "medium")
    avg_auc = sum(p["metrics"]["roc_auc"] for p in PATIENTS) / len(PATIENTS)
    avg_score = sum(p["wellness_score"] for p in PATIENTS) // len(PATIENTS)
    return render_template("doctor_dashboard.html", patients=PATIENTS,
                           high_risk=high, medium_risk=medium,
                           avg_auc=round(avg_auc, 3), avg_score=avg_score,
                           total=len(PATIENTS))

@app.route("/doctor/patient/<pid>")
def patient_detail(pid):
    if session.get("role") != "doctor":
        return redirect(url_for("landing"))
    patient = next((p for p in PATIENTS if p["id"] == pid), None)
    if not patient:
        return redirect(url_for("doctor_dashboard"))
    return render_template("patient_detail.html", patient=patient,
                           week=_week_days(),
                           today=datetime.now().strftime("%A, %d %b %Y"))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("landing"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
