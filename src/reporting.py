"""Clinical Dossier and Report Generation Engine using ReportLab, CSV, and JSON."""
from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from typing import Any, Dict, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def generate_pdf_report(
    prediction_data: Dict[str, Any],
    benchmark_data: List[Dict[str, Any]] | None = None,
    importance_data: List[Dict[str, Any]] | None = None,
) -> bytes:
    """
    Generate a ReportLab PDF Clinical Intelligence & Risk Dossier.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )
    styles = getSampleStyleSheet()

    # Custom styles
    header_style = ParagraphStyle(
        "HeaderTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748b"),
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#0f766e"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#64748b"),
    )

    story = []

    # Title & Metadata
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    story.append(Paragraph("Smart Healthcare System", header_style))
    story.append(Paragraph("AI-Driven Clinical Risk Analysis Dossier · Educational ML Decision Support", subtitle_style))
    story.append(Paragraph(f"Generated: {now_str} · Protocol: PIMA-INDIANS-ML-2026", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0f766e"), spaceAfter=14))

    # Executive Prediction Summary
    model_name = prediction_data.get("model", "Random Forest")
    probability = float(prediction_data.get("probability", 0.0))
    risk_pct = f"{probability * 100:.1f}%"
    risk_level = prediction_data.get("risk_level", "Moderate")
    prediction_val = prediction_data.get("prediction", 0)
    risk_badge = "Elevated Diabetes Likelihood" if prediction_val == 1 else "Lower Diabetes Likelihood"

    badge_color = colors.HexColor("#dc2626") if prediction_val == 1 else colors.HexColor("#16a34a")

    summary_data = [
        [
            Paragraph("<b>Evaluated ML Model:</b>", body_style),
            Paragraph(f"<b>{model_name}</b>", body_style),
            Paragraph("<b>Risk Classification:</b>", body_style),
            Paragraph(f"<font color='{badge_color.hexval()}'><b>{risk_badge}</b></font>", body_style),
        ],
        [
            Paragraph("<b>Estimated Risk Score:</b>", body_style),
            Paragraph(f"<b>{risk_pct}</b> (Probability {probability:.4f})", body_style),
            Paragraph("<b>Risk Tier:</b>", body_style),
            Paragraph(f"<b>{risk_level}</b>", body_style),
        ],
    ]

    summary_table = Table(summary_data, colWidths=[1.8 * inch, 2.0 * inch, 1.6 * inch, 2.0 * inch])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#e2e8f0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))

    story.append(Paragraph("1. Assessment Overview", section_heading))
    story.append(summary_table)
    story.append(Spacer(1, 10))

    # Patient Diagnostic Parameters Table
    inputs = prediction_data.get("inputs", {})
    if inputs:
        story.append(Paragraph("2. Patient Diagnostic Parameters", section_heading))
        input_rows = [
            [
                Paragraph("<b>Diagnostic Parameter</b>", body_style),
                Paragraph("<b>Entered Value</b>", body_style),
                Paragraph("<b>Clinical Unit</b>", body_style),
                Paragraph("<b>Research Baseline Ref</b>", body_style),
            ]
        ]
        units = {
            "Pregnancies": "Count",
            "Glucose": "mg/dL (2h oral glucose)",
            "BloodPressure": "mm Hg (Diastolic)",
            "SkinThickness": "mm (Triceps skinfold)",
            "Insulin": "μU/mL (2h serum)",
            "BMI": "kg/m² (Body Mass Index)",
            "DiabetesPedigreeFunction": "Genetic Score",
            "Age": "Years",
        }
        refs = {
            "Pregnancies": "0 – 17",
            "Glucose": "70 – 140 normal",
            "BloodPressure": "60 – 80 optimal",
            "SkinThickness": "10 – 50",
            "Insulin": "16 – 166 fasting/post",
            "BMI": "18.5 – 24.9 normal",
            "DiabetesPedigreeFunction": "0.08 – 2.42",
            "Age": "21 – 81",
        }
        for feat, val in inputs.items():
            input_rows.append([
                Paragraph(feat, body_style),
                Paragraph(f"<b>{val:.2f}</b>" if isinstance(val, float) else str(val), body_style),
                Paragraph(units.get(feat, "—"), body_style),
                Paragraph(refs.get(feat, "—"), body_style),
            ])

        param_table = Table(input_rows, colWidths=[2.2 * inch, 1.5 * inch, 2.0 * inch, 1.7 * inch])
        param_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(param_table)
        story.append(Spacer(1, 10))

    # Benchmark Summary Table
    if benchmark_data:
        story.append(Paragraph("3. Multi-Model Benchmark & Performance Validation", section_heading))
        bench_rows = [
            [
                Paragraph("<b>Model Name</b>", body_style),
                Paragraph("<b>Accuracy</b>", body_style),
                Paragraph("<b>Precision</b>", body_style),
                Paragraph("<b>Recall</b>", body_style),
                Paragraph("<b>F1 Score</b>", body_style),
                Paragraph("<b>ROC-AUC</b>", body_style),
                Paragraph("<b>PR-AUC</b>", body_style),
            ]
        ]
        for bm in benchmark_data:
            is_active = bm.get("model") == model_name
            prefix = "<b>* " if is_active else ""
            suffix = "</b>" if is_active else ""
            bench_rows.append([
                Paragraph(f"{prefix}{bm.get('model', '')}{suffix}", body_style),
                Paragraph(f"{float(bm.get('accuracy', 0)):.3f}", body_style),
                Paragraph(f"{float(bm.get('precision', 0)):.3f}", body_style),
                Paragraph(f"{float(bm.get('recall', 0)):.3f}", body_style),
                Paragraph(f"{float(bm.get('f1', 0)):.3f}", body_style),
                Paragraph(f"<b>{float(bm.get('roc_auc', 0)):.3f}</b>", body_style),
                Paragraph(f"{float(bm.get('pr_auc', 0)):.3f}", body_style),
            ])

        bench_table = Table(bench_rows, colWidths=[1.8 * inch, 0.9 * inch, 0.9 * inch, 0.9 * inch, 0.9 * inch, 1.0 * inch, 1.0 * inch])
        bench_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(bench_table)
        story.append(Spacer(1, 10))

    # Feature Importance
    if importance_data:
        story.append(Paragraph(f"4. Global Feature Explainability ({model_name})", section_heading))
        imp_rows = [
            [
                Paragraph("<b>Predictive Feature</b>", body_style),
                Paragraph("<b>Relative Importance Weight</b>", body_style),
                Paragraph("<b>Significance Interpretation</b>", body_style),
            ]
        ]
        for item in importance_data[:6]:
            feat = item.get("feature", "")
            imp = float(item.get("importance", 0))
            pct = f"{imp * 100:.1f}%"
            interp = "Primary Risk Factor" if imp > 0.20 else "Moderate Contributor" if imp > 0.10 else "Baseline Diagnostic Factor"
            imp_rows.append([
                Paragraph(feat, body_style),
                Paragraph(f"<b>{pct}</b>", body_style),
                Paragraph(interp, body_style),
            ])

        imp_table = Table(imp_rows, colWidths=[2.5 * inch, 2.0 * inch, 2.9 * inch])
        imp_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(imp_table)
        story.append(Spacer(1, 14))

    # Responsible AI Notice
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=8))
    story.append(Paragraph("<b>Responsible AI & Medical Disclaimer:</b>", disclaimer_style))
    story.append(Paragraph(
        "This analytical report is generated for educational and decision-support demonstration purposes using machine-learning models trained on the PIMA Indians Diabetes Dataset. It does not constitute a clinical diagnosis, prognosis, or medical treatment recommendation. Healthcare decisions must always be made in consultation with certified medical professionals.",
        disclaimer_style,
    ))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
