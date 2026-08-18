"""Generate a machine-readable project report from trained artifacts."""
from __future__ import annotations

import json
from pathlib import Path


def load_training_report(model_dir: str = "models") -> dict:
    root = Path(model_dir)
    metadata_path = root / "metadata.json"
    metrics_path = root / "metrics.json"
    if not metadata_path.exists() or not metrics_path.exists():
        raise FileNotFoundError("Training artifacts are missing. Run the training pipeline first.")
    return {
        "metadata": json.loads(metadata_path.read_text(encoding="utf-8")),
        "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
    }


def write_markdown_report(report: dict, path: str = "models/model_report.md") -> None:
    rows = report["metrics"]
    lines = [
        "# Smart Healthcare Model Report",
        "",
        "This report summarizes reproducible model-training results for the PIMA diabetes prediction project.",
        "",
        f"**Selected model:** {report['metadata']['model']}",
        f"**Model version:** {report['metadata']['model_version']}",
        f"**Selection metric:** {report['metadata']['selection_metric']}",
        "",
        "| Model | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | PR-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['accuracy']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} | "
            f"{row['specificity']:.3f} | {row['f1']:.3f} | {row['roc_auc']:.3f} | {row['pr_auc']:.3f} |"
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
