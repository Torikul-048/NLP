"""Plotting and visualization module for academic IR models.

Generates publication-quality result plots (confusion matrices, classification
metrics, retrieval performance, category breakdowns, score distributions, and
cross-model comparisons) and organizes them into model-wise folders.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .utils import ROOT, read_json, write_json


def setup_matplotlib():
    """Configure matplotlib with a headless backend and aesthetic defaults."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
        "axes.edgecolor": "#cccccc",
        "axes.linewidth": 0.8,
        "grid.color": "#ebebeb",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#fafafa",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.autolayout": False,
    })
    return plt


# Friendly display names for models
MODEL_TITLES = {
    "model_a": "Model A — TF-IDF (Sparse)",
    "model_b": "Model B — Word2Vec (Dense Centroid)",
    "model_c": "Model C — BiLSTM (Neural Matcher)",
    "model_d": "Model D — Sentence-BERT (MiniLM)",
}

COLOR_PALETTE = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "success": "#2ca02c",
    "danger": "#d62728",
    "purple": "#9467bd",
    "teal": "#17becf",
    "retrieval": "#2b5c8f",
    "end_to_end": "#1e824c",
}


def compute_binary_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute 2x2 confusion matrix and classification metrics from per-query rows."""
    tp, fp, tn, fn = 0, 0, 0, 0
    for r in rows:
        actual = bool(r.get("has_answer") in (True, "True", 1, "1"))
        predicted = bool(r.get("accepted") in (True, "True", 1, "1"))
        if actual and predicted:
            tp += 1
        elif not actual and predicted:
            fp += 1
        elif not actual and not predicted:
            tn += 1
        else:
            fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    far = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    frr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Total": total,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "Specificity": specificity,
        "False_Acceptance_Rate": far,
        "False_Rejection_Rate": frr,
    }


def plot_confusion_matrix(
    plt,
    metrics: Dict[str, Any],
    title: str,
    output_path: Path,
    dpi: int = 300,
):
    """Render a polished 2x2 Confusion Matrix heatmap with annotations."""
    tp = metrics["TP"]
    fp = metrics["FP"]
    tn = metrics["TN"]
    fn = metrics["FN"]

    matrix = np.array([[tn, fp], [fn, tp]])
    total = max(matrix.sum(), 1)
    percentages = matrix / total * 100

    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=dpi)

    cmap = plt.cm.Blues
    cax = ax.imshow(matrix, interpolation="nearest", cmap=cmap)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    labels = [
        ["True Negative (TN)\nCorrect Rejection", "False Positive (FP)\nFalse Acceptance"],
        ["False Negative (FN)\nFalse Rejection", "True Positive (TP)\nCorrect Acceptance"],
    ]

    for i in range(2):
        for j in range(2):
            count = matrix[i, j]
            pct = percentages[i, j]
            desc = labels[i][j]
            # Select contrasting text color
            val = matrix[i, j]
            threshold_val = matrix.max() / 2.0
            color = "white" if val > threshold_val else "#1a252f"
            ax.text(
                j, i,
                f"{desc}\n\n{count} ({pct:.1f}%)",
                ha="center", va="center",
                color=color, fontsize=10, weight="bold",
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Rejected (Negative)", "Accepted (Positive)"], fontsize=10)
    ax.set_yticklabels(["Unanswerable / OOD", "Answerable Query"], fontsize=10)
    ax.set_xlabel("System Decision (Predicted)", fontsize=11, labelpad=8, weight="semibold")
    ax.set_ylabel("Ground Truth (Actual)", fontsize=11, labelpad=8, weight="semibold")

    subtitle = (
        f"Acc: {metrics['Accuracy']*100:.1f}% | Prec: {metrics['Precision']*100:.1f}% | "
        f"Rec: {metrics['Recall']*100:.1f}% | F1: {metrics['F1_Score']*100:.1f}%"
    )
    ax.set_title(f"{title}\n{subtitle}", pad=14, fontsize=11)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_classification_metrics_bar(
    plt,
    metrics_retrieval: Dict[str, Any],
    metrics_e2e: Dict[str, Any],
    model_name: str,
    output_path: Path,
    dpi: int = 300,
):
    """Bar chart comparing Accuracy, Precision, Recall, F1, Specificity across modes."""
    metric_keys = ["Accuracy", "Precision", "Recall", "F1_Score", "Specificity"]
    labels = ["Accuracy", "Precision", "Recall\n(Sensitivity)", "F1-Score", "Specificity\n(TNR)"]

    retrieval_vals = [metrics_retrieval[k] * 100 for k in metric_keys]
    e2e_vals = [metrics_e2e[k] * 100 for k in metric_keys]

    x = np.arange(len(metric_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=dpi)
    rects1 = ax.bar(x - width / 2, retrieval_vals, width, label="Retrieval Only", color=COLOR_PALETTE["retrieval"], alpha=0.9)
    rects2 = ax.bar(x + width / 2, e2e_vals, width, label="End-to-End", color=COLOR_PALETTE["end_to_end"], alpha=0.9)

    ax.set_ylabel("Score (%)", fontsize=11, weight="semibold")
    ax.set_title(f"{model_name}\nAcceptance & Rejection Classification Metrics", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", alpha=0.4)

    def attach_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=9, weight="bold",
            )

    attach_labels(rects1)
    attach_labels(rects2)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_retrieval_ranking_metrics(
    plt,
    retrieval_row: Dict[str, Any],
    e2e_row: Dict[str, Any],
    model_name: str,
    output_path: Path,
    dpi: int = 300,
):
    """Bar chart comparing Hit@1, Hit@3, MRR, Recall@3 across modes."""
    metric_keys = ["Hit@1", "Hit@3", "MRR", "Recall@3"]
    labels = ["Hit@1", "Hit@3", "MRR", "Recall@3"]

    def to_float(val):
        try:
            return float(val) if val is not None and val != "" else 0.0
        except (ValueError, TypeError):
            return 0.0

    retrieval_vals = [to_float(retrieval_row.get(k)) * 100 for k in metric_keys]
    e2e_vals = [to_float(e2e_row.get(k)) * 100 for k in metric_keys]

    x = np.arange(len(metric_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=dpi)
    rects1 = ax.bar(x - width / 2, retrieval_vals, width, label="Retrieval Only", color=COLOR_PALETTE["retrieval"], alpha=0.9)
    rects2 = ax.bar(x + width / 2, e2e_vals, width, label="End-to-End", color=COLOR_PALETTE["end_to_end"], alpha=0.9)

    ax.set_ylabel("Performance (%)", fontsize=11, weight="semibold")
    ax.set_title(f"{model_name}\nInformation Retrieval Ranking Performance", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, weight="semibold")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", alpha=0.4)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, weight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, weight="bold")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_category_breakdown(
    plt,
    category_rows: List[Dict[str, Any]],
    model_name: str,
    output_path: Path,
    dpi: int = 300,
):
    """Bar chart of Hit@1 and MRR across query categories for a specific model."""
    # Filter for end_to_end or retrieval_only
    e2e_cats = [r for r in category_rows if r.get("mode") == "end_to_end"]
    if not e2e_cats:
        e2e_cats = [r for r in category_rows if r.get("mode") == "retrieval_only"]

    categories = []
    hit1_vals = []
    mrr_vals = []

    for r in e2e_cats:
        cat = r.get("category", "")
        if cat in ("negative", "out_of_domain"):
            continue
        categories.append(cat.replace("_", " ").title())
        try:
            h1 = float(r.get("Hit@1")) * 100 if r.get("Hit@1") not in (None, "") else 0.0
            mrr = float(r.get("MRR")) * 100 if r.get("MRR") not in (None, "") else 0.0
        except (ValueError, TypeError):
            h1, mrr = 0.0, 0.0
        hit1_vals.append(h1)
        mrr_vals.append(mrr)

    if not categories:
        return

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=dpi)
    rects1 = ax.bar(x - width / 2, hit1_vals, width, label="Hit@1 (%)", color="#2980b9", alpha=0.9)
    rects2 = ax.bar(x + width / 2, mrr_vals, width, label="MRR (%)", color="#27ae60", alpha=0.9)

    ax.set_ylabel("Score (%)", fontsize=11, weight="semibold")
    ax.set_title(f"{model_name}\nCategory-Wise Performance (End-to-End)", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9, rotation=15, ha="right")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", alpha=0.4)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.0f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8, weight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.0f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8, weight="bold")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_score_distribution(
    plt,
    rows: List[Dict[str, Any]],
    threshold: Optional[float],
    model_name: str,
    output_path: Path,
    dpi: int = 300,
):
    """Plot distribution of top scores comparing Answerable vs Unanswerable queries."""
    # Filter retrieval_only for score analysis
    subset = [r for r in rows if r.get("mode") == "retrieval_only"]
    if not subset:
        subset = rows

    pos_scores = []
    neg_scores = []

    for r in subset:
        try:
            score = float(r.get("top_score"))
            has_ans = bool(r.get("has_answer") in (True, "True", 1, "1"))
            if has_ans:
                pos_scores.append(score)
            else:
                neg_scores.append(score)
        except (ValueError, TypeError):
            continue

    if not pos_scores and not neg_scores:
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=dpi)

    all_scores = pos_scores + neg_scores
    min_val, max_val = min(all_scores), max(all_scores)
    span = max_val - min_val if max_val > min_val else 1.0
    bins = np.linspace(max(0, min_val - 0.05 * span), max_val + 0.05 * span, 18)

    if pos_scores:
        ax.hist(pos_scores, bins=bins, alpha=0.65, color="#2ecc71", label=f"Answerable ({len(pos_scores)})", edgecolor="white")
    if neg_scores:
        ax.hist(neg_scores, bins=bins, alpha=0.65, color="#e74c3c", label=f"Unanswerable / OOD ({len(neg_scores)})", edgecolor="white")

    if threshold is not None:
        ax.axvline(threshold, color="#2c3e50", linestyle="--", linewidth=1.8, label=f"Threshold ({threshold:.4f})")

    ax.set_xlabel("Top Retrieval / Relevance Score", fontsize=11, weight="semibold", labelpad=8)
    ax.set_ylabel("Query Count", fontsize=11, weight="semibold")
    ax.set_title(f"{model_name}\nTop Score Distribution & Decision Boundary", pad=12)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", alpha=0.4)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_training_loss_curve(
    plt,
    history: List[Dict[str, Any]],
    output_path: Path,
    dpi: int = 300,
):
    """Plot training and validation loss curves over epochs for neural models."""
    if not history:
        return

    epochs = [h.get("epoch", i + 1) for i, h in enumerate(history)]
    train_losses = [h.get("train_loss", 0.0) for h in history]
    val_losses = [h.get("validation_loss", 0.0) for h in history]

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=dpi)
    ax.plot(epochs, train_losses, marker="o", markersize=4, label="Training Loss", color="#3498db", linewidth=2)
    ax.plot(epochs, val_losses, marker="s", markersize=4, label="Validation Loss", color="#e67e22", linewidth=2)

    best_idx = int(np.argmin(val_losses))
    best_epoch = epochs[best_idx]
    best_loss = val_losses[best_idx]
    ax.scatter([best_epoch], [best_loss], color="#e74c3c", s=90, zorder=5, label=f"Best Model (Epoch {best_epoch})")

    ax.set_xlabel("Epoch", fontsize=11, weight="semibold")
    ax.set_ylabel("BCE Loss", fontsize=11, weight="semibold")
    ax.set_title("Model C (BiLSTM Neural Matcher)\nTraining & Validation Loss Convergence", pad=12)
    ax.set_xticks(epochs)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, alpha=0.4)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# CROSS-MODEL COMPARISON PLOTS
# =========================================================================

def plot_cross_model_comparison(
    plt,
    comparison_rows: List[Dict[str, Any]],
    output_dir: Path,
    dpi: int = 300,
):
    """Generate comparative charts across all models."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Retrieval Only Comparison & End to End Comparison
    for target_mode, title_mode in [("retrieval_only", "Retrieval-Only Mode (Raw Ranking)"),
                                   ("end_to_end", "End-to-End Mode (Final System Answers)")]:
        sub_rows = [r for r in comparison_rows if r.get("mode") == target_mode]
        if not sub_rows:
            continue

        model_order = ["model_a", "model_b", "model_c", "model_d"]
        sub_dict = {r["Model"]: r for r in sub_rows if "Model" in r}

        models_present = [m for m in model_order if m in sub_dict]
        if not models_present:
            continue

        labels = [m.replace("_", " ").upper() for m in models_present]
        metrics = ["Hit@1", "Hit@3", "MRR", "Recall@3", "OOD_Accuracy"]
        metric_names = ["Hit@1", "Hit@3", "MRR", "Recall@3", "OOD Acc"]

        x = np.arange(len(labels))
        width = 0.16
        fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=dpi)

        colors = ["#2980b9", "#27ae60", "#f39c12", "#8e44ad", "#e74c3c"]
        for idx, (m_key, m_name) in enumerate(zip(metrics, metric_names)):
            vals = []
            for m in models_present:
                try:
                    val = float(sub_dict[m].get(m_key, 0.0)) * 100
                except (ValueError, TypeError):
                    val = 0.0
                vals.append(val)
            offset = (idx - len(metrics) / 2 + 0.5) * width
            rects = ax.bar(x + offset, vals, width, label=m_name, color=colors[idx % len(colors)], alpha=0.9)

        ax.set_ylabel("Score (%)", fontsize=11, weight="semibold")
        ax.set_title(f"Cross-Model Benchmark Comparison\n{title_mode}", pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10, weight="bold")
        ax.set_ylim(0, 115)
        ax.legend(loc="upper right", ncol=len(metrics), frameon=True)
        ax.grid(axis="y", alpha=0.4)

        fig.tight_layout()
        filename = f"model_comparison_{target_mode}.png"
        fig.savefig(output_dir / filename, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    # 2. Error Trade-Offs: False Acceptance Rate vs False Rejection Rate
    e2e_rows = [r for r in comparison_rows if r.get("mode") == "end_to_end"]
    if e2e_rows:
        fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=dpi)
        models = [r.get("Model", "").replace("_", " ").upper() for r in e2e_rows]
        far_vals = [float(r.get("False_Acceptance_Rate", 0.0)) * 100 for r in e2e_rows]
        frr_vals = [float(r.get("False_Rejection_Rate", 0.0)) * 100 for r in e2e_rows]

        x = np.arange(len(models))
        width = 0.35

        r1 = ax.bar(x - width / 2, far_vals, width, label="False Acceptance Rate (FAR)", color="#e74c3c", alpha=0.85)
        r2 = ax.bar(x + width / 2, frr_vals, width, label="False Rejection Rate (FRR)", color="#f39c12", alpha=0.85)

        ax.set_ylabel("Error Rate (%)", fontsize=11, weight="semibold")
        ax.set_title("Rejection Error Trade-Off Across Models (End-to-End)\nLower is Better", pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10, weight="bold")
        ax.set_ylim(0, max(max(far_vals + frr_vals, default=20) + 15, 30))
        ax.legend(loc="upper right", frameon=True)
        ax.grid(axis="y", alpha=0.4)

        for rect in r1:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=9, weight="bold")
        for rect in r2:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=9, weight="bold")

        fig.tight_layout()
        fig.savefig(output_dir / "error_rates_far_frr.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    # 3. Radar Chart (Spider Chart) for Multi-Metric Comparison
    if e2e_rows:
        try:
            plot_radar_chart(plt, e2e_rows, output_dir / "overall_radar_chart.png", dpi=dpi)
        except Exception:
            pass


def plot_radar_chart(
    plt,
    e2e_rows: List[Dict[str, Any]],
    output_path: Path,
    dpi: int = 300,
):
    """Generate a multi-metric radar chart for all models in end-to-end mode."""
    categories = ["Hit@1", "Hit@3", "MRR", "Recall@3", "OOD Acc", "1 - FAR"]
    n_cats = len(categories)

    angles = [n / float(n_cats) * 2 * math.pi for n in range(n_cats)]
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True), dpi=dpi)
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categories, fontsize=10, weight="bold")
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80, 100], ["20%", "40%", "60%", "80%", "100%"], color="grey", size=8)
    plt.ylim(0, 105)

    colors = {"model_a": "#3498db", "model_b": "#2ecc71", "model_c": "#e67e22", "model_d": "#9b59b6"}

    for r in e2e_rows:
        m = r.get("Model", "")
        h1 = float(r.get("Hit@1", 0.0)) * 100
        h3 = float(r.get("Hit@3", 0.0)) * 100
        mrr = float(r.get("MRR", 0.0)) * 100
        rec3 = float(r.get("Recall@3", 0.0)) * 100
        ood = float(r.get("OOD_Accuracy", 0.0)) * 100
        far_inv = (1.0 - float(r.get("False_Acceptance_Rate", 0.0))) * 100

        values = [h1, h3, mrr, rec3, ood, far_inv]
        values += values[:1]

        c = colors.get(m, "#34495e")
        label = m.replace("_", " ").upper()
        ax.plot(angles, values, linewidth=1.8, linestyle="solid", label=label, color=c)
        ax.fill(angles, values, color=c, alpha=0.1)

    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=9, frameon=True)
    ax.set_title("Overall Model Capability Radar Profile (End-to-End)", pad=24, fontsize=12, weight="bold")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# MAIN GENERATOR
# =========================================================================

def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def generate_all_plots(output_dir: Optional[Path | str] = None, dpi: int = 300) -> Dict[str, Any]:
    """Generate all performance and evaluation plots into model-wise folders."""
    plt = setup_matplotlib()

    base_dir = Path(output_dir) if output_dir else (ROOT / "plotting")
    base_dir.mkdir(parents=True, exist_ok=True)

    eval_dir = ROOT / "artifacts/evaluation"
    per_query_path = eval_dir / "per_query.csv"
    comparison_path = eval_dir / "comparison.csv"
    category_path = eval_dir / "by_category.csv"
    thresholds_path = ROOT / "artifacts/thresholds.json"

    per_query_rows = read_csv_rows(per_query_path)
    comparison_rows = read_csv_rows(comparison_path)
    category_rows = read_csv_rows(category_path)
    thresholds = read_json(thresholds_path) if thresholds_path.exists() else {}

    generated_files = []
    models = ["model_a", "model_b", "model_c", "model_d"]

    for model_id in models:
        model_p_rows = [r for r in per_query_rows if r.get("Model") == model_id]
        if not model_p_rows:
            continue

        model_dir = base_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        model_title = MODEL_TITLES.get(model_id, model_id.replace("_", " ").title())

        # Split into modes
        ro_rows = [r for r in model_p_rows if r.get("mode") == "retrieval_only"]
        e2e_rows = [r for r in model_p_rows if r.get("mode") == "end_to_end"]

        metrics_ro = compute_binary_metrics(ro_rows)
        metrics_e2e = compute_binary_metrics(e2e_rows)

        # 1. Confusion Matrix - Retrieval Only
        p1 = model_dir / "confusion_matrix_retrieval_only.png"
        plot_confusion_matrix(plt, metrics_ro, f"{model_title}\nConfusion Matrix (Retrieval Only)", p1, dpi=dpi)
        generated_files.append(p1)

        # 2. Confusion Matrix - End to End
        p2 = model_dir / "confusion_matrix_end_to_end.png"
        plot_confusion_matrix(plt, metrics_e2e, f"{model_title}\nConfusion Matrix (End-to-End)", p2, dpi=dpi)
        generated_files.append(p2)

        # 3. Acceptance/Rejection Classification Metrics
        p3 = model_dir / "classification_metrics.png"
        plot_classification_metrics_bar(plt, metrics_ro, metrics_e2e, model_title, p3, dpi=dpi)
        generated_files.append(p3)

        # 4. Retrieval Ranking Metrics (Hit@1, Hit@3, MRR, Recall@3)
        comp_ro = next((r for r in comparison_rows if r.get("Model") == model_id and r.get("mode") == "retrieval_only"), {})
        comp_e2e = next((r for r in comparison_rows if r.get("Model") == model_id and r.get("mode") == "end_to_end"), {})
        p4 = model_dir / "retrieval_metrics.png"
        plot_retrieval_ranking_metrics(plt, comp_ro, comp_e2e, model_title, p4, dpi=dpi)
        generated_files.append(p4)

        # 5. Category-Wise Performance
        model_cats = [r for r in category_rows if r.get("Model") == model_id]
        p5 = model_dir / "category_performance.png"
        plot_category_breakdown(plt, model_cats, model_title, p5, dpi=dpi)
        generated_files.append(p5)

        # 6. Score Distribution
        th = thresholds.get(model_id, {}).get("threshold")
        p6 = model_dir / "score_distribution.png"
        plot_score_distribution(plt, model_p_rows, th, model_title, p6, dpi=dpi)
        generated_files.append(p6)

        # 7. Model C Specific: Training Loss Curve
        if model_id == "model_c":
            hist_path = ROOT / "artifacts/model_c/training_history.json"
            if hist_path.exists():
                hist = read_json(hist_path)
                p7 = model_dir / "training_loss_curve.png"
                plot_training_loss_curve(plt, hist, p7, dpi=dpi)
                generated_files.append(p7)

        # Write metrics summary JSON
        write_json(model_dir / "metrics_summary.json", {
            "model": model_id,
            "title": model_title,
            "threshold": th,
            "retrieval_only": {
                "classification": metrics_ro,
                "ranking": comp_ro,
            },
            "end_to_end": {
                "classification": metrics_e2e,
                "ranking": comp_e2e,
            },
        })

    # Cross-Model Comparison Plots
    if comparison_rows:
        comp_dir = base_dir / "comparison"
        plot_cross_model_comparison(plt, comparison_rows, comp_dir, dpi=dpi)
        write_json(comp_dir / "comparison_summary.json", {
            "models_evaluated": list(set(r.get("Model", "") for r in comparison_rows)),
            "benchmarks": comparison_rows,
        })

    # Write helpful README inside plotting folder
    write_plotting_readme(base_dir)

    return {
        "status": "success",
        "output_directory": str(base_dir),
        "generated_files_count": len(generated_files),
        "models": [m for m in models if (base_dir / m).exists()],
    }


def write_plotting_readme(base_dir: Path):
    """Write comprehensive documentation inside the plotting directory."""
    readme_path = base_dir / "README.md"
    content = """# 📊 Evaluation & Performance Visualizations

This directory contains automatically generated publication-quality evaluation plots and performance breakdowns for the KUET Academic Information Retrieval System.

---

## Directory Organization

```text
plotting/
├── README.md                                  # This guide
├── model_a/                                   # Model A: TF-IDF (Sparse Unigram + Bigram)
│   ├── confusion_matrix_retrieval_only.png    # 2x2 Confusion matrix for query acceptance
│   ├── confusion_matrix_end_to_end.png        # 2x2 Confusion matrix after metadata routing
│   ├── classification_metrics.png             # Accuracy, Precision, Recall, F1, Specificity
│   ├── retrieval_metrics.png                  # Hit@1, Hit@3, MRR, Recall@3
│   ├── category_performance.png               # Category-wise Hit@1 & MRR
│   ├── score_distribution.png                 # Top scores vs calibrated rejection threshold
│   └── metrics_summary.json                   # Machine-readable metric values
├── model_b/                                   # Model B: Word2Vec (Dense Centroid)
│   └── ...
├── model_c/                                   # Model C: BiLSTM (Deep Neural Matcher)
│   ├── ...
│   └── training_loss_curve.png                # Epoch-by-epoch loss convergence
├── model_d/                                   # Model D: Sentence-BERT (MiniLM-L6-v2)
│   └── ...
└── comparison/                                # Cross-Model Comparative Visualizations
    ├── model_comparison_retrieval_only.png    # Side-by-side comparison (Raw Ranking)
    ├── model_comparison_end_to_end.png        # Side-by-side comparison (Final Application)
    ├── error_rates_far_frr.png                # False Acceptance Rate vs False Rejection Rate
    ├── overall_radar_chart.png                # Multi-dimensional radar profile
    └── comparison_summary.json                # Benchmark summary table
```

---

## Metric Definitions & Interpretations

### 1. Confusion Matrix (Query Acceptance vs Rejection)
In an Academic Information Retrieval system with out-of-domain (OOD) protection:
- **Actual Positive**: The query has a legitimate answer in the curriculum (`has_answer = True`).
- **Actual Negative**: The query is unanswerable or out-of-domain (`has_answer = False`).
- **Predicted Positive**: The model accepted the query for course retrieval (`accepted = True`).
- **Predicted Negative**: The model rejected the query (`accepted = False`).

| Cell | Meaning in Academic IR |
|---|---|
| **True Positive (TP)** | Legitimate syllabus question correctly accepted and answered. |
| **False Positive (FP)** | Out-of-domain or unanswerable query incorrectly accepted (**False Acceptance** / Hallucination hazard). |
| **True Negative (TN)** | Out-of-domain or invalid query correctly rejected by threshold (**OOD Protection**). |
| **False Negative (FN)** | Valid syllabus question mistakenly rejected (**False Rejection**). |

### 2. Retrieval Ranking Metrics
- **Hit@1**: Proportion of queries where the top-ranked retrieved course contains the gold answer course.
- **Hit@3**: Proportion where at least one correct gold course is among the top 3 ranked courses.
- **Mean Reciprocal Rank (MRR)**: Average reciprocal rank $\\frac{1}{\\text{rank}}$ of the first relevant document.
- **Recall@3**: Fraction of all acceptable gold courses retrieved in the first 3 positions.
"""
    readme_path.write_text(content, encoding="utf-8")
