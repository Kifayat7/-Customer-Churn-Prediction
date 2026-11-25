"""Metrics and plots used to evaluate the churn model."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
METRICS_PATH = ROOT / "reports" / "model_metrics.txt"


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_churn_distribution(df):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df, x="Churn", hue="Churn", palette=["#4C9BE8", "#E8574C"], legend=False, ax=ax)
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    ax.set_title("Churn Distribution")
    return _save(fig, "churn_distribution.png")


def plot_correlation_matrix(df):
    d = df[["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]].copy()
    d["Churn"] = (df["Churn"] == "Yes").astype(int)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(d.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix")
    return _save(fig, "correlation_matrix.png")


def plot_confusion_matrix(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    return _save(fig, "confusion_matrix.png")


def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_prob):.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    return _save(fig, "roc_curve.png")


def write_metrics(results: dict, best_name: str, path=METRICS_PATH):
    """results = {model_name: metrics_dict}."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Customer Churn Model Metrics (hold-out test set)", "=" * 50, ""]
    for name, m in results.items():
        lines.append(f"{name}{'  <-- selected' if name == best_name else ''}")
        lines += [f"  {k:<10}: {v:.4f}" for k, v in m.items()]
        lines.append("")
    path.write_text("\n".join(lines))
    return path
