from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve


def _save_plot(fig: Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")

def plot_precision_recall_curve(y_true, proba, name: str, output_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, proba)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(recall, precision, color="#4e79a7", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall curve: {name}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, min(1, precision.max() * 1.1) if len(precision) else 1)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    _save_plot(fig, output_path)


def plot_feature_importance(model, feature_cols: list[str], name: str,
                             output_path: Path, top_n: int = 15) -> None:
    if not hasattr(model, "feature_importances_"):
        return

    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp.index[::-1], imp.values[::-1], color="#4e79a7")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} features: {name}")
    ax.spines[["top", "right"]].set_visible(False)

    _save_plot(fig, output_path)