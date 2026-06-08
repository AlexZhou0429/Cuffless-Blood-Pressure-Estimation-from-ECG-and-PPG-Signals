from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluation import plot_bland_altman, plot_true_vs_predicted


def _resolve_output_root(output_root):
    if output_root is not None:
        return Path(output_root)
    candidates = [Path.cwd() / "outputs", Path.cwd().parent / "outputs"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def show_saved_result(method, output_root=None):
    """Display a trained method's metrics and standard plots in a notebook."""
    method_dir = _resolve_output_root(output_root) / method
    metrics = pd.read_csv(method_dir / "metrics.csv")
    predictions = pd.read_csv(method_dir / "predictions.csv")
    y_true = predictions[["true_sbp", "true_dbp"]].to_numpy()
    y_pred = predictions[["pred_sbp", "pred_dbp"]].to_numpy()

    try:
        from IPython.display import display

        display(metrics)
        display(predictions.head(10))
    except ImportError:
        print(metrics.to_string(index=False))

    plot_true_vs_predicted(y_true, y_pred, method)
    plt.show()
    for index, target in [(0, "SBP"), (1, "DBP")]:
        plot_bland_altman(y_true, y_pred, method, index, target)
        plt.show()
    return metrics, predictions
