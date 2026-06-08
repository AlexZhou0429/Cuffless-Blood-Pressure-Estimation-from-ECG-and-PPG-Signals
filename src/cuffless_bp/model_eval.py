from pathlib import Path
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "cuffless_bp_matplotlib"))

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


METRIC_COLUMNS = [
    "sbp_rmse",
    "dbp_rmse",
    "sbp_mae",
    "dbp_mae",
    "overall_mae",
    "sbp_r2",
    "dbp_r2",
]


def calculate_metrics(y_true, y_pred):
    """Calculate the common SBP/DBP regression metrics in mmHg."""
    return {
        "sbp_rmse": float(np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))),
        "dbp_rmse": float(np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))),
        "sbp_mae": float(mean_absolute_error(y_true[:, 0], y_pred[:, 0])),
        "dbp_mae": float(mean_absolute_error(y_true[:, 1], y_pred[:, 1])),
        "overall_mae": float(mean_absolute_error(y_true, y_pred)),
        "sbp_r2": float(r2_score(y_true[:, 0], y_pred[:, 0])),
        "dbp_r2": float(r2_score(y_true[:, 1], y_pred[:, 1])),
    }


def predictions_frame(y_true, y_pred, model_name):
    frame = pd.DataFrame(
        {
            "model": model_name,
            "true_sbp": y_true[:, 0],
            "true_dbp": y_true[:, 1],
            "pred_sbp": y_pred[:, 0],
            "pred_dbp": y_pred[:, 1],
        }
    )
    frame["sbp_error"] = frame["pred_sbp"] - frame["true_sbp"]
    frame["dbp_error"] = frame["pred_dbp"] - frame["true_dbp"]
    return frame


def plot_true_vs_predicted(y_true, y_pred, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, index, target in [(axes[0], 0, "SBP"), (axes[1], 1, "DBP")]:
        truth, prediction = y_true[:, index], y_pred[:, index]
        lower = min(truth.min(), prediction.min())
        upper = max(truth.max(), prediction.max())
        axis.scatter(truth, prediction, s=18, alpha=0.68)
        axis.plot([lower, upper], [lower, upper], color="tab:red")
        axis.set(
            title=f"{model_name}: {target} true vs predicted",
            xlabel=f"True {target} (mmHg)",
            ylabel=f"Predicted {target} (mmHg)",
        )
        axis.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_bland_altman(y_true, y_pred, model_name, target_index, target_name):
    truth, prediction = y_true[:, target_index], y_pred[:, target_index]
    means = (truth + prediction) / 2.0
    differences = prediction - truth
    bias = np.mean(differences)
    standard_deviation = np.std(differences, ddof=1)

    fig, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(means, differences, s=18, alpha=0.68)
    axis.axhline(bias, color="tab:red", label=f"Bias: {bias:.2f}")
    axis.axhline(bias + 1.96 * standard_deviation, color="tab:gray", linestyle="--", label="+1.96 SD")
    axis.axhline(bias - 1.96 * standard_deviation, color="tab:gray", linestyle="--", label="-1.96 SD")
    axis.set(
        title=f"{model_name}: Bland-Altman {target_name}",
        xlabel="Mean of predicted and true BP (mmHg)",
        ylabel="Predicted - true BP (mmHg)",
    )
    axis.legend()
    axis.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def save_feature_distributions(raw_mapping, filtered_mapping, selected_features, output_dir):
    """Save target and selected-feature distributions used for data review."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for column, label, row in [
        ("target_sbp", "SBP (mmHg)", 0),
        ("target_dbp", "DBP (mmHg)", 1),
    ]:
        axes[row, 0].hist(raw_mapping[column], bins=35, color="#5271a6", alpha=0.85)
        axes[row, 0].set_title(f"{label} before outlier filtering")
        axes[row, 1].hist(filtered_mapping[column], bins=35, color="#3b8c6e", alpha=0.85)
        axes[row, 1].set_title(f"{label} after outlier filtering")
        for axis in axes[row]:
            axis.set_xlabel(label)
            axis.set_ylabel("Beat count")
            axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "target_distributions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    displayed_features = selected_features[:8]
    rows = int(np.ceil(len(displayed_features) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(12, 3.4 * rows), squeeze=False)
    for axis, feature in zip(axes.flat, displayed_features):
        axis.hist(filtered_mapping[feature], bins=35, color="#c46b3c", alpha=0.85)
        axis.set_title(feature)
        axis.set_ylabel("Beat count")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(displayed_features) :]:
        axis.axis("off")
    fig.suptitle("Distributions of top selected statistical features", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "selected_feature_distributions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_evaluation(y_true, y_pred, model_name, output_dir):
    """Save metrics, predictions, true-vs-predicted, and Bland-Altman plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = calculate_metrics(y_true, y_pred)
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)
    predictions_frame(y_true, y_pred, model_name).to_csv(output_dir / "predictions.csv", index=False)

    figure = plot_true_vs_predicted(y_true, y_pred, model_name)
    figure.savefig(output_dir / "true_vs_predicted.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    for index, target in [(0, "SBP"), (1, "DBP")]:
        figure = plot_bland_altman(y_true, y_pred, model_name, index, target)
        figure.savefig(output_dir / f"bland_altman_{target.lower()}.png", dpi=180, bbox_inches="tight")
        plt.close(figure)
    return metrics


def load_saved_result(output_dir):
    """Load one method's metrics and predictions for notebook visualization."""
    output_dir = Path(output_dir)
    return pd.read_csv(output_dir / "metrics.csv"), pd.read_csv(output_dir / "predictions.csv")
