import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .data import denormalize_y


def _finite(value, default=0.0):
    value = float(value)
    return value if np.isfinite(value) else default


def _channel_features(values, prefix):
    values = np.asarray(values, dtype=float)
    centered = values - np.mean(values)
    std = np.std(values)
    if std < 1e-8:
        skew, kurtosis = 0.0, 0.0
    else:
        standardized = centered / std
        skew = np.mean(standardized**3)
        kurtosis = np.mean(standardized**4) - 3.0

    first_difference = np.diff(values)
    second_difference = np.diff(first_difference)
    peaks, _ = find_peaks(values)
    valleys, _ = find_peaks(-values)
    return {
        f"{prefix}_mean": _finite(np.mean(values)),
        f"{prefix}_std": _finite(std),
        f"{prefix}_min": _finite(np.min(values)),
        f"{prefix}_max": _finite(np.max(values)),
        f"{prefix}_median": _finite(np.median(values)),
        f"{prefix}_iqr": _finite(np.percentile(values, 75) - np.percentile(values, 25)),
        f"{prefix}_range": _finite(np.ptp(values)),
        f"{prefix}_rms": _finite(np.sqrt(np.mean(values**2))),
        f"{prefix}_area": _finite(np.trapezoid(values)),
        f"{prefix}_energy": _finite(np.sum(values**2)),
        f"{prefix}_skew": _finite(skew),
        f"{prefix}_kurtosis": _finite(kurtosis),
        f"{prefix}_diff_mean": _finite(np.mean(first_difference)) if len(first_difference) else 0.0,
        f"{prefix}_diff_std": _finite(np.std(first_difference)) if len(first_difference) else 0.0,
        f"{prefix}_diff_abs_mean": _finite(np.mean(np.abs(first_difference))) if len(first_difference) else 0.0,
        f"{prefix}_diff2_abs_mean": _finite(np.mean(np.abs(second_difference))) if len(second_difference) else 0.0,
        f"{prefix}_peak_count": float(len(peaks)),
        f"{prefix}_valley_count": float(len(valleys)),
    }


def extract_statistical_features(x):
    """Extract statistical and morphology descriptors from padded beats."""
    rows = []
    for sample_id, beat in enumerate(np.asarray(x)):
        valid = beat[np.any(beat != 0, axis=-1)]
        if len(valid) == 0:
            rows.append({"sample_id": sample_id, "beat_length": 0.0})
            continue

        ppg, ecg = valid[:, 0], valid[:, 1]
        ppg_peaks, _ = find_peaks(ppg)
        ecg_peaks, _ = find_peaks(ecg)
        row = {
            "sample_id": sample_id,
            "beat_length": float(len(valid)),
            "ppg_ecg_corr": _finite(np.corrcoef(ppg, ecg)[0, 1]) if len(valid) > 2 else 0.0,
            "ppg_first_peak_pos": float(ppg_peaks[0] / len(ppg)) if len(ppg_peaks) else -1.0,
            "ecg_first_peak_pos": float(ecg_peaks[0] / len(ecg)) if len(ecg_peaks) else -1.0,
        }
        row["ecg_to_ppg_first_peak_lag"] = (
            row["ppg_first_peak_pos"] - row["ecg_first_peak_pos"]
            if len(ppg_peaks) and len(ecg_peaks)
            else 0.0
        )
        row.update(_channel_features(ppg, "ppg"))
        row.update(_channel_features(ecg, "ecg"))
        rows.append(row)
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_feature_mapping(x, y_normalized, scalers):
    mapping = extract_statistical_features(x)
    targets = denormalize_y(y_normalized, scalers)
    mapping["target_sbp"] = targets[:, 0]
    mapping["target_dbp"] = targets[:, 1]
    return mapping


def filter_outliers(
    mapping,
    iqr_multiplier=1.5,
    sbp_bounds=(70.0, 220.0),
    dbp_bounds=(40.0, 140.0),
):
    """Filter physiologically implausible and target-IQR outliers."""
    mask = (
        mapping["target_sbp"].between(*sbp_bounds)
        & mapping["target_dbp"].between(*dbp_bounds)
        & (mapping["target_sbp"] > mapping["target_dbp"])
    )
    for column in ["target_sbp", "target_dbp"]:
        q1, q3 = mapping[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        if np.isfinite(iqr) and iqr > 1e-12:
            mask &= mapping[column].between(
                q1 - iqr_multiplier * iqr,
                q3 + iqr_multiplier * iqr,
            )
    return mapping.loc[mask].reset_index(drop=True), mapping.loc[~mask].reset_index(drop=True)


def select_features(mapping, max_features=24, min_absolute_correlation=0.02):
    """Rank features by their strongest absolute SBP/DBP correlation."""
    excluded = {"sample_id", "target_sbp", "target_dbp"}
    rows = []
    for feature in [column for column in mapping.columns if column not in excluded]:
        if mapping[feature].nunique(dropna=True) < 2:
            continue
        sbp_correlation = mapping[feature].corr(mapping["target_sbp"])
        dbp_correlation = mapping[feature].corr(mapping["target_dbp"])
        correlations = np.abs([sbp_correlation, dbp_correlation])
        if np.isnan(correlations).all():
            continue
        score = np.nanmax(correlations)
        if np.isfinite(score) and score >= min_absolute_correlation:
            rows.append(
                {
                    "feature": feature,
                    "score": float(score),
                    "sbp_correlation": _finite(sbp_correlation),
                    "dbp_correlation": _finite(dbp_correlation),
                }
            )
    scores = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return scores.head(max_features)["feature"].tolist(), scores
