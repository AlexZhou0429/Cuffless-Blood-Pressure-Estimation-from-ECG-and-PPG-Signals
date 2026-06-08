from pathlib import Path

import numpy as np
from scipy.signal import find_peaks


FS = 125
MAX_BEAT_LENGTH = 200


def load_beat_data(data_dir="data"):
    """Load normalized beat-wise arrays and training min/max statistics."""
    data_dir = Path(data_dir)
    required = ["x_main.npy", "y_main.npy", "x_test.npy", "y_test.npy", "min_max.npy"]
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing data files in {data_dir}: {', '.join(missing)}. "
            "See README.md for the expected data layout."
        )
    return {
        "x_main": np.load(data_dir / "x_main.npy"),
        "y_main": np.load(data_dir / "y_main.npy"),
        "x_test": np.load(data_dir / "x_test.npy"),
        "y_test": np.load(data_dir / "y_test.npy"),
        "min_max": np.load(data_dir / "min_max.npy", allow_pickle=True).item(),
    }


def shuffled_split(x, y, sample_ids, validation_fraction=0.2, seed=487):
    """Shuffle aligned arrays reproducibly and split train/validation data."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    x = np.asarray(x)[order]
    y = np.asarray(y)[order]
    sample_ids = np.asarray(sample_ids)[order]
    validation_count = max(1, int(round(len(x) * validation_fraction)))
    return {
        "x_train": x[validation_count:],
        "x_valid": x[:validation_count],
        "y_train": y[validation_count:],
        "y_valid": y[:validation_count],
        "train_sample_ids": sample_ids[validation_count:],
        "valid_sample_ids": sample_ids[:validation_count],
    }


def detect_abp_beat_boundaries(
    abp,
    fs=FS,
    min_peak_distance_sec=0.35,
    prominence_fraction=0.08,
):
    """Detect systolic peaks and the intervening ABP valleys."""
    abp = np.asarray(abp, dtype=float)
    if len(abp) < 3:
        return np.array([], dtype=int), np.array([], dtype=int)

    distance = max(1, int(min_peak_distance_sec * fs))
    signal_range = np.percentile(abp, 95) - np.percentile(abp, 5)
    prominence = max(1e-6, prominence_fraction * signal_range)
    peaks, _ = find_peaks(abp, distance=distance, prominence=prominence)
    if len(peaks) < 3:
        return peaks, np.array([], dtype=int)

    valleys = []
    for left_peak, right_peak in zip(peaks[:-1], peaks[1:]):
        segment = abp[left_peak : right_peak + 1]
        valleys.append(left_peak + int(np.argmin(segment)))
    return peaks, np.array(sorted(set(valleys)), dtype=int)


def pad_beat(ppg_segment, ecg_segment, max_beat_length=MAX_BEAT_LENGTH):
    """Zero-pad one PPG/ECG beat without temporal resampling."""
    beat = np.zeros((max_beat_length, 2), dtype=float)
    beat_length = min(len(ppg_segment), max_beat_length)
    beat[:beat_length, 0] = ppg_segment[:beat_length]
    beat[:beat_length, 1] = ecg_segment[:beat_length]
    return beat


def extract_subject_beats(
    subject_x,
    subject_abp,
    subject_id,
    fs=FS,
    max_beat_length=MAX_BEAT_LENGTH,
    min_beat_sec=0.35,
    max_beat_sec=1.6,
):
    """Convert one continuous recording into beat-wise PPG/ECG samples."""
    ppg = np.asarray(subject_x[0], dtype=float)
    ecg = np.asarray(subject_x[1], dtype=float)
    abp = np.asarray(subject_abp, dtype=float)
    peaks, valleys = detect_abp_beat_boundaries(abp, fs=fs)
    min_length = int(min_beat_sec * fs)
    max_length = min(int(max_beat_sec * fs), max_beat_length)

    beat_x, beat_y, metadata = [], [], []
    for start, end in zip(valleys[:-1], valleys[1:]):
        beat_length = end - start
        if beat_length < min_length or beat_length > max_length:
            continue
        abp_segment = abp[start:end]
        sbp = float(np.max(abp_segment))
        dbp = float(np.min(abp_segment))
        if not np.isfinite(sbp) or not np.isfinite(dbp) or sbp <= dbp:
            continue
        beat_x.append(pad_beat(ppg[start:end], ecg[start:end], max_beat_length))
        beat_y.append([sbp, dbp])
        metadata.append(
            {
                "subject": int(subject_id),
                "start": int(start),
                "end": int(end),
                "peak_count": int(np.sum((peaks >= start) & (peaks < end))),
            }
        )

    if not beat_x:
        return (
            np.empty((0, max_beat_length, 2), dtype=float),
            np.empty((0, 2), dtype=float),
            [],
        )
    return np.asarray(beat_x), np.asarray(beat_y), metadata


def compute_scalers(x_train, y_train):
    """Compute min/max values from non-padded training samples."""
    valid_mask = np.any(x_train != 0, axis=-1)
    valid_x = x_train[valid_mask]
    return {
        "x_min": np.min(valid_x, axis=0),
        "x_max": np.max(valid_x, axis=0),
        "y_min": np.min(y_train, axis=0),
        "y_max": np.max(y_train, axis=0),
    }


def normalize_x(x, scalers):
    valid_mask = np.any(x != 0, axis=-1)
    denominator = np.maximum(scalers["x_max"] - scalers["x_min"], 1e-8)
    normalized = np.zeros_like(x, dtype=float)
    normalized[valid_mask] = (x[valid_mask] - scalers["x_min"]) / denominator
    return normalized


def normalize_y(y, scalers):
    denominator = np.maximum(scalers["y_max"] - scalers["y_min"], 1e-8)
    return (y - scalers["y_min"]) / denominator


def denormalize_y(y_normalized, scalers):
    return y_normalized * (scalers["y_max"] - scalers["y_min"]) + scalers["y_min"]


def denormalize_target(y_normalized, scalers, target_index):
    y_normalized = np.asarray(y_normalized).reshape(-1)
    y_min = scalers["y_min"][target_index]
    y_max = scalers["y_max"][target_index]
    return y_normalized * (y_max - y_min) + y_min
