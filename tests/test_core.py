import numpy as np

from cuffless_bp.data import shuffled_split
from cuffless_bp.evaluation import calculate_metrics
from cuffless_bp.features import extract_statistical_features, select_features


def test_shuffled_split_is_reproducible_and_aligned():
    sample_ids = np.arange(10)
    x = sample_ids.reshape(-1, 1, 1)
    y = np.column_stack([sample_ids, sample_ids + 100])

    first = shuffled_split(x, y, sample_ids, validation_fraction=0.2, seed=7)
    second = shuffled_split(x, y, sample_ids, validation_fraction=0.2, seed=7)

    np.testing.assert_array_equal(first["train_sample_ids"], second["train_sample_ids"])
    np.testing.assert_array_equal(first["valid_sample_ids"], second["valid_sample_ids"])
    np.testing.assert_array_equal(first["x_train"][:, 0, 0], first["y_train"][:, 0])
    assert set(first["train_sample_ids"]).isdisjoint(first["valid_sample_ids"])


def test_statistical_features_ignore_zero_padding():
    beats = np.zeros((2, 8, 2), dtype=float)
    beats[0, :4, 0] = [0.1, 0.5, 0.2, 0.1]
    beats[0, :4, 1] = [0.2, 0.8, 0.3, 0.1]
    beats[1, :6, 0] = [0.1, 0.3, 0.6, 0.4, 0.2, 0.1]
    beats[1, :6, 1] = [0.1, 0.2, 0.9, 0.3, 0.2, 0.1]

    features = extract_statistical_features(beats)

    assert features["beat_length"].tolist() == [4.0, 6.0]
    assert {"ppg_mean", "ecg_mean", "ppg_ecg_corr"}.issubset(features.columns)
    assert np.isfinite(features.to_numpy()).all()


def test_feature_selection_ranks_training_correlations():
    sample_ids = np.arange(20)
    mapping = extract_statistical_features(
        np.column_stack(
            [
                np.tile(sample_ids[:, None], (1, 6)),
                np.tile((sample_ids * 2)[:, None], (1, 6)),
            ]
        ).reshape(20, 6, 2)
    )
    mapping["target_sbp"] = sample_ids * 3
    mapping["target_dbp"] = sample_ids * 2

    selected, scores = select_features(mapping, max_features=5, min_absolute_correlation=0.01)

    assert 0 < len(selected) <= 5
    assert scores["score"].is_monotonic_decreasing


def test_metrics_are_reported_per_target():
    y_true = np.array([[120.0, 80.0], [130.0, 85.0]])
    y_pred = np.array([[118.0, 82.0], [132.0, 83.0]])

    metrics = calculate_metrics(y_true, y_pred)

    assert metrics["sbp_rmse"] == 2.0
    assert metrics["dbp_rmse"] == 2.0
    assert metrics["overall_mae"] == 2.0
