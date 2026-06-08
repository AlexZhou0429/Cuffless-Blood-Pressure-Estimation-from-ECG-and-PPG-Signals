import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data import denormalize_target, denormalize_y, load_beat_data, shuffled_split
from .evaluation import save_evaluation, save_feature_distributions
from .features import build_feature_mapping, extract_statistical_features, filter_outliers, select_features
from .models import create_lstm_model, create_traditional_model, install_pyarrow_stub


TRADITIONAL_MODELS = ("svm", "random_forest", "adaboost")


def prepare_experiment(config):
    data = load_beat_data(config.data_dir)
    raw_mapping = build_feature_mapping(data["x_main"], data["y_main"], data["min_max"])
    filtered_mapping, outliers = filter_outliers(raw_mapping, config.iqr_multiplier)

    sample_ids = filtered_mapping["sample_id"].astype(int).to_numpy()
    split = shuffled_split(
        data["x_main"][sample_ids],
        data["y_main"][sample_ids],
        sample_ids,
        config.validation_fraction,
        config.seed,
    )
    train_mapping = filtered_mapping.set_index("sample_id").loc[split["train_sample_ids"]]
    selected_features, feature_scores = select_features(
        train_mapping,
        config.max_features,
        config.min_feature_correlation,
    )
    if not selected_features:
        raise ValueError("No features passed selection. Lower min_feature_correlation.")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    feature_scores.to_csv(config.output_dir / "feature_scores.csv", index=False)
    (config.output_dir / "selected_features.json").write_text(
        json.dumps(selected_features, indent=2),
        encoding="utf-8",
    )
    if config.save_feature_mapping:
        raw_mapping.to_csv(config.output_dir / "feature_mapping_raw.csv", index=False)
        filtered_mapping.to_csv(config.output_dir / "feature_mapping_filtered.csv", index=False)
        outliers.to_csv(config.output_dir / "feature_mapping_outliers.csv", index=False)
    save_feature_distributions(
        raw_mapping,
        filtered_mapping,
        selected_features,
        config.output_dir,
    )

    return {
        "data": data,
        "mapping": filtered_mapping,
        "selected_features": selected_features,
        "split": split,
        "raw_count": len(raw_mapping),
        "filtered_count": len(filtered_mapping),
        "outlier_count": len(outliers),
    }


def _traditional_matrices(experiment):
    split = experiment["split"]
    selected = experiment["selected_features"]
    indexed = experiment["mapping"].set_index("sample_id")
    x_train = indexed.loc[split["train_sample_ids"], selected].reset_index(drop=True)
    x_valid = indexed.loc[split["valid_sample_ids"], selected].reset_index(drop=True)
    x_test = extract_statistical_features(experiment["data"]["x_test"])[selected]
    medians = x_train.median()
    return {
        "x_train": x_train.fillna(medians),
        "x_valid": x_valid.fillna(medians),
        "x_test": x_test.fillna(medians),
        "y_train": denormalize_y(split["y_train"], experiment["data"]["min_max"]),
        "y_valid": denormalize_y(split["y_valid"], experiment["data"]["min_max"]),
        "y_test": denormalize_y(experiment["data"]["y_test"], experiment["data"]["min_max"]),
        "medians": medians,
    }


def train_traditional_models(experiment, config):
    matrices = _traditional_matrices(experiment)
    rng = np.random.default_rng(config.seed)
    limit = min(config.traditional_max_train, len(matrices["x_train"]))
    subset = rng.choice(len(matrices["x_train"]), size=limit, replace=False)
    x_fit = matrices["x_train"].iloc[subset]
    y_fit = matrices["y_train"][subset]

    rows = []
    for model_name in TRADITIONAL_MODELS:
        model = create_traditional_model(model_name, config.seed)
        model.fit(x_fit, y_fit)
        config.model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, config.model_dir / f"{model_name}.joblib")
        method_dir = config.output_dir / model_name
        metrics = save_evaluation(
            matrices["y_test"],
            model.predict(matrices["x_test"]),
            model_name,
            method_dir,
        )
        rows.append({"model": model_name, **metrics})

    matrices["medians"].to_csv(config.model_dir / "feature_medians.csv", header=["median"])
    (config.model_dir / "selected_features.json").write_text(
        json.dumps(experiment["selected_features"], indent=2),
        encoding="utf-8",
    )
    return pd.DataFrame(rows)


def train_lstm_models(experiment, config):
    install_pyarrow_stub()
    from keras.callbacks import EarlyStopping

    split = experiment["split"]
    predictions = {}
    histories = []
    config.model_dir.mkdir(parents=True, exist_ok=True)
    for target_index, target_name in [(0, "sbp"), (1, "dbp")]:
        model = create_lstm_model(
            beat_length=split["x_train"].shape[1],
            num_channels=split["x_train"].shape[2],
            target_name=target_name,
            seed=config.seed + target_index,
        )
        model_path = config.model_dir / f"{target_name}_lstm.keras"
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=config.patience, restore_best_weights=True),
        ]
        history = model.fit(
            split["x_train"],
            split["y_train"][:, target_index],
            validation_data=(split["x_valid"], split["y_valid"][:, target_index]),
            epochs=config.epochs,
            batch_size=config.batch_size,
            shuffle=True,
            callbacks=callbacks,
            verbose=config.keras_verbose,
        )
        histories.append(pd.DataFrame(history.history).assign(target=target_name))
        model.save(model_path)
        prediction = model.predict(experiment["data"]["x_test"], batch_size=config.batch_size, verbose=0)
        predictions[target_name] = denormalize_target(
            prediction,
            experiment["data"]["min_max"],
            target_index,
        )

    pd.concat(histories, ignore_index=True).to_csv(config.output_dir / "lstm_history.csv", index=False)
    y_true = denormalize_y(experiment["data"]["y_test"], experiment["data"]["min_max"])
    y_pred = np.column_stack([predictions["sbp"], predictions["dbp"]])
    return save_evaluation(y_true, y_pred, "separated_lstm", config.output_dir / "lstm")


def run_pipeline(config, run_traditional=True, run_lstm=True):
    """Run feature preparation once, then train the requested model branches."""
    experiment = prepare_experiment(config)
    config.model_dir.mkdir(parents=True, exist_ok=True)
    normalization = {
        key: np.asarray(value).tolist()
        for key, value in experiment["data"]["min_max"].items()
    }
    (config.model_dir / "normalization.json").write_text(
        json.dumps(normalization, indent=2),
        encoding="utf-8",
    )
    summary = {
        "raw_beats": experiment["raw_count"],
        "filtered_beats": experiment["filtered_count"],
        "outlier_beats": experiment["outlier_count"],
        "train_beats": len(experiment["split"]["x_train"]),
        "validation_beats": len(experiment["split"]["x_valid"]),
        "test_beats": len(experiment["data"]["x_test"]),
        "selected_features": len(experiment["selected_features"]),
    }
    metrics = []
    if run_traditional:
        metrics.append(train_traditional_models(experiment, config))
    if run_lstm:
        lstm_metrics = train_lstm_models(experiment, config)
        metrics.append(pd.DataFrame([{"model": "separated_lstm", **lstm_metrics}]))
    if metrics:
        pd.concat(metrics, ignore_index=True).to_csv(config.output_dir / "benchmark_metrics.csv", index=False)
    pd.DataFrame([summary]).to_csv(config.output_dir / "run_summary.csv", index=False)
    return summary
