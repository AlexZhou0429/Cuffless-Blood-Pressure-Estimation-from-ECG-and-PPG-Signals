import argparse
from pathlib import Path

from .config import ExperimentConfig
from .pipeline import run_pipeline


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train statistical-feature regressors and separated SBP/DBP LSTMs."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--seed", type=int, default=487)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--max-features", type=int, default=24)
    parser.add_argument("--min-feature-correlation", type=float, default=0.02)
    parser.add_argument("--iqr-multiplier", type=float, default=1.5)
    parser.add_argument("--traditional-max-train", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--keras-verbose", type=int, choices=[0, 1, 2], default=2)
    parser.add_argument("--save-feature-mapping", action="store_true")
    parser.add_argument("--skip-traditional", action="store_true")
    parser.add_argument("--skip-lstm", action="store_true")
    args = parser.parse_args(argv)
    if args.skip_traditional and args.skip_lstm:
        parser.error("--skip-traditional and --skip-lstm cannot be used together")
    return args


def main(argv=None):
    args = parse_args(argv)
    config = ExperimentConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        max_features=args.max_features,
        min_feature_correlation=args.min_feature_correlation,
        iqr_multiplier=args.iqr_multiplier,
        traditional_max_train=args.traditional_max_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        keras_verbose=args.keras_verbose,
        save_feature_mapping=args.save_feature_mapping,
    )
    summary = run_pipeline(
        config,
        run_traditional=not args.skip_traditional,
        run_lstm=not args.skip_lstm,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
