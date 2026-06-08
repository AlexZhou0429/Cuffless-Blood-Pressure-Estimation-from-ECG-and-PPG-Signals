from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration shared by feature-based and LSTM experiments."""

    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    model_dir: Path = Path("models")
    seed: int = 487
    validation_fraction: float = 0.2
    max_features: int = 24
    min_feature_correlation: float = 0.02
    iqr_multiplier: float = 1.5
    traditional_max_train: int = 5000
    epochs: int = 30
    batch_size: int = 128
    patience: int = 8
    keras_verbose: int = 2
    save_feature_mapping: bool = False

    def __post_init__(self):
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1.")
        if self.max_features <= 0:
            raise ValueError("max_features must be positive.")
        if self.iqr_multiplier <= 0:
            raise ValueError("iqr_multiplier must be positive.")
        if self.traditional_max_train <= 0:
            raise ValueError("traditional_max_train must be positive.")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive.")
        if self.patience < 0:
            raise ValueError("patience cannot be negative.")
