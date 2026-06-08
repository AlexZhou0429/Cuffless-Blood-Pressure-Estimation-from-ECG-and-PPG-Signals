import os
import sys
from pathlib import Path


class _PyArrowStub:
    __version__ = "0.0.0"

    class Table:
        pass

    class RecordBatch:
        pass

    class Array:
        pass

    class ChunkedArray:
        pass


def install_pyarrow_stub():
    """Prevent optional PyArrow binary issues during TensorFlow/Keras import."""
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(os.environ.get("TMPDIR", "/tmp")) / "cuffless_bp_matplotlib"),
    )
    if "pyarrow" in sys.modules and sys.modules["pyarrow"] is None:
        del sys.modules["pyarrow"]
    sys.modules.setdefault("pyarrow", _PyArrowStub())


def create_lstm_model(beat_length=200, num_channels=2, target_name="sbp", seed=487):
    """Create one independent bidirectional LSTM for SBP or DBP."""
    install_pyarrow_stub()
    import keras

    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    inputs = keras.layers.Input(shape=(beat_length, num_channels), name="ecg_ppg_beat")
    x = keras.layers.Masking(mask_value=0.0)(inputs)
    x = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True))(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(32))(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    output = keras.layers.Dense(1, activation="linear", name=target_name)(x)
    model = keras.Model(inputs=inputs, outputs=output, name=f"{target_name}_lstm")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def create_traditional_model(model_name, seed=487):
    """Create an SVM, Random Forest, or AdaBoost multi-output regressor."""
    from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR

    if model_name == "svm":
        return MultiOutputRegressor(make_pipeline(StandardScaler(), SVR(C=10.0, epsilon=1.0)))
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=160,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
    if model_name == "adaboost":
        return MultiOutputRegressor(
            AdaBoostRegressor(
                n_estimators=120,
                learning_rate=0.05,
                random_state=seed,
            )
        )
    raise ValueError(f"Unknown model '{model_name}'. Choose svm, random_forest, or adaboost.")
