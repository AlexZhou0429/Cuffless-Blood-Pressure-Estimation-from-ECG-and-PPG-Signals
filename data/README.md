# Data layout

The repository does not include waveform data or derived NumPy arrays.

Place the following normalized beat-wise files in this directory:

```text
data/
├── x_main.npy   # shape: (n_train, beat_length, 2), channels are PPG and ECG
├── y_main.npy   # shape: (n_train, 2), normalized [SBP, DBP]
├── x_test.npy   # shape: (n_test, beat_length, 2)
├── y_test.npy   # shape: (n_test, 2)
└── min_max.npy  # dict containing x_min, x_max, y_min, and y_max
```

The current model uses a maximum beat length of 200 samples. Shorter beats are
zero-padded. The project is being prepared for migration to MIMIC waveform data;
MIMIC files are not redistributed by this repository.
