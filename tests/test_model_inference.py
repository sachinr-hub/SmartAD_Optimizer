import os
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

MODEL_PATH = "models/ctr_prediction_model.keras"
DL_CSV = "data/dl_data.csv"
INPUT_COLS_CSV = "data/model_input_columns.csv"


@pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason="Trained model not found. Run preprocess_data.py and train_ctr_model.py first.",
)
@pytest.mark.skipif(
    not (os.path.exists(DL_CSV) and os.path.exists(INPUT_COLS_CSV)),
    reason="Preprocessed data or model input columns missing. Run preprocess_data.py first.",
)
def test_model_inference_small_batch():
    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Load data
    data = pd.read_csv(DL_CSV)
    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])  # defensive drop if present

    X_cols = pd.read_csv(INPUT_COLS_CSV).columns.tolist()
    missing = [c for c in X_cols if c not in data.columns]
    assert not missing, f"Missing expected columns in dl_data.csv: {missing}"

    X = data[X_cols].to_numpy(dtype=np.float64)

    # Use a small batch
    batch_size = min(8, X.shape[0])
    assert batch_size > 0, "Dataset appears empty."

    X_batch = X[:batch_size]

    # Reshape for Conv1D input (samples, timesteps, features) -> (samples, features, 1)
    X_batch_reshaped = X_batch.reshape(X_batch.shape[0], X_batch.shape[1], 1)

    # Inference
    y_pred = model.predict(X_batch_reshaped)

    # Validate shape: (batch_size, 1)
    assert y_pred.shape[0] == batch_size, "Prediction batch size mismatch."
    assert y_pred.shape[1] == 1, "Prediction output should have shape (batch, 1)."

    # Validate numeric type and finite values
    y_pred = np.asarray(y_pred).reshape(-1)
    assert np.issubdtype(y_pred.dtype, np.floating), "Predictions must be float."
    assert np.all(np.isfinite(y_pred)), "Predictions contain non-finite values."
