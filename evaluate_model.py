import os
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf


def ensure_results_dir(path: str = "results") -> str:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    return path


def load_model(model_path: str = "models/ctr_prediction_model.keras"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Train the model first by running train_ctr_model.py"
        )
    return tf.keras.models.load_model(model_path)


def load_data(
    dl_csv: str = "data/dl_data.csv",
    input_cols_csv: str = "data/model_input_columns.csv",
):
    if not os.path.exists(dl_csv):
        raise FileNotFoundError(
            f"Preprocessed data not found at {dl_csv}. Run preprocess_data.py first."
        )
    if not os.path.exists(input_cols_csv):
        raise FileNotFoundError(
            f"Model input columns file not found at {input_cols_csv}. Run preprocess_data.py first."
        )

    data = pd.read_csv(dl_csv)
    # Drop optional artifact column
    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])  # defensive

    X_cols = pd.read_csv(input_cols_csv).columns.tolist()

    # Ensure columns exist
    missing = [c for c in X_cols if c not in data.columns]
    if missing:
        raise ValueError(
            f"The following expected model input columns are missing from {dl_csv}: {missing}"
        )

    X = data[X_cols]
    y = data["CTR"]

    # Convert to numeric numpy arrays
    X_np = X.to_numpy(dtype=np.float64)
    y_np = y.to_numpy(dtype=np.float64)
    return X_np, y_np


essential_metrics = [
    ("mse", mean_squared_error),
    ("mae", mean_absolute_error),
    ("r2", r2_score),
]


def evaluate(model_path: str = "models/ctr_prediction_model.keras") -> dict:
    model = load_model(model_path)
    X, y = load_data()

    # Create a reproducible train/test split (just for evaluation parity)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Reshape for Conv1D: (samples, timesteps, features) -> here (samples, features, 1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    y_pred = model.predict(X_test_reshaped)
    # Flatten predictions to 1D
    y_pred = np.asarray(y_pred).reshape(-1)

    results = {name: float(func(y_test, y_pred)) for name, func in essential_metrics}
    results.update(
        {
            "n_test": int(X_test.shape[0]),
            "n_features": int(X_test.shape[1]),
            "timestamp": datetime.now().isoformat(),
            "model_path": model_path,
        }
    )

    # Persist artifacts
    results_dir = ensure_results_dir()
    ts = time.strftime("%Y%m%d_%H%M%S")

    # Save JSON report
    report_path = os.path.join(results_dir, f"evaluation_report_{ts}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, "r--")
    plt.xlabel("Actual CTR")
    plt.ylabel("Predicted CTR")
    plt.title("CTR Model: Actual vs Predicted (Test Set)")
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f"evaluation_scatter_{ts}.png")
    plt.savefig(plot_path)
    plt.close()

    print("\nModel evaluation completed.")
    print(json.dumps(results, indent=2))
    print(f"\nSaved report: {report_path}")
    print(f"Saved scatter plot: {plot_path}")

    return {"metrics": results, "report": report_path, "plot": plot_path}


if __name__ == "__main__":
    evaluate()
