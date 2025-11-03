import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# Load the preprocessed data
data = pd.read_csv("data/dl_data.csv")

# Define features (X) and target (y)
# Ensure 'Unnamed: 0' is dropped if it's an artifact from saving/loading CSVs with index=True
if 'Unnamed: 0' in data.columns:
    X = data.drop(columns=["CTR", "Unnamed: 0"])
else:
    X = data.drop(columns=["CTR"])
y = data["CTR"]

# Get the expected column order from the saved file
model_input_columns = pd.read_csv("data/model_input_columns.csv").columns.tolist()
X = X[model_input_columns]  # Ensure X has the correct columns in the correct order

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(dtype=np.float32), y.to_numpy(dtype=np.float32), test_size=0.2, random_state=42)

# ========== CLASS WEIGHTING FOR IMBALANCED DATA ========== 
# Convert CTR to binary (click/no-click) for class weighting
# Using median as threshold
median_ctr = np.median(y_train)
y_train_binary = (y_train > median_ctr).astype(int)

# Compute class weights
classes = np.array([0, 1])
class_weights = compute_class_weight('balanced', classes=classes, y=y_train_binary)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

print(f"\nClass Distribution:")
print(f"No-Click (0): {np.sum(y_train_binary == 0)} samples")
print(f"Click (1): {np.sum(y_train_binary == 1)} samples")
print(f"Class Weights: {class_weight_dict}")
print(f"This gives {class_weights[1]/class_weights[0]:.1f}x more weight to clicks\n")

# Reshape input for Conv1D: (samples, timesteps, features) -> (samples, features, 1)
X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# ========== SIMPLIFIED CNN ARCHITECTURE ========== 
# Simpler architecture to prevent overfitting on sparse interaction features
model = Sequential([
    Input(shape=(X_train.shape[1], 1)),
    Conv1D(filters=32, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    
    # Dense layers with moderate regularization
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)  # Linear output for regression
])

# Compile with standard MSE and better metrics
initial_lr = 1e-3
use_cosine_restarts = True
if use_cosine_restarts:
    lr_schedule = CosineDecayRestarts(initial_lr, first_decay_steps=1000, t_mul=2.0, m_mul=0.7)
    optimizer = Adam(learning_rate=lr_schedule)
else:
    optimizer = Adam(learning_rate=initial_lr)

model.compile(
    optimizer=optimizer,
    loss='mse',  # Standard MSE for regression
    metrics=['mae', tf.keras.metrics.AUC(name='auc')]
)

# Model summary
model.summary()

# Train the model with early stopping
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
callbacks = [early_stop] if use_cosine_restarts else [early_stop, reduce_lr]

history = model.fit(
    X_train_reshaped, y_train,
    epochs=100,
    batch_size=64,  # Larger batch size for stability
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# Evaluate the model
# Note: model.evaluate returns [loss, mae, auc] due to metrics
eval_results = model.evaluate(X_test_reshaped, y_test, verbose=0)
loss = eval_results[0]
mae = eval_results[1]
auc = eval_results[2]
print(f"\nTest Loss: {loss}, Test MAE: {mae}, Test AUC: {auc}")

# Predict CTR on the test set
y_pred = model.predict(X_test_reshaped).ravel()

# Calculate and print metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Baseline: predict mean of training CTR
baseline_pred = np.full_like(y_test, fill_value=float(np.mean(y_train)))
baseline_mae = mean_absolute_error(y_test, baseline_pred)

print(f"MSE: {mse}")
print(f"MAE: {mae}")
print(f"R²: {r2}")
print(f"Baseline MAE (predict mean CTR): {baseline_mae}")

# Ensure output directories exist
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Plot Actual vs. Predicted CTR
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual CTR")
plt.ylabel("Predicted CTR")
plt.title("Actual vs. Predicted CTR")
plt.savefig("results/actual_vs_predicted_ctr.png")
plt.close()

# Plot training history (loss and MAE)
def plot_training_history(history):
    metrics = [key for key in history.history.keys() if not key.startswith("val_")]
    plt.figure(figsize=(12, 5))
    for i, metric in enumerate(metrics):
        plt.subplot(1, len(metrics), i + 1)
        # Plot from epoch 10 onwards for better visualization if initial values are too high
        plt.plot(history.history[metric][10:], label=f'Training {metric}', marker='o')
        if f'val_{metric}' in history.history:
            plt.plot(history.history[f'val_{metric}'][10:], label=f'Validation {metric}', marker='o')
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.title(metric.capitalize())
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/training_history.png")
    plt.close()

plot_training_history(history)
print("Training history plot saved to results/training_history.png")

# Save the trained model
model.save("models/ctr_prediction_model.keras")   # new recommended format

print("Trained CTR prediction model saved to models/ctr_prediction_model.keras")
