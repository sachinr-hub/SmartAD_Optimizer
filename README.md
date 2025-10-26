# SmartAd Optimizer

An AI-powered ad optimization platform built with Streamlit. It predicts CTR using a CNN model and selects ads using Thompson Sampling. Includes optional authentication with MongoDB and a JSON fallback.

## Project Structure

```
AdOptimization/
├─ app.py                     # Streamlit app (CTR prediction + Thompson Sampling simulation)
├─ auth/
│  └─ __init__.py             # Package marker (auth modules will be moved here)
├─ database.py                # MongoDB + fallback wiring (to be moved to auth/)
├─ user_auth.py               # Streamlit auth UI and session handling (to be moved to auth/)
├─ fallback_auth.py           # JSON-based fallback auth (to be moved to auth/)
├─ preprocess_data.py         # Data preprocessing script to generate training/simulation data
├─ train_ctr_model.py         # Model training script (saves model and plots)
├─ rl_simulation.py           # Standalone simulation demo (optional)
├─ data/
│  ├─ Dataset_Ads.csv
│  ├─ dl_data.csv
│  ├─ rl_data.csv
│  ├─ rl_data_processed_for_simulation.csv
│  └─ model_input_columns.csv
├─ models/
│  ├─ ctr_prediction_model.h5
│  └─ numerical_scaler.pkl
├─ results/                   # Training plots will be written here
├─ requirements.txt
├─ .gitignore
└─ .env                       # Environment variables (optional)
```

## Setup

1) Create and activate a virtual environment
- Windows (PowerShell):
```
python -m venv venv
./venv/Scripts/Activate.ps1
```

2) Install dependencies
```
pip install -r requirements.txt
```

3) Environment variables (optional)
- Create a `.env` file if you want to use MongoDB for auth:
```
MONGODB_URI=mongodb+srv://<user>:<password>@<cluster-url>/?retryWrites=true&w=majority
```
If not provided, the app will automatically fall back to JSON-based local auth in `data/users.json`.

## Data Preparation

- Place your raw dataset at `data/Dataset_Ads.csv`.
- Run preprocessing to generate model inputs and RL simulation data:
```
python preprocess_data.py
```
This will create:
- `data/dl_data.csv`
- `data/rl_data.csv`
- `data/rl_data_processed_for_simulation.csv`
- `data/model_input_columns.csv`
- `models/numerical_scaler.pkl`

## Train the CTR Model

```
python train_ctr_model.py
```
Outputs:
- `models/ctr_prediction_model.h5`
- `results/training_history.png`
- `results/actual_vs_predicted_ctr.png`

## Run the App

```
streamlit run app.py
```
Features:
- CTR prediction for a single user + ad configuration
- Thompson Sampling simulation across ad variants
- Optional login/register + profile management

## Recommended Structure Changes (planned)

To improve maintainability, move auth-related files into the `auth/` package:
- `database.py` -> `auth/database.py`
- `user_auth.py` -> `auth/user_auth.py`
- `fallback_auth.py` -> `auth/fallback_auth.py`

Optionally, move standalone scripts to a `scripts/` folder:
- `preprocess_data.py`, `train_ctr_model.py`, `rl_simulation.py` -> `scripts/`

The app’s imports are already made robust to support both the current and the proposed structure.

## Notes
- `venv/` and `__pycache__/` are ignored via `.gitignore`.
- `plotly` was removed from `requirements.txt` as it wasn't used.
- Added `joblib` to `requirements.txt` (used to load the scaler).
