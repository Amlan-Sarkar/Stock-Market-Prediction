# ======================================================================
# STOCK PRICE PREDICTION — SERVING DASHBOARD
# ----------------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
)

import joblib
import xgboost as xgb

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import shap

np.random.seed(42)
tf.random.set_seed(42)

# ======================================================================
# PATHS
# ======================================================================
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DEFAULT_TRAIN_PATH = DATA_DIR / "Trainset.xlsx"
DEFAULT_TEST_PATH  = DATA_DIR / "Testset.xlsx"

# ======================================================================
# PAGE CONFIG
# ======================================================================
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

matplotlib.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.edgecolor":   "#D1D5DB",
    "axes.labelcolor":  "#374151",
    "axes.titlecolor":  "#111827",
    "xtick.color":      "#6B7280",
    "ytick.color":      "#6B7280",
    "axes.grid":        False,
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
})

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }

.hero-title { font-size: 2.2rem; font-weight: 800; letter-spacing: -0.02em;
              margin-bottom: 0.1rem; color: #111827; }
.hero-sub   { color: #6B7280; font-size: 1.02rem; margin-bottom: 0.5rem; }

.section-eyebrow { text-transform: uppercase; letter-spacing: 0.08em;
                    font-size: 0.72rem; font-weight: 700; color: #9CA3AF;
                    margin-bottom: 2px; }
.section-title   { font-size: 1.3rem; font-weight: 700; color: #111827;
                    margin-bottom: 6px; margin-top: 0px; }

.insight-box, .warning-box, .verdict-box {
    padding: 14px 18px; border-radius: 8px; margin: 10px 0 22px 0;
    font-size: 0.94rem; line-height: 1.55; color: #1F2937;
}
.insight-box  { background: #F0F5FF; border-left: 4px solid #4472C4; }
.warning-box  { background: #FFF8EB; border-left: 4px solid #D97706; }
.verdict-box  { background: #F0FDF4; border-left: 4px solid #16A34A; }
.insight-box b, .warning-box b, .verdict-box b { color: #111827; }

.decision-grid { display: flex; gap: 14px; margin: 10px 0 24px 0; flex-wrap: wrap; }
.decision-card { flex: 1; min-width: 220px; background: white;
                  border: 1px solid #E5E7EB; border-radius: 10px; padding: 16px 18px; }
.decision-card .dc-label { font-size: 0.72rem; font-weight: 700; color: #9CA3AF;
                             text-transform: uppercase; letter-spacing: 0.06em; }
.decision-card .dc-model { font-size: 1.1rem; font-weight: 800; color: #111827;
                             margin: 4px 0 6px 0; }
.decision-card .dc-why   { font-size: 0.85rem; color: #4B5563; line-height: 1.4; }

.pill { display:inline-block; padding: 2px 10px; border-radius: 20px;
        font-size: 0.76rem; font-weight: 600; color: white; margin-right: 6px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ======================================================================
# CONSTANTS — mirrored exactly from STMP.ipynb
# ======================================================================
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

FEATURES = [
    "Open_ret", "High_ret", "Low_ret", "Close_ret",
    "Volume_log", "RSI", "MACD_norm", "MACD_Signal_norm", "BB_PctB",
]
TARGET_COL  = FEATURES.index("Close_ret")
SEQ_LEN     = 60
N_FEAT      = len(FEATURES)

NN_FEATURES = ["Close_ret", "Volume_log", "RSI", "MACD_norm", "MACD_Signal_norm"]
NN_INDICES  = [FEATURES.index(f) for f in NN_FEATURES]
N_FEAT_NN   = len(NN_FEATURES)

TRAIN_VAL_SPLIT = 0.8

MODEL_FILES = {
    "BiLSTM":            "best_bilstm.keras",
    "GRU":               "best_gru.keras",
    "XGBoost":           "best_xgb.json",
    "Linear Regression": "model_linear_regression.pkl",
    "Random Forest":     "model_random_forest.pkl",
}
HISTORY_FILES = {"BiLSTM": "history_bilstm.json", "GRU": "history_gru.json"}

BASE_MODEL_ORDER = ["BiLSTM", "GRU", "XGBoost", "Linear Regression", "Random Forest"]
ALL_MODEL_ORDER  = BASE_MODEL_ORDER + ["Ensemble"]

MODEL_COLORS = {
    "BiLSTM":            "#4472C4",
    "GRU":               "#2E9E5B",
    "XGBoost":           "#C0392B",
    "Linear Regression": "#7D3C98",
    "Random Forest":     "#784212",
    "Ensemble":          "#B7950B",
}
MODEL_TAGLINE = {
    "BiLSTM":            "3-layer bidirectional LSTM (64 units/direction)",
    "GRU":               "3-layer GRU (64 units)",
    "XGBoost":           "Gradient-boosted trees (1,000 rounds, early-stopped)",
    "Linear Regression": "Ordinary least squares baseline",
    "Random Forest":     "Bagged decision trees (200 trees)",
    "Ensemble":          "Inverse-validation-MAE weighted blend of the five models",
}
METRIC_LABELS = {
    "RMSE": "RMSE  (lower is better)",
    "MAE":  "MAE  (lower is better)",
    "MAPE": "MAPE %  (lower is better)",
    "R2":   "R²  (higher is better)",
    "EVS":  "Explained Variance  (higher is better)",
    "DA":   "Directional Accuracy %  (higher is better)",
}
LOWER_IS_BETTER = {"RMSE", "MAE", "MAPE"}

# ======================================================================
# SMALL UI HELPERS
# ======================================================================
def insight(text):
    st.markdown(f'<div class="insight-box">💡 <b>Insight —</b> {text}</div>', unsafe_allow_html=True)

def caveat(text):
    st.markdown(f'<div class="warning-box">⚠️ <b>Caveat —</b> {text}</div>', unsafe_allow_html=True)

def verdict(text):
    st.markdown(f'<div class="verdict-box">✅ <b>Bottom line —</b> {text}</div>', unsafe_allow_html=True)

def eyebrow_title(eyebrow, title):
    st.markdown(
        f'<div class="section-eyebrow">{eyebrow}</div><div class="section-title">{title}</div>',
        unsafe_allow_html=True,
    )

def model_pill(name):
    return f'<span class="pill" style="background:{MODEL_COLORS[name]}">{name}</span>'

def models_at_extreme(metrics, key, want="min"):
    """Return every model name tied for the best value of `key` (handles ties honestly)."""
    values = {n: m[key] for n, m in metrics.items()}
    target = min(values.values()) if want == "min" else max(values.values())
    return [n for n, v in values.items() if np.isclose(v, target)]

# ======================================================================
# PIPELINE FUNCTIONS — mirrored line-for-line from STMP.ipynb
# ======================================================================
def validate_required_columns(df: pd.DataFrame, label: str):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"**{label}** is missing required column(s): {missing}")

def chronology_warning(combined: pd.DataFrame):
    """Returns a warning string if date ordering can't be verified/confirmed, else None."""
    if "Date" not in combined.columns:
        return ("No 'Date' column found in the data — chronological ordering of "
                "train → test is assumed, not verified.")
    dates = pd.to_datetime(combined["Date"])
    if not dates.is_monotonic_increasing:
        return ("Dates are not sorted ascending — the indicator/lookback logic assumes "
                "the training set is immediately followed by the test set in time.")
    return None

@st.cache_data(show_spinner=False)
def engineer_features(train_data: pd.DataFrame, test_data: pd.DataFrame):
    """Cell #2 TECHNICAL INDICATORS — RSI, MACD, Bollinger Bands, then the
    stationary return/log/normalized transforms that are the actual model
    features. Returns the same train/test split shape, minus warm-up rows."""
    train_len    = len(train_data)
    combined_len = train_len + len(test_data)
    combined = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)

    chron_warning = chronology_warning(combined)

    delta    = combined["Close"].diff()
    gain     = delta.where(delta > 0, 0)
    loss     = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs       = avg_gain / avg_loss.replace(0, 1e-10)
    combined["RSI"] = 100 - (100 / (1 + rs))

    ema12 = combined["Close"].ewm(span=12, adjust=False).mean()
    ema26 = combined["Close"].ewm(span=26, adjust=False).mean()
    combined["MACD"]        = ema12 - ema26
    combined["MACD_Signal"] = combined["MACD"].ewm(span=9, adjust=False).mean()

    rolling_mean = combined["Close"].rolling(20).mean()
    rolling_std  = combined["Close"].rolling(20).std()
    combined["BB_Upper"] = rolling_mean + 2 * rolling_std
    combined["BB_Lower"] = rolling_mean - 2 * rolling_std

    prev_close = combined["Close"].shift(1)
    combined["Open_ret"]  = (combined["Open"]  - prev_close) / prev_close
    combined["High_ret"]  = (combined["High"]  - prev_close) / prev_close
    combined["Low_ret"]   = (combined["Low"]   - prev_close) / prev_close
    combined["Close_ret"] = (combined["Close"] - prev_close) / prev_close
    combined["Volume_log"] = np.log1p(combined["Volume"])
    combined["MACD_norm"]        = combined["MACD"] / combined["Close"]
    combined["MACD_Signal_norm"] = combined["MACD_Signal"] / combined["Close"]

    bb_width = (combined["BB_Upper"] - combined["BB_Lower"]).replace(0, 1e-10)
    combined["BB_PctB"] = (combined["Close"] - combined["BB_Lower"]) / bb_width

    combined.dropna(inplace=True)
    combined.reset_index(drop=True, inplace=True)

    rows_dropped  = combined_len - len(combined)
    new_train_len = train_len - rows_dropped
    if new_train_len <= SEQ_LEN + 10:
        raise ValueError(
            "After dropping indicator warm-up rows, too little training data remains "
            f"({new_train_len} rows) to build {SEQ_LEN}-day sequences. Provide a longer "
            "training series."
        )

    train_out = combined.iloc[:new_train_len].reset_index(drop=True)
    test_out  = combined.iloc[new_train_len:].reset_index(drop=True)
    return train_out, test_out, rows_dropped, chron_warning

def split_train_val(train_data: pd.DataFrame):
    split_idx = int(TRAIN_VAL_SPLIT * len(train_data))
    train_df = train_data.iloc[:split_idx].reset_index(drop=True)
    val_df   = train_data.iloc[split_idx:].reset_index(drop=True)
    return train_df, val_df

def build_sequences(data: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i, TARGET_COL])
    return np.array(X), np.array(y)

def build_raw_targets(series, seq_len):
    return np.asarray(series)[seq_len:]

def inverse_return(scaled_preds, scaler):
    dummy = np.zeros((len(scaled_preds), N_FEAT))
    dummy[:, TARGET_COL] = scaled_preds
    return scaler.inverse_transform(dummy)[:, TARGET_COL]

def predict_returns(model, X, scaler):
    raw = model.predict(X, verbose=0).flatten()
    return inverse_return(raw, scaler)

def reconstruct_from_returns(pred_returns, prev_closes):
    return prev_closes * (1 + pred_returns)

def directional_accuracy(real, pred):
    real_dir = np.diff(real) > 0
    pred_dir = np.diff(pred) > 0
    return float(np.mean(real_dir == pred_dir) * 100)

def compute_regression_metrics(real, pred):
    rmse = float(np.sqrt(mean_squared_error(real, pred)))
    mae  = float(mean_absolute_error(real, pred))
    mape = float(np.mean(np.abs((real - pred) / np.where(real != 0, real, np.nan))) * 100)
    r2   = float(r2_score(real, pred))
    evs  = float(explained_variance_score(real, pred))
    da   = directional_accuracy(real, pred)
    return dict(RMSE=rmse, MAE=mae, MAPE=mape, R2=r2, EVS=evs, DA=da)

def build_all_sequences(train_df, val_df, train_data, test_data, scaler):
    """Reproduces notebook cells #6-#9 and #14: normalization + sequence
    construction for train / validation / test, all from one fitted scaler."""
    train_scaled = scaler.transform(train_df[FEATURES].values)
    X_train, y_train = build_sequences(train_scaled, SEQ_LEN)
    X_train_nn  = X_train[:, :, NN_INDICES]
    y_train_ret = build_raw_targets(train_df["Close_ret"].values, SEQ_LEN)
    X_train_flat = X_train.reshape(len(X_train), -1)

    val_inputs = pd.concat(
        (train_df[FEATURES].tail(SEQ_LEN), val_df[FEATURES]), axis=0
    ).reset_index(drop=True)
    val_scaled_full = scaler.transform(val_inputs.values)
    X_val, y_val = build_sequences(val_scaled_full, SEQ_LEN)
    X_val_nn  = X_val[:, :, NN_INDICES]
    y_val_ret = build_raw_targets(val_inputs["Close_ret"].values, SEQ_LEN)
    X_val_flat = X_val.reshape(len(X_val), -1)

    dataset_total = pd.concat(
        (train_data[FEATURES], test_data[FEATURES]), axis=0
    ).reset_index(drop=True)
    inputs = dataset_total[len(dataset_total) - len(test_data) - SEQ_LEN:].values
    inputs = scaler.transform(inputs)
    X_test, _ = build_sequences(inputs, SEQ_LEN)
    X_test_flat = X_test.reshape(len(X_test), -1)
    X_test_nn   = X_test[:, :, NN_INDICES]
    real_close  = test_data["Close"].values

    if len(X_test) != len(real_close):
        raise ValueError(
            f"Length mismatch: X_test has {len(X_test)} rows but real_close has "
            f"{len(real_close)} rows. Check test data alignment after indicator computation."
        )

    prev_closes_test = np.concatenate([[train_data["Close"].values[-1]], test_data["Close"].values[:-1]])
    prev_closes_val  = np.concatenate([[train_df["Close"].values[-1]], val_df["Close"].values[:-1]])
    val_real = val_df["Close"].values

    return dict(
        X_train=X_train, y_train=y_train, X_train_nn=X_train_nn,
        y_train_ret=y_train_ret, X_train_flat=X_train_flat,
        X_val=X_val, y_val=y_val, X_val_nn=X_val_nn,
        y_val_ret=y_val_ret, X_val_flat=X_val_flat,
        X_test=X_test, X_test_flat=X_test_flat, X_test_nn=X_test_nn,
        real_close=real_close, prev_closes_test=prev_closes_test,
        prev_closes_val=prev_closes_val, val_real=val_real,
    )

# ======================================================================
# DATA LOADING
# ======================================================================
@st.cache_data(show_spinner="Reading Excel file…")
def read_excel_cached(source, cache_tag):
    return pd.read_excel(source)

def resolve_data_source():
    """Returns (train_raw, test_raw, source_label) or (None, None, None)."""
    with st.sidebar:
        st.markdown("### 📁 Data")
        bundled_available = DEFAULT_TRAIN_PATH.exists() and DEFAULT_TEST_PATH.exists()
        options = ["Use bundled dataset (data/)"] if bundled_available else []
        options.append("Upload my own Excel files")
        default_index = 0
        choice = st.radio("Source", options, index=default_index, label_visibility="collapsed")

        if choice == "Use bundled dataset (data/)":
            st.caption(f"`{DEFAULT_TRAIN_PATH.name}` + `{DEFAULT_TEST_PATH.name}` from `data/`")
            train_raw = read_excel_cached(str(DEFAULT_TRAIN_PATH), DEFAULT_TRAIN_PATH.stat().st_mtime)
            test_raw  = read_excel_cached(str(DEFAULT_TEST_PATH), DEFAULT_TEST_PATH.stat().st_mtime)
            return train_raw, test_raw, "bundled"
        else:
            train_file = st.file_uploader("Training set (.xlsx)", type=["xlsx"], key="train_upload")
            test_file  = st.file_uploader("Test set (.xlsx)", type=["xlsx"], key="test_upload")
            if train_file is None or test_file is None:
                st.caption("Upload both files to continue.")
                return None, None, None
            train_raw = read_excel_cached(train_file, train_file.name + str(train_file.size))
            test_raw  = read_excel_cached(test_file, test_file.name + str(test_file.size))
            return train_raw, test_raw, "custom"

# ======================================================================
# MODEL ARCHITECTURES — identical hyperparameters to STMP.ipynb
# ======================================================================
def build_bilstm():
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(SEQ_LEN, N_FEAT_NN)),
        Dropout(0.2),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(64)),
        Dropout(0.2),
        Dense(1),
    ], name="BiLSTM")
    model.compile(optimizer="adam", loss="mse")
    return model

def build_gru():
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=(SEQ_LEN, N_FEAT_NN)),
        Dropout(0.2),
        GRU(64, return_sequences=True),
        Dropout(0.2),
        GRU(64),
        Dropout(0.2),
        Dense(1),
    ], name="GRU")
    model.compile(optimizer="adam", loss="mse")
    return model

def build_xgb_model():
    return xgb.XGBRegressor(
        n_estimators=1000, learning_rate=0.01, max_depth=8,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        reg_alpha=0.1, reg_lambda=1, verbosity=0, early_stopping_rounds=20,
    )

def build_rf_model():
    return RandomForestRegressor(
        n_estimators=200, max_depth=12, max_features="sqrt",
        min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1, verbose=0,
    )

# ======================================================================
# MODEL LOADING (cached; `version` busts the cache after retraining)
# ======================================================================
def _load_one_model(name, path):
    try:
        if name in ("BiLSTM", "GRU"):
            return load_model(path), None
        elif name == "XGBoost":
            m = xgb.XGBRegressor()
            m.load_model(str(path))
            return m, None
        else:
            return joblib.load(path), None
    except Exception as e:  # noqa: BLE001 - surfaced to the UI, not swallowed
        return None, str(e)

@st.cache_resource(show_spinner="Loading trained models…")
def load_models(models_dir_str, version):
    models_dir = Path(models_dir_str)
    models, errors = {}, {}
    for name, fname in MODEL_FILES.items():
        path = models_dir / fname
        if not path.exists():
            errors[name] = "not found"
            continue
        model, err = _load_one_model(name, path)
        if model is not None:
            models[name] = model
        else:
            errors[name] = err
    return models, errors

def load_histories(models_dir: Path):
    histories = {}
    for name, fname in HISTORY_FILES.items():
        path = models_dir / fname
        if path.exists():
            try:
                with open(path) as f:
                    histories[name] = json.load(f)
            except Exception:
                pass
    return histories

# ======================================================================
# LIVE TRAINING (on-demand — mirrors notebook cells #9-#13 exactly)
# ======================================================================
class StreamlitEpochProgress(tf.keras.callbacks.Callback):
    """Updates existing placeholders in place — does not trigger reruns."""
    def __init__(self, text_ph, bar_ph, total_epochs, label):
        super().__init__()
        self.text_ph, self.bar_ph = text_ph, bar_ph
        self.total_epochs, self.label = total_epochs, label

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        frac = min((epoch + 1) / self.total_epochs, 1.0)
        self.bar_ph.progress(frac)
        self.text_ph.markdown(
            f"`{self.label}` epoch **{epoch + 1}/{self.total_epochs}** — "
            f"loss `{logs.get('loss', float('nan')):.5f}` · "
            f"val_loss `{logs.get('val_loss', float('nan')):.5f}`"
        )

def train_all_models(seq, models_dir: Path):
    """Trains BiLSTM, GRU, XGBoost, Linear Regression, and Random Forest with
    the exact architectures/hyperparameters/callbacks used in STMP.ipynb, and
    persists them (plus BiLSTM/GRU loss history, which the notebook doesn't
    normally persist) so the Diagnostics tab has real loss curves next time."""
    models_dir.mkdir(parents=True, exist_ok=True)
    trained, histories = {}, {}
    overall = st.progress(0.0, text="Starting training pipeline…")

    # ---- BiLSTM ----
    st.markdown("**Step 1 / 5 — Bidirectional LSTM** (up to 100 epochs, early-stopped)")
    text_ph, bar_ph = st.empty(), st.progress(0.0)
    model_bilstm = build_bilstm()
    hist = model_bilstm.fit(
        seq["X_train_nn"], seq["y_train"], epochs=100, batch_size=32,
        validation_data=(seq["X_val_nn"], seq["y_val"]),
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=0),
            ModelCheckpoint(str(models_dir / "best_bilstm.keras"), monitor="val_loss", save_best_only=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=0),
            StreamlitEpochProgress(text_ph, bar_ph, 100, "BiLSTM"),
        ],
        verbose=0,
    )
    trained["BiLSTM"] = model_bilstm
    histories["BiLSTM"] = {"loss": hist.history["loss"], "val_loss": hist.history["val_loss"]}
    overall.progress(0.2, text="BiLSTM done — training GRU…")

    # ---- GRU ----
    st.markdown("**Step 2 / 5 — GRU** (up to 100 epochs, early-stopped)")
    text_ph, bar_ph = st.empty(), st.progress(0.0)
    model_gru = build_gru()
    hist = model_gru.fit(
        seq["X_train_nn"], seq["y_train"], epochs=100, batch_size=32,
        validation_data=(seq["X_val_nn"], seq["y_val"]),
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=0),
            ModelCheckpoint(str(models_dir / "best_gru.keras"), monitor="val_loss", save_best_only=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=0),
            StreamlitEpochProgress(text_ph, bar_ph, 100, "GRU"),
        ],
        verbose=0,
    )
    trained["GRU"] = model_gru
    histories["GRU"] = {"loss": hist.history["loss"], "val_loss": hist.history["val_loss"]}
    overall.progress(0.4, text="GRU done — training XGBoost…")

    # ---- XGBoost ----
    st.markdown("**Step 3 / 5 — XGBoost** (1,000 rounds, early-stopped)")
    model_xgb = build_xgb_model()
    model_xgb.fit(
        seq["X_train_flat"], seq["y_train_ret"],
        eval_set=[(seq["X_val_flat"], seq["y_val_ret"])], verbose=False,
    )
    model_xgb.save_model(str(models_dir / "best_xgb.json"))
    trained["XGBoost"] = model_xgb
    overall.progress(0.7, text="XGBoost done — fitting Linear Regression…")

    # ---- Linear Regression ----
    st.markdown("**Step 4 / 5 — Linear Regression**")
    model_lr = LinearRegression()
    model_lr.fit(seq["X_train_flat"], seq["y_train_ret"])
    joblib.dump(model_lr, models_dir / "model_linear_regression.pkl")
    trained["Linear Regression"] = model_lr
    overall.progress(0.85, text="Linear Regression done — training Random Forest…")

    # ---- Random Forest ----
    st.markdown("**Step 5 / 5 — Random Forest**")
    model_rf = build_rf_model()
    model_rf.fit(seq["X_train_flat"], seq["y_train_ret"])
    joblib.dump(model_rf, models_dir / "model_random_forest.pkl")
    trained["Random Forest"] = model_rf
    overall.progress(1.0, text="All 5 models trained and saved to models/.")

    for name, fname in HISTORY_FILES.items():
        if name in histories:
            with open(models_dir / fname, "w") as f:
                json.dump(histories[name], f)

    return trained, histories

# ======================================================================
# INFERENCE
# ======================================================================
def run_inference(models: dict, seq: dict, scaler):
    """Reproduces notebook cells #15-#16: validation predictions -> inverse-MAE
    ensemble weights -> test predictions -> regression metrics, restricted to
    whichever models actually loaded/trained successfully."""
    available = [n for n in BASE_MODEL_ORDER if n in models]

    def predict(name, X_nn, X_flat, prev_closes):
        model = models[name]
        if name in ("BiLSTM", "GRU"):
            returns = predict_returns(model, X_nn, scaler)
        else:
            returns = model.predict(X_flat)
        return reconstruct_from_returns(returns, prev_closes)

    val_preds = {n: predict(n, seq["X_val_nn"], seq["X_val_flat"], seq["prev_closes_val"]) for n in available}
    val_mae   = {n: mean_absolute_error(seq["val_real"], p) for n, p in val_preds.items()}

    weights = {}
    if len(available) >= 2:
        inv_mae   = {n: 1.0 / mae if mae > 0 else 0.0 for n, mae in val_mae.items()}
        total_inv = sum(inv_mae.values())
        weights   = {n: w / total_inv for n, w in inv_mae.items()} if total_inv > 0 else {}

    preds = {n: predict(n, seq["X_test_nn"], seq["X_test_flat"], seq["prev_closes_test"]) for n in available}
    if weights:
        preds["Ensemble"] = sum(weights[n] * preds[n] for n in weights)

    metrics = {n: compute_regression_metrics(seq["real_close"], p) for n, p in preds.items()}
    return dict(
        available=available, preds=preds, val_preds=val_preds,
        val_mae=val_mae, weights=weights, metrics=metrics,
    )

@st.cache_data(show_spinner="Computing SHAP values for XGBoost…")
def compute_shap(_model_xgb, X_train_flat, sample_size=1000, seed=42):
    """Underscore-prefixed arg tells Streamlit not to hash the model object
    itself; X_train_flat's content is what actually determines cache validity."""
    n = min(sample_size, len(X_train_flat))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_train_flat), size=n, replace=False)
    X_shap = X_train_flat[idx]
    explainer = shap.TreeExplainer(_model_xgb)
    shap_values = explainer.shap_values(X_shap)
    return shap_values, X_shap

# ======================================================================
# PLOTTING HELPERS
# ======================================================================
def render_grid(names, plot_fn, n_cols=3):
    cols = st.columns(n_cols)
    for i, name in enumerate(names):
        with cols[i % n_cols]:
            fig = plot_fn(name)
            st.pyplot(fig)
            plt.close(fig)

def fig_actual_vs_predicted(real_close, preds, selected):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(real_close, label="Actual", color="black", linewidth=2.5, zorder=5)
    for name in selected:
        ax.plot(preds[name], label=name, alpha=0.85, color=MODEL_COLORS[name], linewidth=1.5)
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Close Price")
    ax.set_title("Actual vs Predicted Stock Price", fontsize=14)
    ax.legend(fontsize=10, frameon=False)
    plt.tight_layout()
    return fig

def fig_residual_hist(name, real_close, pred):
    fig, ax = plt.subplots(figsize=(5, 4))
    residuals = real_close - pred
    ax.hist(residuals, bins=30, color=MODEL_COLORS[name], edgecolor="black", alpha=0.75)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.8)
    ax.set_title(name, fontsize=11)
    ax.set_xlabel("Error (Actual − Predicted)")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    return fig

def fig_residual_scatter(name, real_close, pred):
    fig, ax = plt.subplots(figsize=(5, 4))
    residuals = real_close - pred
    ax.scatter(pred, residuals, color=MODEL_COLORS[name], alpha=0.6, s=20)
    ax.axhline(0, color="red", linestyle="--", linewidth=1.8)
    ax.set_title(name, fontsize=11)
    ax.set_xlabel("Predicted Close Price")
    ax.set_ylabel("Residuals")
    plt.tight_layout()
    return fig

def fig_pred_vs_actual_scatter(name, real_close, pred, r2):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(real_close, pred, color=MODEL_COLORS[name], alpha=0.6, s=20)
    mn, mx = min(real_close.min(), pred.min()), max(real_close.max(), pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.8)
    ax.set_title(name, fontsize=11)
    ax.set_xlabel("Actual Close")
    ax.set_ylabel("Predicted Close")
    ax.text(0.05, 0.92, f"R²={r2:.3f}", transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.7))
    plt.tight_layout()
    return fig

def fig_metric_bar(metric, metrics, names):
    fig, ax = plt.subplots(figsize=(5, 4))
    values = [metrics[n][metric] for n in names]
    colors = [MODEL_COLORS[n] for n in names]
    bars = ax.bar(names, values, color=colors, edgecolor="black", width=0.6)
    ax.set_title(METRIC_LABELS[metric], fontsize=11)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    span = (ax.get_ylim()[1] - ax.get_ylim()[0]) or 1.0
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + span * 0.015,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    return fig

def fig_loss_curve(name, history):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["loss"], label="Train Loss", linewidth=2)
    ax.plot(history["val_loss"], label="Val Loss", linewidth=2, linestyle="--")
    ax.set_title(f"{name} — Loss Curve", fontsize=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend(frameon=False)
    plt.tight_layout()
    return fig

def fig_correlation_heatmap(corr, title, cmap, mask_upper=False):
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool)) if mask_upper else None
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, mask=mask, square=True,
                linewidths=0.5, annot_kws={"size": 9}, ax=ax)
    ax.set_title(title, fontsize=13, pad=10)
    plt.tight_layout()
    return fig

def fig_barh_importance(labels, values, color, xlabel, title):
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh([labels[i] for i in order], np.array(values)[order], color=color, edgecolor="black", alpha=0.9)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    return fig

# ======================================================================
# SIDEBAR — status & actions (rendered after data/model state is known)
# ======================================================================
def render_sidebar_header():
    with st.sidebar:
        st.markdown("## 📈 Stock Price Prediction")
        st.caption("Five models + a weighted ensemble — served straight from STMP.ipynb.")
        st.divider()

def render_sidebar_footer(source_label, rows_dropped, chron_warning):
    with st.sidebar:
        st.divider()
        with st.expander("ℹ️ About this dashboard"):
            st.markdown(
                "This app re-runs the exact feature engineering and inference logic "
                "from `STMP.ipynb` — RSI/MACD/Bollinger-derived return features, a "
                "60-day lookback window, and five independently trained models "
                "blended into an inverse-MAE ensemble.\n\n"
                "Model binaries and plots are intentionally excluded from version "
                "control (see `.gitignore`) and regenerated locally — either by "
                "running the notebook, or with the **Train All Models** action here."
            )
        if chron_warning:
            st.caption(f"⚠️ {chron_warning}")
        st.caption(f"Data source: **{source_label}** · {rows_dropped} row(s) dropped for indicator warm-up.")
        st.caption("Built with Streamlit · TensorFlow/Keras · XGBoost · scikit-learn · SHAP")

# ======================================================================
# LANDING STATES
# ======================================================================
def render_no_data_landing():
    eyebrow_title("SETUP", "No dataset found yet")
    st.markdown(
        "This dashboard needs OHLCV data with columns "
        f"`{', '.join(REQUIRED_COLUMNS)}` (a `Date` column is recommended but optional)."
    )
    st.info(
        f"Place **Trainset.xlsx** and **Testset.xlsx** in `{DATA_DIR.relative_to(BASE_DIR)}/`, "
        "or switch to **Upload my own Excel files** in the sidebar."
    )

def render_data_error(err: Exception):
    eyebrow_title("DATA ERROR", "The uploaded/bundled data couldn't be processed")
    st.error(str(err))
    with st.expander("Full traceback"):
        st.code(traceback.format_exc())

def render_no_models_state(errors, seq, models_dir):
    eyebrow_title("ONE-TIME SETUP", "No trained models found yet")
    st.markdown(
        "`models/` is empty or unreadable — expected on a fresh clone, since trained "
        "weights are `.gitignore`d and regenerated locally. Train all five models now "
        "with the exact architecture and hyperparameters from `STMP.ipynb`:\n\n"
        "- **BiLSTM** & **GRU** — up to 100 epochs each, early-stopped (patience 20)\n"
        "- **XGBoost** — 1,000 rounds, early-stopped (patience 20)\n"
        "- **Linear Regression** and **Random Forest** — fit directly (seconds)\n\n"
        "This runs once and takes a few minutes on CPU. Every future load will read "
        "the saved weights from `models/` instantly instead of retraining."
    )
    with st.expander("Why isn't a model already loaded?"):
        for name, err in errors.items():
            st.caption(f"**{name}** — {err}")
    if st.button("🚀 Train All 5 Models Now", type="primary"):
        st.session_state["_run_training"] = True
        st.rerun()

# ======================================================================
# MAIN DASHBOARD
# ======================================================================
def render_dashboard(train_data, val_df_full, train_df, val_df, test_data,
                      seq, models, model_errors, results, histories, models_dir):
    available   = results["available"]
    preds       = results["preds"]
    metrics     = results["metrics"]
    weights     = results["weights"]
    val_mae     = results["val_mae"]
    real_close  = seq["real_close"]
    shown_names = [n for n in ALL_MODEL_ORDER if n in metrics]

    if model_errors:
        missing = ", ".join(model_errors.keys())
        st.warning(
            f"Showing results for **{len(available)} of 5** base models — "
            f"unavailable: {missing}. Train the full set for the complete comparison."
        )
        if st.button("🚀 Train All 5 Models Now", key="retrain_partial"):
            st.session_state["_run_training"] = True
            st.rerun()

    # ---- derived story facts ----
    best_rmse  = models_at_extreme(metrics, "RMSE", "min")
    worst_rmse = models_at_extreme(metrics, "RMSE", "max")
    best_r2    = models_at_extreme(metrics, "R2", "max")
    best_mape  = models_at_extreme(metrics, "MAPE", "min")
    best_da    = models_at_extreme(metrics, "DA", "max")
    da_values  = [metrics[n]["DA"] for n in shown_names]

    ens_rank_line = ""
    if "Ensemble" in metrics:
        rank = sorted(metrics, key=lambda n: metrics[n]["RMSE"]).index("Ensemble") + 1
        ens_rank_line = f" The Ensemble ranks #{rank} of {len(shown_names)} on RMSE."

    # ---- hero verdict ----
    eyebrow_title("RESULT", "The Bottom Line")
    verdict(
        f"<b>{' / '.join(best_rmse)}</b> produced the lowest test-set price-tracking error "
        f"(RMSE {metrics[best_rmse[0]]['RMSE']:.2f}, R²={metrics[best_rmse[0]]['R2']:.3f}), "
        f"while <b>{' / '.join(worst_rmse)}</b> trailed at RMSE {metrics[worst_rmse[0]]['RMSE']:.2f}."
        + ens_rank_line
    )
    caveat(
        f"Directional Accuracy sits at {min(da_values):.1f}–{max(da_values):.1f}% across "
        f"models — barely above a coin flip. This is expected: day-to-day price "
        f"<i>direction</i> behaves close to a random walk, while price <i>level</i> is "
        f"highly autocorrelated (today's close is close to yesterday's). The strong R² "
        f"scores reflect the latter, not a trading edge — <b>don't read these models as "
        f"directional trading signals.</b>"
    )

    # ---- decision guide ----
    eyebrow_title("DECISION GUIDE", "Which model should you actually use?")
    ensemble_card = ""
    if "Ensemble" in metrics:
        ensemble_card = f'''<div class="decision-card">
            <div class="dc-label">Want a single blended estimate</div>
            <div class="dc-model" style="color:{MODEL_COLORS['Ensemble']}">Ensemble</div>
            <div class="dc-why">Inverse-validation-MAE blend of all base models — hedges
            against any one model's regime-specific failures, though on this run it does
            not beat the single best model on RMSE.</div>
        </div>'''
    st.markdown(
        f'''<div class="decision-grid">
        <div class="decision-card">
            <div class="dc-label">Track price level accurately</div>
            <div class="dc-model" style="color:{MODEL_COLORS[best_rmse[0]]}">{' / '.join(best_rmse)}</div>
            <div class="dc-why">Lowest RMSE/MAPE on the held-out test set — best suited to
            valuation-style tracking, not entry/exit timing.</div>
        </div>
        {ensemble_card}
        <div class="decision-card">
            <div class="dc-label">Calling next-day direction</div>
            <div class="dc-model" style="color:#DC2626">None reliably</div>
            <div class="dc-why">All models cluster near {np.mean(da_values):.0f}% directional
            accuracy — indistinguishable from chance. Don't use any of these for buy/sell timing.</div>
        </div>
        </div>''',
        unsafe_allow_html=True,
    )

    # ---- KPI row ----
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Best RMSE", " / ".join(best_rmse)); k1.caption(f"↓ {metrics[best_rmse[0]]['RMSE']:.4f}")
    k2.metric("Best R²", " / ".join(best_r2));     k2.caption(f"↑ {metrics[best_r2[0]]['R2']:.4f}")
    k3.metric("Best MAPE", " / ".join(best_mape)); k3.caption(f"↓ {metrics[best_mape[0]]['MAPE']:.2f}%")
    k4.metric("Best Directional Acc.", " / ".join(best_da)); k4.caption(f"↑ {metrics[best_da[0]]['DA']:.2f}%")

    st.write("".join(model_pill(n) for n in shown_names), unsafe_allow_html=True)
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📖 Overview", "🏆 Model Showdown", "🔬 Diagnostics", "🧬 Feature Insights",
    ])

    # ==================================================================
    # TAB 1 — OVERVIEW
    # ==================================================================
    with tab1:
        eyebrow_title("DATASET", "Snapshot")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Train rows (post-warmup)", f"{len(train_data):,}")
        c2.metric("→ Train / Val split", f"{len(train_df)} / {len(val_df)}")
        c3.metric("Test rows", f"{len(test_data):,}")
        c4.metric("Lookback window", f"{SEQ_LEN} days")

        full_close = pd.concat([train_data["Close"], test_data["Close"]], ignore_index=True)
        chart_df = pd.DataFrame({
            "Close": full_close,
            "Split": ["Train"] * len(train_df) + ["Validation"] * len(val_df) + ["Test"] * len(test_data),
        })
        st.line_chart(chart_df["Close"], height=180)
        st.caption(
            f"Full Close-price series — first {len(train_df)} rows train, next "
            f"{len(val_df)} validation, final {len(test_data)} held-out test."
        )

        st.divider()
        eyebrow_title("OVERVIEW", "Actual vs Predicted (test set)")
        selected = st.multiselect(
            "Models to plot", options=shown_names, default=shown_names,
            key="overview_model_select", label_visibility="collapsed",
        )
        if not selected:
            st.warning("Select at least one model to plot.")
            selected = shown_names
        st.pyplot(fig_actual_vs_predicted(real_close, preds, selected))

        shown_m = {n: metrics[n] for n in selected}
        tightest = min(shown_m, key=lambda n: shown_m[n]["RMSE"])
        loosest  = max(shown_m, key=lambda n: shown_m[n]["RMSE"])
        if len(selected) > 1:
            insight(
                f"<b>{tightest}</b> (RMSE {metrics[tightest]['RMSE']:.2f}) hugs the actual "
                f"price line most closely among the models shown. <b>{loosest}</b> "
                f"(RMSE {metrics[loosest]['RMSE']:.2f}) diverges the most — check the "
                f"residual plots in <i>Diagnostics</i> for where it goes wrong."
            )
        else:
            insight(f"<b>{tightest}</b> alone — RMSE {metrics[tightest]['RMSE']:.2f}, "
                    f"R²={metrics[tightest]['R2']:.3f}.")

        st.divider()
        eyebrow_title("SCORECARD", "Regression Metrics — Test Set")
        metrics_df = pd.DataFrame({n: metrics[n] for n in shown_names}).T
        metrics_df.index.name = "Model"

        def highlight_best(df):
            styles = pd.DataFrame("", index=df.index, columns=df.columns)
            for col in df.columns:
                best = df[col].min() if col in LOWER_IS_BETTER else df[col].max()
                styles.loc[np.isclose(df[col], best), col] = (
                    "background-color: #d4edda; color: #155724; font-weight: bold"
                )
            return styles

        st.dataframe(
            metrics_df.style.apply(highlight_best, axis=None).format({
                "RMSE": "{:.4f}", "MAE": "{:.4f}", "MAPE": "{:.2f}%",
                "R2": "{:.4f}", "EVS": "{:.4f}", "DA": "{:.2f}%",
            }),
            width="stretch", height=min(280, 45 * (len(shown_names) + 1)),
        )
        caveat(
            "Directional Accuracy (DA) near 50% is expected for daily price moves. "
            "Judge these models on RMSE/MAE/R² for level-tracking quality, not DA."
        )

        if weights:
            st.divider()
            eyebrow_title("ENSEMBLE", "Weights (inverse validation-MAE)")
            wdf = pd.DataFrame({"Validation MAE": val_mae, "Weight": weights}).sort_values("Weight", ascending=False)
            st.dataframe(
                wdf.style.format({"Validation MAE": "{:.4f}", "Weight": "{:.4f}"})
                    .bar(subset=["Weight"], color="#4472C4"),
                width="stretch",
            )
            top_w = wdf.index[0]
            worst_w = wdf.index[-1]
            insight(
                f"<b>{top_w}</b> — the most accurate on validation — dominates the blend "
                f"at {wdf.loc[top_w, 'Weight']:.1%}. <b>{worst_w}</b> contributes the least "
                f"({wdf.loc[worst_w, 'Weight']:.1%}) but still pulls the average toward its "
                f"weaker predictions, which is part of why the Ensemble doesn't automatically "
                f"beat the single best base model on this run."
            )

    # ==================================================================
    # TAB 2 — MODEL SHOWDOWN
    # ==================================================================
    with tab2:
        eyebrow_title("HEAD TO HEAD", "Metric Comparison — All Models")
        metric_keys = ["RMSE", "MAE", "MAPE", "R2", "EVS", "DA"]
        cols = st.columns(3)
        for i, m in enumerate(metric_keys):
            with cols[i % 3]:
                st.pyplot(fig_metric_bar(m, metrics, shown_names))
        insight(
            f"<b>{' / '.join(best_rmse)}</b> leads the error-based metrics (RMSE, MAE, "
            f"MAPE), while Directional Accuracy stays flat across every model — model "
            f"choice changes <i>how precisely</i> you track price, not <i>whether</i> you "
            f"can call its direction."
        )

        st.divider()
        eyebrow_title("FIT QUALITY", "Predicted vs Actual — Test Set")
        st.caption("Points hugging the red diagonal = accurate predictions.")
        render_grid(shown_names, lambda n: fig_pred_vs_actual_scatter(n, real_close, preds[n], metrics[n]["R2"]))
        insight(
            f"The tightest cluster around the diagonal belongs to <b>{' / '.join(best_r2)}</b> "
            f"(R²={metrics[best_r2[0]]['R2']:.3f}). Any cloud bending away from the line at "
            f"price extremes is over/under-shooting during large moves."
        )

    # ==================================================================
    # TAB 3 — DIAGNOSTICS
    # ==================================================================
    with tab3:
        eyebrow_title("TRAINING BEHAVIOR", "Loss Curves — BiLSTM & GRU")
        nn_histories = {n: histories[n] for n in ("BiLSTM", "GRU") if n in histories}
        if nn_histories:
            cols = st.columns(2)
            overfit_notes = []
            for i, (name, hist) in enumerate(nn_histories.items()):
                with cols[i % 2]:
                    st.pyplot(fig_loss_curve(name, hist))
                final_train, final_val = hist["loss"][-1], hist["val_loss"][-1]
                gap_pct = (final_val - final_train) / max(final_train, 1e-9) * 100
                overfit_notes.append((name, gap_pct, len(hist["loss"])))
            widest = max(overfit_notes, key=lambda t: abs(t[1]))
            insight(
                f"<b>{widest[0]}</b> shows the widest train/validation loss gap "
                f"({widest[1]:+.0f}%) after {widest[2]} epochs. A modest positive gap is "
                f"normal; a large one signals memorization rather than generalization."
            )
        else:
            st.info(
                "Loss history isn't available for the currently loaded models — it isn't "
                "saved by a plain `model.save()`. Use **Train All 5 Models Now** in the "
                "sidebar to generate fresh curves for this session (and future ones)."
            )

        st.divider()
        eyebrow_title("ERROR BEHAVIOR", "Residual Distributions")
        st.caption("Centered on zero with no skew = the model isn't systematically biased high or low.")
        render_grid(shown_names, lambda n: fig_residual_hist(n, real_close, preds[n]))
        bias = {n: float(np.mean(real_close - preds[n])) for n in shown_names}
        most_biased = max(bias, key=lambda n: abs(bias[n]))
        direction = "under" if bias[most_biased] > 0 else "over"
        insight(
            f"<b>{most_biased}</b> shows the strongest systematic bias — its residuals "
            f"average {bias[most_biased]:+.2f}, meaning it tends to <b>{direction}predict</b> "
            f"on net rather than erring randomly in both directions."
        )

        st.divider()
        eyebrow_title("HETEROSKEDASTICITY CHECK", "Residual Scatter Plots")
        st.caption("A random horizontal band = well-behaved errors. A funnel shape = error grows with price level.")
        render_grid(shown_names, lambda n: fig_residual_scatter(n, real_close, preds[n]))
        insight(
            "Watch for a funnel widening at higher predicted prices — that pattern "
            "(heteroskedasticity) means error scales with price level, so a flat MAE "
            "understates risk during high-price periods."
        )

    # ==================================================================
    # TAB 4 — FEATURE INSIGHTS
    # ==================================================================
    with tab4:
        eyebrow_title("FEATURE RELATIONSHIPS", "Correlation Heatmap (full training set)")
        corr_full = train_data[FEATURES].corr()
        st.pyplot(fig_correlation_heatmap(corr_full, "Feature Correlation Matrix", "RdYlGn"))

        best_pair, best_val = None, 0.0
        for i, f1 in enumerate(FEATURES):
            for f2 in FEATURES[i + 1:]:
                v = abs(corr_full.loc[f1, f2])
                if v > best_val:
                    best_val, best_pair = v, (f1, f2)
        insight(
            f"The strongest relationship outside a feature and itself is "
            f"<b>{best_pair[0]} ↔ {best_pair[1]}</b> at {best_val:.2f}. The neural nets "
            f"train on a trimmed 5-feature set (<code>{', '.join(NN_FEATURES)}</code>) "
            f"specifically to reduce redundancy like this before it reaches a sequence model."
        )
        with st.expander("Pre-validation-split EDA heatmap (train_df only)"):
            corr_eda = train_df[FEATURES].corr()
            st.pyplot(fig_correlation_heatmap(corr_eda, "Feature Correlation Matrix (train only)", "coolwarm", mask_upper=True))

        if "XGBoost" in models:
            st.divider()
            eyebrow_title("MODEL ATTENTION", "XGBoost Feature Importance")
            importances = models["XGBoost"].feature_importances_
            agg = np.zeros(N_FEAT)
            for idx, imp in enumerate(importances):
                agg[idx % N_FEAT] += imp
            st.pyplot(fig_barh_importance(
                FEATURES, agg, "steelblue",
                "Aggregated Importance (summed over all timesteps)", "XGBoost Feature Importance",
            ))
            top_feat = FEATURES[int(np.argmax(agg))]
            insight(
                f"<b>{top_feat}</b> dominates XGBoost's split decisions across the "
                f"{SEQ_LEN}-day window. Since XGBoost predicts <i>returns</i> rather than "
                f"raw price, this is the input that most consistently helps forecast the "
                f"next percentage move — not just what correlates with price level."
            )

            st.divider()
            eyebrow_title("MODEL ATTENTION", "SHAP Feature Importance (XGBoost)")
            with st.spinner("Computing SHAP values…"):
                shap_values, X_shap = compute_shap(models["XGBoost"], seq["X_train_flat"])
            shap_feature_names = [f"{FEATURES[f]}_t-{SEQ_LEN - t}" for t in range(SEQ_LEN) for f in range(N_FEAT)]

            fig, _ = plt.subplots(figsize=(9, 7))
            shap.summary_plot(shap_values, X_shap, feature_names=shap_feature_names, max_display=20, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            shap_agg = np.zeros(N_FEAT)
            for f in range(N_FEAT):
                cols_idx = np.arange(f, X_shap.shape[1], N_FEAT)
                shap_agg[f] = np.mean(np.abs(shap_values[:, cols_idx]))
            st.pyplot(fig_barh_importance(
                FEATURES, shap_agg, "purple", "Mean Absolute SHAP Value",
                "SHAP Feature Importance (Aggregated)",
            ))
            top_shap = FEATURES[int(np.argmax(shap_agg))]
            agree = "agrees with" if top_shap == top_feat else "differs from"
            insight(
                f"SHAP's top driver, <b>{top_shap}</b>, {agree} the raw split-count "
                f"ranking. SHAP accounts for feature interactions and direction of "
                f"effect, not just split frequency — it's the better citation for "
                f"<i>why</i> the model predicts what it does."
            )
        else:
            st.info("XGBoost isn't loaded, so feature-importance and SHAP views aren't available.")

# ======================================================================
# MAIN
# ======================================================================
def main():
    render_sidebar_header()

    train_raw, test_raw, source_label = resolve_data_source()
    if train_raw is None:
        render_no_data_landing()
        return

    try:
        validate_required_columns(train_raw, "Training set")
        validate_required_columns(test_raw, "Test set")
        train_data, test_data, rows_dropped, chron_warning = engineer_features(train_raw, test_raw)
        train_df, val_df = split_train_val(train_data)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_df[FEATURES].values)
        seq = build_all_sequences(train_df, val_df, train_data, test_data, scaler)
    except Exception as e:
        render_data_error(e)
        return

    st.session_state.setdefault("model_version", 0)

    if st.session_state.get("_run_training"):
        with st.status("Running full training pipeline (mirrors STMP.ipynb)…", expanded=True) as status:
            train_all_models(seq, MODELS_DIR)
            load_models.clear()
            st.session_state["model_version"] += 1
            status.update(label="Training complete — models saved to models/.", state="complete")
        st.session_state["_run_training"] = False
        st.rerun()

    models, model_errors = load_models(str(MODELS_DIR), st.session_state["model_version"])
    histories = load_histories(MODELS_DIR)

    st.markdown('<div class="hero-title">📈 Stock Price Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Five architecturally different models, one dataset — '
        'a controlled test of what\'s actually learnable in daily price data.</div>',
        unsafe_allow_html=True,
    )

    if not models:
        render_no_models_state(model_errors, seq, MODELS_DIR)
        render_sidebar_footer(source_label, rows_dropped, chron_warning)
        return

    try:
        results = run_inference(models, seq, scaler)
        render_dashboard(train_data, val_df, train_df, val_df, test_data,
                          seq, models, model_errors, results, histories, MODELS_DIR)
    except Exception as e:
        st.error(f"Inference failed: {e}")
        with st.expander("Full traceback"):
            st.code(traceback.format_exc())

    render_sidebar_footer(source_label, rows_dropped, chron_warning)


if __name__ == "__main__":
    main()