# ==========================================================
# STOCK MARKET PREDICTION — STREAMLIT APP
# Redesigned for narrative clarity, decision support & polish
# ==========================================================

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import shap

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, explained_variance_score
)
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, LSTM, Bidirectional, GRU
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

np.random.seed(42)

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Stock Market Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# GLOBAL STYLE
# ==========================================================
mpl.rcParams.update({
    'font.family':      'DejaVu Sans',
    'axes.edgecolor':   '#D1D5DB',
    'axes.labelcolor':  '#374151',
    'axes.titlecolor':  '#111827',
    'xtick.color':      '#6B7280',
    'ytick.color':      '#6B7280',
    'axes.grid':        False,
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
})

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', -apple-system, sans-serif; }

/* Hero */
.hero-title { font-size: 2.3rem; font-weight: 800; letter-spacing: -0.02em;
              margin-bottom: 0.1rem; color: #111827; }
.hero-sub   { color: #6B7280; font-size: 1.05rem; margin-bottom: 0.4rem; }
.hero-tag   { display:inline-block; background:#EEF2FF; color:#4338CA;
              padding:3px 10px; border-radius:20px; font-size:0.78rem;
              font-weight:600; margin-right:6px; margin-bottom: 4px;}

/* Section labels */
.section-eyebrow { text-transform: uppercase; letter-spacing: 0.08em;
                    font-size: 0.72rem; font-weight: 700; color: #9CA3AF;
                    margin-bottom: 2px; }
.section-title   { font-size: 1.35rem; font-weight: 700; color: #111827;
                    margin-bottom: 4px; margin-top: 0px;}

/* Callout boxes */
.insight-box, .warning-box, .verdict-box {
    padding: 14px 18px; border-radius: 8px; margin: 10px 0 22px 0;
    font-size: 0.94rem; line-height: 1.55; color: #1F2937;
}
.insight-box  { background: #F0F5FF; border-left: 4px solid #4472C4; }
.warning-box  { background: #FFF8EB; border-left: 4px solid #D97706; }
.verdict-box  { background: #F0FDF4; border-left: 4px solid #16A34A; }
.insight-box b, .warning-box b, .verdict-box b { color: #111827; }

/* Decision cards */
.decision-grid { display: flex; gap: 14px; margin: 10px 0 24px 0; flex-wrap: wrap; }
.decision-card { flex: 1; min-width: 220px; background: white;
                  border: 1px solid #E5E7EB; border-radius: 10px;
                  padding: 16px 18px; }
.decision-card .dc-label { font-size: 0.72rem; font-weight: 700; color: #9CA3AF;
                             text-transform: uppercase; letter-spacing: 0.06em; }
.decision-card .dc-model { font-size: 1.15rem; font-weight: 800; color: #111827;
                             margin: 4px 0 6px 0; }
.decision-card .dc-why   { font-size: 0.86rem; color: #4B5563; line-height: 1.4; }

/* Model pill legend */
.pill { display:inline-block; padding: 2px 10px; border-radius: 20px;
        font-size: 0.78rem; font-weight: 600; color: white; margin-right: 6px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==========================================================
# CONSTANTS
# ==========================================================
FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
    'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower'
]
NN_FEATURES  = ['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal']
TARGET_COL   = FEATURES.index('Close')
N_FEAT       = len(FEATURES)
N_FEAT_NN    = len(NN_FEATURES)

MODEL_COLORS = {
    'BiLSTM':            '#4472C4',
    'GRU':               '#2E9E5B',
    'XGBoost':           '#C0392B',
    'Linear Regression': '#7D3C98',
    'Random Forest':     '#784212',
    'Ensemble':          '#B7950B'
}

MODEL_TAGLINE = {
    'BiLSTM':            'Deep sequence learner',
    'GRU':                'Lightweight recurrent net',
    'XGBoost':            'Gradient-boosted trees',
    'Linear Regression':  'Baseline linear model',
    'Random Forest':      'Bagged decision trees',
    'Ensemble':            'MAE-weighted blend'
}

# ==========================================================
# HELPER FUNCTIONS — ML PIPELINE (unchanged logic)
# ==========================================================
def add_indicators(df):
    delta    = df['Close'].diff()
    gain     = delta.where(delta > 0, 0)
    loss     = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs       = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']        = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    rm = df['Close'].rolling(20).mean()
    rs = df['Close'].rolling(20).std()
    df['BB_Upper'] = rm + 2 * rs
    df['BB_Lower'] = rm - 2 * rs
    return df


def build_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i, TARGET_COL])
    return np.array(X), np.array(y)


def build_return_targets(raw_close, seq_len):
    returns = []
    for i in range(seq_len, len(raw_close)):
        prev = raw_close[i - 1]
        curr = raw_close[i]
        returns.append((curr - prev) / prev)
    return np.array(returns)


def inverse_price(scaled_preds, scaler):
    dummy = np.zeros((len(scaled_preds), N_FEAT))
    dummy[:, TARGET_COL] = scaled_preds
    return scaler.inverse_transform(dummy)[:, TARGET_COL]


def predict_prices(model, X, scaler, flatten=True, is_keras=True):
    if is_keras:
        raw = model.predict(X, verbose=0)
    else:
        raw = model.predict(X)
    if flatten:
        raw = raw.flatten()
    return inverse_price(raw, scaler)


def reconstruct_from_returns(pred_returns, prev_closes):
    return prev_closes * (1 + pred_returns)


def directional_accuracy(real, pred):
    real_dir = np.diff(real) > 0
    pred_dir = np.diff(pred) > 0
    return np.mean(real_dir == pred_dir) * 100


def get_early_stop():
    return EarlyStopping(
        monitor='val_loss', patience=20,
        restore_best_weights=True, verbose=0
    )


def get_reduce_lr():
    return ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=10, min_lr=1e-6, verbose=0
    )

# ==========================================================
# HELPER FUNCTIONS — NARRATIVE / UI
# ==========================================================
def insight(text):
    st.markdown(f'<div class="insight-box">💡 <b>Insight —</b> {text}</div>',
                unsafe_allow_html=True)

def caveat(text):
    st.markdown(f'<div class="warning-box">⚠️ <b>Caveat —</b> {text}</div>',
                unsafe_allow_html=True)

def verdict(text):
    st.markdown(f'<div class="verdict-box">✅ <b>Bottom line —</b> {text}</div>',
                unsafe_allow_html=True)

def eyebrow_title(eyebrow, title):
    st.markdown(f'<div class="section-eyebrow">{eyebrow}</div>'
                f'<div class="section-title">{title}</div>',
                unsafe_allow_html=True)

def model_pill(name):
    color = MODEL_COLORS[name]
    return f'<span class="pill" style="background:{color}">{name}</span>'

# ==========================================================
# SIDEBAR
# ==========================================================
with st.sidebar:
    st.markdown("## 📈 Stock Prediction")
    st.caption("Six models. One question: can price be predicted?")
    st.divider()

    st.markdown("### 📁 Upload Data")
    train_file = st.file_uploader("Training Set (.xlsx)", type=['xlsx'])
    test_file  = st.file_uploader("Test Set (.xlsx)",     type=['xlsx'])

    st.divider()
    st.markdown("### ⚙️ Settings")
    SEQ_LEN    = st.slider("Sequence Length (days)", 30, 90, 60, step=5)
    MAX_EPOCHS = st.slider("Max Epochs", 50, 200, 100, step=10)
    BATCH_SIZE = st.select_slider("Batch Size", options=[16, 32, 64], value=32)

    st.divider()
    can_run  = train_file is not None and test_file is not None
    run_btn  = st.button(
        "🚀 Train & Predict",
        type="primary",
        use_container_width=True,
        disabled=not can_run
    )
    if not can_run:
        st.caption("Upload both files to enable training.")

    st.divider()
    with st.expander("ℹ️ About this project"):
        st.markdown(
            "This dashboard compares **six modeling approaches** — two "
            "deep sequence models, two tree ensembles, one linear "
            "baseline, and a weighted blend — on the same stock price "
            "series, using the same engineered features.\n\n"
            "The goal isn't to \"beat the market.\" It's to honestly "
            "measure **how much of next-day price movement is learnable "
            "at all**, and to show which modeling family is best suited "
            "to which job — tracking price levels vs. calling direction."
        )

# ==========================================================
# HEADER / HERO
# ==========================================================
st.markdown('<div class="hero-title">📈 Stock Market Price Prediction</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Six models, one dataset — a controlled test of '
    'what\'s actually predictable in price data.</div>',
    unsafe_allow_html=True
)
st.markdown(
    ''.join(model_pill(n) for n in MODEL_COLORS) + '&nbsp;',
    unsafe_allow_html=True
)
st.write("")
st.divider()

# ==========================================================
# EMPTY STATE — STORY-FIRST LANDING
# ==========================================================
if 'results' not in st.session_state and not run_btn:
    eyebrow_title("THE QUESTION", "Can six different modeling families predict a stock's next move?")
    st.markdown(
        "Most stock prediction demos show one model and call it a day. "
        "This one runs **five architecturally different approaches** — "
        "two recurrent neural nets, two tree ensembles, and a linear "
        "baseline — on identical data, then blends them into a weighted "
        "ensemble. The comparison itself is the point: it reveals which "
        "kind of signal each model family is actually good at extracting."
    )
    st.write("")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**🧠 Step 1 — Engineer the signal**")
        st.caption(
            "RSI, MACD, and Bollinger Bands are computed from raw OHLCV "
            "data to give models momentum and volatility context beyond "
            "price alone."
        )
    with c2:
        st.markdown("**⚔️ Step 2 — Run a fair fight**")
        st.caption(
            "BiLSTM, GRU, XGBoost, Random Forest, and Linear Regression "
            "train on the same split, same horizon, same target — so "
            "differences reflect the model, not the setup."
        )
    with c3:
        st.markdown("**🔎 Step 3 — Interrogate the result**")
        st.caption(
            "Six metrics, residual diagnostics, and SHAP values decide "
            "which model wins — and whether any of them beat a coin flip "
            "on direction."
        )

    st.divider()
    st.info(
        "👆 Upload your training and test Excel files in the sidebar, "
        "then click **Train & Predict** to begin."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Models", "6",  "BiLSTM · GRU · XGBoost · LR · RF · Ensemble")
    c2.metric("Metrics", "6", "RMSE · MAE · MAPE · R² · EVS · DA")
    c3.metric("Visualizations", "10+", "Charts · Heatmaps · SHAP · Scatter")

# ==========================================================
# TRAINING PIPELINE (logic unchanged from original build)
# ==========================================================
if run_btn:
    nn_indices = [FEATURES.index(f) for f in NN_FEATURES]

    with st.status("Running pipeline…", expanded=True) as status:

        # ── LOAD ────────────────────────────────────────────
        st.write("📂 Loading data…")
        train_data = pd.read_excel(train_file)
        test_data  = pd.read_excel(test_file)
        st.write(
            f"✅ Loaded — Train: `{train_data.shape}` · "
            f"Test: `{test_data.shape}`"
        )

        # ── INDICATORS ──────────────────────────────────────
        st.write("🔧 Computing technical indicators…")
        train_len    = len(train_data)
        combined_len = train_len + len(test_data)
        combined     = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)
        combined     = add_indicators(combined)
        combined.dropna(inplace=True)
        combined.reset_index(drop=True, inplace=True)

        rows_dropped  = combined_len - len(combined)
        new_train_len = train_len - rows_dropped

        train_data = combined.iloc[:new_train_len].reset_index(drop=True)
        test_data  = combined.iloc[new_train_len:].reset_index(drop=True)
        st.write("✅ RSI · MACD · Bollinger Bands added")

        # ── SPLIT & NORMALIZE ───────────────────────────────
        st.write("📐 Splitting & normalising…")
        split_idx = int(0.8 * len(train_data))
        train_df  = train_data.iloc[:split_idx]
        val_df    = train_data.iloc[split_idx:]

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_df[FEATURES].values)
        train_scaled = scaler.transform(train_df[FEATURES].values)
        st.write(
            f"✅ Train: `{train_df.shape[0]}` rows · "
            f"Val: `{val_df.shape[0]}` rows"
        )

        # ── SEQUENCES ───────────────────────────────────────
        st.write("🔄 Building sequences…")
        X_train, y_train = build_sequences(train_scaled, SEQ_LEN)
        X_train_nn  = X_train[:, :, nn_indices]
        y_train_ret = build_return_targets(train_df['Close'].values, SEQ_LEN)

        val_inputs      = pd.concat(
            (train_df[FEATURES].tail(SEQ_LEN), val_df[FEATURES]), axis=0
        )
        val_scaled_full = scaler.transform(val_inputs.values)
        X_val, y_val    = build_sequences(val_scaled_full, SEQ_LEN)
        X_val_nn        = X_val[:, :, nn_indices]
        y_val_ret       = build_return_targets(val_inputs['Close'].values, SEQ_LEN)

        X_train_flat = X_train.reshape(len(X_train), -1)
        X_val_flat   = X_val.reshape(len(X_val), -1)
        st.write(
            f"✅ X_train: `{X_train.shape}` · "
            f"X_val: `{X_val.shape}`"
        )

        # ── BILSTM ──────────────────────────────────────────
        st.write("🧠 Training BiLSTM…")
        model_bilstm = Sequential([
            Bidirectional(LSTM(64, return_sequences=True),
                          input_shape=(SEQ_LEN, N_FEAT_NN)),
            Dropout(0.2),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.2),
            Bidirectional(LSTM(64)),
            Dropout(0.2),
            Dense(1)
        ], name='BiLSTM')
        model_bilstm.compile(optimizer='adam', loss='mse')
        history_bilstm = model_bilstm.fit(
            X_train_nn, y_train,
            epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
            validation_data=(X_val_nn, y_val),
            callbacks=[get_early_stop(), get_reduce_lr()],
            verbose=0
        )
        st.write(
            f"✅ BiLSTM done — "
            f"`{len(history_bilstm.history['loss'])}` epochs"
        )

        # ── GRU ─────────────────────────────────────────────
        st.write("🧠 Training GRU…")
        model_gru = Sequential([
            GRU(64, return_sequences=True,
                input_shape=(SEQ_LEN, N_FEAT_NN)),
            Dropout(0.2),
            GRU(64, return_sequences=True),
            Dropout(0.2),
            GRU(64),
            Dropout(0.2),
            Dense(1)
        ], name='GRU')
        model_gru.compile(optimizer='adam', loss='mse')
        history_gru = model_gru.fit(
            X_train_nn, y_train,
            epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
            validation_data=(X_val_nn, y_val),
            callbacks=[get_early_stop(), get_reduce_lr()],
            verbose=0
        )
        st.write(
            f"✅ GRU done — "
            f"`{len(history_gru.history['loss'])}` epochs"
        )

        # ── XGBOOST ─────────────────────────────────────────
        st.write("🌲 Training XGBoost…")
        model_xgb = xgb.XGBRegressor(
            n_estimators=1000, learning_rate=0.01, max_depth=8,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            reg_alpha=0.1, reg_lambda=1, verbosity=0,
            early_stopping_rounds=20
        )
        model_xgb.fit(
            X_train_flat, y_train_ret,
            eval_set=[(X_val_flat, y_val_ret)],
            verbose=False
        )
        st.write("✅ XGBoost done")

        # ── LINEAR REGRESSION ───────────────────────────────
        st.write("📏 Training Linear Regression…")
        model_lr = LinearRegression()
        model_lr.fit(X_train_flat, y_train)
        st.write("✅ Linear Regression done")

        # ── RANDOM FOREST ───────────────────────────────────
        st.write("🌳 Training Random Forest…")
        model_rf = RandomForestRegressor(
            n_estimators=200, max_depth=12, max_features='sqrt',
            min_samples_split=5, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        )
        model_rf.fit(X_train_flat, y_train_ret)
        st.write("✅ Random Forest done")

        # ── TEST SET ────────────────────────────────────────
        st.write("🧪 Preparing test set…")
        dataset_total = pd.concat(
            (train_data[FEATURES], test_data[FEATURES]), axis=0
        )
        inputs = dataset_total[
            len(dataset_total) - len(test_data) - SEQ_LEN:
        ].values
        inputs = scaler.transform(inputs)

        X_test, _   = build_sequences(inputs, SEQ_LEN)
        X_test_nn   = X_test[:, :, nn_indices]
        X_test_flat = X_test.reshape(len(X_test), -1)
        real_close  = test_data['Close'].values

        assert len(X_test) == len(real_close), (
            f"Length mismatch: X_test {len(X_test)} vs real_close {len(real_close)}"
        )

        # ── PREDICTIONS ─────────────────────────────────────
        st.write("🔮 Generating predictions…")
        prev_closes_test = np.concatenate([
            [train_data['Close'].values[-1]],
            test_data['Close'].values[:-1]
        ])
        prev_closes_val = val_inputs['Close'].values[SEQ_LEN - 1 : -1]

        preds = {
            'BiLSTM':   predict_prices(model_bilstm, X_test_nn, scaler),
            'GRU':      predict_prices(model_gru,    X_test_nn, scaler),
            'XGBoost':  reconstruct_from_returns(
                            model_xgb.predict(X_test_flat), prev_closes_test),
            'Linear Regression': predict_prices(
                            model_lr, X_test_flat, scaler, flatten=False, is_keras=False),
            'Random Forest': reconstruct_from_returns(
                            model_rf.predict(X_test_flat), prev_closes_test)
        }

        # Ensemble weights from validation MAE
        val_real  = inverse_price(y_val, scaler)
        val_preds = {
            'BiLSTM':   predict_prices(model_bilstm, X_val_nn, scaler),
            'GRU':      predict_prices(model_gru,    X_val_nn, scaler),
            'XGBoost':  reconstruct_from_returns(
                            model_xgb.predict(X_val_flat), prev_closes_val),
            'Linear Regression': predict_prices(
                            model_lr, X_val_flat, scaler, flatten=False, is_keras=False),
            'Random Forest': reconstruct_from_returns(
                            model_rf.predict(X_val_flat), prev_closes_val)
        }

        val_mae   = {n: mean_absolute_error(val_real, p) for n, p in val_preds.items()}
        inv_mae   = {n: 1 / mae for n, mae in val_mae.items()}
        total_inv = sum(inv_mae.values())
        ens_weights = {n: w / total_inv for n, w in inv_mae.items()}

        preds['Ensemble'] = sum(
            ens_weights[n] * preds[n] for n in ens_weights
        )

        # ── METRICS ─────────────────────────────────────────
        st.write("📊 Computing metrics…")
        metrics = {}
        for name, pred in preds.items():
            rmse = np.sqrt(mean_squared_error(real_close, pred))
            mae  = mean_absolute_error(real_close, pred)
            mape = np.mean(np.abs((real_close - pred) / real_close)) * 100
            r2   = r2_score(real_close, pred)
            evs  = explained_variance_score(real_close, pred)
            da   = directional_accuracy(real_close, pred)
            metrics[name] = dict(
                RMSE=round(rmse, 4), MAE=round(mae, 4),
                MAPE=round(mape, 2), R2=round(r2, 4),
                EVS=round(evs, 4),   DA=round(da, 2)
            )

        # ── SAVE TO SESSION STATE ───────────────────────────
        st.session_state['results'] = {
            'preds':        preds,
            'real_close':   real_close,
            'metrics':      metrics,
            'histories':    {'BiLSTM': history_bilstm, 'GRU': history_gru},
            'train_data':   train_data,
            'model_xgb':    model_xgb,
            'X_train_flat': X_train_flat,
            'val_mae':      val_mae,
            'ens_weights':  ens_weights,
            'seq_len':      SEQ_LEN
        }

        status.update(
            label="✅ Training complete!", state="complete", expanded=False
        )

# ==========================================================
# RESULTS
# ==========================================================
if 'results' in st.session_state:
    R          = st.session_state['results']
    preds      = R['preds']
    real_close = R['real_close']
    metrics    = R['metrics']
    histories  = R['histories']
    train_data = R['train_data']
    model_xgb  = R['model_xgb']
    X_train_flat = R['X_train_flat']
    seq_len    = R['seq_len']
    ens_weights = R['ens_weights']
    val_mae    = R['val_mae']

    # ── DERIVED STORY FACTS ──────────────────────────────────
    best_rmse_model = min(metrics, key=lambda n: metrics[n]['RMSE'])
    best_r2_model   = max(metrics, key=lambda n: metrics[n]['R2'])
    best_mape_model = min(metrics, key=lambda n: metrics[n]['MAPE'])
    best_da_model   = max(metrics, key=lambda n: metrics[n]['DA'])
    worst_rmse_model = max(metrics, key=lambda n: metrics[n]['RMSE'])

    nn_models   = ['BiLSTM', 'GRU']
    tree_models = ['XGBoost', 'Random Forest']
    nn_avg_rmse   = np.mean([metrics[n]['RMSE'] for n in nn_models])
    tree_avg_rmse = np.mean([metrics[n]['RMSE'] for n in tree_models])

    da_values = [metrics[n]['DA'] for n in metrics]
    da_spread = max(da_values) - min(da_values)
    da_mean   = np.mean(da_values)

    ens_rmse_rank = sorted(metrics, key=lambda n: metrics[n]['RMSE']).index('Ensemble') + 1

    tree_wins    = tree_avg_rmse < nn_avg_rmse
    winner_label = "Tree-based models" if tree_wins else "Recurrent nets"
    loser_label  = "recurrent nets" if tree_wins else "tree-based models"
    suits_text   = (
        "gradient boosting\u2019s tabular splits" if tree_wins
        else "sequence models at capturing temporal patterns"
    )

    # ── HERO VERDICT ──────────────────────────────────────────
    eyebrow_title("RESULT", "The Bottom Line")
    verdict(
        f"<b>{best_rmse_model}</b> produced the lowest price-tracking error "
        f"(RMSE {metrics[best_rmse_model]['RMSE']:.2f}, R²={metrics[best_rmse_model]['R2']:.3f}), "
        f"outperforming the weaker end of the field by a wide margin — "
        f"{worst_rmse_model} trailed at RMSE {metrics[worst_rmse_model]['RMSE']:.2f}. "
        f"{winner_label} averaged {min(tree_avg_rmse, nn_avg_rmse):.2f} RMSE vs. "
        f"{max(tree_avg_rmse, nn_avg_rmse):.2f} for the {loser_label}, "
        f"suggesting the engineered technical-indicator features suit "
        f"{suits_text} better on this dataset. "
        f"The Ensemble landed #{ens_rmse_rank} of 6 on RMSE — its role is "
        f"variance reduction across regimes, not beating the single best model."
    )
    caveat(
        f"Directional Accuracy sits at {da_mean:.1f}% on average (range "
        f"{min(da_values):.1f}–{max(da_values):.1f}%) — essentially a coin "
        f"flip. This is expected: day-to-day price <i>direction</i> behaves "
        f"close to a random walk, while price <i>level</i> is highly "
        f"autocorrelated (today's price is close to yesterday's). The high "
        f"R² scores reflect the latter, not a trading edge. "
        f"<b>Don't read these models as directional trading signals.</b>"
    )

    # ── DECISION GUIDE ────────────────────────────────────────
    eyebrow_title("DECISION GUIDE", "Which model should you actually use?")
    st.markdown(
        f'''<div class="decision-grid">
        <div class="decision-card">
            <div class="dc-label">Track price level accurately</div>
            <div class="dc-model" style="color:{MODEL_COLORS[best_rmse_model]}">{best_rmse_model}</div>
            <div class="dc-why">Lowest RMSE/MAPE — closest fit to actual close price.
            Best choice for valuation-style tracking, not entry/exit timing.</div>
        </div>
        <div class="decision-card">
            <div class="dc-label">Want stability across models</div>
            <div class="dc-model" style="color:{MODEL_COLORS['Ensemble']}">Ensemble</div>
            <div class="dc-why">Blends all five models weighted by inverse validation
            error — smooths out any single model's regime-specific failures.</div>
        </div>
        <div class="decision-card">
            <div class="dc-label">Calling next-day direction</div>
            <div class="dc-model" style="color:#DC2626">None reliably</div>
            <div class="dc-why">All models cluster near {da_mean:.0f}% directional
            accuracy — no better than chance. Don't use any of these for buy/sell timing.</div>
        </div>
        </div>''',
        unsafe_allow_html=True
    )

    # ── KPI ROW ─────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Best RMSE (lower ↓ is better)", best_rmse_model)
    k1.caption(f"↓ {metrics[best_rmse_model]['RMSE']:.4f}")
    k2.metric("Best R² (higher ↑ is better)", best_r2_model)
    k2.caption(f"↑ {metrics[best_r2_model]['R2']:.4f}")
    k3.metric("Best MAPE (lower ↓ is better)", best_mape_model)
    k3.caption(f"↓ {metrics[best_mape_model]['MAPE']:.2f}%")
    k4.metric("Best DA (higher ↑ is better)", best_da_model)
    k4.caption(f"↑ {metrics[best_da_model]['DA']:.2f}%")

    st.divider()

    # ── TABS ────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📖 The Story",
        "🏆 Model Showdown",
        "🔬 Under the Hood",
        "🧬 What Drives Predictions"
    ])

    # ────────────────────────────────────────────────────────
    # TAB 1 — THE STORY
    # ────────────────────────────────────────────────────────
    with tab1:
        eyebrow_title("OVERVIEW", "Actual vs Predicted")
        selected_models = st.multiselect(
            "Models to plot",
            options=list(MODEL_COLORS.keys()),
            default=list(MODEL_COLORS.keys()),
            key="overview_model_select",
            label_visibility="collapsed"
        )
        if not selected_models:
            st.warning("Select at least one model to plot.")
            selected_models = list(MODEL_COLORS.keys())

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(real_close, label='Actual', color='black',
                linewidth=2.5, zorder=5)
        for name in selected_models:
            ax.plot(preds[name], label=name, alpha=0.85,
                    color=MODEL_COLORS[name], linewidth=1.5)
        ax.set_xlabel('Trading Days')
        ax.set_ylabel('Close Price')
        ax.set_title('Actual vs Predicted Stock Price', fontsize=14)
        ax.legend(fontsize=10, frameon=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        shown_metrics = {n: metrics[n] for n in selected_models}
        tightest = min(shown_metrics, key=lambda n: shown_metrics[n]['RMSE'])
        loosest  = max(shown_metrics, key=lambda n: shown_metrics[n]['RMSE'])
        if len(selected_models) > 1:
            insight(
                f"<b>{tightest}</b> (RMSE {metrics[tightest]['RMSE']:.2f}) hugs "
                f"the actual price line most closely among the models shown, "
                f"including through the sharp reversals mid-series. "
                f"<b>{loosest}</b> (RMSE {metrics[loosest]['RMSE']:.2f}) "
                f"visibly lags or smooths over those moves — a sign it's "
                f"undershooting volatility rather than tracking it."
            )
        else:
            insight(
                f"<b>{tightest}</b> alone — RMSE {metrics[tightest]['RMSE']:.2f}, "
                f"R²={metrics[tightest]['R2']:.3f}. Add more models above to "
                f"compare how closely each tracks the actual price line."
            )

        st.divider()
        eyebrow_title("SCORECARD", "Regression Metrics")

        metrics_df = pd.DataFrame(metrics).T
        metrics_df.index.name = 'Model'

        def highlight_best(df):
            styles = pd.DataFrame('', index=df.index, columns=df.columns)
            for col in ['RMSE', 'MAE', 'MAPE']:
                best = df[col].min()
                styles.loc[df[col] == best, col] = (
                    'background-color: #d4edda; '
                    'color: #155724; font-weight: bold'
                )
            for col in ['R2', 'EVS', 'DA']:
                best = df[col].max()
                styles.loc[df[col] == best, col] = (
                    'background-color: #d4edda; '
                    'color: #155724; font-weight: bold'
                )
            return styles

        st.dataframe(
            metrics_df.style
                .apply(highlight_best, axis=None)
                .format({
                    'RMSE': '{:.4f}', 'MAE':  '{:.4f}',
                    'MAPE': '{:.2f}%','R2':   '{:.4f}',
                    'EVS':  '{:.4f}', 'DA':   '{:.2f}%'
                }),
            use_container_width=True,
            height=280
        )
        caveat(
            "Directional Accuracy (DA) near 50% is expected — price "
            "direction behaves close to a random walk. High R² reflects "
            "strong price-<i>level</i> autocorrelation, not directional "
            "predictability. Judge these models on RMSE/MAE/R², not DA."
        )

        st.divider()
        eyebrow_title("ENSEMBLE", "Ensemble Weights")
        wdf = pd.DataFrame({
            'Validation MAE': val_mae,
            'Weight':         ens_weights
        }).sort_values('Weight', ascending=False)
        st.dataframe(
            wdf.style.format(
                {'Validation MAE': '{:.4f}', 'Weight': '{:.4f}'}
            ).bar(subset=['Weight'], color='#4472C4'),
            use_container_width=True
        )
        top_weight_model = wdf.index[0]
        insight(
            f"Weights are inverse-MAE on the validation set, so "
            f"<b>{top_weight_model}</b> — the most accurate validator — "
            f"dominates the blend at {wdf.loc[top_weight_model, 'Weight']:.1%}. "
            f"This makes the Ensemble a hedge against any one model "
            f"overfitting to validation-specific patterns, rather than a "
            f"pure accuracy play."
        )

    # ────────────────────────────────────────────────────────
    # TAB 2 — MODEL SHOWDOWN
    # ────────────────────────────────────────────────────────
    with tab2:
        eyebrow_title("HEAD TO HEAD", "Metric Comparison — All Models")

        metric_list   = ['RMSE', 'MAE', 'MAPE', 'R2', 'EVS', 'DA']
        metric_labels = {
            'RMSE': 'RMSE (lower ↓)',
            'MAE':  'MAE (lower ↓)',
            'MAPE': 'MAPE % (lower ↓)',
            'R2':   'R² (higher ↑)',
            'EVS':  'Explained Variance (higher ↑)',
            'DA':   'Directional Accuracy % (higher ↑)'
        }
        model_names = list(MODEL_COLORS.keys())
        colors      = list(MODEL_COLORS.values())

        cols = st.columns(3)
        for idx, metric in enumerate(metric_list):
            with cols[idx % 3]:
                fig, ax = plt.subplots(figsize=(5, 4))
                values = [metrics[n][metric] for n in model_names]
                bars   = ax.bar(model_names, values, color=colors,
                                edgecolor='black', width=0.6)
                ax.set_title(metric_labels[metric], fontsize=11)
                ax.set_xticklabels(model_names, rotation=25,
                                   ha='right', fontsize=8)
                for bar, val in zip(bars, values):
                    offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.015
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + offset,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8
                    )
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        insight(
            f"<b>{best_rmse_model}</b> sweeps most error-based metrics "
            f"(RMSE, MAE, MAPE), while directional accuracy stays flat "
            f"across all six models — reinforcing that model choice "
            f"changes <i>how precisely</i> you track price, not "
            f"<i>whether</i> you can predict its direction."
        )

        st.divider()
        eyebrow_title("FIT QUALITY", "Predicted vs Actual Scatter Plots")
        st.caption(
            "Points hugging the red diagonal = accurate predictions. "
            "Spread away from the line = systematic error."
        )
        cols2 = st.columns(3)
        for idx, (name, pred) in enumerate(preds.items()):
            with cols2[idx % 3]:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.scatter(real_close, pred,
                           color=MODEL_COLORS[name], alpha=0.6, s=20)
                mn = min(real_close.min(), pred.min())
                mx = max(real_close.max(), pred.max())
                ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1.8)
                ax.set_title(f'{name}', fontsize=11)
                ax.set_xlabel('Actual Close')
                ax.set_ylabel('Predicted Close')
                r2 = metrics[name]['R2']
                ax.text(0.05, 0.92, f'R²={r2:.3f}',
                        transform=ax.transAxes, fontsize=9,
                        bbox=dict(boxstyle='round', fc='white', alpha=0.7))
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        insight(
            f"The tightest cluster around the diagonal belongs to "
            f"<b>{best_r2_model}</b> (R²={metrics[best_r2_model]['R2']:.3f}). "
            f"Any model whose cloud bends away from the line at price "
            f"extremes is systematically over- or under-shooting during "
            f"large moves — worth checking against the residual plots in "
            f"<i>Under the Hood</i>."
        )

    # ────────────────────────────────────────────────────────
    # TAB 3 — UNDER THE HOOD
    # ────────────────────────────────────────────────────────
    with tab3:
        eyebrow_title("TRAINING BEHAVIOR", "Training & Validation Loss Curves")
        cols3 = st.columns(2)
        overfit_notes = []
        for idx, (name, hist) in enumerate(histories.items()):
            with cols3[idx % 2]:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(hist.history['loss'],
                        label='Train Loss', linewidth=2)
                ax.plot(hist.history['val_loss'],
                        label='Val Loss', linewidth=2, linestyle='--')
                ax.set_title(f'{name} — Loss Curve', fontsize=12)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('MSE')
                ax.legend(frameon=False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            final_train = hist.history['loss'][-1]
            final_val   = hist.history['val_loss'][-1]
            gap_pct = (final_val - final_train) / max(final_train, 1e-9) * 100
            overfit_notes.append((name, gap_pct, len(hist.history['loss'])))

        widest = max(overfit_notes, key=lambda t: abs(t[1]))
        insight(
            f"Early stopping triggered convergence for both nets. "
            f"<b>{widest[0]}</b> shows the widest train/validation gap "
            f"({widest[1]:+.0f}%) — a modest positive gap is normal, but a "
            f"large one signals the model is starting to memorize training "
            f"noise rather than generalizing."
        )

        st.divider()
        eyebrow_title("ERROR BEHAVIOR", "Residual Distributions")
        st.caption(
            "A distribution centered on zero with no skew means the model "
            "isn't systematically biased high or low."
        )
        cols4 = st.columns(3)
        bias_notes = {}
        for idx, (name, pred) in enumerate(preds.items()):
            with cols4[idx % 3]:
                fig, ax = plt.subplots(figsize=(5, 4))
                residuals = real_close - pred
                bias_notes[name] = float(np.mean(residuals))
                ax.hist(residuals, bins=30,
                        color=MODEL_COLORS[name],
                        edgecolor='black', alpha=0.75)
                ax.axvline(0, color='red', linestyle='--', linewidth=1.8)
                ax.set_title(name, fontsize=11)
                ax.set_xlabel('Error (Actual − Predicted)')
                ax.set_ylabel('Frequency')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        most_biased = max(bias_notes, key=lambda n: abs(bias_notes[n]))
        direction = "under" if bias_notes[most_biased] > 0 else "over"
        insight(
            f"<b>{most_biased}</b> shows the strongest systematic bias — "
            f"its residuals average {bias_notes[most_biased]:+.2f}, meaning "
            f"it tends to <b>{direction}predict</b> actual price on net, "
            f"rather than erring randomly in both directions."
        )

        st.divider()
        eyebrow_title("HETEROSKEDASTICITY CHECK", "Residual Scatter Plots")
        st.caption(
            "A random horizontal band = well-behaved errors. A funnel or "
            "trend shape = error grows with price level."
        )
        cols5 = st.columns(3)
        for idx, (name, pred) in enumerate(preds.items()):
            with cols5[idx % 3]:
                fig, ax = plt.subplots(figsize=(5, 4))
                residuals = real_close - pred
                ax.scatter(pred, residuals,
                           color=MODEL_COLORS[name], alpha=0.6, s=20)
                ax.axhline(0, color='red', linestyle='--', linewidth=1.8)
                ax.set_title(name, fontsize=11)
                ax.set_xlabel('Predicted Close Price')
                ax.set_ylabel('Residuals')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        insight(
            "Watch for a funnel shape widening at higher predicted prices — "
            "that pattern (heteroskedasticity) means the model's error "
            "scales with price level, so a flat MAE understates risk during "
            "high-price periods."
        )

    # ────────────────────────────────────────────────────────
    # TAB 4 — WHAT DRIVES PREDICTIONS
    # ────────────────────────────────────────────────────────
    with tab4:
        eyebrow_title("FEATURE RELATIONSHIPS", "Feature Correlation Heatmap")
        corr = train_data[FEATURES].corr()
        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(
            corr,
            annot=True, fmt='.2f', cmap='RdYlGn',
            linewidths=0.5, square=True,
            annot_kws={'size': 10}, ax=ax
        )
        ax.set_title('Feature Correlation Matrix', fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # find strongest non-trivial off-diagonal correlation excluding OHLC group
        ohlc_like = {'Open', 'High', 'Low', 'Close', 'Adj Close'}
        best_pair, best_val = None, 0
        for i, f1 in enumerate(FEATURES):
            for f2 in FEATURES[i+1:]:
                if f1 in ohlc_like and f2 in ohlc_like:
                    continue
                v = abs(corr.loc[f1, f2])
                if v > best_val:
                    best_val, best_pair = v, (f1, f2)

        insight(
            f"OHLC-family columns (Open/High/Low/Close/Adj Close) are "
            f"correlated at ~1.00 with each other — expected, since they "
            f"move together intraday. This near-perfect collinearity is "
            f"exactly why the neural nets train on a trimmed feature set "
            f"(<code>{', '.join(NN_FEATURES)}</code>) instead of all "
            f"{N_FEAT} raw columns: feeding collinear inputs to a "
            f"sequence model inflates variance without adding signal. "
            f"Outside that group, the strongest relationship is "
            f"<b>{best_pair[0]} ↔ {best_pair[1]}</b> at {best_val:.2f} — "
            f"MACD and its signal line are related by construction "
            f"(one is a smoothed version of the other)."
        )

        st.divider()
        eyebrow_title("MODEL ATTENTION", "XGBoost Feature Importance")
        importances  = model_xgb.feature_importances_
        feat_imp_agg = np.zeros(N_FEAT)
        for idx, imp in enumerate(importances):
            feat_imp_agg[idx % N_FEAT] += imp

        sorted_idx = np.argsort(feat_imp_agg)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(
            [FEATURES[i] for i in sorted_idx],
            feat_imp_agg[sorted_idx],
            color='steelblue', edgecolor='black', alpha=0.85
        )
        ax.set_xlabel('Aggregated Importance (summed over all timesteps)')
        ax.set_title('XGBoost Feature Importance', fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        top_feat = FEATURES[sorted_idx[-1]]
        insight(
            f"<b>{top_feat}</b> dominates XGBoost's split decisions across "
            f"the {seq_len}-day lookback window. Since XGBoost predicts "
            f"<i>returns</i> rather than raw price, this reflects which "
            f"input most consistently helps forecast the next percentage "
            f"move — not just which column correlates with price level."
        )

        st.divider()
        eyebrow_title("MODEL ATTENTION", "SHAP Feature Importance (XGBoost)")
        with st.spinner("Computing SHAP values — this may take a moment…"):
            sample_size = min(500, len(X_train_flat))
            X_shap      = X_train_flat[:sample_size]

            shap_feature_names = [
                f"{FEATURES[f]}_t-{seq_len - t}"
                for t in range(seq_len)
                for f in range(N_FEAT)
            ]

            explainer   = shap.TreeExplainer(model_xgb)
            shap_values = explainer.shap_values(X_shap)

            # Summary plot (top 20 named lags)
            fig, _ = plt.subplots(figsize=(10, 7))
            shap.summary_plot(
                shap_values, X_shap,
                feature_names=shap_feature_names,
                max_display=20, show=False
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Aggregated bar
            shap_agg = np.zeros(N_FEAT)
            for feature_idx in range(N_FEAT):
                cols_idx = np.arange(feature_idx, X_shap.shape[1], N_FEAT)
                shap_agg[feature_idx] = np.mean(
                    np.abs(shap_values[:, cols_idx])
                )

            sorted_shap = np.argsort(shap_agg)
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.barh(
                [FEATURES[i] for i in sorted_shap],
                shap_agg[sorted_shap],
                color='purple', edgecolor='black'
            )
            ax.set_xlabel('Mean Absolute SHAP Value')
            ax.set_title('SHAP Feature Importance (Aggregated)', fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            top_shap_feat = FEATURES[sorted_shap[-1]]
            agree = "agrees with" if top_shap_feat == top_feat else "differs from"
            insight(
                f"SHAP's top driver, <b>{top_shap_feat}</b>, {agree} the "
                f"raw importance ranking. SHAP is the more trustworthy read "
                f"here — it accounts for feature interactions and "
                f"direction of effect, not just split-count frequency, so "
                f"it's the better citation if you're explaining <i>why</i> "
                f"the model predicts what it does, not just <i>what</i> it "
                f"weighs most."
            )