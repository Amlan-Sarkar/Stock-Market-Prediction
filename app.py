# ==========================================================
# STOCK MARKET PREDICTION — STREAMLIT APP
# ==========================================================

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# ==========================================================
# HELPER FUNCTIONS
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
# SIDEBAR
# ==========================================================
with st.sidebar:
    st.markdown("## 📈 Stock Prediction")
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

# ==========================================================
# HEADER
# ==========================================================
st.markdown("# 📈 Stock Market Price Prediction")
st.markdown(
    "**Models:** BiLSTM · GRU · XGBoost · Linear Regression · "
    "Random Forest · Ensemble"
)
st.divider()

# ==========================================================
# TRAINING PIPELINE
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

    # ── KPI ROW ─────────────────────────────────────────────
    best_rmse_model = min(metrics, key=lambda n: metrics[n]['RMSE'])
    best_r2_model   = max(metrics, key=lambda n: metrics[n]['R2'])
    best_mape_model = min(metrics, key=lambda n: metrics[n]['MAPE'])
    best_da_model   = max(metrics, key=lambda n: metrics[n]['DA'])
    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Best RMSE (lower ↓ is better)", best_rmse_model,
          delta=-metrics[best_rmse_model]['RMSE'],
          delta_color="inverse")
    k2.metric("Best R² (higher ↑ is better)", best_r2_model,
          delta=metrics[best_r2_model]['R2'])
    k3.metric("Best MAPE (lower ↓ is better)", best_mape_model,
          delta=-metrics[best_mape_model]['MAPE'],
          delta_color="inverse")
    k4.metric("Best DA (higher ↑ is better)", best_da_model,
          delta=metrics[best_da_model]['DA'])

    st.divider()

    # ── TABS ────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "📈 Model Comparison",
        "🔍 Diagnostics",
        "🧬 Features"
    ])

    # ────────────────────────────────────────────────────────
    # TAB 1 — OVERVIEW
    # ────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Actual vs Predicted — All Models")
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(real_close, label='Actual', color='black',
                linewidth=2.5, zorder=5)
        for name, pred in preds.items():
            ax.plot(pred, label=name, alpha=0.85,
                    color=MODEL_COLORS[name], linewidth=1.5)
        ax.set_xlabel('Trading Days')
        ax.set_ylabel('Close Price')
        ax.set_title('Actual vs Predicted Stock Price', fontsize=14)
        ax.legend(fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.divider()
        st.subheader("Regression Metrics")

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

        st.divider()
        st.subheader("Ensemble Weights")
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

        st.caption(
            "ℹ️ Directional Accuracy near 50% is expected — "
            "price direction is a near-random walk. "
            "High R² reflects strong price-level autocorrelation, "
            "not directional predictability."
        )

    # ────────────────────────────────────────────────────────
    # TAB 2 — MODEL COMPARISON
    # ────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Metric Comparison — All Models")

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

        st.divider()
        st.subheader("Predicted vs Actual Scatter Plots")
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

    # ────────────────────────────────────────────────────────
    # TAB 3 — DIAGNOSTICS
    # ────────────────────────────────────────────────────────
    with tab3:
        st.subheader("Training & Validation Loss Curves")
        cols3 = st.columns(2)
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
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        st.divider()
        st.subheader("Residual Distributions")
        cols4 = st.columns(3)
        for idx, (name, pred) in enumerate(preds.items()):
            with cols4[idx % 3]:
                fig, ax = plt.subplots(figsize=(5, 4))
                residuals = real_close - pred
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

        st.divider()
        st.subheader("Residual Scatter Plots")
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

    # ────────────────────────────────────────────────────────
    # TAB 4 — FEATURES
    # ────────────────────────────────────────────────────────
    with tab4:
        st.subheader("Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(
            train_data[FEATURES].corr(),
            annot=True, fmt='.2f', cmap='RdYlGn',
            linewidths=0.5, square=True,
            annot_kws={'size': 10}, ax=ax
        )
        ax.set_title('Feature Correlation Matrix', fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.divider()
        st.subheader("XGBoost Feature Importance")
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

        st.divider()
        st.subheader("SHAP Feature Importance (XGBoost)")
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

# ==========================================================
# EMPTY STATE
# ==========================================================
else:
    st.info(
        "👆 Upload your training and test Excel files in the sidebar, "
        "then click **Train & Predict** to begin."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Models", "6",  "BiLSTM · GRU · XGBoost · LR · RF · Ensemble")
    c2.metric("Metrics", "6", "RMSE · MAE · MAPE · R² · EVS · DA")
    c3.metric("Visualizations", "10+", "Charts · Heatmaps · SHAP · Scatter")
