# 📈 Stock Market Price Prediction

A multi-model machine learning system for stock price prediction using historical OHLCV data and technical indicators.

---

## 🧠 Models
| Model | Type |
|---|---|
| Bidirectional LSTM | Deep Learning |
| GRU | Deep Learning |
| XGBoost | Machine Learning |
| Linear Regression | ML Baseline |
| Random Forest | Machine Learning |
| Weighted Ensemble | Combined |

---

## 📊 Features Used
- Raw OHLCV: Open, High, Low, Close, Adj Close, Volume
- Technical Indicators: RSI, MACD, MACD Signal, Bollinger Bands

---

## ⚙️ Methodology
- 80/20 train-validation split
- MinMax normalization on training data only
- 60-day look-back sequences
- Tree models trained on % return targets to handle extrapolation
- Neural nets trained on reduced 5-feature set to reduce overfitting
- Inverse-MAE weighted ensemble

---

## 📈 Results
| Model | RMSE | MAPE | R² |
|---|---|---|---|
| XGBoost | 18.55 | 1.27% | 0.85 |
| Random Forest | 19.11 | 1.37% | 0.84 |
| Ensemble | 21.00 | 1.57% | 0.80 |
| Linear Regression | 26.12 | 1.90% | 0.70 |
| GRU | 29.93 | 2.23% | 0.65 |
| BiLSTM | 44.70 | 3.37% | 0.13 |

---

## 🗂️ Project Structure
├── STMPfinalfr.ipynb       # Main notebook

├── app.py                  # Streamlit web app

├── Trainset.xlsx           # Training data

├── Testset.xlsx            # Test data

├── requirements.txt        # Dependencies

---

## 🚀 How to Run

### Notebook (Google Colab)
1. Upload `Trainset.xlsx` and `Testset.xlsx` to Colab
2. Run all cells in order

### Streamlit App (Local)
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📦 Dependencies
Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 🔍 Key Findings
- XGBoost and Random Forest outperform deep learning models on this dataset
- Directional accuracy (~53%) is near random (50%) — consistent with weak-form market efficiency
- High R² reflects price-level autocorrelation, not directional predictability
  
