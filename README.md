
# FALCON - Financial Analysis with Cross-asset Optimization Network

FALCON is a deep learning framework designed for multi-asset stock prediction and portfolio optimization. It leverages a hierarchical transformer architecture combining individual asset analysis, cross-asset attention, and market regime detection to build robust investment strategies.

---

## ✨ Key Features

- **Hierarchical Transformer Model**: Three-layer architecture with asset-specific encoders, cross-asset attention, and portfolio optimization agent.
- **Market Regime Detection**: Integrated module to detect bullish, bearish, and volatile market states.
- **Multi-Objective Training**: Balances return prediction, asset scoring, portfolio allocation, and regime detection.
- **Comprehensive Backtesting**: Simulates weekly rebalancing and tracks portfolio performance over time.

---

## 🏗️ Project Structure

- `Main.py` — Main FALCON system, model architecture, training loops, data preparation, and backtesting.
- `stock_data/` — Folder for raw, processed data and saved models.
- `falcon.log` — Logging file to track model training and evaluation.
- `models/` — Directory to store trained model checkpoints and scalers.

---

## 📚 Model Architecture

1. **Base Layer**: Asset-specific Transformer encoders.
2. **Middle Layer**: Cross-asset attention mechanism and market regime detection.
3. **Top Layer**: Portfolio optimization agent that generates stock weights and ranking scores.

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

Make sure to install the following libraries:
- `torch`
- `yfinance`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `tqdm`

---

## 🏃‍♂️ How to Run

1. **Initialize FALCON System**

```python
from Main import FALCONSystem

tickers = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA']
falcon = FALCONSystem(ticker_list=tickers)
```

2. **Download and Process Data**

```python
falcon.download_stock_data(start_date='2005-01-01', end_date='2022-01-01')
falcon.process_stock_data()
```

3. **Prepare Training Data**

```python
X_daily, y_returns = falcon.prepare_data()
```

4. **Train Base Layer (Pretraining)**

```python
falcon.train_base_layer(X_daily, y_returns, epochs=100)
```

5. **Train Full Model**

```python
falcon.train_full_model(X_daily, y_returns, epochs=200)
```

6. **Backtest Portfolio Strategy**

```python
# Load model if needed
# falcon.load_model(model_path="path/to/model.pt", scalers_path="path/to/scalers.pkl")

# Backtest
results = falcon.backtest(test_data=falcon.process_stock_data(),
                          start_date='2022-01-01',
                          end_date='2025-01-01')

print(results)
```

---

## 📈 Example Performance

- **Total Return**: 46.7%
- **Annualized Return**: 13.6%
- **Sharpe Ratio**: 0.82
- **Maximum Drawdown**: 10.8%
- **Portfolio Turnover**: 32.4%

*(Based on backtesting on S&P 500 stocks from 2022 to 2025)*

---

## 🧠 Future Work

- Integration of multi-asset classes (commodities, bonds, etc.)
- Explainable AI techniques for better model interpretability
- Reinforcement Learning based dynamic portfolio rebalancing
- Alternative data sources (sentiment analysis, macroeconomic indicators)

---

## 📜 Citation

If you use FALCON in your research or projects, please cite:

> Anay Praveen, Aravindh Palaniguru. *Financial Analysis with Cross-asset Optimization Network for Portfolio Management*. University of Nottingham Malaysia, 2025.

---

## 📧 Contact

For any questions or feedback, reach out:
- anaypraveen@example.com
- aravindh.palaniguru@example.com
