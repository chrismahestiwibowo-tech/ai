"""
Quick Start Guide - XAU/USD Price Prediction System
====================================================

SETUP STEPS:
-----------

1. CONFIGURE METATRADER5 (Required for live data):
   - Copy .env.example to .env
   - Edit .env with your MT5 credentials:
     
     MT5_LOGIN=your_account_number
     MT5_PASSWORD=your_password
     MT5_SERVER=your_broker_server
     MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
   
   Note: If you don't have MT5 credentials, the system will still try to connect
         to MT5 in read-only mode if the terminal is installed.


2. INSTALL PACKAGES:
   pip install -r requirements.txt


3. RUN THE APPLICATION:
   python main.py
   
   This will:
   ✓ Load 3 years of XAU/USD data from MetaTrader5
   ✓ Calculate 11 technical indicators
   ✓ Train XGBoost model (80/20 split)
   ✓ Evaluate with MSE, RMSE, MAE, R²
   ✓ Run 10,000 Monte Carlo simulations
   ✓ Generate reports and visualizations


TESTING INDIVIDUAL MODULES:
---------------------------

Test data loading:
  python data_loader.py

Test indicator calculations:
  python indicators.py

Test model training:
  python model.py

Test Monte Carlo simulation:
  python monte_carlo.py


EXPECTED OUTPUT:
---------------

After running main.py, you'll find:

📁 models/
  - xauusd_xgboost_model.pkl    (Trained model)
  - scaler.pkl                   (Feature scaler)

📁 plots/
  - predictions.png              (Actual vs Predicted)
  - feature_importance.png       (Top features chart)
  - monte_carlo_paths.png        (10,000 simulation paths)
  - monte_carlo_distribution.png (Price distribution)

📁 results/
  - model_metrics.csv            (MSE, RMSE, MAE, R²)
  - feature_importance.csv       (Feature rankings)
  - monte_carlo_forecast.csv     (Forecast statistics)
  - price_probabilities.csv      (Target price probabilities)
  - summary_report.txt           (Complete analysis report)


CUSTOMIZATION:
-------------

Edit config.py to change:
  - Symbol (default: XAUUSD)
  - Timeframe (default: H1 - 1 hour)
  - Historical period (default: 3 years)
  - Train/test split (default: 80/20)
  - XGBoost parameters
  - Monte Carlo settings


TROUBLESHOOTING:
---------------

❌ "Failed to initialize MT5"
   → Check MT5 is installed and running
   → Verify .env credentials are correct
   → Try demo account first

❌ "Module not found"
   → Ensure virtual environment is activated
   → Run: pip install -r requirements.txt

❌ "No data fetched"
   → Check internet connection
   → Verify XAUUSD symbol is available on your broker
   → Ensure MT5 has historical data downloaded


TECHNICAL INDICATORS INCLUDED:
-----------------------------

1.  RSI (Relative Strength Index)
2.  MACD (Moving Average Convergence Divergence)
3.  MACD Signal Line
4.  Bollinger Bands (Upper & Lower)
5.  ATR (Average True Range)
6.  SMA (Simple Moving Average 20, 50)
7.  EMA (Exponential Moving Average 20, 50)
8.  Stochastic Oscillator
9.  CCI (Commodity Channel Index)
10. ADX (Average Directional Index)
11. Williams %R

Plus: OBV, ROC, price changes, rolling statistics, lag features


PERFORMANCE METRICS:
-------------------

The system evaluates using:
  • MSE  - Mean Squared Error (lower is better)
  • RMSE - Root Mean Squared Error (in price units)
  • MAE  - Mean Absolute Error
  • R²   - Coefficient of determination (closer to 1 is better)


MONTE CARLO SIMULATION:
----------------------

Provides:
  • 10,000 price path simulations
  • Mean, median, percentile forecasts
  • 90% confidence intervals
  • Value at Risk (VaR) calculations
  • Probability analysis for target prices


WORKFLOW:
---------

main.py orchestrates the complete workflow:

1. Data Loading (data_loader.py)
   ↓
2. Feature Engineering (indicators.py)
   ↓
3. Model Training (model.py)
   ↓
4. Monte Carlo Simulation (monte_carlo.py)
   ↓
5. Report Generation


EXAMPLE OUTPUT:
--------------

MODEL EVALUATION RESULTS
========================
TRAIN SET:
  MSE:  0.123456
  RMSE: 0.351362
  MAE:  0.245678
  R²:   0.987654

TEST SET:
  MSE:  0.234567
  RMSE: 0.484426
  MAE:  0.356789
  R²:   0.976543

MONTE CARLO FORECAST (30 days ahead)
====================================
Current Price:        $2,050.00
Mean:                 $2,065.23
Median:               $2,063.45
5th Percentile:       $1,995.67
95th Percentile:      $2,135.89


⚠️  DISCLAIMER:
--------------
This system is for EDUCATIONAL PURPOSES ONLY.
Past performance does not guarantee future results.
Always conduct your own research before trading.


For more details, see README.md
"""

if __name__ == "__main__":
    print(__doc__)
