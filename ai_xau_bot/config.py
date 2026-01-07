"""
Configuration file for XAU/USD price prediction
"""
import os
from dotenv import load_dotenv

load_dotenv()

# MetaTrader5 Configuration
MT5_CONFIG = {
    'login': int(os.getenv('MT5_LOGIN', 0)),
    'password': os.getenv('MT5_PASSWORD', ''),
    'server': os.getenv('MT5_SERVER', ''),
    'path': os.getenv('MT5_PATH', '')
}

# Trading Symbol
SYMBOL = 'XAUUSD'

# Data Configuration
TIMEFRAME = 'H1'  # 1 Hour timeframe
YEARS_BACK = 3
TRAIN_SIZE = 0.8
TEST_SIZE = 0.2

# Model Configuration
XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'max_depth': 7,
    'learning_rate': 0.01,
    'n_estimators': 1000,
    'min_child_weight': 3,
    'gamma': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'early_stopping_rounds': 50
}

# Monte Carlo Configuration
MC_SIMULATIONS = 10000
MC_DAYS_AHEAD = 30

# Feature Engineering
INDICATORS = [
    'RSI',           # Relative Strength Index
    'MACD',          # Moving Average Convergence Divergence
    'MACD_Signal',   # MACD Signal Line
    'BB_Upper',      # Bollinger Bands Upper
    'BB_Lower',      # Bollinger Bands Lower
    'ATR',           # Average True Range
    'SMA_20',        # Simple Moving Average 20
    'EMA_20',        # Exponential Moving Average 20
    'Stochastic',    # Stochastic Oscillator
    'CCI',           # Commodity Channel Index
    'ADX'            # Average Directional Index
]

# Output Configuration
MODEL_PATH = 'models/xauusd_xgboost_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
RESULTS_PATH = 'results/'
PLOTS_PATH = 'plots/'
