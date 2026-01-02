import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from xgboost import XGBRegressor

# Try to import TensorFlow, but make it optional for Python 3.14+
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.warning("⚠️ TensorFlow is not available (requires Python 3.12 or earlier). LSTM model will be disabled.")

import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="☀️ SOL Price Prediction Dashboard",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with responsive design and dark/light theme support
st.markdown("""
<style>
    /* ===== RESPONSIVE & THEME VARIABLES ===== */
    :root {
        --header-size: 48px;
        --padding-base: 20px;
        --border-radius: 10px;
    }
    
    /* Dark theme (default for Streamlit dark mode) */
    @media (prefers-color-scheme: dark) {
        :root {
            --bg-primary: rgba(14, 17, 23, 1);
            --bg-secondary: rgba(38, 39, 48, 0.8);
            --bg-card: rgba(240, 242, 246, 0.1);
            --text-primary: #ffffff;
            --text-secondary: #a0a0a0;
            --accent-color: #4da6ff;
            --border-color: rgba(255, 255, 255, 0.1);
            --shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
        }
    }
    
    /* Light theme */
    @media (prefers-color-scheme: light) {
        :root {
            --bg-primary: #ffffff;
            --bg-secondary: #f0f2f6;
            --bg-card: rgba(0, 0, 0, 0.02);
            --text-primary: #0e1117;
            --text-secondary: #555555;
            --accent-color: #0066cc;
            --border-color: rgba(0, 0, 0, 0.1);
            --shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
        }
    }
    
    /* ===== RESPONSIVE TYPOGRAPHY ===== */
    .main-header {
        font-size: var(--header-size);
        font-weight: bold;
        color: var(--accent-color);
        text-align: center;
        margin-bottom: 30px;
        text-shadow: var(--shadow);
        transition: all 0.3s ease;
    }
    
    /* ===== RESPONSIVE CARDS ===== */
    .metric-card {
        background-color: var(--bg-card);
        padding: var(--padding-base);
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
        border: 1px solid var(--border-color);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.2);
    }
    
    .stMetric {
        background-color: var(--bg-card);
        padding: 15px;
        border-radius: 8px;
        border: 1px solid var(--border-color);
    }
    
    [data-testid="stMetricValue"] {
        color: var(--accent-color);
        font-size: clamp(1.2rem, 3vw, 2rem);
    }
    
    /* ===== MOBILE RESPONSIVE (< 768px) ===== */
    @media only screen and (max-width: 768px) {
        :root {
            --header-size: 28px;
            --padding-base: 12px;
            --border-radius: 8px;
        }
        
        .main-header {
            font-size: 28px !important;
            margin-bottom: 20px;
            padding: 10px;
        }
        
        .metric-card {
            padding: 12px;
            margin-bottom: 10px;
        }
        
        [data-testid="stMetricValue"] {
            font-size: 1.5rem !important;
        }
        
        /* Stack columns on mobile */
        [data-testid="column"] {
            width: 100% !important;
            margin-bottom: 1rem;
        }
        
        /* Better button sizing for mobile */
        .stButton > button {
            width: 100%;
            padding: 12px;
            font-size: 14px;
        }
        
        /* Sidebar adjustments */
        [data-testid="stSidebar"] {
            width: 100%;
        }
        
        /* Smaller font sizes for mobile */
        body {
            font-size: 14px;
        }
    }
    
    /* ===== TABLET RESPONSIVE (768px - 1024px) ===== */
    @media only screen and (min-width: 768px) and (max-width: 1024px) {
        :root {
            --header-size: 36px;
            --padding-base: 16px;
        }
        
        .main-header {
            font-size: 36px !important;
        }
        
        [data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
        }
    }
    
    /* ===== DESKTOP RESPONSIVE (> 1024px) ===== */
    @media only screen and (min-width: 1024px) {
        .main-header {
            font-size: 48px;
        }
    }
    
    /* ===== GENERAL RESPONSIVE ELEMENTS ===== */
    /* Make tables responsive */
    .dataframe {
        width: 100%;
        overflow-x: auto;
        display: block;
    }
    
    /* Responsive images and plots */
    img, canvas {
        max-width: 100%;
        height: auto;
    }
    
    /* Better spacing on all devices */
    .block-container {
        padding: clamp(1rem, 3vw, 3rem);
        max-width: 100%;
    }
    
    /* Responsive font sizes */
    body {
        font-size: clamp(14px, 2vw, 16px);
    }
    
    h1 { font-size: clamp(1.8rem, 4vw, 2.5rem); }
    h2 { font-size: clamp(1.5rem, 3.5vw, 2rem); }
    h3 { font-size: clamp(1.2rem, 3vw, 1.75rem); }
    
    /* Better touch targets for mobile */
    @media (hover: none) and (pointer: coarse) {
        button, a, input, select {
            min-height: 44px;
            min-width: 44px;
        }
    }
    
    /* Smooth transitions for theme changes */
    * {
        transition: background-color 0.3s ease, color 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# Azure Foundry API Key - Use Streamlit secrets for security
try:
    api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    azure_endpoint = st.secrets.get("AZURE_ENDPOINT", "https://chrismawibowo-251229-resource.services.ai.azure.com/models")
except:
    api_key = os.getenv("OPENAI_API_KEY", "")
    azure_endpoint = "https://chrismawibowo-251229-resource.services.ai.azure.com/models"

if not api_key:
    st.sidebar.warning("⚠️ Azure API key not configured. Set it in Streamlit secrets or environment variables.")

# Sidebar
st.sidebar.title("⚙️ Configuration")

# Historical data period
st.sidebar.markdown("### 📅 Historical Data Period")
historical_years = st.sidebar.slider(
    "Training Data (Years)", 
    min_value=1, 
    max_value=3, 
    value=1,
    help="Select how many years of historical data to use for training models. More data can improve accuracy but takes longer to process."
)

# Forecast period
st.sidebar.markdown("### 🔮 Forecast Period")
forecast_days = st.sidebar.slider("Forecast Days", 30, 365, 365)

st.sidebar.markdown("---")
st.sidebar.info("📊 Advanced AI models analyze SOL price patterns to provide accurate predictions and trading insights. Read README.md for more details.")

# Main header
st.markdown('<div class="main-header">☀️ SOL Price Prediction Dashboard</div>', unsafe_allow_html=True)

# Load data function with caching
@st.cache_data(ttl=3600)
def load_sol_data(years=1):
    """Load SOL historical data with enhanced features
    
    Args:
        years (int): Number of years of historical data to load (1-3)
    """
    with st.spinner(f"📥 Loading {years} year{'s' if years > 1 else ''} of SOL data..."):
        try:
            ticker = yf.Ticker('SOL-USD')
            df = ticker.history(period=f'{years}y')
            
            if df is None or df.empty:
                st.error("Failed to fetch SOL data from Yahoo Finance. Please check your internet connection.")
                return None
                
            df = df.reset_index()
            df['ds'] = df['Date'].dt.tz_localize(None)
            
            # Basic features
            df['X'] = df['Open']
            df['y'] = df['Close']
            df['High'] = df['High']
            df['Low'] = df['Low']
            df['Volume'] = df['Volume']
            
            # Technical indicators
            df['MA7'] = df['Close'].rolling(window=7).mean()
            df['MA21'] = df['Close'].rolling(window=21).mean()
            df['MA50'] = df['Close'].rolling(window=50).mean()
            df['RSI'] = calculate_rsi(df['Close'], 14)
            df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
            df['Price_Range'] = df['High'] - df['Low']
            df['Daily_Return'] = df['Close'].pct_change()
            df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
            
            # Fill NaN values
            df = df.fillna(method='bfill').fillna(method='ffill')
            
            return df
        except Exception as e:
            st.error(f"Error loading SOL data: {str(e)}")
            return None

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

@st.cache_data
def get_current_price():
    """Get current SOL price"""
    try:
        current_sol = yf.Ticker('SOL-USD')
        history = current_sol.history(period='1d')
        
        if history is None or history.empty:
            st.warning("Could not fetch current price. Using latest historical price.")
            return None
            
        current_price = history['Close'].iloc[-1]
        return current_price
    except Exception as e:
        st.error(f"Error fetching current price: {str(e)}")
        return None

@st.cache_data
def train_models(_df):
    """Train LinearRegression and Prophet models"""
    X = _df[['X']]
    y = _df['y']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    
    # Train Linear Regression
    lrm = LinearRegression()
    lrm.fit(X_train, y_train)
    
    # Create LRM predictions for full dataset
    y_pred_lrm = lrm.predict(X)
    df_lrm = _df[['ds']].copy()
    df_lrm['y'] = y_pred_lrm
    
    # Train Prophet on LRM predictions
    model_prophet = Prophet()
    model_prophet.fit(df_lrm)
    
    return lrm, model_prophet, df_lrm, X_train, X_test, y_train, y_test

def train_xgboost_model(df):
    """Train XGBoost model with enhanced features"""
    feature_cols = ['Open', 'High', 'Low', 'Volume', 'MA7', 'MA21', 'MA50', 'RSI', 'MACD', 'Price_Range', 'Volatility']
    
    # Prepare data
    df_clean = df[feature_cols + ['Close']].dropna()
    X = df_clean[feature_cols]
    y = df_clean['Close']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42, shuffle=False)
    
    # Train XGBoost
    xgb_model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    
    return xgb_model, X_train, X_test, y_train, y_test, feature_cols

def train_lstm_model(df, lookback=60):
    """Train LSTM model"""
    if not TENSORFLOW_AVAILABLE:
        # Return dummy values if TensorFlow is not available
        return None, None, None, None, None, None, lookback
    
    # Prepare data
    data = df[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Build LSTM model
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(100, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train model (with reduced epochs for faster execution)
    model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=0)
    
    return model, scaler, X_train, X_test, y_train, y_test, lookback

def generate_forecast(model_prophet, df_lrm, periods):
    """Generate forecast using Prophet"""
    future_lrm = model_prophet.make_future_dataframe(periods=periods)
    forecast_lrm = model_prophet.predict(future_lrm)
    future_predictions = forecast_lrm[len(df_lrm):].copy()
    return forecast_lrm, future_predictions

def generate_xgb_forecast(xgb_model, df, feature_cols, periods):
    """Generate XGBoost forecast"""
    last_data = df[feature_cols].iloc[-1:].copy()
    predictions = []
    
    for i in range(periods):
        pred = xgb_model.predict(last_data)[0]
        predictions.append(pred)
        
        # Update features for next prediction (only update features in feature_cols)
        last_data = last_data.copy()
        if 'Open' in feature_cols:
            last_data['Open'] = pred
        if 'High' in feature_cols:
            last_data['High'] = pred * 1.02  # Estimate high
        if 'Low' in feature_cols:
            last_data['Low'] = pred * 0.98   # Estimate low
        # Keep other features relatively stable (they'll use previous values)
    
    future_dates = pd.date_range(start=df['ds'].iloc[-1] + pd.Timedelta(days=1), periods=periods)
    forecast_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': predictions,
        'yhat_lower': np.array(predictions) * 0.95,
        'yhat_upper': np.array(predictions) * 1.05
    })
    
    return forecast_df

def generate_lstm_forecast(lstm_model, scaler, df, lookback, periods):
    """Generate LSTM forecast"""
    if not TENSORFLOW_AVAILABLE or lstm_model is None:
        # Return empty forecast if TensorFlow is not available
        future_dates = pd.date_range(start=df['ds'].iloc[-1] + pd.Timedelta(days=1), periods=periods)
        current_price = df['Close'].iloc[-1]
        return pd.DataFrame({
            'ds': future_dates,
            'yhat': [current_price] * periods,
            'yhat_lower': [current_price * 0.92] * periods,
            'yhat_upper': [current_price * 1.08] * periods
        })
    
    # Get last lookback values
    last_data = df[['Close']].tail(lookback).values
    scaled_data = scaler.transform(last_data)
    
    predictions = []
    current_batch = scaled_data.reshape(1, lookback, 1)
    
    for i in range(periods):
        pred = lstm_model.predict(current_batch, verbose=0)[0, 0]
        predictions.append(pred)
        
        # Update batch with prediction
        current_batch = np.append(current_batch[:, 1:, :], [[[pred]]], axis=1)
    
    # Inverse transform predictions
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    
    future_dates = pd.date_range(start=df['ds'].iloc[-1] + pd.Timedelta(days=1), periods=periods)
    forecast_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': predictions,
        'yhat_lower': predictions * 0.92,
        'yhat_upper': predictions * 1.08
    })
    
    return forecast_df

def calculate_model_metrics(lrm, xgb_model, lstm_model, scaler, X_train_lr, X_test_lr, y_train_lr, y_test_lr, 
                           X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb,
                           X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, lookback):
    """Calculate metrics for all models"""
    results = []
    
    # LinearRegression
    y_pred_lr = lrm.predict(X_test_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test_lr, y_pred_lr))
    results.append({'model': '🌸 Orchid', 'rmse': rmse_lr, 'rank': 0})
    
    # XGBoost
    y_pred_xgb = xgb_model.predict(X_test_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_test_xgb, y_pred_xgb))
    results.append({'model': '🌼 Jasmine', 'rmse': rmse_xgb, 'rank': 0})
    
    # LSTM (only if TensorFlow is available)
    if TENSORFLOW_AVAILABLE and lstm_model is not None:
        y_pred_lstm_scaled = lstm_model.predict(X_test_lstm, verbose=0)
        y_pred_lstm = scaler.inverse_transform(y_pred_lstm_scaled).flatten()
        y_test_lstm_actual = scaler.inverse_transform(y_test_lstm.reshape(-1, 1)).flatten()
        rmse_lstm = np.sqrt(mean_squared_error(y_test_lstm_actual, y_pred_lstm))
        results.append({'model': '🌺 Bougainvillea', 'rmse': rmse_lstm, 'rank': 0})
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('rmse')
    results_df['rank'] = range(1, len(results_df) + 1)
    
    return results_df

def calculate_advanced_metrics(df, forecast_lrm, df_lrm):
    """Calculate crypto-specific advanced benchmarking metrics"""
    historical_prices = df['y'].values
    mean_price = historical_prices.mean()
    price_range = historical_prices.max() - historical_prices.min()
    
    # Calculate volatility
    df_calc = df.copy()
    df_calc['returns'] = df_calc['y'].pct_change()
    daily_volatility = df_calc['returns'].std() * 100
    annual_volatility = daily_volatility * np.sqrt(252)
    
    # Calculate ATR (14-day)
    df_calc['high'] = df_calc['y'] * 1.02
    df_calc['low'] = df_calc['y'] * 0.98
    df_calc['prev_close'] = df_calc['y'].shift(1)
    df_calc['tr1'] = df_calc['high'] - df_calc['low']
    df_calc['tr2'] = abs(df_calc['high'] - df_calc['prev_close'])
    df_calc['tr3'] = abs(df_calc['low'] - df_calc['prev_close'])
    df_calc['true_range'] = df_calc[['tr1', 'tr2', 'tr3']].max(axis=1)
    atr_14 = df_calc['true_range'].rolling(window=14).mean().iloc[-1]
    
    # Baseline models
    naive_predictions = df['y'].shift(1)[1:]
    naive_actuals = df['y'][1:]
    naive_rmse = np.sqrt(mean_squared_error(naive_actuals, naive_predictions))
    
    ma_predictions = df['y'].rolling(window=7).mean().shift(1)[7:]
    ma_actuals = df['y'][7:]
    ma_rmse = np.sqrt(mean_squared_error(ma_actuals, ma_predictions))
    
    # Our model metrics (Prophet on LinearRegression predictions)
    historical_forecast = forecast_lrm[:len(df_lrm)]
    our_predictions = historical_forecast['yhat'].values
    
    # Ensure lengths match by using only the length of predictions
    min_len = min(len(our_predictions), len(df['y']))
    our_predictions = our_predictions[:min_len]
    our_actuals = df['y'].values[:min_len]
    
    rmse_actual = np.sqrt(mean_squared_error(our_actuals, our_predictions))
    mae_actual = np.sqrt(mean_squared_error(our_actuals, our_predictions))
    
    rmse_percent = (rmse_actual / mean_price) * 100
    nrmse = rmse_actual / price_range
    rmse_volatility_pct = (rmse_actual / (mean_price * daily_volatility / 100)) * 100
    nrmse_atr = rmse_actual / atr_14
    mape = mean_absolute_percentage_error(our_actuals, our_predictions) * 100
    
    # Directional accuracy
    actual_direction = np.diff(our_actuals) > 0
    pred_direction = np.diff(our_predictions) > 0
    directional_accuracy = (actual_direction == pred_direction).mean() * 100
    
    # Improvements
    improvement_vs_naive = ((naive_rmse - rmse_actual) / naive_rmse) * 100
    improvement_vs_ma = ((ma_rmse - rmse_actual) / ma_rmse) * 100
    
    return {
        'mean_price': mean_price,
        'price_range': price_range,
        'daily_volatility': daily_volatility,
        'annual_volatility': annual_volatility,
        'atr_14': atr_14,
        'naive_rmse': naive_rmse,
        'ma_rmse': ma_rmse,
        'rmse_actual': rmse_actual,
        'mae_actual': mae_actual,
        'rmse_percent': rmse_percent,
        'nrmse': nrmse,
        'rmse_volatility_pct': rmse_volatility_pct,
        'nrmse_atr': nrmse_atr,
        'mape': mape,
        'directional_accuracy': directional_accuracy,
        'improvement_vs_naive': improvement_vs_naive,
        'improvement_vs_ma': improvement_vs_ma
    }

def classify_rmse_performance(rmse_pct):
    """Classify RMSE% based on crypto industry benchmarks (short-term 1-3 days)"""
    if rmse_pct < 5:
        return '🌟 EXCELLENT', '#2ecc71'
    elif rmse_pct < 8:
        return '✅ GOOD', '#3498db'
    elif rmse_pct < 12:
        return '⚠️ ACCEPTABLE', '#f39c12'
    else:
        return '❌ POOR', '#e74c3c'

# Load data
try:
    df = load_sol_data(historical_years)
    
    if df is None:
        st.stop()
    
    current_price = get_current_price()
    
    if current_price is None:
        # Use latest historical price if current price fetch fails
        current_price = df['Close'].iloc[-1]
        st.info(f"Using latest historical price: ${current_price:.2f}")
    
    # Train models with countdown timer
    import time
    
    # Create placeholder for countdown
    countdown_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    # Estimate total training time (in seconds)
    total_time = 15  # Adjust based on your actual training time
    
    # Start countdown in a separate container
    for i in range(total_time, 0, -1):
        countdown_placeholder.info(f"🤖 Training and learning the data with our machines... ⏱️ {i}s remaining")
        progress_bar.progress((total_time - i) / total_time)
        time.sleep(1)
    
    # Clear countdown and show training
    countdown_placeholder.empty()
    progress_bar.empty()
    
    with st.spinner("🤖 Finalizing model training..."):
        # LinearRegression + Prophet
        lrm, model_prophet, df_lrm, X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_models(df)
        forecast_lrm, future_predictions_prophet = generate_forecast(model_prophet, df_lrm, forecast_days)
        
        # XGBoost
        xgb_model, X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb, feature_cols = train_xgboost_model(df)
        future_predictions_xgb = generate_xgb_forecast(xgb_model, df, feature_cols, forecast_days)
        
        # LSTM
        lstm_model, scaler, X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, lookback = train_lstm_model(df)
        future_predictions_lstm = generate_lstm_forecast(lstm_model, scaler, df, lookback, forecast_days)
    
    # Calculate model performance to find best model
    with st.spinner("📊 Calculating model metrics..."):
        results_df = calculate_model_metrics(lrm, xgb_model, lstm_model, scaler,
                                             X_train_lr, X_test_lr, y_train_lr, y_test_lr,
                                             X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb,
                                             X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, lookback)
        
        best_model_name = results_df.iloc[0]['model']
        best_rmse = results_df.iloc[0]['rmse']
    
    # Use the best performing model for advanced metrics
    with st.spinner(f"📊 Calculating advanced benchmarks using {best_model_name}..."):
        # Select forecast based on best model
        if best_model_name == '🌼 Jasmine':
            # Create forecast dataframe for XGBoost historical predictions
            xgb_historical = xgb_model.predict(df[['Open', 'High', 'Low', 'Volume', 'MA7', 'MA21', 'MA50', 'RSI', 'MACD', 'Price_Range', 'Volatility']].dropna())
            df_clean = df.dropna()
            df_best = pd.DataFrame({'ds': df_clean['ds'], 'yhat': xgb_historical})
            forecast_best = pd.concat([df_best, future_predictions_xgb.rename(columns={'yhat': 'yhat'})], ignore_index=True)
            best_forecast = future_predictions_xgb
        elif best_model_name == '🌺 Bougainvillea':
            # Use LSTM predictions
            best_forecast = future_predictions_lstm
            forecast_best = forecast_lrm  # Use prophet structure for compatibility
        else:  # Linear Regression
            best_forecast = future_predictions_prophet
            forecast_best = forecast_lrm
        
        metrics = calculate_advanced_metrics(df, forecast_best, df_lrm)
        metrics['best_model_name'] = best_model_name
        metrics['best_model_rmse'] = best_rmse
        rating, rating_color = classify_rmse_performance(metrics['rmse_percent'])
    
    # Top metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("💰 Current SOL Price", f"${current_price:.2f}")
    
    with col2:
        next_price_prophet = future_predictions_prophet.iloc[0]['yhat']
        change = next_price_prophet - current_price
        st.metric("🌸 Orchid Prediction", f"${next_price_prophet:.2f}", 
                 f"{change:+.2f} ({(change/current_price*100):+.2f}%)")
    
    with col3:
        next_price_xgb = future_predictions_xgb.iloc[0]['yhat']
        change_xgb = next_price_xgb - current_price
        st.metric("🌼 Jasmine Prediction", f"${next_price_xgb:.2f}",
                 f"{change_xgb:+.2f} ({(change_xgb/current_price*100):+.2f}%)")
    
    with col4:
        next_price_lstm = future_predictions_lstm.iloc[0]['yhat']
        change_lstm = next_price_lstm - current_price
        st.metric("🌺 Bougainvillea Prediction", f"${next_price_lstm:.2f}",
                 f"{change_lstm:+.2f} ({(change_lstm/current_price*100):+.2f}%)")
    
    with col5:
        avg_next = (next_price_prophet + next_price_xgb + next_price_lstm) / 3
        avg_change = avg_next - current_price
        st.metric("📊 Ensemble Avg", f"${avg_next:.2f}",
                 f"{avg_change:+.2f} ({(avg_change/current_price*100):+.2f}%)")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview & All Models", "🤖 Model Performance", "📈 Advanced Benchmarks", "🔮 Forecast & Trading Insights"])
    
    with tab1:
        st.subheader("Historical Data & Multi-Model Predictions")
        
        # Get LRM predictions for historical data
        lrm_historical_predictions = lrm.predict(df[['X']].values)
        
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(df['ds'], df['y'], label='Actual Prices', color='blue', linewidth=2.5, marker='o', markersize=2)
        ax.plot(df['ds'], lrm_historical_predictions, label='🌸 Orchid Fit', color='green', linewidth=1.5, linestyle='--', alpha=0.7)
        ax.plot(forecast_lrm['ds'][:len(df)], forecast_lrm['yhat'][:len(df)], label='🌸 Orchid (Historical)', color='orange', linewidth=1.5, linestyle=':')
        
        # Future forecasts
        ax.plot(future_predictions_prophet['ds'], future_predictions_prophet['yhat'], label='🌸 Orchid Forecast', color='red', linewidth=2)
        ax.plot(future_predictions_xgb['ds'], future_predictions_xgb['yhat'], label='🌼 Jasmine Forecast', color='purple', linewidth=2)
        ax.plot(future_predictions_lstm['ds'], future_predictions_lstm['yhat'], label='🌺 Bougainvillea Forecast', color='magenta', linewidth=2)
        
        # Ensemble average
        ensemble_forecast = (future_predictions_prophet['yhat'].values + future_predictions_xgb['yhat'].values + future_predictions_lstm['yhat'].values) / 3
        ax.plot(future_predictions_prophet['ds'], ensemble_forecast, label='🌿 Ensemble Average', color='darkblue', linewidth=2.5, linestyle='-.', alpha=0.8)
        
        ax.fill_between(future_predictions_prophet['ds'], future_predictions_prophet['yhat_lower'], future_predictions_prophet['yhat_upper'], 
                        alpha=0.15, color='red', label='🌸 Orchid Confidence')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.set_title('SOL Price: Multi-Model Forecast Comparison', fontsize=16, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Model comparison table
        st.subheader("📋 Tomorrow's Predictions - All Models")
        tomorrow_data = {
            'Model': ['🌸 Orchid', '🌼 Jasmine', '🌺 Bougainvillea', '🌿 Ensemble'],
            'Predicted Price': [
                f'${next_price_prophet:.2f}',
                f'${next_price_xgb:.2f}',
                f'${next_price_lstm:.2f}',
                f'${avg_next:.2f}'
            ],
            'Change from Current': [
                f'{change:+.2f} ({(change/current_price*100):+.2f}%)',
                f'{change_xgb:+.2f} ({(change_xgb/current_price*100):+.2f}%)',
                f'{change_lstm:+.2f} ({(change_lstm/current_price*100):+.2f}%)',
                f'{avg_change:+.2f} ({(avg_change/current_price*100):+.2f}%)'
            ],
            'Confidence Range': [
                f"${future_predictions_prophet.iloc[0]['yhat_lower']:.2f} - ${future_predictions_prophet.iloc[0]['yhat_upper']:.2f}",
                f"${future_predictions_xgb.iloc[0]['yhat_lower']:.2f} - ${future_predictions_xgb.iloc[0]['yhat_upper']:.2f}",
                f"${future_predictions_lstm.iloc[0]['yhat_lower']:.2f} - ${future_predictions_lstm.iloc[0]['yhat_upper']:.2f}",
                "Composite of all models"
            ]
        }
        tomorrow_df = pd.DataFrame(tomorrow_data)
        st.dataframe(tomorrow_df, width='stretch', hide_index=True)
        
        # 30-day forecast comparison
        st.subheader("📊 30-Day Average Forecast Comparison")
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            avg_30d_prophet = future_predictions_prophet.head(30)['yhat'].mean()
            st.metric("🌸 Orchid 30-Day Avg", f"${avg_30d_prophet:.2f}")
        
        with col_b:
            avg_30d_xgb = future_predictions_xgb.head(30)['yhat'].mean()
            st.metric("🌼 Jasmine 30-Day Avg", f"${avg_30d_xgb:.2f}")
        
        with col_c:
            avg_30d_lstm = future_predictions_lstm.head(30)['yhat'].mean()
            st.metric("🌺 Bougainvillea 30-Day Avg", f"${avg_30d_lstm:.2f}")
        
        with col_d:
            avg_30d_ensemble = (avg_30d_prophet + avg_30d_xgb + avg_30d_lstm) / 3
            st.metric("Ensemble 30-Day Avg", f"${avg_30d_ensemble:.2f}")
    
    with tab2:
        st.subheader("🏆 Multi-Model Performance Comparison")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📈 Model Rankings (by RMSE)")
            st.dataframe(results_df.style.format({'rmse': '{:.4f}'}), 
                        width='stretch', hide_index=True)
            
            st.success(f"🏆 {best_model_name} - Most Accurate Model (RMSE: {best_rmse:.4f})")
            
            st.markdown("### 🔍 Model Descriptions")
            st.markdown("""
            - **🌸 Orchid (Linear Regression + Prophet)**: Elegant baseline using opening price with time-series forecasting
            - **🌼 Jasmine (XGBoost)**: Advanced model with 11 technical indicators (MA7, MA21, MA50, RSI, MACD, etc.)
            - **🌺 Bougainvillea (LSTM)**: Deep learning with 60-day price history pattern recognition
            
            *Flower names represent different model characteristics - just as each flower has unique beauty, each model has unique strengths.*
            """)
        
        with col2:
            # RMSE comparison bar chart
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['#2ecc71' if i == 0 else '#3498db' if i == 1 else '#e67e22' for i in range(len(results_df))]
            ax.barh(results_df['model'], results_df['rmse'], color=colors)
            ax.set_xlabel('RMSE (Lower is Better)', fontsize=11)
            ax.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            for i, (model, rmse) in enumerate(zip(results_df['model'], results_df['rmse'])):
                ax.text(rmse, i, f' {rmse:.2f}', va='center', fontweight='bold')
            st.pyplot(fig)
    
    with tab3:
        st.subheader("📈 Advanced Crypto-Specific Benchmarks")
        
        # Performance Rating
        st.markdown(f"### {rating}")
        st.markdown(f"**Model Quality Assessment for Short-term Forecast (1-3 days)**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📊 Core Metrics")
            st.metric("RMSE", f"${metrics['rmse_actual']:.2f}")
            st.metric("RMSE%", f"{metrics['rmse_percent']:.2f}%")
            st.metric("MAE", f"${metrics['mae_actual']:.2f}")
            st.metric("MAPE", f"{metrics['mape']:.2f}%")
        
        with col2:
            st.markdown("#### 📈 Normalized Metrics")
            st.metric("NRMSE (Range)", f"{metrics['nrmse']:.4f}")
            st.metric("NRMSE (ATR)", f"{metrics['nrmse_atr']:.2f}")
            st.metric("RMSE as % of Volatility", f"{metrics['rmse_volatility_pct']:.1f}%")
            st.metric("Daily Volatility", f"{metrics['daily_volatility']:.2f}%")
        
        with col3:
            st.markdown("#### 🎯 Performance vs Baselines")
            st.metric("vs Naive Model", f"{metrics['improvement_vs_naive']:+.1f}%",
                     delta="Better" if metrics['improvement_vs_naive'] > 0 else "Worse")
            st.metric("vs Moving Average", f"{metrics['improvement_vs_ma']:+.1f}%",
                     delta="Better" if metrics['improvement_vs_ma'] > 0 else "Worse")
            st.metric("Directional Accuracy", f"{metrics['directional_accuracy']:.1f}%",
                     delta="Good" if metrics['directional_accuracy'] > 55 else "Random")
        
        # Benchmark comparison table
        st.markdown("### 📋 Comprehensive Benchmark Table")
        benchmark_data = {
            'Metric': ['RMSE ($)', 'RMSE%', 'NRMSE (Range)', 'NRMSE (ATR)', 'Performance'],
            'Our Model': [
                f'${metrics["rmse_actual"]:.2f}',
                f'{metrics["rmse_percent"]:.2f}%',
                f'{metrics["nrmse"]:.4f}',
                f'{metrics["nrmse_atr"]:.2f}',
                rating
            ],
            'Naive Baseline': [
                f'${metrics["naive_rmse"]:.2f}',
                f'{(metrics["naive_rmse"]/metrics["mean_price"]*100):.2f}%',
                f'{(metrics["naive_rmse"]/metrics["price_range"]):.4f}',
                f'{(metrics["naive_rmse"]/metrics["atr_14"]):.2f}',
                '❌ POOR'
            ],
            'MA Baseline': [
                f'${metrics["ma_rmse"]:.2f}',
                f'{(metrics["ma_rmse"]/metrics["mean_price"]*100):.2f}%',
                f'{(metrics["ma_rmse"]/metrics["price_range"]):.4f}',
                f'{(metrics["ma_rmse"]/metrics["atr_14"]):.2f}',
                '❌ POOR'
            ]
        }
        benchmark_df = pd.DataFrame(benchmark_data)
        st.dataframe(benchmark_df, width='stretch', hide_index=True)
        
        # Visualizations
        st.markdown("### 📊 Benchmark Visualizations")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        
        # Plot 1: RMSE Comparison
        models = ['Our Model', 'Naive', 'Moving Avg']
        rmse_values = [metrics['rmse_actual'], metrics['naive_rmse'], metrics['ma_rmse']]
        colors_bar = ['#2ecc71', '#e74c3c', '#e67e22']
        ax1.bar(models, rmse_values, color=colors_bar, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('RMSE ($)', fontsize=11, fontweight='bold')
        ax1.set_title('RMSE: Our Model vs Baselines', fontsize=13, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        for i, v in enumerate(rmse_values):
            ax1.text(i, v, f'${v:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Performance Scores
        categories = ['RMSE%', 'Vol Adj', 'ATR Norm', 'Dir Acc']
        scores = [
            min(metrics['rmse_percent'] / 12 * 100, 100),
            min(metrics['rmse_volatility_pct'] / 100 * 100, 100),
            min(metrics['nrmse_atr'] / 2 * 100, 100),
            metrics['directional_accuracy']
        ]
        ax2.barh(categories, scores, color=['#3498db', '#9b59b6', '#e67e22', '#1abc9c'])
        ax2.set_xlabel('Score', fontsize=11)
        ax2.set_title('Multi-Dimensional Score', fontsize=13, fontweight='bold')
        ax2.axvline(x=50, color='red', linestyle='--', alpha=0.5)
        ax2.grid(axis='x', alpha=0.3)
        
        # Plot 3: Improvements
        improvements = [metrics['improvement_vs_naive'], metrics['improvement_vs_ma']]
        baseline_names = ['vs Naive', 'vs MA']
        colors3 = ['#2ecc71' if x > 0 else '#e74c3c' for x in improvements]
        ax3.bar(baseline_names, improvements, color=colors3, alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Improvement (%)', fontsize=11, fontweight='bold')
        ax3.set_title('Model Improvement', fontsize=13, fontweight='bold')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax3.axhline(y=15, color='green', linestyle='--', alpha=0.5, label='Target: 15-20%')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        for i, v in enumerate(improvements):
            ax3.text(i, v, f'{v:+.1f}%', ha='center', 
                    va='bottom' if v > 0 else 'top', fontweight='bold')
        
        # Plot 4: Industry Benchmarks
        horizons = ['Intraday\n(1-6h)', 'Short-term\n(1-3d)', 'Medium\n(1-2w)', 'Long\n(1m+)']
        excellent = [2, 5, 10, 15]
        good = [4, 8, 15, 25]
        acceptable = [7, 12, 25, 30]
        x = np.arange(len(horizons))
        width = 0.25
        ax4.bar(x - width, excellent, width, label='Excellent', color='#2ecc71', alpha=0.7)
        ax4.bar(x, good, width, label='Good', color='#3498db', alpha=0.7)
        ax4.bar(x + width, acceptable, width, label='Acceptable', color='#f39c12', alpha=0.7)
        ax4.axhline(y=metrics['rmse_percent'], color='red', linestyle='--', linewidth=2,
                   label=f'Our Model: {metrics["rmse_percent"]:.2f}%')
        ax4.set_ylabel('RMSE% Threshold', fontsize=11, fontweight='bold')
        ax4.set_title('Crypto Benchmarks by Horizon', fontsize=13, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(horizons, fontsize=9)
        ax4.legend(loc='upper left')
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Investment Decision
        st.markdown("### 💼 Investment-Grade Assessment")
        
        # Determine investment grade
        if metrics['improvement_vs_naive'] > 15 and metrics['directional_accuracy'] > 55:
            assessment_grade = "ADDS VALUE"
            assessment_color = "success"
            recommendation = "✅ RECOMMENDED for Investment Consideration"
        elif metrics['improvement_vs_naive'] > 0:
            assessment_grade = "LIMITED VALUE"
            assessment_color = "warning"
            recommendation = "⚠️ USE WITH CAUTION - Supplementary Signal Only"
        else:
            assessment_grade = "NO VALUE"
            assessment_color = "error"
            recommendation = "❌ NOT RECOMMENDED for Investment"
        
        # Display main assessment card
        if assessment_color == "success":
            st.success(f"### {recommendation}")
        elif assessment_color == "warning":
            st.warning(f"### {recommendation}")
        else:
            st.error(f"### {recommendation}")
        
        # Detailed breakdown
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("#### 📊 Performance Analysis")
            st.markdown(f"""
            **Overall Model Quality:** {rating}
            
            **Key Performance Indicators:**
            - **Improvement vs Naive:** {metrics['improvement_vs_naive']:+.2f}%
              - Target: >15% for reliable predictions
              - Current: {'✅ Exceeds target' if metrics['improvement_vs_naive'] > 15 else '⚠️ Below target' if metrics['improvement_vs_naive'] > 0 else '❌ Underperforming'}
            
            - **Improvement vs Moving Average:** {metrics['improvement_vs_ma']:+.2f}%
              - Current: {'✅ Better than MA' if metrics['improvement_vs_ma'] > 0 else '❌ Worse than MA'}
            
            - **Directional Accuracy:** {metrics['directional_accuracy']:.1f}%
              - Random baseline: 50%
              - Target: >55% for meaningful predictions
              - Current: {'✅ Above random' if metrics['directional_accuracy'] > 55 else '⚠️ Near random' if metrics['directional_accuracy'] > 50 else '❌ Below random'}
            
            - **RMSE Percentage:** {metrics['rmse_percent']:.2f}%
              - Industry benchmark for 1-3 day forecast:
              - Excellent: <5%, Good: <8%, Acceptable: <12%
            """)
        
        with col_b:
            st.markdown("#### 💡 Investment Insights")
            
            if assessment_grade == "ADDS VALUE":
                st.markdown("""
                **Why This Model Adds Value:**
                - ✅ Significantly outperforms simple baseline strategies
                - ✅ Directional accuracy exceeds random chance threshold
                - ✅ RMSE% is within acceptable range for crypto volatility
                
                **Recommended Usage:**
                - Use as a primary signal for short-term trading decisions
                - Combine with risk management (stop-loss, position sizing)
                - Monitor confidence intervals for trade entry/exit
                - Re-evaluate model weekly for performance degradation
                
                **Risk Considerations:**
                - Markets are inherently unpredictable
                - Past performance doesn't guarantee future results
                - Always use proper position sizing
                - Never invest more than you can afford to lose
                """)
            elif assessment_grade == "LIMITED VALUE":
                st.markdown("""
                **Why This Model Has Limited Value:**
                - ⚠️ Marginal improvement over baseline strategies
                - ⚠️ Directional accuracy may not be reliable enough
                - ⚠️ Risk-reward ratio may not justify trading costs
                
                **Recommended Usage:**
                - Use ONLY as a supplementary signal
                - Combine with other technical indicators
                - Require confirmation from multiple sources
                - Use smaller position sizes
                - Focus on high-conviction trades only
                
                **Limitations:**
                - Model barely outperforms simple strategies
                - Transaction costs may eliminate any edge
                - Predictions may not be statistically significant
                """)
            else:
                st.markdown("""
                **Why This Model Should NOT Be Used:**
                - ❌ Underperforms even simple baseline strategies
                - ❌ No predictive edge over "yesterday's price" method
                - ❌ Would likely result in losses after trading costs
                
                **Critical Issues:**
                - The model performs worse than:
                  - Using yesterday's price as today's prediction
                  - Simple 7-day moving average
                - Directional accuracy may be random or worse
                - RMSE indicates predictions are too far from actual values
                
                **What This Means:**
                - Do NOT make investment decisions based on this model
                - Model needs fundamental improvements:
                  - Add more relevant features (volume, market sentiment, etc.)
                  - Try different algorithms (LSTM, XGBoost with more features)
                  - Extend training data period
                  - Include external market indicators
                
                **Alternative Approaches:**
                - Focus on fundamental analysis
                - Use established trading strategies
                - Consult with financial professionals
                - Consider passive investment strategies
                """)
        
        # Summary box
        st.markdown("---")
        st.info(f"""
        **📌 Quick Summary:** This model shows **{assessment_grade}** for investment decisions based on {metrics['improvement_vs_naive']:+.1f}% improvement over baseline and {metrics['directional_accuracy']:.1f}% directional accuracy. 
        
        **⚠️ Important Disclaimer:** All predictions are probabilistic estimates, not guarantees. Always conduct your own research, use proper risk management, and consult with qualified financial advisors before making investment decisions. This dashboard is for educational and analytical purposes only.
        """)
    
    with tab4:
        st.subheader("🔮 Forecast Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Forecast Statistics")
            st.write(f"**Forecast Period:** {len(future_predictions_prophet)} days")
            st.write(f"**Date Range:** {future_predictions_prophet['ds'].min().strftime('%Y-%m-%d')} to {future_predictions_prophet['ds'].max().strftime('%Y-%m-%d')}")
            st.write(f"**Minimum Prediction:** ${future_predictions_prophet['yhat'].min():.2f}")
            st.write(f"**Maximum Prediction:** ${future_predictions_prophet['yhat'].max():.2f}")
            st.write(f"**Average Prediction:** ${future_predictions_prophet['yhat'].mean():.2f}")
        
        with col2:
            st.markdown("### 📉 Confidence Metrics")
            conf_width = (best_forecast['yhat_upper'] - best_forecast['yhat_lower']).mean()
            avg_prediction = best_forecast['yhat'].mean()
            conf_pct = (conf_width / avg_prediction * 100)
            
            st.write(f"**Avg Confidence Range:** ${conf_width:.2f}")
            
            # Show confidence level interpretation instead of raw percentage
            if conf_pct > 50:
                confidence_level = "⚠️ Low Confidence (High Uncertainty)"
                st.write(f"**Prediction Confidence:** {confidence_level}")
                st.caption(f"Uncertainty range: ±${conf_width/2:.2f} (predictions vary widely)")
            elif conf_pct > 25:
                confidence_level = "🔸 Moderate Confidence"
                st.write(f"**Prediction Confidence:** {confidence_level}")
                st.caption(f"Uncertainty: ±{conf_pct/2:.1f}% around predictions")
            else:
                confidence_level = "✅ High Confidence"
                st.write(f"**Prediction Confidence:** {confidence_level}")
                st.caption(f"Uncertainty: ±{conf_pct/2:.1f}% around predictions")
            
            trend = "📈 Bullish" if best_forecast['yhat'].iloc[-1] > current_price else "📉 Bearish"
            st.write(f"**Long-term Trend:** {trend}")
        
        # GPT-4 Analysis for Trading Insights
        st.markdown("---")
        st.markdown("### 🤖 AI-Powered Trading Analysis")
        
        with st.spinner("🧠 Generating AI analysis..."):
            try:
                from openai import OpenAI
                
                # Azure AI Foundry uses standard OpenAI client with base_url
                client = OpenAI(
                    api_key=api_key,
                    base_url=azure_endpoint
                )
                
                # Prepare analysis context
                next_price_best = best_forecast.iloc[0]['yhat'] if len(best_forecast) > 0 else current_price
                price_change_pct = ((next_price_best - current_price) / current_price * 100)
                trend_direction = "bullish" if price_change_pct > 0 else "bearish"
                
                avg_7d = best_forecast.head(7)['yhat'].mean()
                avg_30d = best_forecast.head(30)['yhat'].mean()
                
                prompt = f"""You are a professional cryptocurrency trading analyst. Analyze the following SOL price prediction data and provide actionable insights:

**Current Market Data:**
- Current SOL Price: ${current_price:.2f}
- Predicted Next Price: ${next_price_best:.2f}
- Price Change: {price_change_pct:+.2f}%
- 7-Day Average Forecast: ${avg_7d:.2f}
- 30-Day Average Forecast: ${avg_30d:.2f}
- Market Trend: {trend_direction.upper()}
- Model Performance (RMSE): {best_rmse:.4f}

**Model Metrics:**
- RMSE%: {metrics['rmse_percent']:.2f}%
- Directional Accuracy: {metrics['directional_accuracy']:.1f}%
- Improvement vs Baseline: {metrics['improvement_vs_naive']:+.2f}%

Provide your analysis in simple bullet points covering these 5 areas:

1. **Trading Signal** (BUY/HOLD/SELL with confidence level)
2. **Entry/Exit Strategy** (specific price targets and timing)
3. **Portfolio Rebalancing** (allocation recommendations based on risk profile)
4. **Risk Management** (stop-loss levels, position sizing, volatility considerations)
5. **Market Context** (short-term vs long-term outlook)

Be specific, data-driven, and practical. Format your response clearly with sections."""

                # Try different model names (Azure AI Foundry uses specific deployment names)
                model_names = ["Phi-4", "gpt-4o", "gpt-4o-mini", "gpt-4"]
                response = None
                
                for model_name in model_names:
                    try:
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": "You are an expert cryptocurrency trading analyst specializing in SOL/Solana. Provide clear, actionable trading insights based on quantitative data."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.7,
                            max_tokens=800
                        )
                        break  # Success, exit loop
                    except Exception as model_error:
                        if "unknown_model" in str(model_error).lower() and model_name != model_names[-1]:
                            continue  # Try next model
                        else:
                            raise  # Re-raise if it's the last model or different error
                
                gpt_analysis = response.choices[0].message.content
                
                # Display Ai Analysis
                st.markdown("#### 📊 AI Trading Insights")
                st.markdown(gpt_analysis)
                
            except Exception as e:
                st.warning(f"⚠️ Unable to generate AI analysis: {str(e)}")
                st.info("Displaying model predictions without AI interpretation.")
        
        # Show Prophet components if using Prophet-based model
        if best_model_name == '🌸 Orchid':
            st.markdown("---")
            st.markdown("### 📊 Forecast Components (Time Series Decomposition)")
            fig_components = model_prophet.plot_components(forecast_lrm)
            st.pyplot(fig_components)

except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    st.info("Please ensure you have an internet connection and all required packages are installed.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>☀️ SOL Price Prediction Dashboard | 🌸 Orchid • 🌼 Jasmine • 🌺 Bougainvillea</p>
    <p>⚠️ Disclaimer: This is for educational purposes only. Not financial advice. DYOR</p>
    <p style='font-size: 0.9em; margin-top: 10px;'>
        <a href='https://github.com/chrismahestiwibowo-tech/ai' target='_blank' style='color: #4da6ff; text-decoration: none;'>
            📂 View on GitHub
        </a> | Read README.md for specifications
    </p>
</div>
""", unsafe_allow_html=True)
