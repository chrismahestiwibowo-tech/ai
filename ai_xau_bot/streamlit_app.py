"""
Streamlit Web App for XAU/USD Price Prediction
Cloud-compatible version using yfinance for data
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="XAU/USD Prediction Bot",
    page_icon="💰",
    layout="wide"
)

# Title and description
st.title("💰 XAU/USD Price Prediction System")
st.markdown("### AI-Powered Gold Price Analysis and Forecasting")

@st.cache_data(ttl=3600)
def load_gold_data(days=365):
    """Load XAU/USD data using yfinance"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Download gold data (GC=F is Gold Futures)
        ticker = yf.Ticker("GC=F")
        df = ticker.history(start=start_date, end=end_date, interval="1d")
        
        if df.empty:
            st.error("No data retrieved. Please check your internet connection.")
            return None
        
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def add_technical_indicators(df):
    """Add technical indicators to the dataframe"""
    # Simple Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Exponential Moving Average
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # Volatility
    df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
    
    return df

def prepare_features(df):
    """Prepare features for model training"""
    df = df.copy()
    
    # Price changes
    df['Price_Change'] = df['Close'].pct_change()
    df['Price_Change_2d'] = df['Close'].pct_change(periods=2)
    df['Price_Change_5d'] = df['Close'].pct_change(periods=5)
    
    # Volume features
    df['Volume_Change'] = df['Volume'].pct_change()
    
    # Lag features
    for i in [1, 2, 3, 5, 10]:
        df[f'Close_Lag_{i}'] = df['Close'].shift(i)
    
    # Target: next day's price
    df['Target'] = df['Close'].shift(-1)
    
    # Drop rows with NaN
    df = df.dropna()
    
    return df

def train_model(df):
    """Train prediction model"""
    feature_cols = [col for col in df.columns if col not in ['Date', 'Target', 'Dividends', 'Stock Splits']]
    
    X = df[feature_cols]
    y = df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    return model, train_score, test_score, feature_cols

def predict_next_price(model, df, feature_cols):
    """Predict next price"""
    latest_data = df[feature_cols].iloc[-1:].values
    prediction = model.predict(latest_data)[0]
    return prediction

# Main app
def main():
    # Sidebar
    st.sidebar.header("Settings")
    days_to_load = st.sidebar.slider("Days of historical data", 30, 730, 365)
    
    # Load data
    with st.spinner("Loading gold price data..."):
        df = load_gold_data(days_to_load)
    
    if df is None or df.empty:
        st.stop()
    
    # Add indicators
    with st.spinner("Calculating technical indicators..."):
        df = add_technical_indicators(df)
        df_features = prepare_features(df)
    
    # Display current price
    current_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2]
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"${current_price:.2f}", f"{price_change_pct:.2f}%")
    with col2:
        st.metric("24h High", f"${df['High'].iloc[-1]:.2f}")
    with col3:
        st.metric("24h Low", f"${df['Low'].iloc[-1]:.2f}")
    
    # Train model and predict
    with st.spinner("Training prediction model..."):
        model, train_score, test_score, feature_cols = train_model(df_features)
        next_price = predict_next_price(model, df_features, feature_cols)
    
    st.markdown("---")
    
    # Prediction
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔮 Next Day Prediction")
        predicted_change = next_price - current_price
        predicted_change_pct = (predicted_change / current_price) * 100
        st.metric("Predicted Price", f"${next_price:.2f}", f"{predicted_change_pct:.2f}%")
        
    with col2:
        st.subheader("📊 Model Performance")
        st.write(f"Training Score: {train_score:.4f}")
        st.write(f"Testing Score: {test_score:.4f}")
    
    # Price chart
    st.subheader("📈 Price History")
    
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
    
    # Add moving averages
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['SMA_20'],
        name='SMA 20',
        line=dict(color='blue', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['SMA_50'],
        name='SMA 50',
        line=dict(color='red', width=1)
    ))
    
    fig.update_layout(
        title='XAU/USD Price Chart',
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        height=600,
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Technical indicators
    st.subheader("📊 Technical Indicators")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # RSI Chart
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=df['Date'],
            y=df['RSI'],
            name='RSI',
            line=dict(color='purple')
        ))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig_rsi.update_layout(title='RSI (Relative Strength Index)', height=300)
        st.plotly_chart(fig_rsi, use_container_width=True)
    
    with col2:
        # MACD Chart
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], name='MACD'))
        fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['Signal'], name='Signal'))
        fig_macd.update_layout(title='MACD', height=300)
        st.plotly_chart(fig_macd, use_container_width=True)
    
    # Data table
    with st.expander("View Raw Data"):
        st.dataframe(df.tail(20))
    
    # Footer
    st.markdown("---")
    st.markdown("**Disclaimer:** This is for educational purposes only. Not financial advice.")

if __name__ == "__main__":
    main()
