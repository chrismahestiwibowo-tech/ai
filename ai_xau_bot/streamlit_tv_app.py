"""
Streamlit App for XAU/USD Price Prediction using TradingView
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
from tradingview_api import TradingView
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="XAU/USD TradingView Prediction",
    page_icon="💰",
    layout="wide"
)

# Title and description
st.title("💰 XAU/USD Price Prediction System (TradingView)")
st.markdown("### AI-Powered Gold Price Analysis using TradingView & Yahoo Finance")

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_tradingview_data(days=365):
    """Load data from Yahoo Finance for XAU/USD"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        df = yf.download(
            "XAUUSD=X",
            start=start_date,
            end=end_date,
            interval="1d",
            progress=False
        )
        
        if not df.empty:
            df.reset_index(inplace=True)
            # Rename columns to match format
            df.rename(columns={
                'Date': 'time',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'tick_volume'
            }, inplace=True)
            return df
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data(ttl=60)  # Cache for 1 minute
def get_tradingview_signals():
    """Get TradingView technical analysis signals for XAU/USD"""
    try:
        tv = TradingView()
        # Get technical indicators data
        symbol_data = tv.get_technicals(
            symbol="XAUUSD",
            exchange="FOREXCOM",
            interval="1D"
        )
        return symbol_data
    except Exception as e:
        st.warning(f"TradingView signals unavailable: {str(e)}")
        return None

def add_technical_indicators_simple(df):
    """Add basic technical indicators"""
    # Simple Moving Averages
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    
    # Exponential Moving Average
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    return df

def prepare_features_simple(df):
    """Prepare features for model training"""
    df = df.copy()
    
    # Price changes
    df['Price_Change'] = df['close'].pct_change()
    df['Price_Change_2d'] = df['close'].pct_change(periods=2)
    df['Price_Change_5d'] = df['close'].pct_change(periods=5)
    
    # Volume features
    df['Volume_Change'] = df['tick_volume'].pct_change()
    
    # Lag features
    for i in [1, 2, 3, 5, 10]:
        df[f'Close_Lag_{i}'] = df['close'].shift(i)
    
    # Target: next day's price
    df['Target'] = df['close'].shift(-1)
    
    # Drop rows with NaN
    df = df.dropna()
    
    return df

def train_simple_model(df):
    """Train prediction model"""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    
    feature_cols = [col for col in df.columns if col not in 
                    ['time', 'Target', 'spread', 'real_volume']]
    
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
    st.sidebar.header("⚙️ Settings")
    days_to_load = st.sidebar.slider("Days of historical data", 30, 730, 365)
    
    # Get TradingView signals
    tv_analysis = get_tradingview_signals()
    
    # Load data
    with st.spinner("Loading XAU/USD data from Yahoo Finance..."):
        df = load_tradingview_data(days_to_load)
    
    if df is None or df.empty:
        st.error("❌ No data retrieved")
        st.stop()
    
    st.success(f"✅ Loaded {len(df)} candles from Yahoo Finance")
    
    # Add indicators
    with st.spinner("Calculating technical indicators..."):
        df = add_technical_indicators_simple(df)
        df_features = prepare_features_simple(df)
    
    # Display current price
    current_price = df['close'].iloc[-1]
    prev_price = df['close'].iloc[-2]
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Price", f"${current_price:.2f}", f"{price_change_pct:.2f}%")
    with col2:
        st.metric("High", f"${df['high'].iloc[-1]:.2f}")
    with col3:
        st.metric("Low", f"${df['low'].iloc[-1]:.2f}")
    with col4:
        st.metric("Volume", f"{int(df['tick_volume'].iloc[-1]):,}")
    
    # Train model and predict
    with st.spinner("Training prediction model..."):
        model, train_score, test_score, feature_cols = train_simple_model(df_features)
        next_price = predict_next_price(model, df_features, feature_cols)
    
    st.markdown("---")
    
    # Prediction and TradingView signals
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("🔮 AI Prediction")
        predicted_change = next_price - current_price
        predicted_change_pct = (predicted_change / current_price) * 100
        st.metric("Next Day Price", f"${next_price:.2f}", f"{predicted_change_pct:.2f}%")
        
    with col2:
        st.subheader("📊 Model Score")
        st.write(f"Training: {train_score:.4f}")
        st.write(f"Testing: {test_score:.4f}")
        st.write(f"Source: Yahoo Finance")
    
    with col3:
        st.subheader("📡 TradingView Signals")
        if tv_analysis:
            try:
                recommendation = tv_analysis.get('summary', {}).get('RECOMMENDATION', 'NEUTRAL')
                if recommendation == 'BUY' or recommendation == 'STRONG_BUY':
                    st.success(f"🟢 {recommendation}")
                elif recommendation == 'SELL' or recommendation == 'STRONG_SELL':
                    st.error(f"🔴 {recommendation}")
                else:
                    st.warning(f"🟡 {recommendation}")
                buy_signals = tv_analysis.get('summary', {}).get('BUY', 0)
                sell_signals = tv_analysis.get('summary', {}).get('SELL', 0)
                st.write(f"Buy: {buy_signals}")
                st.write(f"Sell: {sell_signals}")
            except:
                st.info("Signal format not recognized")
        else:
            st.info("Signals unavailable")
    
    # Price chart
    st.subheader("📈 Price History (Yahoo Finance)")
    
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['time'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='XAU/USD'
    ))
    
    # Add moving averages
    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['SMA_20'],
        name='SMA 20',
        line=dict(color='blue', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['SMA_50'],
        name='SMA 50',
        line=dict(color='red', width=1)
    ))
    
    fig.update_layout(
        title='XAU/USD Price Chart (Yahoo Finance)',
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
            x=df['time'],
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
        fig_macd.add_trace(go.Scatter(x=df['time'], y=df['MACD'], name='MACD'))
        fig_macd.add_trace(go.Scatter(x=df['time'], y=df['Signal'], name='Signal'))
        fig_macd.update_layout(title='MACD', height=300)
        st.plotly_chart(fig_macd, use_container_width=True)
    
    # Data table
    with st.expander("📋 View Raw Data"):
        st.dataframe(df.tail(50))
    
    # TradingView detailed indicators
    if tv_analysis:
        with st.expander("📊 TradingView Detailed Analysis"):
            try:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write("**Oscillators**")
                    osc = tv_analysis.get('oscillators', {})
                    st.write(f"Buy: {osc.get('BUY', 0)}")
                    st.write(f"Neutral: {osc.get('NEUTRAL', 0)}")
                    st.write(f"Sell: {osc.get('SELL', 0)}")
                with col2:
                    st.write("**Moving Averages**")
                    ma = tv_analysis.get('moving_averages', {})
                    st.write(f"Buy: {ma.get('BUY', 0)}")
                    st.write(f"Neutral: {ma.get('NEUTRAL', 0)}")
                    st.write(f"Sell: {ma.get('SELL', 0)}")
                with col3:
                    st.write("**Summary**")
                    summary = tv_analysis.get('summary', {})
                    st.write(f"Recommendation: {summary.get('RECOMMENDATION', 'N/A')}")
            except Exception as e:
                st.write(f"Technical data: {str(tv_analysis)[:200]}...")
    
    # Footer
    st.markdown("---")
    st.markdown("**Data Source:** Yahoo Finance & TradingView | **Disclaimer:** Educational purposes only. Not financial advice.")

if __name__ == "__main__":
    main()
