import streamlit as st
import os
import requests
import json
from datetime import datetime
from openai import OpenAI
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Bitcoin Price Prediction Bot",
    page_icon="🚀",
    layout="wide"
)

# Initialize OpenAI client with Streamlit secrets
@st.cache_resource
def get_openai_client():
    """Initialize OpenAI client with API key from secrets"""
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        return OpenAI(api_key=api_key)
    except Exception as e:
        st.error("⚠️ OpenAI API key not configured. Please add OPENAI_API_KEY to your Streamlit secrets.")
        st.stop()

def get_btc_price_data():
    """Fetch current Bitcoin price and recent data from multiple sources"""
    
    # Try CoinGecko first
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            'ids': 'bitcoin',
            'vs_currencies': 'usd',
            'include_24hr_vol': 'true',
            'include_24hr_change': 'true',
            'include_market_cap': 'true'
        }
        response = requests.get(url, params=params, timeout=10, verify=False)
        current_data = response.json()['bitcoin']
        
        return {
            'current_price': current_data['usd'],
            'market_cap': current_data['usd_market_cap'],
            'volume_24h': current_data['usd_24h_vol'],
            'price_change_24h': current_data['usd_24h_change'],
            'history': None
        }
    except Exception as e:
        st.warning(f"⚠️ CoinGecko API failed: {str(e)[:100]}")
        
    # Try alternative API - CoinCap
    try:
        url = "https://api.coincap.io/v2/assets/bitcoin"
        response = requests.get(url, timeout=10)
        data = response.json()['data']
        
        return {
            'current_price': float(data['priceUsd']),
            'market_cap': float(data['marketCapUsd']),
            'volume_24h': float(data['volumeUsd24Hr']),
            'price_change_24h': float(data['changePercent24Hr']),
            'history': None
        }
    except Exception as e:
        st.warning(f"⚠️ CoinCap API failed: {str(e)[:100]}")
    
    # Try Binance API as last resort
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        params = {'symbol': 'BTCUSDT'}
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        return {
            'current_price': float(data['lastPrice']),
            'market_cap': 0,  # Not available from Binance
            'volume_24h': float(data['volume']) * float(data['lastPrice']),
            'price_change_24h': float(data['priceChangePercent']),
            'history': None
        }
    except Exception as e:
        st.error(f"❌ All APIs failed. Last error: {str(e)[:100]}")
        
    return None

def analyze_btc_with_ai(btc_data, client):
    """Use OpenAI to analyze Bitcoin data and make predictions"""
    
    # Prepare the market data summary
    market_summary = f"""
Current Bitcoin Market Data (as of {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}):
- Current Price: ${btc_data['current_price']:,.2f}
- 24h Price Change: {btc_data['price_change_24h']:.2f}%
- Market Cap: ${btc_data['market_cap']:,.0f}
- 24h Trading Volume: ${btc_data['volume_24h']:,.0f}

7-Day Price Trend:
Recent prices show {"an upward" if btc_data['price_change_24h'] > 0 else "a downward"} trend.
"""

    try:
        # Create the AI prediction request
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert cryptocurrency analyst with deep knowledge of Bitcoin markets, 
                    technical analysis, and market trends. Provide detailed, data-driven predictions while acknowledging 
                    the inherent volatility and risks in cryptocurrency markets."""
                },
                {
                    "role": "user",
                    "content": f"""Based on the following Bitcoin market data, provide a comprehensive analysis and price prediction:

{market_summary}

Please provide:
1. Technical Analysis: What do these indicators suggest?
2. Short-term Prediction (24-48 hours): Expected price movement and range
3. Medium-term Outlook (7 days): Potential price targets
4. Key Factors: What market factors should traders watch?
5. Risk Assessment: What are the major risks to your prediction?

Be specific with price targets but also note confidence levels and disclaimers about market volatility."""
                }
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        st.error(f"❌ OpenAI API Error: {str(e)}")
        return None

def main():
    """Main Streamlit app"""
    
    # Title and header
    st.title("🚀 Bitcoin Price Prediction Bot")
    st.markdown("### Powered by OpenAI GPT-5")
    st.divider()
    
    # Get OpenAI client
    client = get_openai_client()
    
    # Create columns for layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📊 Market Data")
        
        # Add a button to fetch data
        if st.button("🔄 Get Latest Analysis", type="primary", use_container_width=True):
            with st.spinner("Fetching Bitcoin data..."):
                btc_data = get_btc_price_data()
                
                if btc_data:
                    st.session_state.btc_data = btc_data
                    st.session_state.analysis_ready = True
                else:
                    st.error("Failed to fetch Bitcoin data. Please try again.")
                    st.session_state.analysis_ready = False
        
        # Display current data if available
        if hasattr(st.session_state, 'btc_data') and st.session_state.btc_data:
            data = st.session_state.btc_data
            
            st.metric(
                "Current Price",
                f"${data['current_price']:,.2f}",
                f"{data['price_change_24h']:.2f}%"
            )
            
            if data['market_cap'] > 0:
                st.metric(
                    "Market Cap",
                    f"${data['market_cap']:,.0f}"
                )
            
            st.metric(
                "24h Volume",
                f"${data['volume_24h']:,.0f}"
            )
            
            st.info(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    
    with col2:
        st.subheader("🤖 AI Analysis & Prediction")
        
        # Generate AI analysis if data is ready
        if hasattr(st.session_state, 'analysis_ready') and st.session_state.analysis_ready:
            with st.spinner("Analyzing with OpenAI GPT-5..."):
                prediction = analyze_btc_with_ai(st.session_state.btc_data, client)
                
                if prediction:
                    st.markdown(prediction)
                    st.session_state.analysis_ready = False  # Reset flag
        else:
            st.info("👈 Click 'Get Latest Analysis' to fetch Bitcoin data and generate AI predictions")
    
    # Footer with disclaimer
    st.divider()
    st.warning("""
    ⚠️ **DISCLAIMER**: This is an AI-generated prediction for educational purposes only.
    Not financial advice. Cryptocurrency investments carry high risk. Always do your own research
    and consult with financial professionals before making investment decisions.
    """)

if __name__ == "__main__":
    main()
