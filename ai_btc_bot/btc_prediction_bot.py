import os
import requests
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError(
        "OpenAI API key not found! Please set OPENAI_API_KEY environment variable.\n"
        "You can do this by:\n"
        "1. Creating a .env file with: OPENAI_API_KEY=your_key_here\n"
        "2. Or setting it as an environment variable in your system/deployment platform"
    )
client = OpenAI(api_key=api_key)

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
        print(f"⚠️  CoinGecko API failed: {e}")
        
    # Try alternative API - CoinCap
    try:
        print("Trying alternative source (CoinCap)...")
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
        print(f"⚠️  CoinCap API failed: {e}")
    
    # Try Binance API as last resort
    try:
        print("Trying Binance API...")
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
        print(f"⚠️  Binance API failed: {e}")
        
    return None

def analyze_btc_with_ai(btc_data):
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
            model="gpt-4",
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
        print(f"Error with OpenAI API: {e}")
        return None

def main():
    """Main function to run the BTC prediction bot"""
    print("=" * 80)
    print("🚀 Bitcoin Price Prediction Bot (Powered by OpenAI)")
    print("=" * 80)
    print()
    
    # Fetch Bitcoin data
    print("📊 Fetching current Bitcoin market data...")
    btc_data = get_btc_price_data()
    
    if not btc_data:
        print("❌ Failed to fetch Bitcoin data. Please try again later.")
        return
    
    print(f"✅ Current BTC Price: ${btc_data['current_price']:,.2f}")
    print(f"📈 24h Change: {btc_data['price_change_24h']:.2f}%")
    print()
    
    # Get AI analysis
    print("🤖 Analyzing data with OpenAI GPT-4...")
    print()
    
    prediction = analyze_btc_with_ai(btc_data)
    
    if prediction:
        print("=" * 80)
        print("📝 AI ANALYSIS & PREDICTION:")
        print("=" * 80)
        print()
        print(prediction)
        print()
        print("=" * 80)
        print("⚠️  DISCLAIMER: This is an AI-generated prediction for educational purposes only.")
        print("    Not financial advice. Cryptocurrency investments carry high risk.")
        print("=" * 80)
    else:
        print("❌ Failed to generate prediction. Please check your OpenAI API key.")

if __name__ == "__main__":
    main()
