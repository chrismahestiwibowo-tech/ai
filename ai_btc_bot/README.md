# Bitcoin Price Prediction Bot 🚀

An intelligent Bitcoin price prediction bot powered by OpenAI GPT-4 that provides real-time market analysis and price predictions.

## Features

- 📊 **Real-time Data**: Fetches live Bitcoin market data from multiple sources (CoinGecko, CoinCap, Binance)
- 🤖 **AI Analysis**: Uses OpenAI GPT-4 for intelligent market analysis
- 📈 **Price Predictions**: Provides short-term (24-48h) and medium-term (7-day) predictions
- 💡 **Technical Analysis**: Detailed technical indicators and market insights
- ⚠️ **Risk Assessment**: Identifies key factors and potential risks
- 🔄 **Fallback APIs**: Automatically switches to alternative data sources if one fails

## Installation

1. Clone the repository:
```bash
git clone https://github.com/chrismahestiwibowo-tech/ai.git
cd ai/ai_btc_bot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your OpenAI API key:
- Copy `.env.example` to `.env`
- Add your OpenAI API key to the `.env` file:
```
OPENAI_API_KEY=your_actual_api_key_here
```

## Usage

Run the bot:
```bash
python btc_prediction_bot.py
```

The bot will:
1. Fetch current Bitcoin market data
2. Analyze the data using OpenAI GPT-4
3. Display comprehensive predictions and analysis

## Output Example

```
================================================================================
🚀 Bitcoin Price Prediction Bot (Powered by OpenAI)
================================================================================

📊 Fetching current Bitcoin market data...
✅ Current BTC Price: $88,136.00
📈 24h Change: 0.30%

🤖 Analyzing data with OpenAI GPT-4...

================================================================================
📝 AI ANALYSIS & PREDICTION:
================================================================================

1. Technical Analysis: [Detailed analysis...]
2. Short-term Prediction (24-48 hours): [Price predictions...]
3. Medium-term Outlook (7 days): [Weekly outlook...]
4. Key Factors: [Market factors to watch...]
5. Risk Assessment: [Risk analysis...]
```

## Requirements

- Python 3.7+
- OpenAI API key
- Internet connection for API calls

## Dependencies

- `openai>=1.0.0` - OpenAI API client
- `requests>=2.31.0` - HTTP library
- `python-dotenv>=1.0.0` - Environment variable management
- `pandas>=2.0.0` - Data manipulation (optional)

## Disclaimer

⚠️ **Important**: This bot is for educational and informational purposes only. The predictions are AI-generated and should not be considered as financial advice. Cryptocurrency investments carry high risk. Always do your own research and consult with financial professionals before making investment decisions.

## License

MIT License - Feel free to use and modify as needed.

## Author

Created by chrismahestiwibowo-tech

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.
