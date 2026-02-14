# HBDB Banking Assistant Bot

An intelligent AI-powered banking assistant built with Streamlit and Mistral AI that provides instant answers to banking questions using a comprehensive FAQ database.

## Features

- 🤖 **AI-Powered Responses** - Powered by Mistral Large language model
- 💬 **Conversational Chat** - Multi-turn conversations with context awareness
- 📚 **FAQ Database** - Built on HBDB banking FAQs
- 🎯 **Quick Links** - Pre-built buttons for common banking topics
- ✨ **Beautiful UI** - Dark theme with gradient design and smooth animations
- 📊 **Session Management** - Track conversations and clear chat history
- 🔐 **Professional Design** - Foolproof and attractive interface

## Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/chrismahestiwibowo-tech/ai.git
cd ai/agent_bank
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install streamlit pandas mistralai
```

### Configuration

**Set your Mistral AI API key using environment variables:**

1. Create a `.env` file (copy from `.env.example`):
```bash
cp .env.example .env
```

2. Add your Mistral API key:
```
MISTRAL_API_KEY=your_mistral_api_key_here
```

3. For local development, the app also has a fallback key. For production, always use environment variables.

**Get your API key:**
- Visit https://console.mistral.ai/
- Create an account or sign in
- Generate an API key
- Copy it to your `.env` file

### Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Project Structure

```
agent_bank/
├── app.py                    # Streamlit web interface
├── banking_bot.py            # Core bot logic with Mistral AI integration
├── hbdb_banking_faqs.csv     # FAQ database
└── README.md                 # This file
```

## Usage

1. Open the app in your browser
2. Ask banking questions about HBDB services
3. Use Quick Links for common topics
4. Clear chat or start new sessions as needed

### Example Questions
- "How do I open a savings account?"
- "What is HBDB Premier?"
- "How do I apply for a credit card?"
- "What's the SWIFT code?"

## Files Description

### app.py
- Streamlit web interface with chat UI
- Custom CSS styling for attractive design
- Session state management
- Message handling and display logic

### banking_bot.py
- Core banking bot class
- Integration with Mistral AI API
- FAQ data loading and formatting
- Conversation history management

### hbdb_banking_faqs.csv
- CSV file containing 50+ Q&A pairs
- Banking-related frequently asked questions
- Used as context for the AI model

## Requirements

- streamlit >= 1.28.0
- pandas >= 2.0.0
- mistralai >= 0.0.11
- python-dotenv >= 1.0.0

## Deployment

### Streamlit Cloud Deployment

1. Push your code to GitHub
2. Go to https://share.streamlit.io/
3. Sign in with your GitHub account
4. Click "New app"
5. Select your repository and branch
6. Set the main file to `agent_bank/app.py`
7. Click "Deploy"
8. In the app settings, add your secrets:
   - Go to Settings → Secrets
   - Add `MISTRAL_API_KEY = "your_api_key_here"`

### Docker Deployment

Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py"]
```

Build and run:
```bash
docker build -t hbdb-banking-bot .
docker run -e MISTRAL_API_KEY=your_key -p 8501:8501 hbdb-banking-bot
```

## Troubleshooting

### Requirements Installation Error
- Update package versions to latest compatible versions
- Try: `pip install --upgrade pip setuptools wheel`

### API Key Issues
- Verify `MISTRAL_API_KEY` environment variable is set
- Check your API key is valid at https://console.mistral.ai/
- For local development, the app has a fallback key

### CSV File Not Found
- Ensure `hbdb_banking_faqs.csv` is in the same directory as `app.py`
- Check file permissions

## Author

**chrismahestiwibowo-tech**  
Email: chrismahestiwibowo.ae@gmail.com

## License

MIT License - Feel free to use this project for educational purposes.

## Support

For issues or questions about the banking data, contact HBDB customer service.

## Deployment

For production deployment:
1. Update API key to use environment variables
2. Deploy using Streamlit Cloud or Docker
3. Configure CI/CD pipeline for automated deployments
