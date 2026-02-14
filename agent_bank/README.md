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

Set your Mistral AI API key in `app.py`:
```python
api_key = "your_mistral_api_key_here"
```

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

- streamlit
- pandas
- mistralai
- python 3.8+

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
