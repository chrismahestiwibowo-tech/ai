# 🏦 AI Banking Assistant

An intelligent banking chatbot powered by Mistral AI Large Language Model, providing personalized banking assistance with a beautiful web interface.

## 🌟 Features

### 🎨 User-Friendly Web Interface
- Beautiful gradient design with intuitive layout
- Responsive chat interface
- Quick action buttons for common banking tasks
- Real-time account information display
- Dark mode compatible

### 🔐 Security & Access
- Secure account login system
- Guest mode for general inquiries
- Demo accounts for testing
- Session management with logout functionality

### 💡 Capabilities
- ✅ Check account balance instantly
- ✅ View transaction history
- ✅ Get banking product information
- ✅ Security tips and fraud prevention
- ✅ General banking inquiries
- ✅ AI-powered intelligent responses

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Mistral AI API key

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/chrismahestiwibowo-tech/ai.git
cd ai/agen_bank
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the web interface**
```bash
streamlit run banking_bot_web.py
```

The application will open in your browser at `http://localhost:8501`

### Alternative: Command Line Version
```bash
python banking_bot.py
```

## 📝 Usage

### Demo Accounts
For testing purposes, use these demo accounts:

| Account Number | Name | Balance | Type |
|---------------|------|---------|------|
| 12345 | John Doe | $5,000.00 | Savings |
| 67890 | Jane Smith | $7,500.00 | Checking |

### Features Available

1. **Login Mode**
   - Enter account number for personalized service
   - View account balance and transaction history
   - Get account-specific assistance

2. **Guest Mode**
   - General banking questions
   - Product information
   - Security tips
   - No login required

### Quick Actions
- 💵 Check Balance
- 📊 Recent Transactions
- 🔒 Security Tips

## 🛠️ Technology Stack

- **AI Model**: Mistral Large (latest)
- **Web Framework**: Streamlit
- **API Client**: Mistral AI Python SDK
- **Language**: Python 3.8+

## 📦 Project Structure

```
agen_bank/
├── banking_bot_web.py      # Web interface with Streamlit
├── banking_bot.py          # Command-line version
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🔧 Configuration

The Mistral AI API key is configured in the Python files. For production use, it's recommended to use environment variables:

```python
import os
API_KEY = os.getenv("MISTRAL_API_KEY")
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Chris Mahesti Wibowo**
- GitHub: [@chrismahestiwibowo-tech](https://github.com/chrismahestiwibowo-tech)
- Email: chrismahestiwibowo.ae@gmail.com

## 🙏 Acknowledgments

- Powered by [Mistral AI](https://mistral.ai/)
- Built with [Streamlit](https://streamlit.io/)

---

**⚠️ Note**: This is a demo application with mock data. For production use, integrate with a real banking database and implement proper security measures.
