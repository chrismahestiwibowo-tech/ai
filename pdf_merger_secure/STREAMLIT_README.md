# 📄 PDF Merger Secure

A sleek, secure PDF merging application built with Streamlit. Combine multiple PDF files into one document effortlessly!

## ✨ Features

- 📤 Upload multiple PDF files at once
- 🔄 Arrange files in custom order
- 🔗 Merge PDFs into a single document
- 📥 Download merged PDF instantly
- 🎨 Beautiful, user-friendly interface
- 🔒 Secure file handling
- 💨 Fast processing with streaming

## 🚀 Quick Start

### Local Setup

```bash
# Clone the repository
git clone https://github.com/chrismahestiwibowo-tech/ai.git
cd ai/pdf_merger_secure

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

Visit `http://localhost:8501` in your browser.

### Deploy to Streamlit Cloud

1. **Push to GitHub** (already done ✅)

2. **Go to [Streamlit Cloud](https://share.streamlit.io/)**

3. **Create New App:**
   - Select repository: `chrismahestiwibowo-tech/ai`
   - Select branch: `main`
   - Set main file path: `pdf_merger_secure/streamlit_app.py`

4. **Deploy** and share your link!

## 📋 Requirements

- Python 3.8+
- Streamlit 1.28.1+
- PyPDF 3.17.1+

## 📁 Project Structure

```
pdf_merger_secure/
├── streamlit_app.py          # Main Streamlit app
├── app.py                    # Flask app (alternative)
├── pdf_processor.py          # PDF processing logic
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── .gitignore               # Git ignore rules
├── static/                  # Static files (CSS, JS)
├── templates/               # HTML templates (Flask)
└── uploads/                 # Temporary uploads
```

## 🛠️ How It Works

1. **Upload**: Select one or more PDF files
2. **Arrange**: Use arrow buttons to reorder files
3. **Merge**: Click the merge button to combine PDFs
4. **Download**: Get your merged PDF instantly

## 🔒 Security

- Files are processed locally
- Temporary files are cleaned up automatically
- No data is stored permanently
- Secure filename handling

## 📧 Contact

- **Email**: chrismahestiwibowo.ae@gmail.com
- **GitHub**: [@chrismahestiwibowo-tech](https://github.com/chrismahestiwibowo-tech)

## 📄 License

MIT License - Feel free to use and modify!

---

**Made with ❤️ using Streamlit**
