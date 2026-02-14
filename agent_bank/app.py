import streamlit as st
from banking_bot import BankingBot
import os

st.set_page_config(
    page_title="HBDB Banking Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Simple CSS
st.markdown("""
<style>
    .main { background: #fff; }
    .header { text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
              color: white; border-radius: 10px; margin-bottom: 2rem; }
    .msg-box { padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid; }
    .user-msg { background: #e3f2fd; border-color: #2196f3; color: #1565c0; }
    .bot-msg { background: #f3e5f5; border-color: #9c27b0; color: #6a1b9a; }
</style>
""", unsafe_allow_html=True)

# Session state
if 'bot' not in st.session_state:
    st.session_state.bot = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# Header
st.markdown("<div class='header'><h1>🏦 HBDB Banking Assistant</h1><p>Your AI-Powered Banking Support</p></div>", unsafe_allow_html=True)

# Initialize bot
if not st.session_state.initialized:
    try:
        api_key = os.getenv("MISTRAL_API_KEY", "hKjvYtwfSKR7Ysd7WKvmItCtPL6YfjdR")
        st.session_state.bot = BankingBot(api_key, "hbdb_banking_faqs.csv")
        st.session_state.initialized = True
        st.session_state.messages.append({
            "role": "bot",
            "content": "👋 Welcome! I'm the HBDB Banking Assistant. How can I help you today?"
        })
    except Exception as e:
        st.error(f"❌ Failed to initialize: {str(e)}")

# Layout
col1, col2 = st.columns([1, 3])

# Sidebar
with col1:
    st.markdown("### ⚙️ Settings")
    st.metric("Messages", len(st.session_state.messages))
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.bot:
            st.session_state.bot.clear_history()
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📚 Topics")
    
    topics = [
        ("💳 Credit Card", "How do I apply for a credit card?"),
        ("💰 Savings", "How do I open a savings account?"),
        ("📱 Mobile", "How do I download the mobile app?"),
        ("🔐 Security", "What is HBDB Secure Key?"),
    ]
    
    for title, question in topics:
        if st.button(title, use_container_width=True, key=f"topic_{title}"):
            st.session_state.messages.append({"role": "user", "content": question})
            if st.session_state.bot:
                response = st.session_state.bot.get_response(question)
                st.session_state.messages.append({"role": "bot", "content": response})
            st.rerun()

# Chat area
with col2:
    st.markdown("### 💬 Chat")
    
    # Display messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"<div class='msg-box user-msg'><b>👤 You:</b><br>{message['content']}</div>", unsafe_allow_html=True)
        else:
            # Bot messages - render with markdown support for **bold**, lists, etc.
            st.markdown(f"<div class='msg-box bot-msg'><b>🤖 Assistant:</b></div>", unsafe_allow_html=True)
            st.markdown(message['content'])
    
    st.markdown("---")
    
    # Input
    st.markdown("### 📝 Ask a Question")
    
    col_input, col_btn = st.columns([5, 1])
    
    with col_input:
        user_input = st.text_input(
            "Your question:",
            placeholder="E.g., How do I open an account?",
            label_visibility="collapsed"
        )
    
    with col_btn:
        send_btn = st.button("Send ➤", use_container_width=True)
    
    # Process input
    if send_btn and user_input and user_input.strip():
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get bot response
        if st.session_state.bot:
            with st.spinner("Processing..."):
                try:
                    response = st.session_state.bot.get_response(user_input)
                    st.session_state.messages.append({"role": "bot", "content": response})
                except Exception as e:
                    st.session_state.messages.append({
                        "role": "bot",
                        "content": f"Sorry, I encountered an error: {str(e)}"
                    })
        else:
            st.error("Bot not initialized")
        
        st.rerun()
