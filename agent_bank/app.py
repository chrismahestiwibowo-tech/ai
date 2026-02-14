import streamlit as st
import pandas as pd
from banking_bot import BankingBot
import os
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="HBDB Banking Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.hbdb.com/support',
        'Report a bug': 'https://www.hbdb.com/feedback',
        'About': '# HBDB Banking Bot\nPowered by Mistral AI'
    }
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Chat messages */
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        word-wrap: break-word;
        line-height: 1.6;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        color: #1565c0;
    }
    
    .user-message h1, .user-message h2, .user-message h3, .user-message h4, .user-message h5, .user-message h6 {
        color: #1565c0;
    }
    
    .user-message p, .user-message li {
        color: #1565c0;
    }
    
    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
        color: #6a1b9a;
    }
    
    .bot-message h1, .bot-message h2, .bot-message h3, .bot-message h4, .bot-message h5, .bot-message h6 {
        color: #6a1b9a;
        margin-top: 0.5rem;
        margin-bottom: 0.3rem;
    }
    
    .bot-message p, .bot-message li {
        color: #6a1b9a;
        margin: 0.3rem 0;
    }
    
    .bot-message strong {
        color: #6a1b9a;
        font-weight: 700;
    }
    
    .error-message {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        color: #c62828;
    }
    
    .error-message p {
        color: #c62828;
    }
    
    .info-message {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        color: #2e7d32;
    }
    
    .info-message p {
        color: #2e7d32;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Input field */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #667eea;
    }
    
    /* Header styling */
    .header-container {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .header-container h1 {
        margin: 0;
        font-size: 2.5rem;
    }
    
    .header-container p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Sidebar styling */
    .sidebar-info {
        background-color: #f0f4ff;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid #667eea;
    }
    
    /* Status badge */
    .status-badge {
        display: inline-block;
        background-color: #4caf50;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        margin-left: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'bot' not in st.session_state:
        st.session_state.bot = None
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'bot_initialized' not in st.session_state:
        st.session_state.bot_initialized = False
    
    if 'api_key_set' not in st.session_state:
        st.session_state.api_key_set = False

def init_bot():
    """Initialize the banking bot"""
    try:
        csv_path = "hbdb_banking_faqs.csv"
        
        # Check if CSV file exists
        if not os.path.exists(csv_path):
            st.error(f"❌ FAQ data file not found: {csv_path}")
            return False
        
        api_key = "hKjvYtwfSKR7Ysd7WKvmItCtPL6YfjdR"
        
        st.session_state.bot = BankingBot(api_key, csv_path)
        st.session_state.bot_initialized = True
        st.session_state.api_key_set = True
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error initializing bot: {str(e)}")
        return False

def display_message(role, content):
    """Display a message in the chat"""
    if role == "user":
        message_class = "user-message"
        icon = "👤"
        label = "You"
    elif role == "error":
        message_class = "error-message"
        icon = "⚠️"
        label = "System Error"
    elif role == "info":
        message_class = "info-message"
        icon = "ℹ️"
        label = "System Info"
    else:  # bot
        message_class = "bot-message"
        icon = "🤖"
        label = "HBDB Assistant"
    
    with st.container():
        col_icon, col_content = st.columns([0.08, 0.92])
        
        with col_icon:
            st.markdown(f"<div style='font-size: 1.5rem; margin-top: 0.5rem;'>{icon}</div>", unsafe_allow_html=True)
        
        with col_content:
            st.markdown(f"""
            <div class="chat-message {message_class}">
                <div style="font-weight: bold; margin-bottom: 0.5rem; color: inherit;">{label}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Render content with markdown support for bot and info messages
            if role in ["bot", "info"]:
                st.markdown(content)
            else:
                st.markdown(f"<div style='color: inherit;'>{content}</div>", unsafe_allow_html=True)

def main():
    """Main Streamlit app"""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="header-container">
        <h1>🏦 HBDB Banking Assistant</h1>
        <p>Your AI-Powered Banking Support</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings & Info")
        
        # Initialize bot on first load
        if not st.session_state.bot_initialized:
            with st.spinner("🔄 Initializing HBDB Banking Assistant..."):
                time.sleep(1)
                if init_bot():
                    st.success("✅ Bot initialized successfully!")
                    st.session_state.messages.append({
                        "role": "info",
                        "content": "Welcome to HBDB Banking Assistant! I'm here to help you with banking questions. How can I assist you today?"
                    })
                else:
                    st.error("Failed to initialize bot. Please refresh the page.")
        
        # Bot status
        if st.session_state.bot_initialized:
            st.markdown('<div class="status-badge">Active</div>', unsafe_allow_html=True)
            st.markdown("**Bot Status:** Online")
        else:
            st.warning("Bot Status: Initializing...")
        
        # Divider
        st.divider()
        
        # Session info
        st.markdown("### 📊 Session Info")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", len(st.session_state.messages))
        with col2:
            st.metric("Time", datetime.now().strftime("%H:%M"))
        
        # Actions
        st.markdown("### 🎯 Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                if st.session_state.bot:
                    st.session_state.bot.clear_history()
                st.success("Chat cleared!")
                st.rerun()
        
        with col2:
            if st.button("🔄 New Session", use_container_width=True):
                st.session_state.messages = []
                st.session_state.bot = None
                st.session_state.bot_initialized = False
                st.success("New session started!")
                st.rerun()
        
        # FAQ Preview
        st.markdown("### 📚 Quick Links")
        
        faq_samples = [
            ("💳 Credit Cards", "How do I apply for a credit card?"),
            ("💰 Savings Account", "How do I open a savings account?"),
            ("📱 Mobile App", "How do I download HBDB Mobile Banking app?"),
            ("🔐 Security", "What is HBDB Secure Key?"),
            ("💸 Transfers", "How do I make a wire transfer?"),
        ]
        
        for emoji_title, question in faq_samples:
            if st.button(emoji_title, use_container_width=True, key=question):
                st.session_state.user_input = question
                st.rerun()
        
        # About section
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info(
            "**HBDB Banking Assistant** is powered by Mistral AI's language model. "
            "Get instant answers to your banking questions anytime!"
        )
    
    # Main chat area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 💬 Chat with HBDB Assistant")
    
    with col2:
        theme = st.selectbox("Theme", ["Light", "Dark"], label_visibility="collapsed", key="theme_select")
    
    # Chat history display
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            display_message(message["role"], message["content"])
    
    # Input area with visual separation
    st.divider()
    
    # Input form
    input_col1, input_col2 = st.columns([4, 1])
    
    user_input = None
    
    with input_col1:
        user_input = st.text_input(
            "Ask your banking question...",
            placeholder="E.g., How do I open a savings account? or What's the SWIFT code?",
            key="user_input"
        )
    
    with input_col2:
        send_button = st.button("Send", use_container_width=True, key="send_button")
    
    # Process user input
    if send_button and user_input:
        # Validate input
        if len(user_input.strip()) == 0:
            st.warning("⚠️ Please enter a question before sending.")
        elif len(user_input) > 5000:
            st.error("❌ Message is too long. Please keep it under 5000 characters.")
        else:
            # Add user message to history
            st.session_state.messages.append({
                "role": "user",
                "content": user_input
            })
            
            # Get bot response
            if st.session_state.bot_initialized and st.session_state.bot:
                with st.spinner("🔄 Processing your question..."):
                    try:
                        response = st.session_state.bot.get_response(user_input)
                        st.session_state.messages.append({
                            "role": "bot",
                            "content": response
                        })
                        st.rerun()
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.session_state.messages.append({
                            "role": "error",
                            "content": error_msg
                        })
                        st.rerun()
            else:
                st.error("❌ Bot not initialized. Please refresh the page.")
    
    # Tips section
    if len(st.session_state.messages) == 0:
        st.markdown("---")
        st.markdown("### 💡 Tips for Better Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **✅ Do:**
            - Ask specific questions
            - Include relevant details
            - Ask one question at a time
            """)
        
        with col2:
            st.markdown("""
            **❌ Don't:**
            - Use offensive language
            - Ask for personal account details
            - Share passwords or PINs
            """)
        
        with col3:
            st.markdown("""
            **🎯 Examples:**
            - How do I reset my password?
            - What is HBDB Premier?
            - Can I apply for a loan?
            """)

if __name__ == "__main__":
    main()
