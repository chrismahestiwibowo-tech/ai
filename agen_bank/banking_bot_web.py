"""
Banking Bot Web Interface using Mistral Large Model
A user-friendly web interface for banking assistance
"""

import streamlit as st
from mistralai import Mistral
from datetime import datetime

# Initialize Mistral client
API_KEY = "UWChdZ7ltO5yxNpAtUb8yDknna4CWWIw"
client = Mistral(api_key=API_KEY)

# Mock banking data
CUSTOMER_DATA = {
    "12345": {
        "name": "John Doe",
        "balance": 5000.00,
        "account_type": "Savings",
        "transactions": [
            {"date": "2026-02-05", "type": "deposit", "amount": 1000.00, "description": "Salary"},
            {"date": "2026-02-04", "type": "withdrawal", "amount": 200.00, "description": "ATM Withdrawal"},
            {"date": "2026-02-01", "type": "deposit", "amount": 500.00, "description": "Transfer from Jane"}
        ]
    },
    "67890": {
        "name": "Jane Smith",
        "balance": 7500.00,
        "account_type": "Checking",
        "transactions": [
            {"date": "2026-02-06", "type": "deposit", "amount": 2000.00, "description": "Freelance Payment"},
            {"date": "2026-02-03", "type": "withdrawal", "amount": 150.00, "description": "Online Purchase"},
        ]
    }
}

SYSTEM_PROMPT = """You are a helpful banking assistant chatbot. You help customers with:
- Checking account balances
- Viewing transaction history
- Understanding banking products and services
- General banking inquiries
- Security and fraud prevention tips

Always be professional, clear, and security-conscious. Never ask for or store sensitive information like passwords or full card numbers.
If you're provided with customer data, use it to answer questions. Otherwise, provide general banking guidance.

Keep responses concise, friendly, and helpful. Use emojis occasionally to be more engaging."""


def get_customer_context(account_number):
    """Get customer data if account number is valid"""
    if account_number in CUSTOMER_DATA:
        customer = CUSTOMER_DATA[account_number]
        context = f"""
Customer Information:
- Name: {customer['name']}
- Account Type: {customer['account_type']}
- Current Balance: ${customer['balance']:.2f}

Recent Transactions:
"""
        for txn in customer['transactions']:
            context += f"- {txn['date']}: {txn['type'].title()} ${txn['amount']:.2f} - {txn['description']}\n"
        
        return context, customer
    return None, None


def chat_with_bot(messages, account_number=None):
    """Send messages to Mistral and get response"""
    
    # Add system prompt
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add customer context if available
    if account_number:
        context, _ = get_customer_context(account_number)
        if context:
            full_messages.append({
                "role": "system", 
                "content": f"Current customer context:\n{context}"
            })
    
    # Add conversation history
    full_messages.extend(messages)
    
    # Call Mistral API
    response = client.chat.complete(
        model="mistral-large-latest",
        messages=full_messages
    )
    
    return response.choices[0].message.content


# Page configuration
st.set_page_config(
    page_title="AI Banking Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .account-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: white;
    }
    .balance-display {
        font-size: 2rem;
        font-weight: bold;
        color: #FFD700;
    }
    /* Fix chat message visibility */
    .stChatMessage {
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    /* Ensure text is visible in dark mode */
    [data-testid="stChatMessageContent"] {
        color: inherit !important;
    }
    [data-testid="stChatMessageContent"] p {
        color: inherit !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "account_number" not in st.session_state:
    st.session_state.account_number = None
if "customer_name" not in st.session_state:
    st.session_state.customer_name = None

# Header
st.markdown("""
    <div class="main-header">
        <h1>🏦 AI Banking Assistant</h1>
        <p>Powered by Mistral AI - Your intelligent banking companion</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar for account management
with st.sidebar:
    st.header("Account Access")
    
    if not st.session_state.logged_in:
        st.info("💡 **Demo Accounts:**\n\n12345 - John Doe\n\n67890 - Jane Smith")
        
        login_option = st.radio("Choose access mode:", ["Guest Mode", "Login with Account"])
        
        if login_option == "Login with Account":
            account_input = st.text_input("Enter Account Number:", type="password")
            if st.button("🔐 Login", type="primary"):
                context, customer = get_customer_context(account_input)
                if customer:
                    st.session_state.logged_in = True
                    st.session_state.account_number = account_input
                    st.session_state.customer_name = customer['name']
                    st.session_state.messages = []
                    st.success(f"✅ Welcome, {customer['name']}!")
                    st.rerun()
                else:
                    st.error("❌ Invalid account number. Please try again.")
        else:
            if st.button("Continue as Guest", type="primary"):
                st.session_state.logged_in = True
                st.session_state.account_number = None
                st.session_state.customer_name = "Guest"
                st.session_state.messages = []
                st.rerun()
    else:
        st.success(f"👤 **{st.session_state.customer_name}**")
        
        # Display account info for logged-in users
        if st.session_state.account_number:
            context, customer = get_customer_context(st.session_state.account_number)
            if customer:
                st.markdown(f"""
                    <div class="account-card">
                        <h3>Account Summary</h3>
                        <p><strong>Type:</strong> {customer['account_type']}</p>
                        <p class="balance-display">${customer['balance']:.2f}</p>
                        <p style="font-size: 0.9rem; color: #666;">Current Balance</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with st.expander("📊 Recent Transactions"):
                    for txn in customer['transactions']:
                        icon = "💰" if txn['type'] == "deposit" else "💸"
                        st.markdown(f"""
                            {icon} **{txn['date']}**  
                            {txn['description']}: ${txn['amount']:.2f}
                        """)
        
        if st.button("🚪 Logout", type="secondary"):
            st.session_state.logged_in = False
            st.session_state.account_number = None
            st.session_state.customer_name = None
            st.session_state.messages = []
            st.rerun()
    
    st.divider()
    st.markdown("""
        ### 💡 What I Can Help With:
        - 💵 Check account balance
        - 📜 View transactions
        - 🏦 Banking products info
        - 🔒 Security tips
        - ❓ General questions
    """)

# Main chat interface
if st.session_state.logged_in:
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("💬 Ask me anything about banking..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    response = chat_with_bot(
                        st.session_state.messages,
                        st.session_state.account_number
                    )
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
    
    # Quick action buttons
    if len(st.session_state.messages) == 0:
        st.subheader("🚀 Quick Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💵 Check Balance", use_container_width=True):
                prompt = "What's my current account balance?"
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.rerun()
        
        with col2:
            if st.button("📊 Recent Transactions", use_container_width=True):
                prompt = "Show me my recent transactions"
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.rerun()
        
        with col3:
            if st.button("🔒 Security Tips", use_container_width=True):
                prompt = "Give me some banking security tips"
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.rerun()
    
    # Clear chat button
    if len(st.session_state.messages) > 0:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

else:
    # Welcome screen
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            ### 👋 Welcome!
            
            Please use the sidebar to:
            - **Login** with your account number for personalized service
            - **Continue as Guest** for general banking questions
            
            Your security and privacy are our top priorities! 🔒
        """)

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        🤖 Powered by Mistral AI Large Model | Made with ❤️ using Streamlit
    </div>
    """, unsafe_allow_html=True)
