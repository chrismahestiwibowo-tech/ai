"""
Banking Bot using Mistral Large Model
Handles customer banking queries with AI assistance
"""

import os
from mistralai import Mistral

# Initialize Mistral client
API_KEY = "UWChdZ7ltO5yxNpAtUb8yDknna4CWWIw"
client = Mistral(api_key=API_KEY)

# Mock banking data (in production, this would connect to a real database)
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

Keep responses concise and friendly."""


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
        
        return context
    return None


def chat_with_bot(messages, account_number=None):
    """Send messages to Mistral and get response"""
    
    # Add system prompt
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add customer context if available
    if account_number:
        context = get_customer_context(account_number)
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


def main():
    """Main banking bot interaction loop"""
    print("=" * 60)
    print("🏦 Welcome to AI Banking Assistant")
    print("=" * 60)
    print("\nI'm here to help you with your banking needs!")
    print("\nOptions:")
    print("1. Login with account number (for personalized service)")
    print("2. Continue as guest (general banking questions)")
    print("3. Type 'quit' to exit\n")
    
    account_number = None
    choice = input("Your choice (1/2): ").strip()
    
    if choice == "1":
        account_number = input("Enter your account number: ").strip()
        if account_number in CUSTOMER_DATA:
            print(f"\n✅ Welcome back, {CUSTOMER_DATA[account_number]['name']}!")
        else:
            print("\n⚠️ Account not found. Continuing as guest.")
            account_number = None
    
    print("\n" + "=" * 60)
    print("You can ask me about:")
    print("• Account balance and transactions")
    print("• Banking products and services")
    print("• Security tips and fraud prevention")
    print("• General banking questions")
    print("\nType 'quit' to exit")
    print("=" * 60 + "\n")
    
    # Conversation history
    messages = []
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\n👋 Thank you for using AI Banking Assistant. Have a great day!")
            break
        
        if not user_input:
            continue
        
        # Add user message to history
        messages.append({"role": "user", "content": user_input})
        
        try:
            # Get bot response
            print("\n🤖 Assistant: ", end="", flush=True)
            response = chat_with_bot(messages, account_number)
            print(response)
            print()
            
            # Add assistant response to history
            messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again.\n")
            # Remove the last user message if there was an error
            messages.pop()


if __name__ == "__main__":
    main()
