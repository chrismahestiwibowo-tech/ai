import pandas as pd
from mistralai import Mistral
import os

class BankingBot:
    """Banking FAQ Bot using Mistral AI"""
    
    def __init__(self, api_key, csv_path):
        """
        Initialize the banking bot
        
        Args:
            api_key (str): Mistral AI API key
            csv_path (str): Path to the CSV file with FAQs
        """
        self.api_key = api_key
        self.client = Mistral(api_key=api_key)
        self.model = "mistral-large-latest"
        self.faq_data = self._load_faqs(csv_path)
        self.conversation_history = []
        
    def _load_faqs(self, csv_path):
        """Load FAQ data from CSV"""
        try:
            df = pd.read_csv(csv_path)
            return df
        except Exception as e:
            raise Exception(f"Error loading FAQ data: {str(e)}")
    
    def _format_faq_context(self):
        """Format FAQs as context for the model"""
        faq_text = "Banking FAQs Database:\n\n"
        for idx, row in self.faq_data.iterrows():
            faq_text += f"Q: {row['Question']}\nA: {row['Answer']}\n\n"
        return faq_text
    
    def _build_system_prompt(self):
        """Build system prompt with FAQ context"""
        faq_context = self._format_faq_context()
        system_prompt = f"""You are a helpful HBDB Banking Assistant. You provide accurate information about HBDB banking services, accounts, and FAQs.

{faq_context}

Guidelines:
- Always provide accurate information based on the FAQs provided
- If a question is not covered in the FAQs, provide general banking knowledge but clearly indicate it's not from HBDB FAQs
- Be friendly, professional, and concise
- If you don't know the answer, suggest contacting HBDB customer service
- Help users with their banking needs in a helpful and courteous manner
- Always answer in the same language the user uses"""
        return system_prompt
    
    def get_response(self, user_message):
        """
        Get a response from the bot
        
        Args:
            user_message (str): User's question or message
            
        Returns:
            str: Bot's response
        """
        try:
            # Add user message to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })
            
            # Call Mistral API with conversation history
            response = self.client.chat.complete(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._build_system_prompt()
                    }
                ] + self.conversation_history,
                temperature=0.7,
                max_tokens=1024
            )
            
            # Extract and store assistant response
            assistant_response = response.choices[0].message.content
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_response
            })
            
            return assistant_response
            
        except Exception as e:
            error_msg = f"Error getting response: {str(e)}"
            print(error_msg)
            return "I apologize, but I encountered an error processing your request. Please try again or contact customer service."
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_conversation_history(self):
        """Get the conversation history"""
        return self.conversation_history
