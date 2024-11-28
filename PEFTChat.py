from cmd import PROMPT
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import logging
import json
from typing import List, Dict
import re

class PEFTChatModel:
    def __init__(
        self, 
        base_model_path: str, 
        peft_model_path: str,
        max_length:int = 128,
        max_new_tokens: int = 20, 
        temperature: float = 0.5,
        knowledge_address:str = "./dataset/knowledge-base.md",
        livedataset_address:str ="./dataset/live-datasource.json", 
        device: str = None
    ):
        """
        Initialize the chat model with base and fine-tuned PEFT weights
        
        Args:
            base_model_path (str): Path to the original base model
            peft_model_path (str): Path to the fine-tuned PEFT model
            device (str, optional): Device to load the model on. Defaults to cuda if available.
        """
        self.max_length = max_length
        self.knowledge_address = knowledge_address
        self.livedataset_address = livedataset_address
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        print("*****************  Load knowledge and live database  *****************\n\n\n")
        self.knowledge_base = self._load_knowledge_base()
        self.live_datasource = self._load_live_datasource()
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load base model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, 
            torch_dtype=torch.float16
        ).to(self.device)
        
        # Load PEFT model
        try:
            self.model = PeftModel.from_pretrained(
                self.base_model, 
                peft_model_path
            ).to(self.device)
        except Exception as e:
            logging.error(f"Error loading PEFT model: {e}")
            raise
        
        # Prepare for inference
        self.model.eval()
        
        # Initialize conversation history
        prompt = f"""You are a helpfull digital bank assistant and should give response based on Customer input,
Knowledge base and Live info section.
knowledge section:

{self.knowledge_base}
            """.strip()
        self.conversation_history = []
        self.conversation_history.append(prompt)
        
        
    
    def _load_knowledge_base(self) -> str:
        """Load the knowledge base from markdown file"""
        try:
            with open(self.knowledge_address, 'r') as f:
                return f.read()
        except FileNotFoundError:
            logging.warning("knowledge-base.md not found. Proceeding without knowledge base.")
            return ""
    
    def _load_live_datasource(self) -> List[Dict]:
        """Load the live datasource from JSON file"""
        try:
            with open(self.livedataset_address, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.warning("live-datasource.json not found. Proceeding without live data.")
            return []

    def _query_live_datasource(self, query: str) -> str:
        """Query the live datasource for relevant information"""
        query_lower = query.lower()
        
        # Extract potential date, amount, and merchant from the query
        date = None
        amount = None
        merchant = None
        
        # Simple date extraction (assuming format DD-MMM-YYYY)
        date_match = re.search(r'\d{2}-[A-Z]{3}-\d{4}', query)
        if date_match:
            date = date_match.group()
        
        # Amount extraction
        amount_match = re.search(r'\d+(\.\d{2})?', query)
        if amount_match:
            amount = float(amount_match.group())
        
        # Merchant extraction (simple approach, can be improved)
        merchant_match = re.search(r'from\s+(\w+)', query_lower)
        if merchant_match:
            merchant = merchant_match.group(1)
        
        # Search for matching transactions
        matching_transactions = []
        for transaction in self.live_datasource:
            if (date and transaction['date'] == date) or \
               (amount and abs(transaction['amount'] - amount) < 0.01) or \
               (merchant and transaction['merchant'].lower() == merchant):
                matching_transactions.append(transaction)
        
        if matching_transactions:
            return json.dumps(matching_transactions, indent=2)
        else:
            return "No matching transactions found in the live datasource."
  
    def generate_response(
        self, 
        user_message: str,
    ) -> str:
        """
        Generate a response to the user's message
        
        Args:
            user_message (str): Input message from the user
                    
        Returns:
            str: Model's generated response
        """
        
        # Add user message to conversation history
        live_query_result = self._query_live_datasource(user_message)
        
        if (live_query_result != "No matching transactions found in the live datasource." and 
            live_query_result != ''):
            self.conversation_history.append(f"Live info:\n{live_query_result}")
        
        self.conversation_history.append(f"Customer: {user_message}")
        
        # Prepare full context
        full_context = "\n".join(self.conversation_history)
        
        # Tokenize input
        tokenized_output = self.tokenizer(
                    full_context,
                    padding="max_length",        # Pad sentences to max_length
                    return_tensors="pt",         # Return as PyTorch tensors
                    return_attention_mask=True   # Include attention mask
                )
        input_ids = tokenized_output["input_ids"]
        attention_mask = tokenized_output["attention_mask"]
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                attention_mask=attention_mask,
            )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        # Add model's response to conversation history
        self.conversation_history.append(f"Agent: {response}")
        
        return response
    
    def chat(self):
        """
        Interactive chat interface
        """
        print("PEFT Fine-Tuned Chat Model")
        print("Type 'exit' to end the conversation")
        
        while True:
            user_input = input("Customer: ")
            
            if user_input.lower() == 'exit':
                break
            try:
                response = self.generate_response(user_input)
                print(f"Agent: {response}")
            except Exception as e:
                print(f"Error generating response: {e}")

def main():
    # Example usage
    chat_model = PEFTChatModel(
        base_model_path="gpt2",  # Replace with your base model path
        peft_model_path="./FinetunedModels/gpt2_finetuned"  # Replace with your fine-tuned model path
    )
    
    # Start interactive chat
    chat_model.chat()

if __name__ == "__main__":
    main()

