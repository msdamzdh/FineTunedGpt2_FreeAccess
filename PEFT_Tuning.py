from sympy import true
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType,
)
import json
import logging
from typing import List, Dict
from pathlib import Path
import re
class PEFTTrainer:

    def __init__(
        self,
        model_path: str,
        dataset_address:str="./dataset/conversations.json",
        knowledge_address:str = "./dataset/knowledge-base.md",
        livedataset_address:str ="./dataset/live-datasource.json", 
        max_length = 128,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        print(f"*****************  Initialize PEFTTrainer class  *****************\n")
        self.knowledge_address = knowledge_address
        self.livedataset_address = livedataset_address
        
        self.model_path = model_path
        self.model_name = model_path.split('/')[-1]
        self.device = device
        self.max_length = max_length
        self.dataset_address = dataset_address
        print("*****************  Load knowledge and live database  *****************\n\n\n")
        self.knowledge_base = self._load_knowledge_base()
        self.live_datasource = self._load_live_datasource()
        print("*****************  Initialize tokenizer  *****************\n")
        path = Path(self.model_path)
        local_files_only = True if path.exists() else False
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path,local_files_only=local_files_only)
        if self.tokenizer.pad_token is None:
            # self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("*****************  Download model if it is not available localy  *****************\n")
        print("*****************  Initialize model  *****************\n")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            local_files_only=local_files_only
        )
        
        print("*****************  Prepare model for PEFT  *****************\n")
        self.model = prepare_model_for_kbit_training(self.model)
        
    def _load_knowledge_base(self) -> str:
        """Load the knowledge base from markdown file"""
        try:
            with open(self.knowledge_address, 'r') as f:
                return f.read()
        except FileNotFoundError:
            logging.warning("knowledge-base.md not found. Proceeding without knowledge base.")
            return ""
    
    def _find_target_modules(self) -> List[str]:
        """Dynamically identify all linear/attention layers in the model"""
        target_modules = []
        
        # Helper function to check if module should be targeted
        def check_module(name: str, module: torch.nn.Module) -> bool:
            return (
                isinstance(module, torch.nn.Linear) or
                'attention' in name.lower() or
                any(layer_type in name.lower() for layer_type in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate', 'up_proj', 'down_proj'])
            )
        
        # Iterate through all named modules
        for name, module in self.model.named_modules():
            if check_module(name, module):
                # Extract the module name without parent hierarchy
                module_name = name.split('.')[-1]
                if module_name not in target_modules:
                    target_modules.append(module_name)
        
        logging.info(f"Automatically identified target modules: {target_modules}")
        return target_modules
    
    def _prepare_lora_config(self) -> LoraConfig:
        
        """Configure LoRA """
        
        target_modules = self._find_target_modules()
        # When working with common 🤗 Transformers models, PEFT will know which layers to apply LoRA to
        # but in other cases we should specify target_modules
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules = target_modules if self.model_name=="xlnet-base-cased" else None
        )
    
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
  
    def _prepare_training_data(self, conversations: List[Dict]) -> Dataset:
        """Prepare conversation data for training, incorporating knowledge base and live datasource queries"""
        
        prompt = f"""You are a helpfull digital bank assistant and should give response based on Customer input,
Knowledge base and Live info section.
knowledge section:

{self.knowledge_base}
            """.strip()
            
        formatted_data = []
        for conv in conversations:
            conversation_history = []
            conversation_history.append(prompt)
            messages = conv.get('messages')
            for msg in messages:
                if msg['type'] == 'customer':
                    user ="Customer:"
                    live_query_result = self._query_live_datasource(msg['message'])             
                elif msg['type'] == 'agent':
                    user = "Agent:"
                    live_query_result=''
                if (live_query_result != "No matching transactions found in the live datasource." and 
                    live_query_result != ''):
                    conversation_history.append(f"Live info:\n{live_query_result}")
                
                conversation_history.append(f"{user} {msg['message']}")

            formatted_text = "\n".join(conversation_history).strip()            
            formatted_data.append({"text": formatted_text})
        
        newDataSet = Dataset.from_list(formatted_data)
        
        def tokenize_sentence(examples):
            return self.tokenizer([" ".join(x) for x in examples["text"]],
                                  padding="max_length",  # Ensures padding up to model's max length
                                  return_attention_mask=True,  # Ensure attention mask is returned
                                  truncation=False,
                                )
        
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= self.max_length:
                total_length = (total_length // self.max_length) * self.max_length
            # Split by chunks of max_length.
            result = {
                k: [t[i : i + self.max_length] for i in range(0, total_length, self.max_length)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result
        
        tokenized_dataset = newDataSet.map(
                                            tokenize_sentence,
                                            batched=True,
                                            remove_columns=newDataSet.column_names,
                                        )
        blocked_tokenizer_dataset = tokenized_dataset.map(
                                                            group_texts,
                                                            batched=True,
                                                            remove_columns=tokenized_dataset.column_names
                                                        )
        return blocked_tokenizer_dataset

    def fine_tune(
        self,
        epochs: int = 200,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-3
    ):
        
        print("################ Execute PEFT fine-tuning process ###############\n\n")
        print("***************** Prepare dataset *****************\n")
        # Prepare dataset
        with open(self.dataset_address, 'r') as f:
            conversations = json.load(f)["conversations"]
        dataset = self._prepare_training_data(conversations)
        
        # It’s more efficient to dynamically pad the sentences to the longest length 
        # in a batch during collation, instead of padding the whole dataset to the maximum length.
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        
        print("***************** Configure LoRA *****************\n")
        # Configure LoRA
        lora_config = self._prepare_lora_config()
        
        # Get PEFT model
        peft_model = get_peft_model(self.model, lora_config)
        print("***************** Print trainable parameters info *****************\n")
        # Print trainable parameters info
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in peft_model.parameters())
        print(f"Trainable params: {trainable_params} ({trainable_params/total_params:.2%} of total)")
        
        print("***************** Setting training parameters *****************\n")
        # Training arguments
        output_dir = f"./FinetunedModels/{self.model_name}_finetuned"
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=5,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=5,
            save_strategy="epoch",
            evaluation_strategy="no",
            weight_decay=0.01,
        )
        
        # Initialize trainer with appropriate data collator
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator
        )
        print("***************** Training *****************\n")
        # Train model
        trainer.train()
        print("***************** Saving model *****************\n")
        # Save trained model
        peft_model.save_pretrained(output_dir)
