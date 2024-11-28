import unittest
from unittest.mock import patch, MagicMock, mock_open
from PEFT_Tuning import PEFTTrainer
import json
import torch.nn

class TestPEFTTrainerPrivateMethods(unittest.TestCase):

    def setUp(self):
        # Mock device availability
        self.trainer = PEFTTrainer(
            model_path="./gpt2",
            dataset_address="test_dataset.json",
            knowledge_address="test_knowledge.md",
            livedataset_address="test_live.json"
        )
        self.trainer.model = MagicMock()
    
    @patch("builtins.open", new_callable=mock_open, read_data="Sample Knowledge Base Content")
    def test_load_knowledge_base(self, mock_file):
        knowledge_base = self.trainer._load_knowledge_base()
        self.assertEqual(knowledge_base, "Sample Knowledge Base Content")
        mock_file.assert_called_once_with("test_knowledge.md", 'r')
    
    @patch("builtins.open", new_callable=mock_open, read_data='[{"date": "01-Jan-2024", "amount": 123.45, "merchant": "Store"}]')
    def test_load_live_datasource(self, mock_file):
        live_datasource = self.trainer._load_live_datasource()
        self.assertEqual(live_datasource, [{"date": "01-Jan-2024", "amount": 123.45, "merchant": "Store"}])
        mock_file.assert_called_once_with("test_live.json", 'r')

    @patch("PEFT_Tuning.AutoModelForCausalLM.from_pretrained")
    @patch("PEFT_Tuning.AutoTokenizer.from_pretrained")
    def test_find_target_modules(self, mock_tokenizer, mock_model):
        # Mock model with named_modules returning dummy modules
        linear_mock = MagicMock()
        linear_mock.__class__ = torch.nn.Linear  # Assign valid class type for isinstance checks

        mock_model.return_value.named_modules.return_value = [
            ("layer1", linear_mock),
            ("attention_layer", MagicMock()),
            ("other_layer", MagicMock())
        ]

        # Initialize PEFTTrainer
        trainer = PEFTTrainer(model_path="test_model_path")
        trainer.model = mock_model.return_value

        # Test the target module identification
        target_modules = trainer._find_target_modules()
        self.assertIn("layer1", target_modules)
        self.assertIn("attention_layer", target_modules)
        self.assertNotIn("other_layer", target_modules)

    @patch("PEFT_Tuning.LoraConfig")
    @patch("PEFT_Tuning.PEFTTrainer._find_target_modules", return_value=["module1"])
    def test_prepare_lora_config(self, mock_find_target_modules, mock_lora_config):
        lora_config = self.trainer._prepare_lora_config()
        mock_find_target_modules.assert_called_once()
        mock_lora_config.assert_called_once_with(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=None  # Adjust this if there's a condition in your method that alters it.
        )

    @patch("PEFT_Tuning.AutoTokenizer.from_pretrained")
    def test_query_live_datasource(self, mock_tokenizer):
        # Mock live datasource with transactions
        self.trainer.live_datasource = [
            {"date": "01-Jan-2024", "amount": 123.45, "merchant": "Store"}
        ]
        
        # Mock tokenizer
        mock_tokenizer.return_value = MagicMock()

        # Test query with a match
        query = "Can you find transactions from Store on 01-Jan-2024 for 123.45?"
        result = self.trainer._query_live_datasource(query)
        expected_result = json.dumps([{"date": "01-Jan-2024", "amount": 123.45, "merchant": "Store"}], indent=2)
        self.assertEqual(result, expected_result)

        # Test query with no match
        query_no_match = "Find transactions on 02-Jan-2024"
        result_no_match = self.trainer._query_live_datasource(query_no_match)
        self.assertEqual(result_no_match, "No matching transactions found in the live datasource.")

    @patch("PEFT_Tuning.Dataset.from_list")
    @patch("PEFT_Tuning.AutoTokenizer.from_pretrained")
    def test_prepare_training_data(self, mock_tokenizer, mock_dataset_from_list):
        # Mock tokenizer behavior
        mock_tokenizer.return_value = MagicMock()

        # Mock Dataset.from_list
        mock_dataset = MagicMock()
        mock_dataset_from_list.return_value = mock_dataset

        # Mock input conversations
        conversations = [{"messages": [{"type": "customer", "message": "query"}]}]

        # Call _prepare_training_data
        dataset = self.trainer._prepare_training_data(conversations)

        # Assertions
        mock_dataset_from_list.assert_called_once_with([{
            "text": f"""You are a helpfull digital bank assistant and should give response based on Customer input,
Knowledge base and Live info section.
knowledge section:

{self.trainer.knowledge_base}
            """.strip()+"\nCustomer: query"}])

        self.assertIsNotNone(dataset)

if __name__ == "__main__":
    unittest.main()