# PEFT Fine-Tuned Chatbot

## Project Overview
PEFT-based chatbot using GPT-2, designed as a digital bank assistant with LoRA fine-tuning.

## Prerequisites
- Python 3.8+
- CUDA-compatible GPU recommended

## Setup
1. Clone the repository
2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure
- `PEFT_Tuning.py`: Fine-tuning logic
- `PEFTChat.py`: Chatbot model 
- `InterfaceApp.py`: Streamlit interface
- `GPT2Downloader.py`: Model download utility
- `test_PEFT_Tuning.py`: Unit tests

## Workflow
1. Download models: `python GPT2Downloader.py`
2. Run tests: `python test_PEFT_Tuning.py`
3. Fine-tune: `python FineTuneGPT2.py`
4. Start chatbot: `streamlit run InterfaceApp.py`

## Fine-Tuning Parameter Customization

### In `FineTuneGPT2.py`

#### Key Tunable Parameters
- `epochs`: Total training iterations (default: 200)
  - Increase for more thorough learning
  - Decrease to reduce training time

- `learning_rate`: Model adaptation speed (default: 2e-3)
  - Lower values: More conservative learning
  - Higher values: Faster but potentially less stable adaptation

- `batch_size`: Training data processed in one iteration (default: 4)
  - Larger batches: More stable gradients
  - Smaller batches: Less memory consumption

- `gradient_accumulation_steps`: Virtual batch size multiplier (default: 4)
  - Helps train with larger effective batch sizes on limited GPU memory
  - Increases training stability

#### Example Customization
```python
trainer_gpt.fine_tune(
    epochs=300,           # Extended training
    learning_rate=1e-4,   # More conservative learning
    batch_size=8,         # Larger batch processing
    gradient_accumulation_steps=2  # Adjusted gradient accumulation
)
```

### Considerations
- Adjust parameters based on:
  - Dataset size
  - Available computational resources
  - Desired model performance

## Notes
- Requires substantial computational resources
- Fine-tuning performance depends on dataset quality
