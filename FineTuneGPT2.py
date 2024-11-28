from PEFT_Tuning import PEFTTrainer 

trainer_gpt = PEFTTrainer(
    model_path="./gpt2",
    dataset_address="./dataset/conversations.json",
    knowledge_address = "./dataset/knowledge-base.md",
    livedataset_address ="./dataset/live-datasource.json", 
    max_length = 128
    )

trainer_gpt.fine_tune(epochs = 200,learning_rate=2e-3)

