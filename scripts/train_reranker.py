from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TRAIN_DATASET_PATH = "../data/training/reranker_data.json"
OUTPUT_MODEL_PATH = "../models/reranker"

def load_training_data():
    """Loads the dataset for reranker fine-tuning"""
    dataset = load_dataset("json", data_files={"train": TRAIN_DATASET_PATH})
    return DatasetDict({"train": dataset["train"]})

def train_reranker():
    """Fine-tunes a cross-encoder reranker on query-document pairs"""
    dataset = load_training_data()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess_function(examples):
        return tokenizer(examples["query"], examples["document"], truncation=True, padding="max_length", max_length=512)

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    training_args = TrainingArguments(
        output_dir=OUTPUT_MODEL_PATH,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        tokenizer=tokenizer,
    )

    print("Training reranker...")
    trainer.train()
    trainer.save_model(OUTPUT_MODEL_PATH)
    print(f"Fine-tuned model saved at {OUTPUT_MODEL_PATH}")

if __name__ == "__main__":
    train_reranker()
