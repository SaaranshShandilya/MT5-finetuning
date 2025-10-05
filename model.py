import torch
from datasets import Dataset, DatasetDict
from transformers import MT5Tokenizer
from transformers import MT5ForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import transformers
from datasets import load_dataset

tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")
model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")


ds = load_dataset("matejklemen/clc_fce")
small_dataset = DatasetDict({
    "train": ds["train"].shuffle(seed=42).select(range(1000)),
    "validation": ds["validation"].shuffle(seed=42).select(range(500)),
    "test": ds["test"].shuffle(seed=42).select(range(500)),
})



max_input_length = 128
max_target_length = 128

def preprocess_function(examples):
    inputs = [" ".join(tokens) for tokens in examples["src_tokens"]]
    targets = [" ".join(tokens) for tokens in examples["tgt_tokens"]]

    model_inputs = tokenizer(
        inputs, max_length=128, truncation=True, padding="max_length"
    )

    labels = tokenizer(
        targets, max_length=128, truncation=True, padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = small_dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="./mt5-error-correction",
    eval_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=5,
    predict_with_generate=True,
    logging_dir="./logs",
    logging_steps=100,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

trainer.save_model("./mt5-error-correction-final")
tokenizer.save_pretrained("./mt5-error-correction-final")

