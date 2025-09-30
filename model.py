import torch
from datasets import Dataset, DatasetDict
from transformers import MT5Tokenizer
from transformers import MT5ForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import transformers

tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")
model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
def load_text_data(source_path, target_path):
    with open(source_path, encoding="utf-8") as src_file, open(target_path, encoding="utf-8") as tgt_file:
        source_lines = [line.strip() for line in src_file.readlines()]
        target_lines = [line.strip() for line in tgt_file.readlines()]

    assert len(source_lines) == len(target_lines), "Source and target files must have the same number of lines."

    return Dataset.from_dict({"input_text": source_lines, "target_text": target_lines})

train_dataset = load_text_data("data/train/train.src", "data/train/train.tgt")
test_dataset = load_text_data("data/test/test.src", "data/test/test.tgt")

split_point = len(train_dataset) // 2
divided_datasets = train_dataset.train_test_split(test_size=0.5, seed=42)

train_dataset = divided_datasets['train']

max_input_length = 512
max_target_length = 128

def preprocess(examples):
    inputs = examples['input_text']
    targets = examples['target_text']
    
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding='max_length')
    labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding='max_length')
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = train_dataset.map(preprocess, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="./mt5-hindi-finetuned",
    eval_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,  # if on GPU
    logging_dir="./logs",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=test_dataset.map(preprocess, batched=True),
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
