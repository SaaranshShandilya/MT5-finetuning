from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import torch
from datasets import load_dataset, DatasetDict
from sklearn.metrics import fbeta_score, precision_score, recall_score

model_dir = "./mt5-error-correction-final"
model = MT5ForConditionalGeneration.from_pretrained(model_dir)
tokenizer = MT5Tokenizer.from_pretrained(model_dir)
model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

ds = load_dataset("matejklemen/clc_fce")
small_dataset = DatasetDict({
    "train": ds["train"].shuffle(seed=42).select(range(1000)),
    "validation": ds["validation"].shuffle(seed=42).select(range(500)),
    "test": ds["test"].shuffle(seed=42).select(range(600)),
})

test_dataset = small_dataset['test']

def tokens_to_sentence(tokens):
    return " ".join(tokens)

src_texts = [tokens_to_sentence(x) for x in test_dataset["src_tokens"]]
ref_texts = [tokens_to_sentence(x) for x in test_dataset["tgt_tokens"]]

preds = []
for text in src_texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(model.device)
    outputs = model.generate(**inputs, max_length=128)
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    preds.append(pred)

y_true = []
y_pred = []

for ref, pred in zip(ref_texts, preds):
    ref_tokens = ref.split()
    pred_tokens = pred.split()
    min_len = min(len(ref_tokens), len(pred_tokens))
    y_true.extend(ref_tokens[:min_len])
    y_pred.extend(pred_tokens[:min_len])

precision = precision_score(y_true, y_pred, average="micro", zero_division=0)
recall = recall_score(y_true, y_pred, average="micro", zero_division=0)
f05 = fbeta_score(y_true, y_pred, beta=0.5, average="micro", zero_division=0)

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F0.5:      {f05:.4f}")



