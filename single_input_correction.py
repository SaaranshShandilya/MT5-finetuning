from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import torch

model_dir = "./mt5-error-correction-final"   # path where you saved
model = MT5ForConditionalGeneration.from_pretrained(model_dir)
tokenizer = MT5Tokenizer.from_pretrained(model_dir)
model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

sentence = "Thank for your work."

inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True).to(model.device)
corrected_output = model.generate(**inputs, max_length=128)
pred = tokenizer.decode(corrected_output[0], skip_special_tokens=True)
print(f"Corrected output: {pred}")


