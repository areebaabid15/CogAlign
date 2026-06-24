from transformers import AutoModelForCausalLM, AutoTokenizer
import time

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", low_cpu_mem_usage=True)

prompt = "You are an empathetic coach. The user said they'd write their dissertation intro but is now watching cat videos. Write one gentle, 2-sentence nudge."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")

start = time.time()
output = model.generate(**inputs, max_new_tokens=60)
elapsed = time.time() - start

print(tokenizer.decode(output[0], skip_special_tokens=True))
print(f"Time taken: {elapsed:.1f}s")
