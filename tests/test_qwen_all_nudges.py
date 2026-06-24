from transformers import AutoModelForCausalLM, AutoTokenizer
import time

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
print("Loading model into memory (this happens once)...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", low_cpu_mem_usage=True
)

scenarios = {
    "Level 1: Empathetic Coach": {
        "system": (
            "You are a warm, highly supportive academic mentor. Your goal is to gently motivate the student. "
            "Constraint: Output exactly one single sentence. Speak directly to the user.\n"
            "Example: 'Every paragraph you write right now is a step closer to completing your degree and achieving your long-term goals.'"
        ),
        "user": "I am supposed to be working on my dissertation, but I am caught watching cat videos. Give me a 1-sentence motivational nudge.",
    },
    "Level 2: Sarcastic Friend": {
        "system": (
            "You are a witty, sarcastic best friend. Your humor is sharp, playful, and direct. "
            "Constraint: Output exactly one single sentence of dialogue. Do not include any introductory words, quotation marks, or explanations.\n"
            "Example: 'I'm sure the admissions board will find your deep knowledge of funny cat reactions incredibly relevant to your degree.'"
        ),
        "user": "I am supposed to be working on my dissertation, but I am watching cat videos. Give me a 1-sentence sarcastic roast.",
    },
    "Level 3: Deadline Supervisor": {
        "system": (
            "You are a strict, serious university thesis director. Your tone is formal, urgent, and direct. "
            "Constraint: Output exactly one single sentence giving a professional order to return to work.\n"
            "Example: 'Cease all digital distractions immediately and direct your full attention back to your pending academic writing.'"
        ),
        "user": "I am caught watching cat videos instead of writing my dissertation. Give me a 1-sentence formal command.",
    },
}

print("\n--- STARTING PERSONA BENCHMARKS ---")

for name, context in scenarios.items():
    print(f"\nGenerating for {name}...")

    # Explicitly pulling strings to pass into the template safely
    messages = [
        {"role": "system", "content": str(context["system"])},
        {"role": "user", "content": str(context["user"])},
    ]

    # Rendering template
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt")

    start = time.time()
    # do_sample=True activates the creative temperature setting
    output = model.generate(
        **inputs, max_new_tokens=50, temperature=0.8, do_sample=True
    )
    elapsed = time.time() - start

    raw_out = tokenizer.decode(output[0], skip_special_tokens=True)
    clean_out = raw_out.split("assistant")[-1].strip()

    print(f'[{name} Output]:\n"{clean_out}"')
    print(f"Time taken: {elapsed:.1f}s")
    print("-" * 40)
