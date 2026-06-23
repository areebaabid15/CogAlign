# src/nudge_engine.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import time


class QwenNudgeEngine:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        print(f"Loading {model_name} into memory (this happens once)...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", low_cpu_mem_usage=True
        )
        print("Model loaded successfully!")

        # Store our verified few-shot prompts mapped to escalation levels
        self.personas = {
            1: {
                "system": (
                    "You are a warm, highly supportive academic mentor. Your goal is to gently motivate the student. "
                    "Constraint: Output exactly one single sentence. Speak directly to the user.\n"
                    "Example: 'Every paragraph you write right now is a step closer to completing your degree and achieving your long-term goals.'"
                ),
                "user": "I am supposed to be working on my dissertation, but I am caught watching cat videos. Give me a 1-sentence motivational nudge.",
            },
            2: {
                "system": (
                    "You are a witty, sarcastic best friend. Your humor is sharp, playful, and direct. "
                    "Constraint: Output exactly one single sentence of dialogue. Do not include any introductory words, quotation marks, or explanations.\n"
                    "Example: 'I'm sure the admissions board will find your deep knowledge of funny cat reactions incredibly relevant to your degree.'"
                ),
                "user": "I am supposed to be working on my dissertation, but I am watching cat videos. Give me a 1-sentence sarcastic roast.",
            },
            3: {
                "system": (
                    "You are a strict, serious university thesis director. Your tone is formal, urgent, and direct. "
                    "Constraint: Output exactly one single sentence giving a professional order to return to work.\n"
                    "Example: 'Cease all digital distractions immediately and direct your full attention back to your pending academic writing.'"
                ),
                "user": "I am caught watching cat videos instead of writing my dissertation. Give me a 1-sentence formal command.",
            },
        }

    def generate_nudge(self, level: int) -> str:
        """
        Generates a 1-sentence intervention based on the escalation level (1, 2, or 3).
        """
        # Fallback safely to Level 1 if an invalid level is passed
        if level not in self.personas:
            level = 1

        context = self.personas[level]

        # Structure the payload safely as plain strings for Jinja2
        messages = [
            {"role": "system", "content": str(context["system"])},
            {"role": "user", "content": str(context["user"])},
        ]

        # Render the text format for the model
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt")

        # Run generation
        output = self.model.generate(
            **inputs, max_new_tokens=50, temperature=0.8, do_sample=True
        )

        # Decode and isolate the assistant's response
        raw_out = self.tokenizer.decode(output[0], skip_special_tokens=True)
        clean_out = raw_out.split("assistant")[-1].strip()

        # Clean up any accidental outer quotation marks
        return clean_out.strip("\"'")



# Self-test block to verify the module works independently
if __name__ == "__main__":
    # Initialize the engine
    engine = QwenNudgeEngine()

    print("\n[Test Run] Simulating a Level 2 Intervention:")
    start_time = time.time()

    # Try generating a Level 2 sarcastic nudge
    nudge = engine.generate_nudge(level=2)

    elapsed = time.time() - start_time
    print(f"Result: {nudge}")
    print(f"Time taken: {elapsed:.1f}s")
