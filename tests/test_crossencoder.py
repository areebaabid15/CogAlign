from sentence_transformers import CrossEncoder
import time

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

goal = "I'm going to write the introduction to my dissertation"
on_task_text = "Chapter 1: Introduction. This dissertation examines..."
off_task_text = "Top 10 funniest cat videos this week — you won't believe number 7!"

start = time.time()
scores = model.predict([(goal, on_task_text), (goal, off_task_text)])
elapsed = time.time() - start

print("On-task score:", scores[0])
print("Off-task score:", scores[1])
print(f"Time taken: {elapsed:.2f}s")
