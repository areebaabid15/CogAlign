from sentence_transformers import SentenceTransformer, util
import time

# 1. Load the standard Bi-Encoder embedding model
print("Loading Bi-Encoder model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Define the exact same scenarios
goal = "I'm going to write the introduction to my dissertation"
on_task_text = "Chapter 1: Introduction. This dissertation examines..."
off_task_text = "Top 10 funniest cat videos this week — you won't believe number 7!"

start = time.time()
# 3. Compute vector embeddings for all three texts separately
goal_embedding = model.encode(goal, convert_to_tensor=True)
on_task_embedding = model.encode(on_task_text, convert_to_tensor=True)
off_task_embedding = model.encode(off_task_text, convert_to_tensor=True)

# 4. Calculate the cosine similarity angles between the vectors
on_task_score = util.cos_sim(goal_embedding, on_task_embedding).item()
off_task_score = util.cos_sim(goal_embedding, off_task_embedding).item()
elapsed = time.time() - start

# 5. Print results
print("\n--- BI-ENCODER RESULTS ---")
print("On-task score:", on_task_score)
print("Off-task score:", off_task_score)
print(f"Time taken: {elapsed:.2f}s")
print("--------------------------")
