# src/classifier.py
from sentence_transformers import CrossEncoder
import numpy as np


class WorkspaceClassifier:
    def __init__(self, model_name="mixedbread-ai/mxbai-rerank-xsmall-v1"):
        """
        Initializes the lightweight Cross-Encoder model for real-time
        relevancy checking.
        """
        print("Loading lightweight text classification engine...")
        self.model = CrossEncoder(model_name)
        print("Classifier model loaded successfully.")

    def is_related(self, window_text, core_goal, threshold=0.15):
        """
        Compares the active window text against your core research goal.
        Returns True if the score is above the threshold, False otherwise.
        """
        if not window_text.strip():
            return False

        # Combine pairs for the Cross-Encoder to rank
        pairs = [[core_goal, window_text]]
        score = self.model.predict(pairs)[0]

        # Sigmoid normalization if scores are raw logits
        if score < 0 or score > 1:
            score = 1 / (1 + np.exp(-score))

        return bool(score >= threshold), float(score)


if __name__ == "__main__":
    # Self-test block to verify model predictions
    classifier = GoalClassifier()
    my_goal = (
        "Writing university postgraduate literature review and thesis documentation"
    )

    test_cases = [
        "Microsoft Word - LITERATURE REVIEW v2.docx",
        "Moray Saiyaan Ep 02 - 15th November 2016 - ARY Digital Drama - YouTube",
        "Notes - literature review outline and citations",
        "Google Safari - ChatGPT research prompts",
    ]

    print(f"\n=== Testing Classifier Against Goal: '{my_goal}' ===")
    for case in test_cases:
        matched, raw_score = classifier.is_related(case, my_goal)
        status = "✅ GOAL-ORIENTED" if matched else "❌ DISTRACTION"
        print(f"[{status}] (Score: {raw_score:.2f}) -> '{case}'")
