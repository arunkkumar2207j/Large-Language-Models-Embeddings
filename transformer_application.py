import os

# 1. Define the custom, preferred path explicitly
CUSTOM_CACHE_DIR = r"C:/Arun/IIT-M/hf-cache"

# 2. Ensure the directory exists (as you were doing)
os.makedirs(CUSTOM_CACHE_DIR, exist_ok=True)

# 3. Set the environment variable to your custom path
os.environ['HF_HOME'] = CUSTOM_CACHE_DIR
print(f"Hugging Face cache redirected to: {CUSTOM_CACHE_DIR}")

from transformers import pipeline

# --- Configuration ---
TASK_NAME = "sentiment-analysis"
MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"


try:
    # Load the specialized pipeline for Sentiment Analysis
    sentiment_analyser = pipeline(TASK_NAME, model=MODEL_ID)
    print(f"Pipeline loaded successfully using model: {MODEL_ID}")

    # Custom business-style inputs
    texts = [
        "The customer support was incredibly helpful and resolved my issue quickly.",
        "The product quality is terrible and I want a refund.",
        "The new AI feature is interesting, but the interface is confusing.",
    ]

    print("\nInput texts:")
    for i, t in enumerate(texts, start=1):
        print(f"{i}. {t}")

    # run the pipeline
    results = sentiment_analyser(texts)
    print(f"result: {results}")

    for text, result in zip(texts, results):
        label = result["label"]
        score = result["score"]

        # Check the prediction label and print a categorized output
        sentiment_emoji = "✅ Positive" if label == "POSITIVE" else "❌ Negative"

        print("-" * 50)
        print(f"Input: '{text}'")
        print(f"Prediction: {sentiment_emoji} (Score: {score:.4f})")

except Exception as e:
    print(f"Error loading pipeline: {e}")