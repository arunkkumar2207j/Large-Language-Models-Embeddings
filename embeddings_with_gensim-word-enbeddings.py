import gensim.downloader as api
import numpy as np

# --- Configuration ---
MODEL_NAME = "glove-wiki-gigaword-50"
TARGET_WORDS = ["king", "queen", "diamond"]
VECTOR_SLICE_LENGTH = 10  # Display the first 10 values of the vector
TOP_N_SIMILAR = 5         # Display the top 5 most similar words

def word_embeddings():
    print("--- Word Embeddings with GloVe (Gensim) ---")

    try:
        # 1. Load the pre-trained GloVe vectors. This downloads the model if not cached.
        # The result is a KeyedVectors object which maps words to their vectors.
        word_vectors = api.load(MODEL_NAME)
        print("Model loaded successfully.")
        print(f"Vector Dimension: {word_vectors.vector_size}")

        # 2. Process each target word
        print("\n" + "=" * 50)
        for word in TARGET_WORDS:
            print(f"Processing word: '{word}'")
            if word in word_vectors:
                # a. Display the first 10 values of its vector
                vector = word_vectors[word]
                vector_snippet = vector[:VECTOR_SLICE_LENGTH]
                print(
                    f"  Vector (First {VECTOR_SLICE_LENGTH} values):\n  {vector_snippet.round(4)}")  # Round for cleaner output
                # b. Display the top 5 most similar words and their similarity scores
                similar_words = word_vectors.most_similar(word, topn=TOP_N_SIMILAR)
                print(f"  Top {TOP_N_SIMILAR} most similar words:")
                for similar_word, score in similar_words:
                    print(f"    - {similar_word:<10} (Score: {score:.4f})")
            else:
                print(f"  Word '{word}' not found in the vocabulary of {MODEL_NAME}.")

            print("-" * 50)

    except Exception as e:
        print(f"\nAn error occurred during model loading or processing: {e}")
        print("Please ensure your internet connection is stable and you have Gensim installed.")

if __name__ == "__main__":
    word_embeddings()