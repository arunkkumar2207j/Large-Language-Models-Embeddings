import gensim.downloader as api
import numpy as np

# --- Configuration ---
MODEL_NAME = "glove-wiki-gigaword-50"
TARGET_WORDS = ["king", "queen", "diamond"]
VECTOR_SLICE_LENGTH = 10  # Display the first 10 values of the vector
TOP_N_SIMILAR = 5         # Display the top 5 most similar words

print(f"--- Loading Pre-trained Word Embeddings: {MODEL_NAME} ---")

# 1. Load the pre-trained GloVe vectors. This downloads the model if not cached.
# The result is a KeyedVectors object which maps words to their vectors.
word_vectors = api.load(MODEL_NAME)

def sentence_level_embeddings():
    # 1. Load the pre-trained GloVe vectors. This downloads the model if not cached.
    # The result is a KeyedVectors object which maps words to their vectors.
    # word_vectors = api.load(MODEL_NAME)
    print("Model loaded successfully!!")
    print(f"Vector Dimension: {word_vectors.vector_size}")

    # 2. Process each target word
    print("\n" + "="*100)
    for word in TARGET_WORDS:
        print(f"Processing word: '{word}'")

        if word in word_vectors:
            # a. Display the first 10 values of its vector
            vector = word_vectors[word]
            vector_snippet = vector[VECTOR_SLICE_LENGTH]

            print(f"   Vector (First {VECTOR_SLICE_LENGTH}, values):\n {vector_snippet.round(4)}")
            # b. Display the top 5 most similar words and their similarity scores
            similar_words = word_vectors.most_similar(word, topn=TOP_N_SIMILAR)
            print(f"  Top {TOP_N_SIMILAR} most similar words:")
            print(f"similar_words: {similar_words}")
            for similar_word, score in similar_words:
                print(f"    - {similar_word:<10} (Score: {score:.4f})")
        else:
            print(f"  Word '{word}' not found in the vocabulary of {MODEL_NAME}.")

        print("-" * 50)


# Define five sample sentences
sentences = [
    "AI will transform the future of work.", # S1: AI/Work
    "Machine learning models need large datasets.", # S2: AI/Data
    "The jewelry store sells gold rings and necklaces.", # S3: Jewelry
    "Neural networks are the foundation of deep learning.", # S4: AI/Tech
    "The gold necklace is made of precious metal." # S5: Jewelry
]

def get_sentence_vector(sentence, wv):
    """Computes the average vector for a sentence, skipping OOV words."""
    # Tokenize and normalize the sentence
    words = [word.lower() for word in sentence.split()]

    # Filter for words present in the GloVe vocabulary
    vectors = [wv[word] for word in words if word in wv]
    if not vectors:
        # If no words are found, return a zero vector (avoids division by zero)
        return np.zeros(wv.vector_size)
    # Compute the average vector (centroid)
    return np.mean(vectors, axis=0)

if __name__ == "__main__":
    # sentence_level_embeddings()
    # Compute sentence vectors
    sentence_vectors = [get_sentence_vector(s, word_vectors) for s in sentences]
    print(f"sentence_vectors: {sentence_vectors}")
    # Combine vectors into a matrix for efficient calculation
    sentence_matrix = np.stack(sentence_vectors)

    # Calculate the Cosine Similarity Matrix (C = S @ S.T / ||S|| * ||S||.T)
    # 1. Compute dot products (numerator)
    dot_product_matrix = np.dot(sentence_matrix, sentence_matrix.T)

    # 2. Compute magnitudes (denominators)
    # Calculate the magnitude (L2 norm) of each sentence vector
    norms = np.linalg.norm(sentence_matrix, axis=1)
    # The outer product of norms gives the products of magnitudes (||A|| * ||B||)
    norm_product_matrix = np.outer(norms, norms)

    # 3. Compute the final similarity matrix
    # Handle division by zero for zero vectors (sentences with only OOV words)
    similarity_matrix = np.divide(dot_product_matrix, norm_product_matrix,
                                  out=np.zeros_like(dot_product_matrix),
                                  where=norm_product_matrix != 0)

    print("\nSentence Vectors Computed (Averaged Word Embeddings):")
    for i, s in enumerate(sentences):
        print(f"S{i + 1}: '{s}'")

    print("\nCosine Similarity Matrix (Rounded to 4 decimals):")
    print("        S1      S2      S3      S4      S5")
    print("      " + "------" * 5)

    # Print the matrix row by row
    for i, row in enumerate(similarity_matrix):
        row_str = " | ".join([f"{val:.4f}" for val in row])
        print(f"S{i + 1} | {row_str}")