import os

# Set the cache directory *before* importing transformers.
# This redirects the cache from the potentially problematic C:\Users\... to a local folder
# named '.huggingface_cache' in the same directory as the script.
try:
    # Use os.getcwd() to create a cache folder in the same directory as the script
    cache_path = os.path.join(os.getcwd(), '.huggingface_cache')
    os.environ['HF_HOME'] = cache_path
    print(f"Hugging Face cache redirected to: {cache_path}")
except Exception as e:
    # Fallback/Error handling for path setting
    print(f"Warning: Could not set custom cache path due to {e}. Using default.")

import torch
from transformers import pipeline

# Model Choice: distilgpt2 is small and fast for local testing
MODEL_NAME = "distilgpt2"
try:
    # Initialize the text generation pipeline
    generator = pipeline("text-generation", model=MODEL_NAME)
    print(f"--- Using Hugging Face Model: {MODEL_NAME} ---")

except Exception as e:
    # Note: If this still fails, you may need to check firewall settings or run your terminal as administrator.
    print(f"Error loading model: {e}")
    print("Please ensure you have 'torch' and 'transformers' installed.")
    exit()

# --- INPUT DATA ---
article = "Large Language Models (LLMs) are deep learning algorithms that can recognize, summarize, translate, predict, and generate content. They are trained on massive datasets of text and code. Their applications range from customer service chatbots to complex code generation tools, fundamentally changing how humans interact with technology. The most critical component of an LLMs power is the transformer architecture, which allows them to process long-range dependencies in the text."

# --- TASK 1: SUMMARISATION (limit to <= 30 words) ---
summarization_prompts = {
    "Simple Prompt (Vague)": f"Summarize the following text:\n{article}",
    "Refined Prompt (Specific)": f"Summarize the following text in exactly 30 words or less. Focus on the definition and core application.\nText: {article}"
}

# --- TASK 2: Q&A (Answer a factual question) ---
qa_prompts = {
    "Simple Prompt (No Context)": "What is the primary architectural component that gives LLMs their power?",
    "Refined Prompt (Contextual)": "Based on the provided text, what is the primary architectural component that gives LLMs their power? The key is the...",
}

# --- TASK 3: CREATIVE TEXT GENERATION (4-line poem on AI) ---
creative_prompts = {
    "Simple Prompt (Open)": "Write a short poem about AI.",
    "Refined Prompt (Constrained)": "Write a four-line rhyming poem about AI's potential, in a hopeful and futuristic tone. Do not exceed four lines.",
}


# --- EXECUTION AND COMPARISON FUNCTION ---
def generate_and_compare(task_name, prompts_dict, max_len, temperature):
    """Generates and prints output for both simple and refined prompts for a given task."""
    print(f"\n{'=' * 50}\n## 🤖 Task: {task_name}\n{'=' * 50}")

    for name, prompt in prompts_dict.items():
        # Generate text with controlled parameters
        # max_new_tokens controls output length. temperature controls randomness (creativity).
        output = generator(
            prompt,
            max_new_tokens=max_len,
            do_sample=True,
            temperature=temperature,
            pad_token_id=generator.tokenizer.eos_token_id,
            num_return_sequences=1
        )

        # Clean up output: remove the input prompt from the generated text for clarity
        generated_text = output[0]['generated_text'].replace(prompt, '').strip()

        print(f"### {name}")
        print(f"**Input Snippet:** '{prompt[:80]}...'")
        print(f"**Output:** {generated_text}")
        print("-" * 20)


# Run the tasks
# Lower temperature (0.5-0.7) for factual tasks (less randomness)
generate_and_compare("Summarisation", summarization_prompts, max_len=50, temperature=0.7)
generate_and_compare("Q&A", qa_prompts, max_len=30, temperature=0.5)
# Higher temperature (0.9-1.0) for creative tasks (more randomness)
generate_and_compare("Creative Text Generation", creative_prompts, max_len=60, temperature=0.9)