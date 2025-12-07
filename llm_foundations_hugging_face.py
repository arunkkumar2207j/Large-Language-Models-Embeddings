# Imports
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import gensim.downloader as api
import numpy as np

# Choose a folder that definitely exists / you can control
HF_CACHE_DIR = r"C:\Arun\IIT-M\hf-cache"
os.makedirs(HF_CACHE_DIR, exist_ok=True)

# Section A: LLM Foundations & Hugging Face
# Hugging Face Setup & Text Generation
def hf_setup_text_generation() -> None:
    print("--- 1. Hugging Face: Text Generation ---")
    try:
        # Small causal language model (good for demo)
        model_name = "distilgpt2"
        print(f"\n[HF] Loading model and tokenizer: {model_name} ...")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=HF_CACHE_DIR
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=HF_CACHE_DIR
        )

        prompt = "AI is transforming industries by"
        print(f"\n[HF] Prompt:\n{prompt}\n")

        # Tokenise prompt into tensors
        inputs = tokenizer(prompt, return_tensors="pt")

        # Generate 3 different continuations
        output_ids_list = model.generate(
            **inputs,
            max_length=80,  # max total tokens (prompt + generated)
            do_sample=True,  # sampling instead of greedy
            top_p=0.9,  # nucleus sampling
            temperature=0.8,  # creativity
            num_return_sequences=3  # generate 3 different outputs
        )

        print("[HF] Generated continuations:\n")

        for i, output_ids in enumerate(output_ids_list, start=1):
            generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            # Clean extra whitespace
            generated_text = " ".join(generated_text.split())
            print(f"--- Continuation {i} ---")
            print(generated_text)
            print()

        print("*"*100)

    except Exception as e:
        print(f"Could not run text generation (requires internet download/setup): {e}")
        exit()

# -------------------------------
# Part 2: Hugging Face – Tokenisation
# -------------------------------
def hf_tokenization_demo() -> None:
    print("--- 2: Hugging Face - Tokenization ---")
    try:
        # Small causal language model (good for demo)
        model_name = "distilgpt2"
        print(f"\n[HF] Loading model and tokenizer: {model_name} ...")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=HF_CACHE_DIR
        )

        text = "LLMs are powerful tools for natural language understanding."
        print(f"\n[HF] Original text:\n{text}\n")

        # Encode: text → token IDs
        encoded = tokenizer(text)
        print("[HF] Token IDs:\n", encoded["input_ids"])

        # Show tokens (subword pieces)
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
        print("\n[HF] Tokens:")
        for tok in tokens:
            print(tok, end=" | ")
        print("\n")

        # Decode: IDs → text
        decoded = tokenizer.decode(encoded["input_ids"])
        print("[HF] Decoded back from IDs:\n", decoded)
        print("\n" + "-" * 60)

    except Exception as e:
        print(f"Could not run text generation (requires internet download/setup): {e}")
        exit()

if __name__ == "__main__":
    # Section A: LLM Foundations & Hugging Face
    # hf_setup_text_generation()
    # print("*" * 100)
    # hf_tokenization_demo()
    print("*" * 100)
