import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Choose a folder that definitely exists / you can control
HF_CACHE_DIR = r"C:\Arun\IIT-M\hf-cache"
os.makedirs(HF_CACHE_DIR, exist_ok=True)

# Helper: generate text for a given prompt
def generate_with_model(tokenizer, model, prompt:str, max_length:int=80) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    output_ids = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Clean whitespace
    return " ".join(text.split())

def compare_prompt_outputs() -> None:
    print("--- Prompt Comparison: Original vs Rephrased ---")
    try:
        model_name="distilgpt2"
        print(f"\n[HF] Loading model and tokenizer: {model_name} ...")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=HF_CACHE_DIR
        )
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=HF_CACHE_DIR
        )
        model.config.pad_token_id = tokenizer.eos_token_id

        # ---------- Task 1: Summarisation ----------
        summarization_text(tokenizer, model)

        # ---------- Task 2: Q&A ----------
        qa_text(tokenizer, model)

        # ---------- Task 3: Creative Text Generation ----------
        creative_text_generation_text(tokenizer, model)

    except Exception as e:
        print(e)


# ---------- Task 1: Summarisation ----------
def summarization_text(tokenizer, model):
    summ_original = (
        "Summarise in 30 words or fewer: "
        "AI is transforming industries by automating routine tasks, enabling data-driven decisions, "
        "and creating personalised user experiences across healthcare, finance, education, and retail."
    )
    summ_rephrased = (
        "In at most 30 words, give a concise summary of how AI is changing industries such as healthcare, "
        "finance, education, and retail."
    )

    print("\n=== Task 1: Summarisation (≤ 30 words) ===\n")
    out_summ_orig = generate_with_model(tokenizer, model, summ_original, max_length=60)
    out_summ_reph = generate_with_model(tokenizer, model, summ_rephrased, max_length=60)

    print("[Original Prompt]:")
    print(summ_original)
    print("\n[Output]:")
    print(out_summ_orig)
    print("\n" + "-" * 60)

    print("\n[Rephrased Prompt]:")
    print(summ_rephrased)
    print("\n[Output]:")
    print(out_summ_reph)
    print("\n" + "=" * 80)

# ---------- Task 2: Q&A ----------
def qa_text(tokenizer, model):
    qa_original = (
        "Question: Which planet is known as the Red Planet?\n"
        "Answer:"
    )
    qa_rephrased = (
        "You are a factual assistant. Answer briefly.\n"
        "Q: Which planet in our solar system is commonly called the Red Planet?\n"
        "A:"
    )

    print("\n=== Task 2: Q&A (Factual) ===\n")
    out_qa_orig = generate_with_model(tokenizer, model, qa_original, max_length=40)
    out_qa_reph = generate_with_model(tokenizer, model, qa_rephrased, max_length=40)

    print("[Original Prompt]:")
    print(qa_original)
    print("\n[Output]:")
    print(out_qa_orig)
    print("\n" + "-" * 60)

    print("\n[Rephrased Prompt]:")
    print(qa_rephrased)
    print("\n[Output]:")
    print(out_qa_reph)
    print("\n" + "=" * 80)

# ---------- Task 3: Creative Text Generation ----------
def creative_text_generation_text(tokenizer, model):
    poem_original = "Write a 4-line rhyming poem about AI helping humans at work."
    poem_rephrased = (
        "Compose a short 4-line poem about artificial intelligence supporting people "
        "in their daily jobs. Make it positive and imaginative."
    )

    print("\n=== Task 3: Creative Text Generation (4-line poem) ===\n")
    out_poem_orig = generate_with_model(tokenizer, model, poem_original, max_length=40)
    out_poem_reph = generate_with_model(tokenizer, model, poem_rephrased, max_length=40)

    print("[Original Prompt]:")
    print(poem_original)
    print("\n[Output]:")
    print(out_poem_orig)
    print("\n" + "-" * 60)

    print("\n[Rephrased Prompt]:")
    print(poem_rephrased)
    print("\n[Output]:")
    print(out_poem_reph)
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # other demos...
    compare_prompt_outputs()