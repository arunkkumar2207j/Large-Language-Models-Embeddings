# practice.py

# ✅ STEP 1: Set HF cache directory FIRST — before any imports!
import os
os.environ["HF_HOME"] = "C:/Arun/IIT-M/hf-cache"  # Short, safe path

# ✅ STEP 2: Now import everything
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# ✅ STEP 3: Create cache dir (optional but safe)
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

# ✅ STEP 4: Load model and tokenizer
model_name = "facebook/opt-350m"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set pad_token
tokenizer.pad_token = tokenizer.eos_token

# ✅ STEP 5: Create pipeline
generatorTF = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=60,    # ✅ plural!
    do_sample=False,
    device="cpu"
)

# ✅ STEP 6: Run prompt
prompt_summarization = "Summarize in 30 words: Renewable energy is key to fighting climate change."
output = generatorTF(prompt_summarization)[0]["generated_text"]
response = output[len(prompt_summarization):].strip()
print("Response:", response)