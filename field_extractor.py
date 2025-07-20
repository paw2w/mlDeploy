import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Constants
LOCAL_MODEL_DIR = "./t5_invoice_model"
HUGGINGFACE_REPO = "psabhay2003/t5_invoice_model"

# Globals for lazy loading
model = None
tokenizer = None

def load_model():
    global model, tokenizer

    if model is None or tokenizer is None:
        # Use local if available, else Hugging Face
        model_source = LOCAL_MODEL_DIR if os.path.isdir(LOCAL_MODEL_DIR) else HUGGINGFACE_REPO

        print(f"Loading model from: {model_source}")

        # Load tokenizer
        tokenizer = T5Tokenizer.from_pretrained(model_source)

        # Load model (avoid accelerate dependency)
        model = T5ForConditionalGeneration.from_pretrained(
            model_source,
            torch_dtype=torch.float32
        )

def extract_invoice_fields(raw_text: str) -> dict:
    """
    Extract structured invoice fields from raw OCR text using a fine-tuned T5 model.
    """
    load_model()  # Ensure lazy loading

    input_text = f"Extract invoice fields: {raw_text}"

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )

    # Generate output from model
    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=128
    )

    # Decode the model output
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    # Parse into key-value dictionary
    result = {}
    for item in decoded_output.split(","):
        if ":" in item:
            key, val = item.split(":", 1)
            result[key.strip()] = val.strip()

    return result
