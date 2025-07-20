import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

model = None
tokenizer = None

def load_model():
    """
    Lazily loads the tokenizer and model with low CPU memory usage.
    """
    global model, tokenizer
    if model is None or tokenizer is None:
        model_source = "psabhay2003/t5_invoice_model"

        tokenizer = T5Tokenizer.from_pretrained(model_source)
        model = T5ForConditionalGeneration.from_pretrained(
            model_source,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True  # ✅ reduces RAM usage when loading
        )

def extract_invoice_fields(raw_text: str) -> dict:
    """
    Extracts structured invoice fields from raw OCR text using the fine-tuned T5 model.
    """
    load_model()

    input_text = f"Extract invoice fields: {raw_text}"

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )

    # Generate model output
    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=128
    )

    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    # Convert decoded string into dictionary
    result = {}
    for item in decoded_output.split(","):
        if ":" in item:
            key, val = item.split(":", 1)
            result[key.strip()] = val.strip()

    return result
