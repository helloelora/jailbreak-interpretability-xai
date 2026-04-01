"""
Mistral Small 3.1 model loading via Unsloth (4-bit quantization).
"""

import torch
from unsloth import FastLanguageModel


MODEL_NAME = "unsloth/Mistral-Small-3.1-24B-Instruct-2503-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 2048


def load_model(model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LENGTH):
    """Load Mistral Small 3.1 with 4-bit quantization via Unsloth."""
    model, tokenizer_or_processor = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    # Unsloth may return a PixtralProcessor (multimodal) instead of a tokenizer.
    # Extract the underlying tokenizer for text-only use.
    if hasattr(tokenizer_or_processor, "tokenizer"):
        tokenizer = tokenizer_or_processor.tokenizer
    else:
        tokenizer = tokenizer_or_processor

    return model, tokenizer


def generate(model, tokenizer, prompt, max_new_tokens=256):
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response


def get_first_token_logits(model, tokenizer, prompt):
    """Get logits for the first generated token (used for compliance scoring).

    Returns the full logit vector over the vocabulary for the next token
    after the prompt. This lets us measure the refusal vs. compliance signal.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    # Logits at the last prompt position = distribution over first generated token
    return outputs.logits[0, -1, :]


def format_chat(tokenizer, user_message, system_message=None):
    """Format a prompt using the model's chat template."""
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
