"""
Mistral Small 3.1 model loading via Unsloth (4-bit quantization).

Mistral Small 3.1 is a multimodal model (Mistral3ForConditionalGeneration).
For text-only jailbreak analysis, we extract the underlying language model
(model.language_model) which is a standard MistralForCausalLM.
"""

import torch
from unsloth import FastLanguageModel


MODEL_NAME = "unsloth/Mistral-Small-3.1-24B-Instruct-2503-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 2048


def _disable_flex_attention(module):
    """Steer generation away from flex-attention code paths that break here."""
    if module is None:
        return
    config = getattr(module, "config", None)
    if config is None:
        return

    for attr in ["_attn_implementation", "attn_implementation"]:
        if hasattr(config, attr):
            setattr(config, attr, "eager")


def load_model(model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LENGTH):
    """Load Mistral Small 3.1 with 4-bit quantization via Unsloth.

    Returns the inner language model (MistralForCausalLM) and tokenizer,
    bypassing the multimodal wrapper.
    """
    try:
        full_model, tokenizer_or_processor = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            fix_mistral_regex=True,
        )
    except TypeError:
        full_model, tokenizer_or_processor = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
        )
    FastLanguageModel.for_inference(full_model)

    # Extract the text-only language model from the multimodal wrapper
    if hasattr(full_model, "language_model"):
        model = full_model.language_model
    else:
        model = full_model

    # Extract the tokenizer from the processor if needed
    if hasattr(tokenizer_or_processor, "tokenizer"):
        tokenizer = tokenizer_or_processor.tokenizer
    else:
        tokenizer = tokenizer_or_processor

    _disable_flex_attention(full_model)
    _disable_flex_attention(getattr(full_model, "model", None))
    _disable_flex_attention(getattr(full_model, "language_model", None))
    _disable_flex_attention(model)

    return model, tokenizer


def _greedy_generate_fallback(model, tokenizer, inputs, max_new_tokens):
    """Fallback decoder when model.generate hits a flex-attention regression."""
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    eos_token_id = tokenizer.eos_token_id
    generated = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=generated, attention_mask=attention_mask)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            if attention_mask is not None:
                next_mask = torch.ones_like(next_token, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, next_mask], dim=1)

            if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                break

    return generated


def generate(model, tokenizer, prompt, max_new_tokens=256):
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    except ValueError as exc:
        if "too many values to unpack" not in str(exc):
            raise
        outputs = _greedy_generate_fallback(model, tokenizer, inputs, max_new_tokens)
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
