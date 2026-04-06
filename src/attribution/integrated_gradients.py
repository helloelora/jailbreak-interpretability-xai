"""
Integrated Gradients attribution for analyzing compliance flip.

Computes token-level attribution scores that measure how each input token
contributes to the model's decision to refuse or comply. Uses the logit gap
between refusal tokens and compliance tokens as the target function.

Requires the model to be loaded in float16 (not 4-bit) for gradient computation.

Reference: Sundararajan et al. (2017) "Axiomatic Attribution for Deep Networks"
"""

import logging

import torch
import numpy as np

logger = logging.getLogger(__name__)

# Refusal prefixes — same as in the fuzzer, used to identify refusal token IDs
REFUSAL_PREFIXES = [
    "I cannot", "I can't", "I'm sorry", "Sorry", "I apologize",
    "I'm not able", "I must decline", "I won't", "As an AI",
    "I'm unable", "It is not appropriate", "I don't think",
]


def _get_refusal_token_ids(tokenizer):
    """Get the set of token IDs that correspond to refusal starts."""
    refusal_ids = set()
    for prefix in REFUSAL_PREFIXES:
        tokens = tokenizer.encode(prefix, add_special_tokens=False)
        if tokens:
            refusal_ids.add(tokens[0])
    return refusal_ids


def _refusal_logit_gap(logits, refusal_ids):
    """Compute the logit gap: refusal_logit - compliance_logit.

    A positive value means the model leans toward refusal.
    A negative value means the model leans toward compliance.

    Args:
        logits: shape (vocab_size,) — logits for the next token
        refusal_ids: set of token IDs considered refusal tokens

    Returns:
        Scalar tensor (refusal_mass - compliance_mass in probability space)
    """
    probs = torch.softmax(logits, dim=0)
    refusal_prob = sum(probs[tid] for tid in refusal_ids)
    return refusal_prob


def compute_attribution(model, tokenizer, prompt, n_steps=50):
    """Compute Integrated Gradients attribution for each input token.

    Measures how each token contributes to the refusal probability.
    Positive attribution = token pushes toward refusal.
    Negative attribution = token pushes toward compliance.

    Args:
        model: Language model loaded in float16 (requires gradient support).
        tokenizer: The model's tokenizer.
        prompt: Input prompt string (already formatted with chat template).
        n_steps: Number of interpolation steps (default 50).

    Returns:
        dict with:
            - tokens: list of token strings
            - token_ids: list of token IDs
            - attributions: numpy array of attribution scores per token
            - refusal_prob: float, the refusal probability for this prompt
    """
    refusal_ids = _get_refusal_token_ids(tokenizer)

    # Get the embedding layer and its device
    # With device_map="auto", model.device may not exist — use the embedding's device
    embedding_layer = model.model.embed_tokens
    embed_device = embedding_layer.weight.device

    inputs = tokenizer(prompt, return_tensors="pt").to(embed_device)
    input_ids = inputs["input_ids"]  # (1, seq_len)

    input_embeds = embedding_layer(input_ids).detach()  # (1, seq_len, hidden_dim)

    # Baseline: zero embedding (neutral input)
    baseline_embeds = torch.zeros_like(input_embeds)

    # Accumulate gradients along the interpolation path (memory-efficient)
    accumulated_grads = torch.zeros_like(input_embeds)

    for step in range(n_steps + 1):
        alpha = step / n_steps
        # Interpolated embedding between baseline and actual input
        interp_embeds = baseline_embeds + alpha * (input_embeds - baseline_embeds)
        interp_embeds = interp_embeds.detach().requires_grad_(True)

        # Forward pass using embeddings directly (bypass the embedding lookup)
        outputs = model(inputs_embeds=interp_embeds, attention_mask=inputs["attention_mask"])
        logits = outputs.logits[0, -1, :]  # next-token logits

        # Target: refusal probability
        target = _refusal_logit_gap(logits, refusal_ids)

        # Backward pass to get gradient w.r.t. interpolated embeddings
        model.zero_grad()
        target.backward()

        # Accumulate instead of storing all gradients
        accumulated_grads += interp_embeds.grad.detach()

    # Integrated Gradients: mean of gradients × (input - baseline)
    avg_grads = accumulated_grads / (n_steps + 1)
    diff = input_embeds - baseline_embeds  # (1, seq_len, hidden_dim)
    ig = avg_grads * diff  # (1, seq_len, hidden_dim)

    # Reduce to per-token attribution by summing over the hidden dimension
    attributions = ig.sum(dim=-1).squeeze(0).cpu().numpy()  # (seq_len,)

    # Get the actual refusal probability for this prompt
    with torch.no_grad():
        outputs = model(inputs_embeds=input_embeds, attention_mask=inputs["attention_mask"])
        logits = outputs.logits[0, -1, :]
        refusal_prob = _refusal_logit_gap(logits, refusal_ids).item()

    # Decode tokens for readability
    token_ids = input_ids.squeeze(0).tolist()
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    return {
        "tokens": tokens,
        "token_ids": token_ids,
        "attributions": attributions,
        "refusal_prob": refusal_prob,
    }


def compare_attributions(attr_refused, attr_jailbroken):
    """Compare attributions between a refused prompt and a jailbroken one.

    Identifies tokens in the jailbroken prompt that have the largest
    negative attribution shift (tokens that suppress refusal).

    Args:
        attr_refused: Output of compute_attribution for the refused prompt.
        attr_jailbroken: Output of compute_attribution for the jailbroken prompt.

    Returns:
        dict with:
            - refused: {tokens, attributions, refusal_prob}
            - jailbroken: {tokens, attributions, refusal_prob}
            - refusal_prob_shift: how much the refusal probability changed
            - top_refusal_tokens: tokens with highest refusal attribution in refused prompt
            - top_comply_tokens: tokens with most negative attribution in jailbroken prompt
    """
    # Top tokens pushing toward refusal in the refused prompt
    ref_indices = np.argsort(attr_refused["attributions"])[::-1]
    top_refusal = [
        (attr_refused["tokens"][i], float(attr_refused["attributions"][i]))
        for i in ref_indices[:10]
    ]

    # Top tokens pushing toward compliance in the jailbroken prompt
    jb_indices = np.argsort(attr_jailbroken["attributions"])
    top_comply = [
        (attr_jailbroken["tokens"][i], float(attr_jailbroken["attributions"][i]))
        for i in jb_indices[:10]
    ]

    return {
        "refused": {
            "tokens": attr_refused["tokens"],
            "attributions": attr_refused["attributions"].tolist(),
            "refusal_prob": attr_refused["refusal_prob"],
        },
        "jailbroken": {
            "tokens": attr_jailbroken["tokens"],
            "attributions": attr_jailbroken["attributions"].tolist(),
            "refusal_prob": attr_jailbroken["refusal_prob"],
        },
        "refusal_prob_shift": attr_refused["refusal_prob"] - attr_jailbroken["refusal_prob"],
        "top_refusal_tokens": top_refusal,
        "top_comply_tokens": top_comply,
    }
