"""
Integrated Gradients attribution via Captum's LayerIntegratedGradients.

Hooks into the embedding layer to compute gradients of the compliance score
with respect to the token embeddings. Works on BitsAndBytes 4-bit models via
straight-through gradient estimation (values are approximate but directionally
correct).

Reference: Sundararajan et al. (2017) "Axiomatic Attribution for Deep Networks".
"""

import torch
from captum.attr import LayerIntegratedGradients


def _make_forward_fn(model, comply_ids, refuse_ids):
    """Forward function: input_ids -> compliance score (comply - refuse logit gap)."""

    def forward_fn(input_ids):
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[0, -1, :]
        comply_logit = logits[comply_ids].mean()
        refuse_logit = logits[refuse_ids].mean()
        return (comply_logit - refuse_logit).unsqueeze(0)

    return forward_fn


def _find_user_span(tokenizer, raw_prompt, formatted_input_ids):
    """Find [start, end) token range where raw_prompt appears in chat-formatted IDs.

    Chat templates wrap user content in boilerplate (BOS, system prompt,
    [INST]/[/INST]). For visualization we want to isolate the user span.
    """
    if raw_prompt is None:
        return None
    formatted_ids = formatted_input_ids[0].tolist()

    for variant in [raw_prompt, " " + raw_prompt, "\n" + raw_prompt]:
        raw_ids = tokenizer.encode(variant, add_special_tokens=False)
        n = len(raw_ids)
        if n == 0:
            continue
        for i in range(len(formatted_ids) - n + 1):
            if formatted_ids[i:i + n] == raw_ids:
                return (i, i + n)

    # Fuzzy: longest suffix of raw_ids that appears in formatted
    raw_ids = tokenizer.encode(raw_prompt, add_special_tokens=False)
    for trim in range(1, min(4, len(raw_ids))):
        sub = raw_ids[trim:]
        n = len(sub)
        for i in range(len(formatted_ids) - n + 1):
            if formatted_ids[i:i + n] == sub:
                return (i, i + n)

    return None


def compute_attribution(model, tokenizer, prompt, comply_ids, refuse_ids,
                        n_steps=50, raw_prompt=None):
    """Compute token-level IG attribution for a prompt.

    Args:
        model: Language model.
        tokenizer: Tokenizer.
        prompt: Chat-formatted prompt string.
        comply_ids, refuse_ids: Compliance/refusal token ID tensors.
        n_steps: IG interpolation steps (default 50).
        raw_prompt: Original user content before chat formatting.
            Used to locate the user-content span for visualization.

    Returns:
        dict with:
            - tokens: list of token strings
            - attributions: tensor of per-token IG scores (L1 normalized)
            - compliance_score: scalar comply - refuse score
            - user_span: (start, end) tuple into tokens, or None
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]

    comply_ids = comply_ids.to(model.device)
    refuse_ids = refuse_ids.to(model.device)

    embedding_layer = model.get_input_embeddings()
    forward_fn = _make_forward_fn(model, comply_ids, refuse_ids)

    lig = LayerIntegratedGradients(forward_fn, embedding_layer)
    attributions = lig.attribute(
        inputs=input_ids,
        baselines=None,
        n_steps=n_steps,
        internal_batch_size=1,
    )

    # Sum over embedding dim, normalize by L1
    attr = attributions.sum(dim=-1).squeeze(0).float()
    attr = attr / (attr.abs().sum() + 1e-10)

    tokens = [tokenizer.decode([tid]) for tid in input_ids[0]]
    user_span = _find_user_span(tokenizer, raw_prompt, input_ids.cpu())

    with torch.no_grad():
        score = forward_fn(input_ids).item()

    return {
        "tokens": tokens,
        "attributions": attr.detach().cpu(),
        "compliance_score": score,
        "user_span": user_span,
    }


def compare_attributions(attr_clean, attr_jailbreak, k=10):
    """Summary comparison between clean and jailbreak attributions.

    Prompts have different tokenizations so we report top-k tokens per prompt
    rather than token-aligned diffs.
    """
    def top_tokens(attr_dict):
        scores = attr_dict["attributions"]
        tokens = attr_dict["tokens"]
        span = attr_dict.get("user_span")
        if span is not None:
            start, end = span
            scores = scores[start:end]
            tokens = tokens[start:end]
        if len(scores) == 0:
            return []
        top_idx = scores.abs().topk(min(k, len(scores))).indices
        return [(tokens[i], scores[i].item()) for i in top_idx]

    return {
        "clean_compliance_score": attr_clean["compliance_score"],
        "jailbreak_compliance_score": attr_jailbreak["compliance_score"],
        "score_shift": attr_jailbreak["compliance_score"] - attr_clean["compliance_score"],
        "clean_top_tokens": top_tokens(attr_clean),
        "jailbreak_top_tokens": top_tokens(attr_jailbreak),
    }
