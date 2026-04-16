"""
Activation tracing for mechanistic interpretability of jailbreak compliance flips.

Two techniques:
1. Logit lens: project each layer's hidden state through the lm_head to see
   when the model's "decision" (refuse vs comply) crystallizes across depth.
2. Activation patching: replace a specific layer's output at the last token
   with the jailbreak's hidden state. Layers where this shifts the score
   toward compliance are causally responsible for the flip.

Uses nnsight for clean hook access on Unsloth-loaded 4-bit Mistral models.
Works with both plain HF CausalLM and multimodal wrappers.
"""

import logging

import torch
import torch.nn.functional as F
from nnsight import NNsight

logger = logging.getLogger(__name__)


# Attribute paths to walk when locating model components. Covers plain HF
# CausalLM, Unsloth wrappers, and the Mistral3ForConditionalGeneration
# multimodal wrapper used by Mistral Small 3.1.
_CANDIDATE_PATHS = [
    (),
    ("model",),
    ("language_model",),
    ("language_model", "model"),
    ("model", "language_model"),
    ("model", "language_model", "model"),
]


def _traverse(obj, path):
    for a in path:
        obj = getattr(obj, a)
    return obj


def _find_parent_path(model, attr_name):
    """Return the attr path whose endpoint contains `attr_name`."""
    for path in _CANDIDATE_PATHS:
        obj = model
        ok = True
        for a in path:
            if not hasattr(obj, a):
                ok = False
                break
            obj = getattr(obj, a)
        if ok and hasattr(obj, attr_name):
            return path
    return None


def get_model_info(model):
    """Detect model structure and return component paths.

    Returns dict with: n_layers, layers_path, norm_path, lm_head_path.
    """
    layers_path = _find_parent_path(model, "layers")
    if layers_path is None:
        raise AttributeError(
            f"Cannot find transformer layers. Class: {type(model).__name__}"
        )
    n_layers = len(_traverse(model, layers_path).layers)
    return {
        "n_layers": n_layers,
        "layers_path": layers_path,
        "norm_path": _find_parent_path(model, "norm"),
        "lm_head_path": _find_parent_path(model, "lm_head"),
    }


def _wrapped_layer(wrapped, i, info):
    return _traverse(wrapped, info["layers_path"]).layers[i]


def _raw_lm_head(model, info):
    if info["lm_head_path"] is None:
        return None
    return _traverse(model, info["lm_head_path"]).lm_head


def _raw_norm(model, info):
    if info["norm_path"] is None:
        return None
    return _traverse(model, info["norm_path"]).norm


def _last_token(t):
    """Last token's hidden state from a saved nnsight tensor."""
    if t.dim() == 3:
        return t[0, -1, :]
    if t.dim() == 2:
        return t[-1, :]
    raise ValueError(f"Unexpected shape: {tuple(t.shape)}")


def logit_lens(model, tokenizer, prompt, comply_ids, refuse_ids):
    """Project each layer's last-token hidden state through the lm_head.

    Returns per-layer P(comply) and P(refuse) at the final token position —
    shows when the model's decision to refuse/comply emerges across depth.

    Args:
        model: The language model.
        tokenizer: The model's tokenizer.
        prompt: Formatted prompt string.
        comply_ids: Tensor of compliance token IDs (e.g., "Sure", "Here").
        refuse_ids: Tensor of refusal token IDs (e.g., "I", "Sorry").

    Returns:
        dict with:
            - comply_probs: (n_layers,) P(comply) per layer
            - refuse_probs: (n_layers,) P(refuse) per layer
            - gap: (n_layers,) comply - refuse per layer
            - crossover_layer: first layer where comply > refuse (or None)
            - n_layers: int
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    info = get_model_info(model)
    n_layers = info["n_layers"]

    comply_ids = comply_ids.to(model.device)
    refuse_ids = refuse_ids.to(model.device)

    lm_head = _raw_lm_head(model, info)
    norm = _raw_norm(model, info)
    if lm_head is None:
        raise RuntimeError("logit_lens needs lm_head")

    # Cache per-layer hidden states via nnsight
    wrapped = NNsight(model)
    saved = []
    with torch.no_grad():
        with wrapped.trace(input_ids):
            for i in range(n_layers):
                saved.append(_wrapped_layer(wrapped, i, info).output[0].save())

    comply_probs, refuse_probs = [], []
    with torch.no_grad():
        for i in range(n_layers):
            h = _last_token(saved[i]).to(model.device)
            if norm is not None:
                h = norm(h.unsqueeze(0)).squeeze(0)
            logits = lm_head(h)
            probs = torch.softmax(logits.float(), dim=-1)
            comply_probs.append(probs[comply_ids].sum().item())
            refuse_probs.append(probs[refuse_ids].sum().item())

    comply_probs = torch.tensor(comply_probs)
    refuse_probs = torch.tensor(refuse_probs)
    gap = comply_probs - refuse_probs

    crossover = None
    for i in range(n_layers):
        if gap[i].item() > 0:
            crossover = i
            break

    return {
        "comply_probs": comply_probs,
        "refuse_probs": refuse_probs,
        "gap": gap,
        "crossover_layer": crossover,
        "n_layers": n_layers,
    }


def activation_patch(model, tokenizer, clean_prompt, jailbreak_prompt,
                     comply_ids, refuse_ids):
    """Patch each layer's last-token hidden state with the jailbreak's value.

    For each layer i: run the CLEAN prompt but overwrite layer i's output
    at the last token with the JAILBREAK prompt's layer i hidden state.
    Measure how much the compliance score shifts — layers with large shifts
    are causally responsible for the flip.

    Args:
        model: The language model.
        tokenizer: The model's tokenizer.
        clean_prompt: Prompt that gets refused.
        jailbreak_prompt: Mutated prompt that gets compliance.
        comply_ids, refuse_ids: Token IDs for scoring.

    Returns:
        dict with:
            - baseline_score: compliance score on clean prompt
            - jailbreak_score: compliance score on jailbreak prompt
            - patched_scores: (n_layers,) score after patching each layer
            - causal_effect: patched_scores - baseline_score per layer
            - n_layers: int
    """
    info = get_model_info(model)
    n_layers = info["n_layers"]

    comply_ids = comply_ids.to(model.device)
    refuse_ids = refuse_ids.to(model.device)

    def score(logits):
        return (logits[comply_ids].mean() - logits[refuse_ids].mean()).item()

    # Cache jailbreak hidden states
    jb_inputs = tokenizer(jailbreak_prompt, return_tensors="pt").to(model.device)
    jb_ids = jb_inputs["input_ids"]

    wrapped = NNsight(model)
    jb_saved = []
    with torch.no_grad():
        with wrapped.trace(jb_ids):
            for i in range(n_layers):
                jb_saved.append(_wrapped_layer(wrapped, i, info).output[0].save())

    # Baseline scores (no patching)
    with torch.no_grad():
        jb_out = model(jb_ids)
        jb_score = score(jb_out.logits[0, -1, :])

    clean_inputs = tokenizer(clean_prompt, return_tensors="pt").to(model.device)
    clean_ids = clean_inputs["input_ids"]
    with torch.no_grad():
        clean_out = model(clean_ids)
        baseline = score(clean_out.logits[0, -1, :])

    logger.info(f"Baseline={baseline:.4f}, Jailbreak={jb_score:.4f}")

    # Need lm_head proxy to read patched logits
    lm_head_path = info["lm_head_path"]
    if lm_head_path is None:
        raise RuntimeError("activation_patch needs lm_head")

    patched = []
    for layer_idx in range(n_layers):
        with torch.no_grad():
            with wrapped.trace(clean_ids):
                layer = _wrapped_layer(wrapped, layer_idx, info)
                jb_last = _last_token(jb_saved[layer_idx])
                layer.output[0][-1, :] = jb_last
                lm_head_proxy = _traverse(wrapped, lm_head_path).lm_head
                patched_logits = lm_head_proxy.output.save()

        patched.append(score(_last_token(patched_logits)))

        if (layer_idx + 1) % 10 == 0:
            logger.info(f"Patched {layer_idx + 1}/{n_layers} layers")

    patched = torch.tensor(patched)
    return {
        "baseline_score": baseline,
        "jailbreak_score": jb_score,
        "patched_scores": patched,
        "causal_effect": patched - baseline,
        "n_layers": n_layers,
    }


def get_compliance_token_ids(tokenizer):
    """Standard compliance vs refusal first-token sets.

    Returns (comply_ids, refuse_ids) as tensors.
    """
    comply = ["Sure", "Here", "Okay", "Below", "The", "To"]
    refuse = ["I", "Sorry", "As", "Unfortunately", "This"]

    comply_ids, refuse_ids = [], []
    for t in comply:
        ids = tokenizer.encode(t, add_special_tokens=False)
        if ids:
            comply_ids.append(ids[0])
    for t in refuse:
        ids = tokenizer.encode(t, add_special_tokens=False)
        if ids:
            refuse_ids.append(ids[0])

    return torch.tensor(comply_ids), torch.tensor(refuse_ids)


def _score_from_logits(logits, comply_ids, refuse_ids):
    """Return scalar compliance score and probability masses from next-token logits."""
    probs = F.softmax(logits.float(), dim=-1)
    comply_prob = probs[comply_ids].sum().item()
    refuse_prob = probs[refuse_ids].sum().item()
    score = logits[comply_ids].mean().item() - logits[refuse_ids].mean().item()
    top_token_id = int(probs.argmax().item())
    top_token_prob = float(probs[top_token_id].item())

    return {
        "score": score,
        "comply_prob": comply_prob,
        "refuse_prob": refuse_prob,
        "top_token_id": top_token_id,
        "top_token_prob": top_token_prob,
    }


def score_prompt(model, tokenizer, prompt, comply_ids, refuse_ids):
    """Score the next-token refusal/compliance tendency for a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    comply_ids = comply_ids.to(model.device)
    refuse_ids = refuse_ids.to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]

    result = _score_from_logits(logits, comply_ids, refuse_ids)
    result["top_token"] = tokenizer.decode([result["top_token_id"]])
    return result


def cache_last_token_activations(model, tokenizer, prompt, layer_indices=None):
    """Cache last-token hidden states for selected layers on a given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    info = get_model_info(model)
    n_layers = info["n_layers"]

    if layer_indices is None:
        layer_indices = list(range(n_layers))
    else:
        layer_indices = sorted(set(int(i) for i in layer_indices))

    wrapped = NNsight(model)
    saved = {}
    with torch.no_grad():
        with wrapped.trace(input_ids):
            for i in layer_indices:
                saved[i] = _wrapped_layer(wrapped, i, info).output[0].save()

    cached = {}
    for i, tensor in saved.items():
        cached[i] = _last_token(tensor).detach().to(model.device).clone()
    return cached


def intervene_on_prompt(
    model,
    tokenizer,
    prompt,
    comply_ids,
    refuse_ids,
    zero_layers=None,
    replace_layers=None,
):
    """Apply last-token layer interventions and score the resulting next token.

    Args:
        model: Language model.
        tokenizer: Model tokenizer.
        prompt: Chat-formatted prompt string.
        comply_ids, refuse_ids: Token ID tensors.
        zero_layers: Iterable of layer indices to zero at the last token.
        replace_layers: Dict[layer_idx -> hidden_state_tensor] used to overwrite
            the last-token hidden state at selected layers.

    Returns:
        dict with score / probability masses / top-token info.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    info = get_model_info(model)
    lm_head_path = info["lm_head_path"]
    if lm_head_path is None:
        raise RuntimeError("intervene_on_prompt needs lm_head")

    comply_ids = comply_ids.to(model.device)
    refuse_ids = refuse_ids.to(model.device)
    zero_layers = set(int(i) for i in (zero_layers or []))
    replace_layers = {
        int(i): hidden.to(model.device) for i, hidden in (replace_layers or {}).items()
    }
    zero_layers -= set(replace_layers.keys())

    wrapped = NNsight(model)
    with torch.no_grad():
        with wrapped.trace(input_ids):
            for layer_idx in sorted(zero_layers):
                layer = _wrapped_layer(wrapped, layer_idx, info)
                layer.output[0][-1, :] = 0

            for layer_idx, hidden in sorted(replace_layers.items()):
                layer = _wrapped_layer(wrapped, layer_idx, info)
                layer.output[0][-1, :] = hidden

            lm_head_proxy = _traverse(wrapped, lm_head_path).lm_head
            patched_logits = lm_head_proxy.output.save()

    logits = _last_token(patched_logits)
    result = _score_from_logits(logits, comply_ids, refuse_ids)
    result["top_token"] = tokenizer.decode([result["top_token_id"]])
    result["zero_layers"] = sorted(zero_layers)
    result["replace_layers"] = sorted(replace_layers.keys())
    return result
