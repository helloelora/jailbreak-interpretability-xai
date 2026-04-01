"""
Integrated Gradients attribution for analyzing compliance flip.
Uses Captum to compute token-level attribution scores.
"""


def compute_attribution(model, tokenizer, prompt, target_token_ids=None):
    """Compute Integrated Gradients attribution for input tokens.

    Measures how each input token contributes to the logit gap
    between refusal and compliance output tokens.

    Args:
        model: The loaded language model.
        tokenizer: The model's tokenizer.
        prompt: Input prompt string.
        target_token_ids: Optional dict with 'refuse' and 'comply' token IDs.

    Returns:
        Dictionary with token strings and their attribution scores.
    """
    # TODO: implement IG attribution via Captum
    raise NotImplementedError


def compare_attributions(attribution_clean, attribution_jailbreak):
    """Compare attribution maps between a refused prompt and a successful jailbreak.

    Args:
        attribution_clean: Attribution dict for the clean (refused) prompt.
        attribution_jailbreak: Attribution dict for the jailbroken prompt.

    Returns:
        Dictionary highlighting the tokens with the largest attribution shift.
    """
    # TODO: implement attribution comparison
    raise NotImplementedError
