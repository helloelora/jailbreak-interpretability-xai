"""
Evaluation metrics for jailbreak interpretability analysis.
"""

from scipy.stats import kendalltau


def faithfulness_score(fuzzer_tokens, attribution_tokens):
    """Compute Kendall's Tau between fuzzer-identified trigger tokens
    and Integrated Gradients high-attribution tokens.

    Args:
        fuzzer_tokens: Ranked list of tokens identified by the fuzzer as triggers.
        attribution_tokens: Ranked list of tokens by IG attribution score.

    Returns:
        Kendall's Tau correlation coefficient and p-value.
    """
    # TODO: implement ranking alignment computation
    raise NotImplementedError


def ablation_test(model, tokenizer, prompt, circuits_to_mask):
    """Test whether masking identified circuits restores refusal behavior.

    Args:
        model: The loaded model.
        tokenizer: The model's tokenizer.
        prompt: A previously successful jailbreak prompt.
        circuits_to_mask: List of (layer, head/neuron) tuples to ablate.

    Returns:
        Dictionary with original response, ablated response, and refusal restored flag.
    """
    # TODO: implement circuit ablation
    raise NotImplementedError
