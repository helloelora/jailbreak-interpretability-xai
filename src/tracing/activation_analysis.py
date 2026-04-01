"""
TransformerLens-based activation tracing for Mistral Small 4.
Caches internal activations to locate layers where safety signals are suppressed.
"""


def cache_activations(model, tokenizer, prompt):
    """Run the model and cache all intermediate activations.

    Args:
        model: TransformerLens-wrapped model.
        tokenizer: The model's tokenizer.
        prompt: Input prompt string.

    Returns:
        ActivationCache with all layer activations.
    """
    # TODO: implement activation caching via TransformerLens
    raise NotImplementedError


def compare_traces(cache_refused, cache_jailbroken):
    """Compare activation traces between a refused and jailbroken prompt.

    Identifies layers where the safety signal diverges.

    Args:
        cache_refused: ActivationCache from a refused prompt.
        cache_jailbroken: ActivationCache from a successful jailbreak.

    Returns:
        Dictionary mapping layer indices to divergence scores.
    """
    # TODO: implement trace comparison (cosine similarity, L2 distance per layer)
    raise NotImplementedError


def identify_safety_layers(divergence_map, threshold=0.1):
    """Identify layers most responsible for safety signal suppression.

    Args:
        divergence_map: Output from compare_traces.
        threshold: Minimum divergence to flag a layer.

    Returns:
        List of (layer_index, divergence_score) for flagged layers.
    """
    # TODO: implement layer identification
    raise NotImplementedError
