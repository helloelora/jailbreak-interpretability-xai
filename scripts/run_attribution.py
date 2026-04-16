"""
Run Integrated Gradients attribution on compliance flip pairs.

For each seed that has validated jailbreaks, computes IG attributions
on the refused prompt and the best jailbroken prompt, then compares them.

Usage:
    python -m scripts.run_attribution <harmbench_validated_dir> [--output-dir <dir>]

Example:
    python -m scripts.run_attribution results/Malware_harmbench/ --output-dir results/attributions/
"""

import argparse
import json
import logging
import os
import sys

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutput

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.attribution.integrated_gradients import compute_attribution, compare_attributions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Same model but loaded in float16 for gradient computation
MODEL_NAME = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"


class CausalLMWrapper(nn.Module):
    """Wrap Mistral3 as a standard nn.Module for IG and tracing utilities."""

    def __init__(self, full_model):
        super().__init__()
        self.model = full_model.model.language_model
        self.lm_head = full_model.lm_head

    @property
    def device(self):
        return self.lm_head.weight.device

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden = outputs[0]
        logits = self.lm_head(hidden)
        return CausalLMOutput(logits=logits)

    def get_input_embeddings(self):
        if hasattr(self.model, "get_input_embeddings"):
            return self.model.get_input_embeddings()
        return self.model.embed_tokens


def format_chat(tokenizer, user_message):
    """Format a prompt using the model's chat template."""
    messages = [{"role": "user", "content": user_message}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def load_model_for_ig(model_name="unsloth/Mistral-Small-3.1-24B-Instruct-2503-unsloth-bnb-4bit"):
    """Load model via Unsloth in 4-bit (same as fuzzer).

    Gradients flow through BitsAndBytes 4-bit layers during dequantization
    (same principle as QLoRA). We only need gradients on the embedding layer
    for Integrated Gradients, not on the weights themselves.
    """
    from unsloth import FastLanguageModel

    logger.info(f"Loading model via Unsloth: {model_name}")

    full_model, tokenizer_or_processor = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    # Do NOT call for_inference() — it disables gradient computation
    full_model.eval()

    # Extract the text-only language model from the multimodal wrapper
    # Mistral3ForConditionalGeneration structure:
    #   full_model.model.language_model = MistralModel (embed_tokens, layers, norm)
    #   full_model.lm_head = nn.Linear
    if hasattr(full_model, "model") and hasattr(full_model.model, "language_model"):
        model = CausalLMWrapper(full_model)
    elif hasattr(full_model, "language_model"):
        model = full_model.language_model
    else:
        model = full_model

    # Extract the tokenizer
    if hasattr(tokenizer_or_processor, "tokenizer"):
        tokenizer = tokenizer_or_processor.tokenizer
    else:
        tokenizer = tokenizer_or_processor

    logger.info(f"Model loaded: {type(model).__name__}, ~14GB VRAM")
    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Run IG attribution on compliance flip pairs")
    parser.add_argument("input_dir", type=str,
                        help="Directory with HarmBench-validated seed_*.json files")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: <input_dir>/attributions/)")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--n-steps", type=int, default=50,
                        help="Number of IG interpolation steps (default: 50)")
    parser.add_argument("--max-pairs", type=int, default=3,
                        help="Max jailbreak prompts to analyze per seed (default: 3)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, "attributions")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model (4-bit via Unsloth, same as fuzzer)
    model, tokenizer = load_model_for_ig()

    # Find seed files
    seed_files = sorted(
        f for f in os.listdir(args.input_dir)
        if f.startswith("seed_") and f.endswith(".json")
    )

    if not seed_files:
        logger.error(f"No seed_*.json files found in {args.input_dir}")
        sys.exit(1)

    logger.info(f"Found {len(seed_files)} seed files")

    for fname in seed_files:
        path = os.path.join(args.input_dir, fname)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        seed_prompt = data["seed_prompt"]
        jailbreaks = data.get("jailbreaks", [])

        if not jailbreaks:
            logger.info(f"Skipping {fname}: no validated jailbreaks")
            continue

        logger.info(f"Processing {fname}: seed={seed_prompt[:60]}...")
        logger.info(f"  {len(jailbreaks)} validated jailbreaks, analyzing top {args.max_pairs}")

        # Format the refused prompt
        refused_prompt = format_chat(tokenizer, seed_prompt)

        # Compute attribution on the refused prompt
        logger.info("  Computing IG on refused prompt...")
        attr_refused = compute_attribution(model, tokenizer, refused_prompt, n_steps=args.n_steps)
        logger.info(f"  Refused: refusal_prob={attr_refused['refusal_prob']:.4f}, "
                     f"{len(attr_refused['tokens'])} tokens")

        # Analyze top jailbreak prompts
        comparisons = []
        for i, jb in enumerate(jailbreaks[:args.max_pairs]):
            jb_prompt_text = jb["prompt"]
            jb_formatted = format_chat(tokenizer, jb_prompt_text)

            logger.info(f"  Computing IG on jailbreak {i}: {jb_prompt_text[:60]}...")
            attr_jb = compute_attribution(model, tokenizer, jb_formatted, n_steps=args.n_steps)
            logger.info(f"  Jailbreak {i}: refusal_prob={attr_jb['refusal_prob']:.4f}")

            comparison = compare_attributions(attr_refused, attr_jb)
            comparison["jailbreak_prompt"] = jb_prompt_text
            comparison["jailbreak_score"] = jb.get("score", None)
            comparisons.append(comparison)

            logger.info(f"  Refusal shift: {comparison['refusal_prob_shift']:.4f}")
            logger.info(f"  Top refusal tokens: {comparison['top_refusal_tokens'][:5]}")
            logger.info(f"  Top comply tokens: {comparison['top_comply_tokens'][:5]}")

        # Save results
        result = {
            "seed_prompt": seed_prompt,
            "n_steps": args.n_steps,
            "model": args.model_name,
            "refused": {
                "prompt": seed_prompt,
                "tokens": attr_refused["tokens"],
                "attributions": attr_refused["attributions"].tolist(),
                "refusal_prob": attr_refused["refusal_prob"],
            },
            "comparisons": comparisons,
        }

        out_path = os.path.join(args.output_dir, fname.replace("seed_", "attr_"))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"  Saved to {out_path}")

    logger.info("Attribution analysis complete.")


if __name__ == "__main__":
    main()
