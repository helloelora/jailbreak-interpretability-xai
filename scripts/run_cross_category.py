"""
Cross-category mechanistic analysis of jailbreak compliance flips.

For each validated jailbreak across 4 categories (cybersecurity, malware,
illegal, chemical_biological), compute:
  1. Logit lens on clean and jailbreak prompts (refusal emergence per layer)
  2. Activation patching (causal effect per layer)
  3. IG attribution (top trigger tokens)

The central research question: do the SAME transformer layers control
refusal across all harm categories, or does each category use different
layers? Answer has direct implications for safety robustness.

Usage:
    python -m scripts.run_cross_category \
        --harmbench-dirs dir1 dir2 dir3 ... \
        --output-dir results/cross_category/

Each harmbench-dir should contain seed_*.json files with the
jailbroken_old / jailbroken fields from HarmBench validation.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
from unsloth import FastLanguageModel
from transformers.modeling_outputs import CausalLMOutput

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.attribution.integrated_gradients import (
    compute_attribution, compare_attributions,
)
from src.tracing.activation_analysis import (
    logit_lens, activation_patch, get_compliance_token_ids,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_NAME = "unsloth/Mistral-Small-3.1-24B-Instruct-2503-unsloth-bnb-4bit"


class CausalLMWrapper(nn.Module):
    """Wrap Mistral3 as a nn.Module exposing a standard CausalLM interface.

    nnsight expects a real torch.nn.Module with a ``forward`` method.
    Registering ``model`` and ``lm_head`` as child modules also lets the
    tracing helpers walk the module tree normally.
    """

    def __init__(self, full_model):
        super().__init__()

        if hasattr(full_model, "model") and hasattr(full_model.model, "language_model"):
            self.model = full_model.model.language_model
            self.lm_head = full_model.lm_head
        else:
            self.model = full_model.model if hasattr(full_model, "model") else full_model
            self.lm_head = getattr(full_model, "lm_head", None)

        if self.lm_head is None:
            raise AttributeError("CausalLMWrapper requires an lm_head on the wrapped model")

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


def load_model():
    logger.info(f"Loading {MODEL_NAME}")
    full_model, tok_or_proc = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    full_model.eval()

    # Wrap for uniform CausalLM interface
    if hasattr(full_model, "model") and hasattr(full_model.model, "language_model"):
        model = CausalLMWrapper(full_model)
    elif hasattr(full_model, "language_model"):
        model = full_model.language_model
    else:
        model = full_model

    # Extract tokenizer
    tokenizer = tok_or_proc.tokenizer if hasattr(tok_or_proc, "tokenizer") else tok_or_proc

    logger.info(f"Model ready (class={type(model).__name__})")
    return model, tokenizer


def format_chat(tokenizer, user_message):
    messages = [{"role": "user", "content": user_message}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-category XAI analysis of compliance flips"
    )
    parser.add_argument(
        "--harmbench-dirs", type=str, nargs="+", required=True,
        help="Directories containing HarmBench-validated seed_*.json files",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--n-ig-steps", type=int, default=50)
    parser.add_argument(
        "--max-seeds-per-dir", type=int, default=None,
        help="Skip analysis beyond N seeds per directory (default: all)",
    )
    parser.add_argument(
        "--skip-patching", action="store_true",
        help="Skip activation patching (saves time for quick runs)",
    )
    return parser.parse_args()


def analyze_pair(model, tokenizer, seed_prompt, jailbreak_prompt,
                 comply_ids, refuse_ids, n_ig_steps, skip_patching):
    """Run logit lens (both prompts), activation patching, IG on both prompts."""
    clean_formatted = format_chat(tokenizer, seed_prompt)
    jb_formatted = format_chat(tokenizer, jailbreak_prompt)

    logger.info("  Logit lens (clean)...")
    ll_clean = logit_lens(model, tokenizer, clean_formatted, comply_ids, refuse_ids)

    logger.info("  Logit lens (jailbreak)...")
    ll_jb = logit_lens(model, tokenizer, jb_formatted, comply_ids, refuse_ids)

    patch_result = None
    if not skip_patching:
        logger.info("  Activation patching...")
        patch_result = activation_patch(
            model, tokenizer, clean_formatted, jb_formatted, comply_ids, refuse_ids,
        )

    logger.info("  IG (clean)...")
    ig_clean = compute_attribution(
        model, tokenizer, clean_formatted, comply_ids, refuse_ids,
        n_steps=n_ig_steps, raw_prompt=seed_prompt,
    )

    logger.info("  IG (jailbreak)...")
    ig_jb = compute_attribution(
        model, tokenizer, jb_formatted, comply_ids, refuse_ids,
        n_steps=n_ig_steps, raw_prompt=jailbreak_prompt,
    )

    ig_comparison = compare_attributions(ig_clean, ig_jb)

    return {
        "seed_prompt": seed_prompt,
        "jailbreak_prompt": jailbreak_prompt,
        "logit_lens_clean": {
            "comply_probs": ll_clean["comply_probs"].tolist(),
            "refuse_probs": ll_clean["refuse_probs"].tolist(),
            "gap": ll_clean["gap"].tolist(),
            "crossover_layer": ll_clean["crossover_layer"],
        },
        "logit_lens_jailbreak": {
            "comply_probs": ll_jb["comply_probs"].tolist(),
            "refuse_probs": ll_jb["refuse_probs"].tolist(),
            "gap": ll_jb["gap"].tolist(),
            "crossover_layer": ll_jb["crossover_layer"],
        },
        "activation_patching": None if patch_result is None else {
            "baseline_score": patch_result["baseline_score"],
            "jailbreak_score": patch_result["jailbreak_score"],
            "patched_scores": patch_result["patched_scores"].tolist(),
            "causal_effect": patch_result["causal_effect"].tolist(),
        },
        "ig_comparison": ig_comparison,
        "n_layers": ll_clean["n_layers"],
    }


def category_from_dirname(dirname):
    """Infer category from dir name like 'Cyber_harmbench', 'malware_467668', 'fuzz_505235'."""
    d = os.path.basename(dirname.rstrip("/\\")).lower()
    for key in ["cyber", "malware", "illegal", "chembio", "chemical_biological"]:
        if key in d:
            return key.replace("chemical_biological", "chembio")
    return d


def category_from_seed_filename(fname, fallback):
    """Extract category from 'seed_cybersecurity_0_XXX.json' → 'cybersecurity'."""
    parts = fname.replace("seed_", "").split("_")
    # heuristic: first non-numeric part
    for p in parts:
        if not p[0].isdigit():
            return p
    return fallback


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = load_model()
    comply_ids, refuse_ids = get_compliance_token_ids(tokenizer)
    logger.info(f"Compliance IDs: {comply_ids.tolist()}")
    logger.info(f"Refusal IDs: {refuse_ids.tolist()}")

    all_results = []

    for hb_dir in args.harmbench_dirs:
        dir_category = category_from_dirname(hb_dir)
        logger.info(f"=== Directory: {hb_dir} (category={dir_category}) ===")

        seed_files = sorted(
            f for f in os.listdir(hb_dir)
            if f.startswith("seed_") and f.endswith(".json")
        )

        if args.max_seeds_per_dir is not None:
            seed_files = seed_files[:args.max_seeds_per_dir]

        for fname in seed_files:
            path = os.path.join(hb_dir, fname)
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            seed_prompt = data["seed_prompt"]
            jailbreaks = data.get("jailbreaks", [])

            if not jailbreaks:
                logger.info(f"  {fname}: no validated jailbreaks, skipping")
                continue

            # Pick the highest-scoring jailbreak
            best_jb = max(jailbreaks, key=lambda c: c.get("score", 0))
            jb_prompt = best_jb["prompt"]

            category = category_from_seed_filename(fname, dir_category)
            logger.info(f"  Analyzing {fname} (category={category})")
            logger.info(f"    seed: {seed_prompt[:70]}...")
            logger.info(f"    best jb ({len(jailbreaks)} total): {jb_prompt[:70]}...")

            try:
                result = analyze_pair(
                    model, tokenizer, seed_prompt, jb_prompt,
                    comply_ids, refuse_ids,
                    n_ig_steps=args.n_ig_steps,
                    skip_patching=args.skip_patching,
                )
                result["category"] = category
                result["source_file"] = fname
                result["source_dir"] = hb_dir
                result["n_total_jailbreaks"] = len(jailbreaks)

                all_results.append(result)

                # Save incrementally in case the job gets killed
                out_path = os.path.join(
                    args.output_dir, f"analysis_{category}_{fname.replace('.json', '')}.json"
                )
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False, default=str)

            except Exception as e:
                logger.error(f"Failed on {fname}: {e}", exc_info=True)

    # Per-job summary (unique filename so parallel jobs don't clash).
    # plot_cross_category.py loads individual analysis_*.json files directly
    # so this is just for bookkeeping.
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "n_analyses": len(all_results),
        "categories": sorted(set(r["category"] for r in all_results)),
        "harmbench_dirs": args.harmbench_dirs,
        "analyses": all_results,
    }
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    summary_path = os.path.join(args.output_dir, f"run_summary_{job_id}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"Done. {len(all_results)} analyses saved to {args.output_dir}/")
    logger.info(f"Per-job summary: {summary_path}")


if __name__ == "__main__":
    main()
