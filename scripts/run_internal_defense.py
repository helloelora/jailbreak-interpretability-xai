"""
Run internal defense interventions on jailbreak prompts.

For each analyzed clean/jailbreak pair:
  1. Read the causal layers identified by activation patching
  2. Intervene on the jailbreak prompt at the last token using:
     - zero ablation on the top-k causal layers
     - clean-activation patching on the top-k causal layers
     - matched random-layer controls for both defenses
  3. Measure whether the next-token decision moves back toward refusal

This is a mitigation-oriented follow-up to the mechanistic analysis:
if causal layers really drive jailbreak compliance, intervening on them
should reduce the compliance score and ideally restore a refusal-like token.
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutput
from unsloth import FastLanguageModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.tracing.activation_analysis import (
    cache_last_token_activations,
    get_compliance_token_ids,
    intervene_on_prompt,
    score_prompt,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_NAME = "unsloth/Mistral-Small-3.1-24B-Instruct-2503-unsloth-bnb-4bit"


class CausalLMWrapper(nn.Module):
    """Wrap Mistral3 as a plain causal LM for nnsight tracing."""

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
    """Load and wrap the Unsloth model for tracing-based interventions."""
    logger.info(f"Loading {MODEL_NAME}")
    full_model, tok_or_proc = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    full_model.eval()

    if hasattr(full_model, "model") and hasattr(full_model.model, "language_model"):
        model = CausalLMWrapper(full_model)
    elif hasattr(full_model, "language_model"):
        model = full_model.language_model
    else:
        model = full_model

    tokenizer = tok_or_proc.tokenizer if hasattr(tok_or_proc, "tokenizer") else tok_or_proc
    logger.info(f"Model ready (class={type(model).__name__})")
    return model, tokenizer


def format_chat(tokenizer, user_message):
    messages = [{"role": "user", "content": user_message}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run internal defense interventions")
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Directory containing analysis_*.json files from run_cross_category",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Defaults to <input_dir>/internal_defense/",
    )
    parser.add_argument(
        "--top-k-values", type=int, nargs="+", default=[1, 3, 5],
        help="Evaluate interventions on the top-k causal layers",
    )
    parser.add_argument(
        "--random-trials", type=int, default=5,
        help="Number of matched random-layer control trials per example and k",
    )
    parser.add_argument(
        "--max-examples", type=int, default=None,
        help="Optional cap for debugging or smoke tests",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_analyses(input_dir):
    analyses = []
    for fname in sorted(os.listdir(input_dir)):
        if not (fname.startswith("analysis_") and fname.endswith(".json")):
            continue
        path = os.path.join(input_dir, fname)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("activation_patching") is None:
            continue
        data["analysis_file"] = fname
        analyses.append(data)
    return analyses


def top_positive_layers(causal_effect, k):
    order = np.argsort(np.array(causal_effect))[::-1]
    selected = [int(i) for i in order if causal_effect[i] > 0]
    return selected[:k]


def summarize_intervention(result, baseline_score, refuse_token_ids):
    top_token_id = int(result["top_token_id"])
    restored_by_score = result["score"] < 0
    restored_by_token = top_token_id in refuse_token_ids
    return {
        "score": float(result["score"]),
        "score_drop": float(baseline_score - result["score"]),
        "comply_prob": float(result["comply_prob"]),
        "refuse_prob": float(result["refuse_prob"]),
        "top_token_id": top_token_id,
        "top_token": result["top_token"],
        "top_token_prob": float(result["top_token_prob"]),
        "restored_by_score": bool(restored_by_score),
        "restored_by_token": bool(restored_by_token),
        "zero_layers": result.get("zero_layers", []),
        "replace_layers": result.get("replace_layers", []),
    }


def mean_trial_summary(trials):
    if not trials:
        return None

    top_token = Counter(t["top_token"] for t in trials).most_common(1)[0][0]
    return {
        "score": float(np.mean([t["score"] for t in trials])),
        "score_drop": float(np.mean([t["score_drop"] for t in trials])),
        "comply_prob": float(np.mean([t["comply_prob"] for t in trials])),
        "refuse_prob": float(np.mean([t["refuse_prob"] for t in trials])),
        "top_token": top_token,
        "restored_by_score_rate": float(np.mean([t["restored_by_score"] for t in trials])),
        "restored_by_token_rate": float(np.mean([t["restored_by_token"] for t in trials])),
    }


def run_example(
    model,
    tokenizer,
    analysis,
    comply_ids,
    refuse_ids,
    top_k_values,
    random_trials,
    rng,
):
    seed_prompt = analysis["seed_prompt"]
    jailbreak_prompt = analysis["jailbreak_prompt"]
    category = analysis["category"]
    causal_effect = analysis["activation_patching"]["causal_effect"]
    n_layers = int(analysis["n_layers"])

    clean_formatted = format_chat(tokenizer, seed_prompt)
    jb_formatted = format_chat(tokenizer, jailbreak_prompt)

    clean_baseline = score_prompt(model, tokenizer, clean_formatted, comply_ids, refuse_ids)
    jailbreak_baseline = score_prompt(model, tokenizer, jb_formatted, comply_ids, refuse_ids)
    refuse_token_ids = {int(i) for i in refuse_ids.tolist()}

    max_k = max(top_k_values)
    important_layers = top_positive_layers(causal_effect, max_k)
    clean_cache = cache_last_token_activations(model, tokenizer, clean_formatted)

    result = {
        "analysis_file": analysis["analysis_file"],
        "category": category,
        "seed_prompt": seed_prompt,
        "jailbreak_prompt": jailbreak_prompt,
        "n_layers": n_layers,
        "top_positive_layers": important_layers,
        "clean_baseline": summarize_intervention(clean_baseline, clean_baseline["score"], refuse_token_ids),
        "jailbreak_baseline": summarize_intervention(
            jailbreak_baseline, jailbreak_baseline["score"], refuse_token_ids,
        ),
        "interventions": {},
    }

    all_layers = list(range(n_layers))
    for k in top_k_values:
        selected_layers = important_layers[: min(k, len(important_layers))]
        if not selected_layers:
            logger.warning(f"{analysis['analysis_file']}: no positive causal layers for k={k}")
            continue

        replace_layers = {layer: clean_cache[layer] for layer in selected_layers}
        zero_top = intervene_on_prompt(
            model, tokenizer, jb_formatted, comply_ids, refuse_ids,
            zero_layers=selected_layers,
        )
        patch_top = intervene_on_prompt(
            model, tokenizer, jb_formatted, comply_ids, refuse_ids,
            replace_layers=replace_layers,
        )

        random_zero_trials = []
        random_patch_trials = []
        for _ in range(random_trials):
            pool = [layer for layer in all_layers if layer not in selected_layers]
            if len(pool) < len(selected_layers):
                pool = all_layers
            sampled = sorted(int(x) for x in rng.choice(pool, size=len(selected_layers), replace=False))

            random_zero = intervene_on_prompt(
                model, tokenizer, jb_formatted, comply_ids, refuse_ids,
                zero_layers=sampled,
            )
            random_patch = intervene_on_prompt(
                model, tokenizer, jb_formatted, comply_ids, refuse_ids,
                replace_layers={layer: clean_cache[layer] for layer in sampled},
            )

            random_zero_trials.append(
                {
                    **summarize_intervention(
                        random_zero, jailbreak_baseline["score"], refuse_token_ids,
                    ),
                    "sampled_layers": sampled,
                }
            )
            random_patch_trials.append(
                {
                    **summarize_intervention(
                        random_patch, jailbreak_baseline["score"], refuse_token_ids,
                    ),
                    "sampled_layers": sampled,
                }
            )

        result["interventions"][str(k)] = {
            "selected_layers": selected_layers,
            "zero_ablate_top": summarize_intervention(
                zero_top, jailbreak_baseline["score"], refuse_token_ids,
            ),
            "clean_patch_top": summarize_intervention(
                patch_top, jailbreak_baseline["score"], refuse_token_ids,
            ),
            "zero_ablate_random_trials": random_zero_trials,
            "clean_patch_random_trials": random_patch_trials,
            "zero_ablate_random_mean": mean_trial_summary(random_zero_trials),
            "clean_patch_random_mean": mean_trial_summary(random_patch_trials),
        }

    return result


def aggregate_summary(example_results, top_k_values):
    interventions = [
        "zero_ablate_top",
        "clean_patch_top",
        "zero_ablate_random_mean",
        "clean_patch_random_mean",
    ]

    def aggregate_scope(items):
        summary = {}
        for k in top_k_values:
            rows = [
                ex["interventions"].get(str(k))
                for ex in items
                if str(k) in ex["interventions"]
            ]
            if not rows:
                continue

            summary[str(k)] = {}
            for name in interventions:
                vals = [row[name] for row in rows if row.get(name) is not None]
                if not vals:
                    continue

                restored_score_values = [
                    v.get("restored_by_score", v.get("restored_by_score_rate", 0.0))
                    for v in vals
                ]
                restored_token_values = [
                    v.get("restored_by_token", v.get("restored_by_token_rate", 0.0))
                    for v in vals
                ]

                summary[str(k)][name] = {
                    "n_examples": len(vals),
                    "mean_score": float(np.mean([v["score"] for v in vals])),
                    "mean_score_drop": float(np.mean([v["score_drop"] for v in vals])),
                    "mean_comply_prob": float(np.mean([v["comply_prob"] for v in vals])),
                    "mean_refuse_prob": float(np.mean([v["refuse_prob"] for v in vals])),
                    "restored_by_score_rate": float(np.mean(restored_score_values)),
                    "restored_by_token_rate": float(np.mean(restored_token_values)),
                }
        return summary

    categories = sorted(set(ex["category"] for ex in example_results))
    return {
        "overall": aggregate_scope(example_results),
        "by_category": {
            cat: aggregate_scope([ex for ex in example_results if ex["category"] == cat])
            for cat in categories
        },
    }


def main():
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, "internal_defense")
    os.makedirs(args.output_dir, exist_ok=True)

    analyses = load_analyses(args.input_dir)
    if args.max_examples is not None:
        analyses = analyses[: args.max_examples]
    if not analyses:
        raise SystemExit(f"No analysis_*.json with activation_patching found in {args.input_dir}")

    logger.info(f"Loaded {len(analyses)} analyses from {args.input_dir}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    model, tokenizer = load_model()
    comply_ids, refuse_ids = get_compliance_token_ids(tokenizer)

    example_results = []
    for idx, analysis in enumerate(analyses, start=1):
        logger.info(
            f"[{idx}/{len(analyses)}] {analysis['analysis_file']} "
            f"(category={analysis['category']})"
        )
        result = run_example(
            model, tokenizer, analysis, comply_ids, refuse_ids,
            top_k_values=args.top_k_values,
            random_trials=args.random_trials,
            rng=rng,
        )
        example_results.append(result)

        out_path = os.path.join(
            args.output_dir,
            f"defense_{analysis['analysis_file']}",
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    summary = aggregate_summary(example_results, args.top_k_values)
    final_payload = {
        "input_dir": args.input_dir,
        "model": MODEL_NAME,
        "top_k_values": args.top_k_values,
        "random_trials": args.random_trials,
        "n_examples": len(example_results),
        "examples": example_results,
        "summary": summary,
    }

    results_path = os.path.join(args.output_dir, "defense_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved defense results to {results_path}")


if __name__ == "__main__":
    main()
