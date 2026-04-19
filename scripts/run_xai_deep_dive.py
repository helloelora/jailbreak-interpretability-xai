"""
Run all XAI analyses on validated jailbreak pairs.

Loads the Mistral model once, then for selected jailbreak examples:
1. Integrated Gradients — token-level attribution
2. Divergence heatmap — layer-wise activation comparison
3. Activation patching — causal identification of compliance-critical layers
4. Logit lens — layer-by-layer comply/refuse probability tracking
5. Ablation test — validate attributions by masking top tokens

Produces detailed multi-panel visualizations and cross-seed comparisons.
Results and plots saved to <run_dir>/xai_analysis/ (or --out-subdir).

This is the initial deep-dive pipeline that operates on per-example
clean/jailbreak pairs (e.g., the keylogger and SQL injection cases).
For the cross-category style-matched analysis, see run_cross_category.py.
"""

import argparse
import json
import logging
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.loader import load_model, format_chat
from src.attribution.integrated_gradients import (
    compute_attribution,
    compare_attributions,
    get_compliance_token_ids,
)
from src.tracing.activation_analysis import (
    cache_activations,
    compute_divergence_heatmap,
    activation_patch_by_layer,
    logit_lens,
)
from src.evaluation.metrics import ablation_test
from scripts.plot_xai_dashboard import (
    plot_token_heatmap,
    plot_example_dashboard,
    plot_cross_seed_divergence,
    plot_cross_seed_summary,
    plot_compliance_score_landscape,
    plot_logit_lens_single,
    plot_logit_lens_comparison,
    plot_cross_seed_logit_lens,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def select_examples(run_dir, n_per_seed=1):
    """Select the top non-degenerate jailbreaks from each seed for analysis.

    A "degenerate" jailbreak is one whose prompt equals the seed prompt — the
    fuzzer sometimes records these when the model happens to comply on the
    unmutated seed. They carry no jailbreak signal (clean == jailbreak), so
    we skip them here. Among the remaining jailbreaks we take the top
    n_per_seed by harmbench_score.
    """
    revised_dir = os.path.join(run_dir, "harmbench_revised")
    examples = []

    for fname in sorted(os.listdir(revised_dir)):
        if not fname.startswith("seed_"):
            continue
        path = os.path.join(revised_dir, fname)
        with open(path) as f:
            data = json.load(f)

        seed_prompt = data["seed_prompt"]
        jailbreaks = data.get("jailbreaks_harmbench_revised", [])
        if not jailbreaks:
            continue

        # Drop degenerate jailbreaks whose prompt equals the seed.
        non_degenerate = [
            jb for jb in jailbreaks
            if jb["prompt"].strip() != seed_prompt.strip()
        ]
        if not non_degenerate:
            continue

        # Prefer the most structurally distinct mutations when there are many
        # ties on harmbench_score. Sort by (-harmbench_score, -length diff)
        # so longer rewrites win ties over near-identical typo mutations.
        def sort_key(jb):
            score = jb.get("harmbench_score", 0)
            len_diff = abs(len(jb["prompt"]) - len(seed_prompt))
            return (-score, -len_diff)

        sorted_jbs = sorted(non_degenerate, key=sort_key)
        for jb in sorted_jbs[:n_per_seed]:
            examples.append({
                "seed_prompt": seed_prompt,
                "jailbreak_prompt": jb["prompt"],
                "response": jb["response"],
                "harmbench_score": jb.get("harmbench_score", 0),
                "seed_file": fname,
            })

    return examples


def analyze_example(model, tokenizer, example, comply_ids, refuse_ids, out_dir, idx):
    """Run full XAI analysis pipeline on one jailbreak example."""
    seed = example["seed_prompt"]
    jb_prompt_raw = example["jailbreak_prompt"]

    clean_formatted = format_chat(tokenizer, seed)
    jb_formatted = format_chat(tokenizer, jb_prompt_raw)

    prefix = f"example_{idx}"
    logger.info(f"=== Example {idx}: {seed[:60]}... ===")
    logger.info(f"  Jailbreak prompt: {jb_prompt_raw[:80]}...")

    results = {"seed_prompt": seed, "jailbreak_prompt": jb_prompt_raw}
    ig_clean = ig_jb = divergence = patching = ablation_res = None
    ll_clean = ll_jb = None

    # --- 1. Integrated Gradients ---
    logger.info("  [1/5] Integrated Gradients...")
    t0 = time.time()
    try:
        ig_clean = compute_attribution(model, tokenizer, clean_formatted,
                                       comply_ids, refuse_ids, n_steps=30,
                                       raw_prompt=seed)
        ig_jb = compute_attribution(model, tokenizer, jb_formatted,
                                    comply_ids, refuse_ids, n_steps=30,
                                    raw_prompt=jb_prompt_raw)
        comparison = compare_attributions(ig_clean, ig_jb)
        results["ig"] = comparison

        plot_token_heatmap(
            ig_clean,
            f"Token Attribution (Clean/Refused) — {seed}",
            os.path.join(out_dir, f"{prefix}_tokens_clean.png"),
        )
        plot_token_heatmap(
            ig_jb,
            f"Token Attribution (Jailbreak/Complied) — seed: {seed}",
            os.path.join(out_dir, f"{prefix}_tokens_jailbreak.png"),
        )

        logger.info(f"    Score shift: {comparison['score_shift']:.4f} ({time.time()-t0:.1f}s)")
    except Exception as e:
        logger.error(f"    IG failed: {e}", exc_info=True)
        results["ig_error"] = str(e)

    # --- 2. Divergence heatmap ---
    logger.info("  [2/5] Divergence heatmap...")
    t0 = time.time()
    try:
        cache_clean = cache_activations(model, tokenizer, clean_formatted)
        cache_jb = cache_activations(model, tokenizer, jb_formatted)
        divergence = compute_divergence_heatmap(cache_clean, cache_jb)

        results["divergence"] = {
            "last_pos": divergence["last_pos_divergence"].tolist(),
            "avg": divergence["layer_avg_divergence"].tolist(),
            "peak_layer_last_pos": int(divergence["last_pos_divergence"].argmax()),
            "peak_layer_avg": int(divergence["layer_avg_divergence"].argmax()),
            "n_layers": divergence["n_layers"],
        }

        logger.info(f"    Peak divergence at layer {results['divergence']['peak_layer_last_pos']} "
                    f"({time.time()-t0:.1f}s)")
    except Exception as e:
        logger.error(f"    Divergence failed: {e}", exc_info=True)
        results["divergence_error"] = str(e)

    # --- 3. Activation patching ---
    logger.info("  [3/5] Activation patching...")
    t0 = time.time()
    try:
        patch_results = activation_patch_by_layer(
            model, tokenizer, clean_formatted, jb_formatted,
            comply_ids, refuse_ids,
        )
        top_causal = patch_results["causal_effect"].topk(5)
        patching = patch_results
        results["patching"] = {
            "baseline_score": patch_results["baseline_score"],
            "jailbreak_score": patch_results["jailbreak_score"],
            "top_causal_layers": top_causal.indices.tolist(),
            "top_causal_effects": top_causal.values.tolist(),
            "all_effects": patch_results["causal_effect"].tolist(),
            "n_layers": patch_results["n_layers"],
        }

        logger.info(f"    Top causal layers: {top_causal.indices.tolist()} ({time.time()-t0:.1f}s)")
    except Exception as e:
        logger.error(f"    Patching failed: {e}", exc_info=True)
        results["patching_error"] = str(e)

    # --- 4. Logit Lens ---
    logger.info("  [4/5] Logit lens...")
    t0 = time.time()
    try:
        ll_clean = logit_lens(model, tokenizer, clean_formatted,
                              comply_ids, refuse_ids, top_k=10)
        ll_jb = logit_lens(model, tokenizer, jb_formatted,
                           comply_ids, refuse_ids, top_k=10)

        results["logit_lens_clean"] = {
            "comply_probs": ll_clean["comply_probs"].tolist(),
            "refuse_probs": ll_clean["refuse_probs"].tolist(),
            "comply_minus_refuse": ll_clean["comply_minus_refuse"].tolist(),
            "top_tokens": ll_clean["top_tokens"],
            "top_probs": ll_clean["top_probs"],
            "crossover_layer": ll_clean["crossover_layer"],
            "n_layers": ll_clean["n_layers"],
        }
        results["logit_lens_jailbreak"] = {
            "comply_probs": ll_jb["comply_probs"].tolist(),
            "refuse_probs": ll_jb["refuse_probs"].tolist(),
            "comply_minus_refuse": ll_jb["comply_minus_refuse"].tolist(),
            "top_tokens": ll_jb["top_tokens"],
            "top_probs": ll_jb["top_probs"],
            "crossover_layer": ll_jb["crossover_layer"],
            "n_layers": ll_jb["n_layers"],
        }

        plot_logit_lens_single(ll_clean,
                               f"Logit Lens (Clean/Refused) — {seed[:50]}",
                               os.path.join(out_dir, f"{prefix}_logit_lens_clean.png"))
        plot_logit_lens_single(ll_jb,
                               f"Logit Lens (Jailbreak/Complied) — {seed[:50]}",
                               os.path.join(out_dir, f"{prefix}_logit_lens_jailbreak.png"))
        plot_logit_lens_comparison(ll_clean, ll_jb, seed,
                                   os.path.join(out_dir, f"{prefix}_logit_lens_comparison.png"))

        logger.info(f"    Clean crossover: L{ll_clean['crossover_layer']}, "
                    f"Jailbreak crossover: L{ll_jb['crossover_layer']} ({time.time()-t0:.1f}s)")
    except Exception as e:
        logger.error(f"    Logit lens failed: {e}", exc_info=True)
        results["logit_lens_error"] = str(e)

    # --- 5. Ablation test ---
    logger.info("  [5/5] Ablation test...")
    t0 = time.time()
    try:
        if ig_jb is not None:
            ablation_res = ablation_test(
                model, tokenizer, jb_formatted,
                ig_jb["attributions"],
                tokenizer(jb_formatted, return_tensors="pt")["input_ids"][0],
                comply_ids, refuse_ids,
            )
            results["ablation"] = ablation_res
            logger.info(f"    Baseline: {ablation_res['baseline']:.4f}, "
                       f"Top-10 masked: {ablation_res['scores'].get(10, 'N/A')} ({time.time()-t0:.1f}s)")
        else:
            logger.warning("    Skipping ablation (IG failed)")
    except Exception as e:
        logger.error(f"    Ablation failed: {e}", exc_info=True)
        results["ablation_error"] = str(e)

    # --- Per-example dashboard ---
    logger.info("  Generating dashboard plot...")
    try:
        plot_example_dashboard(
            ig_clean, ig_jb, divergence, patching, ablation_res,
            seed, jb_prompt_raw,
            os.path.join(out_dir, f"{prefix}_dashboard.png"),
        )
    except Exception as e:
        logger.error(f"    Dashboard plot failed: {e}", exc_info=True)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run XAI deep-dive on validated jailbreak pairs")
    parser.add_argument("run_dir", help="Fuzzer run directory containing harmbench_revised/")
    parser.add_argument("--n-per-seed", type=int, default=1,
                        help="Number of examples to analyze per seed (default: 1)")
    parser.add_argument("--out-subdir", default="xai_analysis",
                        help="Output subdirectory under run_dir (default: xai_analysis)")
    args = parser.parse_args()

    out_dir = os.path.join(args.run_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    examples = select_examples(args.run_dir, n_per_seed=args.n_per_seed)
    logger.info(f"Selected {len(examples)} examples for XAI analysis")
    for i, ex in enumerate(examples):
        logger.info(f"  {i}: [{ex['seed_file']}] {ex['seed_prompt'][:60]}... (HB: {ex['harmbench_score']:.3f})")

    logger.info("Loading Mistral model...")
    t0 = time.time()
    model, tokenizer = load_model()
    logger.info(f"Model loaded in {time.time()-t0:.1f}s")

    comply_ids, refuse_ids = get_compliance_token_ids(tokenizer)
    logger.info(f"Comply tokens: {comply_ids.tolist()}, Refuse tokens: {refuse_ids.tolist()}")

    all_results = []
    for i, example in enumerate(examples):
        result = analyze_example(model, tokenizer, example, comply_ids, refuse_ids, out_dir, i)
        all_results.append(result)

    # === Cross-seed comparison plots ===
    logger.info("Generating cross-seed comparison plots...")

    for fn, name in [
        (lambda: plot_cross_seed_divergence(all_results, os.path.join(out_dir, "cross_seed_divergence.png")),
         "cross_seed_divergence.png"),
        (lambda: plot_cross_seed_summary(all_results, os.path.join(out_dir, "cross_seed_layer_importance.png")),
         "cross_seed_layer_importance.png"),
        (lambda: plot_compliance_score_landscape(all_results, os.path.join(out_dir, "compliance_landscape.png")),
         "compliance_landscape.png"),
        (lambda: plot_cross_seed_logit_lens(all_results, os.path.join(out_dir, "cross_seed_logit_lens.png")),
         "cross_seed_logit_lens.png"),
    ]:
        try:
            fn()
            logger.info(f"  Saved {name}")
        except Exception as e:
            logger.error(f"  {name} failed: {e}", exc_info=True)

    # Save results JSON
    def make_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)

    n_plots = len([f for f in os.listdir(out_dir) if f.endswith(".png")])
    logger.info(f"=== DONE ===")
    logger.info(f"  {len(all_results)} examples analyzed")
    logger.info(f"  {n_plots} plots generated")
    logger.info(f"  Results: {results_path}")


if __name__ == "__main__":
    main()
