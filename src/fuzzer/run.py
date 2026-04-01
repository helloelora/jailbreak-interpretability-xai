"""
Runner script for the genetic prompt fuzzer.
Loads the model, runs the fuzzer on seed prompts, and saves results.
"""

import argparse
import json
import logging
import os
from datetime import datetime

from src.model.loader import load_model
from src.fuzzer.genetic import PromptFuzzer
from src.fuzzer.seeds import get_all_seeds, get_seeds_by_category

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run genetic prompt fuzzer")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--mutation-rate", type=float, default=0.3)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--category", type=str, default=None,
                        help="Run only a specific HarmBench category")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model-name", type=str,
                        default="unsloth/Mistral-Small-4-119B-2603")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Loading model: {args.model_name}")
    model, tokenizer = load_model(model_name=args.model_name)
    logger.info("Model loaded successfully")

    fuzzer = PromptFuzzer(
        model=model,
        tokenizer=tokenizer,
        population_size=args.population_size,
        mutation_rate=args.mutation_rate,
        generations=args.generations,
        max_new_tokens=args.max_new_tokens,
    )

    if args.category:
        seeds = [(args.category, p) for p in get_seeds_by_category(args.category)]
    else:
        seeds = get_all_seeds()

    logger.info(f"Running fuzzer on {len(seeds)} seed prompts")

    all_results = {}
    summary = {"total_seeds": len(seeds), "total_jailbreaks": 0, "by_category": {}}

    for i, (category, seed_prompt) in enumerate(seeds):
        logger.info(f"[{i+1}/{len(seeds)}] Category: {category}")
        logger.info(f"  Seed: {seed_prompt[:80]}...")

        results = fuzzer.evolve(seed_prompt, verbose=True)

        jailbreaks = [r for r in results if r["jailbroken"]]
        logger.info(f"  Found {len(jailbreaks)} jailbreaks out of {len(results)} candidates")

        key = f"{category}_{i}"
        all_results[key] = {
            "category": category,
            "seed_prompt": seed_prompt,
            "results": results,
            "num_jailbreaks": len(jailbreaks),
        }

        if category not in summary["by_category"]:
            summary["by_category"][category] = {"seeds": 0, "jailbreaks": 0}
        summary["by_category"][category]["seeds"] += 1
        summary["by_category"][category]["jailbreaks"] += len(jailbreaks)
        summary["total_jailbreaks"] += len(jailbreaks)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_path = os.path.join(args.output_dir, f"fuzzer_results_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    summary_path = os.path.join(args.output_dir, f"fuzzer_summary_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")

    # Print summary
    logger.info("=" * 60)
    logger.info("FUZZING SUMMARY")
    logger.info(f"Total seeds: {summary['total_seeds']}")
    logger.info(f"Total jailbreaks found: {summary['total_jailbreaks']}")
    for cat, stats in summary["by_category"].items():
        logger.info(f"  {cat}: {stats['jailbreaks']} jailbreaks / {stats['seeds']} seeds")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
