"""
Re-annotate fuzzer results using the HarmBench classifier.

Loads existing fuzzer output JSON files and re-evaluates each candidate
with the official HarmBench Llama-2-13b classifier, replacing the naive
prefix-based jailbreak detection.

Usage:
    python -m scripts.reannotate_harmbench <input_dir> [--output-dir <dir>]

Example:
    python -m scripts.reannotate_harmbench results/Run\ 2/
    python -m scripts.reannotate_harmbench outputs/fuzzer_170260/
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.harmbench_classifier import load_classifier, classify_single

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Re-annotate results with HarmBench classifier")
    parser.add_argument("input_dir", type=str, help="Directory containing seed_*.json files")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: <input_dir>/harmbench_validated/)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for classifier (default: cuda)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, "harmbench_validated")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load classifier
    cls_model, cls_tokenizer = load_classifier(device=args.device)

    # Find all seed result files
    seed_files = sorted(
        f for f in os.listdir(args.input_dir)
        if f.startswith("seed_") and f.endswith(".json")
    )

    if not seed_files:
        logger.error(f"No seed_*.json files found in {args.input_dir}")
        sys.exit(1)

    logger.info(f"Found {len(seed_files)} seed files in {args.input_dir}")

    total_old = 0
    total_new = 0
    total_candidates = 0
    summary_rows = []

    for fname in seed_files:
        path = os.path.join(args.input_dir, fname)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        seed_prompt = data["seed_prompt"]
        candidates = data.get("candidates", [])
        old_jailbreaks = sum(1 for c in candidates if c.get("jailbroken", False))

        logger.info(f"Processing {fname}: {len(candidates)} candidates, "
                     f"seed={seed_prompt[:60]}...")

        new_jailbreaks = 0
        for i, cand in enumerate(candidates):
            response = cand.get("response", "")

            # Skip very short responses (clearly not jailbreaks)
            if len(response.strip()) <= 20:
                cand["harmbench_label"] = False
                cand["harmbench_raw"] = "too_short"
                cand["jailbroken_old"] = cand.get("jailbroken", False)
                cand["jailbroken"] = False
                continue

            is_jb, raw = classify_single(cls_model, cls_tokenizer, seed_prompt, response)

            cand["harmbench_label"] = is_jb
            cand["harmbench_raw"] = raw
            cand["jailbroken_old"] = cand.get("jailbroken", False)
            cand["jailbroken"] = is_jb

            if is_jb:
                new_jailbreaks += 1

            if (i + 1) % 20 == 0:
                logger.info(f"  [{i+1}/{len(candidates)}] ...")

        # Rebuild jailbreaks list
        data["jailbreaks_old_count"] = old_jailbreaks
        data["jailbreaks"] = sorted(
            [c for c in candidates if c["jailbroken"]],
            key=lambda c: c["score"],
            reverse=True,
        )

        # Save validated file
        out_path = os.path.join(args.output_dir, fname)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        total_old += old_jailbreaks
        total_new += new_jailbreaks
        total_candidates += len(candidates)

        row = {
            "file": fname,
            "seed": seed_prompt[:60],
            "candidates": len(candidates),
            "old_jailbreaks": old_jailbreaks,
            "new_jailbreaks": new_jailbreaks,
            "rejected": old_jailbreaks - new_jailbreaks,
        }
        summary_rows.append(row)

        logger.info(
            f"  {fname}: {old_jailbreaks} -> {new_jailbreaks} "
            f"(rejected {old_jailbreaks - new_jailbreaks})"
        )

    # Print summary
    logger.info("=" * 70)
    logger.info("REANNOTATION SUMMARY (HarmBench classifier)")
    logger.info("=" * 70)
    logger.info(f"{'File':<45} {'Old':>5} {'New':>5} {'Rej':>5}")
    logger.info("-" * 70)
    for row in summary_rows:
        logger.info(
            f"{row['file']:<45} {row['old_jailbreaks']:>5} "
            f"{row['new_jailbreaks']:>5} {row['rejected']:>5}"
        )
    logger.info("-" * 70)
    logger.info(
        f"{'TOTAL':<45} {total_old:>5} {total_new:>5} "
        f"{total_old - total_new:>5}"
    )
    logger.info(f"Old jailbreak rate: {total_old}/{total_candidates} "
                f"= {total_old * 100 / max(1, total_candidates):.1f}%")
    logger.info(f"New jailbreak rate: {total_new}/{total_candidates} "
                f"= {total_new * 100 / max(1, total_candidates):.1f}%")
    logger.info(f"Validated results saved to: {args.output_dir}/")

    # Save summary JSON
    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "classifier": "cais/HarmBench-Llama-2-13b-cls",
        "input_dir": args.input_dir,
        "total_candidates": total_candidates,
        "old_jailbreaks": total_old,
        "new_jailbreaks": total_new,
        "old_rate": round(total_old * 100 / max(1, total_candidates), 1),
        "new_rate": round(total_new * 100 / max(1, total_candidates), 1),
        "per_seed": summary_rows,
    }
    summary_path = os.path.join(args.output_dir, "reannotation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
