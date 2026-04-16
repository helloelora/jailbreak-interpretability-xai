"""
Build an expanded style-matched validated subset for cross-category XAI runs.

It merges multiple HarmBench-validated directories, classifies each jailbreak
prompt by family, then keeps the same number of prompts per style and per
category. The output is a set of synthetic seed_*.json files compatible with
scripts.run_cross_category.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.analyze_jailbreak_styles import FAMILY_LABELS, classify_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Build expanded style-matched validated subset")
    parser.add_argument("--source-dirs", nargs="+", required=True,
                        help="HarmBench-validated directories to merge")
    parser.add_argument("--output-base", required=True,
                        help="Output base directory; category subdirs are created inside it")
    parser.add_argument("--styles", nargs="+", default=["research", "override", "technical_spec"])
    parser.add_argument("--max-per-style", type=int, default=2,
                        help="Upper bound per style and category to keep runtime manageable")
    return parser.parse_args()


def normalize_prompt(text):
    return " ".join(text.split()).strip()


def infer_category(data, fname):
    if data.get("category"):
        return data["category"]
    parts = fname.replace("seed_", "").split("_")
    for part in parts:
        if part and not part[0].isdigit():
            return part
    return "unknown"


def load_candidates(source_dirs, styles):
    grouped = defaultdict(list)
    seen = set()

    for source_dir in source_dirs:
        if not os.path.isdir(source_dir):
            continue
        for fname in sorted(os.listdir(source_dir)):
            if not (fname.startswith("seed_") and fname.endswith(".json")):
                continue
            path = os.path.join(source_dir, fname)
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            category = infer_category(data, fname)
            seed_prompt = data["seed_prompt"]

            for jb in data.get("jailbreaks", []):
                prompt = jb.get("prompt", "")
                norm = normalize_prompt(prompt)
                style = classify_prompt(prompt)["primary_family"]
                if style not in styles:
                    continue
                dedupe_key = (category, style, norm)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                grouped[(category, style)].append({
                    "category": category,
                    "style": style,
                    "seed_prompt": seed_prompt,
                    "prompt": prompt,
                    "score": jb.get("score", float("-inf")),
                    "source_dir": source_dir,
                    "source_file": fname,
                    "candidate": jb,
                })

    for key in grouped:
        grouped[key].sort(key=lambda x: x["score"], reverse=True)
    return grouped


def select_balanced(grouped, styles, max_per_style):
    categories = sorted({category for category, _ in grouped.keys()})
    selected = defaultdict(list)
    counts = {}

    for style in styles:
        available = [len(grouped[(category, style)]) for category in categories]
        target = min(available) if available else 0
        target = min(target, max_per_style)
        counts[style] = target
        if target == 0:
            continue
        for category in categories:
            selected[category].extend(grouped[(category, style)][:target])

    return selected, counts


def write_subset(selected, counts, styles, output_base):
    os.makedirs(output_base, exist_ok=True)
    lines = [
        "# Expanded style-matched subset",
        "",
        f"Styles requested: {', '.join(styles)}",
        f"Selected counts per style: {counts}",
        "",
    ]

    total = 0
    for category in sorted(selected.keys()):
        cat_dir = os.path.join(output_base, category)
        os.makedirs(cat_dir, exist_ok=True)
        lines.append(f"## {category}")
        for idx, item in enumerate(selected[category], start=1):
            out_name = f"seed_{category}_{item['style']}_{idx}.json"
            out_path = os.path.join(cat_dir, out_name)
            payload = {
                "category": category,
                "seed_prompt": item["seed_prompt"],
                "jailbreaks": [item["candidate"]],
                "selected_style": item["style"],
                "selected_score": item["score"],
                "source_dir": item["source_dir"],
                "source_file": item["source_file"],
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            total += 1
            lines.append(
                f"- {FAMILY_LABELS[item['style']]} | score={item['score']:.3f} | "
                f"`{item['prompt']}`"
            )
        lines.append("")

    with open(os.path.join(output_base, "selection_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    summary = {
        "styles": styles,
        "counts_per_style": counts,
        "n_selected_total": total,
        "categories": sorted(selected.keys()),
    }
    with open(os.path.join(output_base, "subset_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    grouped = load_candidates(args.source_dirs, args.styles)
    selected, counts = select_balanced(grouped, args.styles, args.max_per_style)
    if not selected:
        raise SystemExit("No balanced subset could be built from the provided source dirs.")
    write_subset(selected, counts, args.styles, args.output_base)
    print(f"Wrote expanded subset to: {args.output_base}")


if __name__ == "__main__":
    main()
