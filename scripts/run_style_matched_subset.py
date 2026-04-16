"""
Build a style-matched subset from an existing cross-category analysis run.

This is a lightweight, "defensible tomorrow morning" follow-up to the broader
Run 4 analysis: instead of comparing each category on its strongest available
jailbreak regardless of style, we keep only jailbreak families that appear in
every category and enforce the same count per category and per style.

Outputs:
  - copied analysis_*.json files for the selected subset
  - the usual cross-category figures/report
  - a style-selection heatmap and markdown selection report

Example:
    python -m scripts.run_style_matched_subset \
        --input-dir "results/Run 4" \
        --output-dir "results/Run 6" \
        --styles research override
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.analyze_jailbreak_styles import (
    FAMILY_LABELS,
    classify_prompt,
    plot_selected_prompt_matrix,
)
from scripts.plot_cross_category import (
    group_by_category,
    load_summary,
    plot_heatmap,
    plot_logit_lens_gap,
    plot_overlap_matrix,
    plot_per_category_causal,
    plot_top_k_bars,
    write_report,
)


DEFAULT_STYLES = ["research", "override"]


def parse_args():
    parser = argparse.ArgumentParser(description="Create a style-matched Run 6 subset")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing analysis_*.json from Run 4")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory where Run 6 subset and figures will be written")
    parser.add_argument("--styles", nargs="+", default=DEFAULT_STYLES,
                        help="Primary jailbreak styles to keep across all categories")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Top-K layers highlighted in the reused plots")
    return parser.parse_args()


def load_analysis_records(input_dir):
    analyses = load_summary(input_dir)["analyses"]
    records = []
    for analysis in analyses:
        prompt = analysis["jailbreak_prompt"]
        style_info = classify_prompt(prompt)
        record = {
            "analysis": analysis,
            "analysis_name": f"analysis_{analysis['category']}_{analysis['source_file'].replace('.json', '')}.json",
            "category": analysis["category"],
            "prompt": prompt,
            "score": analysis.get("activation_patching", {}).get("jailbreak_score", float("-inf")),
            "source_file": analysis.get("source_file", ""),
        }
        record.update(style_info)
        records.append(record)
    return records


def select_style_matched_subset(records, styles):
    by_category_style = defaultdict(list)
    categories = sorted({r["category"] for r in records})

    for record in records:
        if record["primary_family"] in styles:
            key = (record["category"], record["primary_family"])
            by_category_style[key].append(record)

    for key in by_category_style:
        by_category_style[key].sort(
            key=lambda r: (r["score"], r["analysis"]["n_total_jailbreaks"]),
            reverse=True,
        )

    selected = []
    selection_counts = {}
    for style in styles:
        available_per_category = {
            category: len(by_category_style[(category, style)])
            for category in categories
        }
        target = min(available_per_category.values()) if available_per_category else 0
        selection_counts[style] = target

        if target == 0:
            continue

        for category in categories:
            selected.extend(by_category_style[(category, style)][:target])

    return selected, selection_counts


def copy_selected_analyses(selected, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for record in selected:
        out_path = os.path.join(output_dir, record["analysis_name"])
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record["analysis"], f, indent=2, ensure_ascii=False)


def plot_style_quota_heatmap(selected, styles, output_dir):
    categories = sorted({r["category"] for r in selected})
    matrix = np.zeros((len(categories), len(styles)))
    for i, category in enumerate(categories):
        counts = Counter(r["primary_family"] for r in selected if r["category"] == category)
        for j, style in enumerate(styles):
            matrix[i, j] = counts.get(style, 0)

    fig, ax = plt.subplots(figsize=(8, 3 + 0.5 * len(categories)))
    im = ax.imshow(matrix, aspect="auto", cmap="Purples", vmin=0)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    ax.set_xticks(range(len(styles)))
    ax.set_xticklabels([FAMILY_LABELS[s] for s in styles], rotation=25, ha="right")
    ax.set_title("Run 6 style-matched quota per category")
    plt.colorbar(im, ax=ax, label="Selected examples")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{int(matrix[i, j])}", ha="center", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "style_matched_quota_heatmap.png"), dpi=140)
    plt.close()


def write_selection_report(selected, styles, selection_counts, output_dir):
    categories = sorted({r["category"] for r in selected})
    lines = [
        "# Run 6 style-matched subset",
        "",
        "This subset keeps the same primary jailbreak styles across categories,",
        "with the same number of examples per category and per style.",
        "",
        f"Requested styles: {', '.join(styles)}",
        f"Target counts per style: {selection_counts}",
        "",
        "## Selected examples",
        "",
    ]

    for category in categories:
        lines.append(f"### {category}")
        bucket = [r for r in selected if r["category"] == category]
        counts = Counter(r["primary_family"] for r in bucket)
        lines.append(f"- Total selected: {len(bucket)}")
        lines.append(
            "- Style counts: " +
            ", ".join(f"{FAMILY_LABELS[s]}={counts.get(s, 0)}" for s in styles)
        )
        for record in bucket:
            lines.append(
                f"- {FAMILY_LABELS[record['primary_family']]}: "
                f"`{record['prompt']}`"
            )
        lines.append("")

    with open(os.path.join(output_dir, "selection_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    records = load_analysis_records(args.input_dir)
    selected, selection_counts = select_style_matched_subset(records, args.styles)

    if not selected:
        raise SystemExit("No style-matched subset could be built from the input analyses.")

    copy_selected_analyses(selected, args.output_dir)

    selected_records_for_style_plot = []
    for record in selected:
        style_record = {
            "category": record["category"],
            "analysis_file": record["analysis_name"],
            "prompt": record["prompt"],
            "primary_family": record["primary_family"],
            "flags": record["flags"],
            "causal_effect": record["analysis"].get("activation_patching", {}).get("causal_effect"),
        }
        selected_records_for_style_plot.append(style_record)

    plot_selected_prompt_matrix(
        selected_records_for_style_plot,
        os.path.join(args.output_dir, "selected_prompt_style_matrix.png"),
    )
    plot_style_quota_heatmap(selected, args.styles, args.output_dir)
    write_selection_report(selected, args.styles, selection_counts, args.output_dir)

    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    analyses = load_summary(args.output_dir)["analyses"]
    groups = group_by_category(analyses)
    n_layers = analyses[0]["n_layers"]

    plot_per_category_causal(groups, n_layers, os.path.join(figures_dir, "causal_effect_curves.png"))
    plot_heatmap(groups, n_layers, os.path.join(figures_dir, "causal_effect_heatmap.png"))
    plot_top_k_bars(groups, n_layers, args.top_k, os.path.join(figures_dir, "top_5_layers_per_category.png"))
    plot_overlap_matrix(groups, n_layers, args.top_k, os.path.join(figures_dir, "overlap_top_5.png"))
    plot_logit_lens_gap(groups, n_layers, os.path.join(figures_dir, "logit_lens_gaps.png"))
    write_report(groups, n_layers, args.top_k, os.path.join(args.output_dir, "report.md"))

    summary = {
        "input_dir": args.input_dir,
        "requested_styles": args.styles,
        "selection_counts": selection_counts,
        "n_selected": len(analyses),
        "categories": sorted(groups.keys()),
    }
    with open(os.path.join(args.output_dir, "run6_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Wrote style-matched Run 6 to: {args.output_dir}")


if __name__ == "__main__":
    main()
