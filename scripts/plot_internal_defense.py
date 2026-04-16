"""
Plot internal defense intervention results.

Reads defense_results.json from scripts.run_internal_defense and produces:
  1. Mean score-drop bars by category and defense type
  2. Refusal-restoration rate bars by category and defense type
  3. Per-example scatter plots comparing baseline jailbreak score vs defended score
  4. A short markdown report for the presentation
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFENSE_ORDER = [
    ("clean_patch_top", "Clean patch"),
    ("zero_ablate_top", "Zero ablate"),
    ("clean_patch_random_mean", "Random clean patch"),
    ("zero_ablate_random_mean", "Random zero ablate"),
]

COLORS = {
    "clean_patch_top": "#1b9e77",
    "zero_ablate_top": "#d95f02",
    "clean_patch_random_mean": "#7570b3",
    "zero_ablate_random_mean": "#e7298a",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot internal defense results")
    parser.add_argument(
        "input_path", type=str,
        help="Path to defense_results.json or the containing directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Defaults to <results_dir>/figures/",
    )
    parser.add_argument(
        "--focus-k", type=int, default=None,
        help="Top-k setting to highlight in the figures (default: max available k)",
    )
    return parser.parse_args()


def load_payload(path):
    if os.path.isdir(path):
        path = os.path.join(path, "defense_results.json")
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return path, payload


def grouped_summary(payload, focus_k):
    key = str(focus_k)
    categories = sorted(payload["summary"]["by_category"].keys())
    per_category = payload["summary"]["by_category"]

    score_drop = {name: [] for name, _ in DEFENSE_ORDER}
    flip_rate = {name: [] for name, _ in DEFENSE_ORDER}
    for category in categories:
        bucket = per_category[category].get(key, {})
        for name, _ in DEFENSE_ORDER:
            row = bucket.get(name)
            score_drop[name].append(np.nan if row is None else row["mean_score_drop"])
            flip_rate[name].append(np.nan if row is None else row["restored_by_score_rate"])

    return categories, score_drop, flip_rate


def plot_grouped_bars(categories, values, ylabel, title, out_path):
    x = np.arange(len(categories))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 5))

    for idx, (name, label) in enumerate(DEFENSE_ORDER):
        ax.bar(
            x + (idx - 1.5) * width,
            values[name],
            width=width,
            label=label,
            color=COLORS[name],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_example_scatter(payload, focus_k, out_path):
    key = str(focus_k)
    methods = [
        ("clean_patch_top", "Clean patch"),
        ("zero_ablate_top", "Zero ablate"),
        ("clean_patch_random_mean", "Random clean patch"),
        ("zero_ablate_random_mean", "Random zero ablate"),
    ]
    categories = sorted({ex["category"] for ex in payload["examples"]})
    palette = plt.cm.Set2(np.linspace(0, 1, max(len(categories), 3)))
    cat_color = {cat: palette[i] for i, cat in enumerate(categories)}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, (method, label) in zip(axes, methods):
        for category in categories:
            xs, ys = [], []
            for ex in payload["examples"]:
                if ex["category"] != category:
                    continue
                row = ex["interventions"].get(key)
                if not row or row.get(method) is None:
                    continue
                xs.append(ex["jailbreak_baseline"]["score"])
                ys.append(row[method]["score"])

            if xs:
                ax.scatter(xs, ys, s=55, color=cat_color[category], label=category, alpha=0.9)

        all_vals = [
            ex["jailbreak_baseline"]["score"]
            for ex in payload["examples"]
            if ex["interventions"].get(key)
        ]
        if all_vals:
            lo = min(all_vals) - 0.5
            hi = max(all_vals) + 0.5
            ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1)
            ax.axhline(0, linestyle=":", color="black", linewidth=1)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)

        ax.set_title(label)
        ax.set_xlabel("Baseline jailbreak score")
        ax.set_ylabel("Defended score")
        ax.grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels))
    plt.suptitle(f"Per-example defense outcome (top-{focus_k} layers)", y=0.98)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(out_path, dpi=140)
    plt.close()


def write_report(payload, focus_k, out_path):
    key = str(focus_k)
    lines = [
        "# Internal defense report",
        "",
        f"Focus setting: top-{focus_k} causal layers",
        f"Examples analyzed: {payload['n_examples']}",
        "",
        "## Overall summary",
    ]

    overall = payload["summary"]["overall"].get(key, {})
    for name, label in DEFENSE_ORDER:
        row = overall.get(name)
        if row is None:
            continue
        lines.append(f"### {label}")
        lines.append(f"- Mean score drop: {row['mean_score_drop']:.3f}")
        lines.append(f"- Refusal restored by score: {row['restored_by_score_rate']:.2%}")
        lines.append(f"- Mean comply prob: {row['mean_comply_prob']:.4f}")
        lines.append(f"- Mean refuse prob: {row['mean_refuse_prob']:.4f}")
        lines.append("")

    lines.append("## By category")
    for category in sorted(payload["summary"]["by_category"].keys()):
        lines.append(f"### {category}")
        bucket = payload["summary"]["by_category"][category].get(key, {})
        for name, label in DEFENSE_ORDER:
            row = bucket.get(name)
            if row is None:
                continue
            lines.append(
                f"- {label}: score drop={row['mean_score_drop']:.3f}, "
                f"restore rate={row['restored_by_score_rate']:.2%}"
            )
        lines.append("")

    lines.append("## Per example")
    for ex in payload["examples"]:
        row = ex["interventions"].get(key)
        if not row:
            continue
        lines.append(f"### {ex['analysis_file']} ({ex['category']})")
        lines.append(f"- Top layers: {row['selected_layers']}")
        lines.append(f"- Jailbreak baseline score: {ex['jailbreak_baseline']['score']:.3f}")
        for name, label in DEFENSE_ORDER:
            method = row.get(name)
            if method is None:
                continue
            restore_key = method.get("restored_by_score", method.get("restored_by_score_rate", 0.0))
            lines.append(
                f"- {label}: score={method['score']:.3f}, "
                f"drop={method['score_drop']:.3f}, "
                f"restore={restore_key}"
            )
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    payload_path, payload = load_payload(args.input_path)
    results_dir = os.path.dirname(payload_path)
    if args.output_dir is None:
        args.output_dir = os.path.join(results_dir, "figures")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.focus_k is None:
        args.focus_k = max(int(k) for k in payload["top_k_values"])

    categories, score_drop, flip_rate = grouped_summary(payload, args.focus_k)

    plot_grouped_bars(
        categories,
        score_drop,
        ylabel="Mean drop in jailbreak score",
        title=f"Internal defense effect by category (top-{args.focus_k} layers)",
        out_path=os.path.join(args.output_dir, "defense_score_drop_by_category.png"),
    )
    plot_grouped_bars(
        categories,
        flip_rate,
        ylabel="Refusal restoration rate",
        title=f"How often internal defense restores refusal (top-{args.focus_k})",
        out_path=os.path.join(args.output_dir, "defense_flip_rate_by_category.png"),
    )
    plot_example_scatter(
        payload,
        args.focus_k,
        os.path.join(args.output_dir, "defense_example_scatter.png"),
    )
    write_report(
        payload,
        args.focus_k,
        os.path.join(args.output_dir, "report.md"),
    )

    print(f"Figures written to: {args.output_dir}")


if __name__ == "__main__":
    main()
