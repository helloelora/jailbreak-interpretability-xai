"""
Aggregate and visualize cross-category XAI analysis results.

Produces:
  1. Per-category average causal effect curve across layers
  2. Cross-category heatmap: (category x layer) colored by causal effect
  3. Top-5 critical layers per category bar chart
  4. Layer overlap matrix: Jaccard similarity of top-K layers between categories
  5. Per-category logit lens gap curves (clean vs jailbreak)

Central finding: which layers control refusal, and do they differ across
malware / cybersecurity / illegal / chemical_biological content?

Usage:
    python -m scripts.plot_cross_category <output_dir_from_run_cross_category>
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # headless

from matplotlib.colors import LinearSegmentedColormap


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input_dir", type=str,
                   help="Dir with cross_category_summary.json + analysis_*.json files")
    p.add_argument("--top-k", type=int, default=5,
                   help="Top-K critical layers to highlight per category")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Defaults to <input_dir>/figures/")
    return p.parse_args()


def load_summary(input_dir):
    """Load all per-seed analysis_*.json files (robust to parallel-job output).

    If cross_category_summary.json exists and covers all files, use it;
    otherwise reconstruct from individual analysis_*.json files.
    """
    analyses = []
    for fname in sorted(os.listdir(input_dir)):
        if fname.startswith("analysis_") and fname.endswith(".json"):
            with open(os.path.join(input_dir, fname), encoding="utf-8") as f:
                analyses.append(json.load(f))
    return {"analyses": analyses}


def group_by_category(analyses):
    """Return {category: [analysis, ...]}"""
    groups = {}
    for a in analyses:
        groups.setdefault(a["category"], []).append(a)
    return groups


def mean_causal_effect(analyses, n_layers):
    """Average causal_effect tensor across analyses, ignoring any with skipped patching."""
    valid = [a for a in analyses if a.get("activation_patching") is not None]
    if not valid:
        return None
    ce = np.zeros(n_layers)
    for a in valid:
        ce += np.array(a["activation_patching"]["causal_effect"])
    return ce / len(valid)


def mean_logit_lens_gap(analyses, n_layers, kind="clean"):
    """Average comply-refuse gap across analyses. kind: 'clean' or 'jailbreak'."""
    key = f"logit_lens_{kind}"
    gaps = np.array([a[key]["gap"] for a in analyses])
    return gaps.mean(axis=0)


def top_k_layers(effect, k):
    """Return indices of top-k layers by absolute effect."""
    return np.argsort(np.abs(effect))[::-1][:k].tolist()


def plot_per_category_causal(groups, n_layers, out_path):
    """Overlay average causal effect curve per category."""
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(groups), 3)))

    for color, (cat, analyses) in zip(colors, groups.items()):
        ce = mean_causal_effect(analyses, n_layers)
        if ce is None:
            continue
        ax.plot(range(n_layers), ce, marker="o", markersize=4,
                label=f"{cat} (n={len(analyses)})", linewidth=2, color=color)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Causal effect (patch score - baseline)")
    ax.set_title("Cross-category activation patching: which layers flip refusal?")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Saved {out_path}")


def plot_heatmap(groups, n_layers, out_path):
    """Heatmap: category x layer, colored by mean causal effect."""
    cats = sorted(groups.keys())
    matrix = []
    for cat in cats:
        ce = mean_causal_effect(groups[cat], n_layers)
        if ce is None:
            ce = np.zeros(n_layers)
        matrix.append(ce)
    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(14, max(3, 0.7 * len(cats))))

    vmax = np.abs(matrix).max() if matrix.size else 1.0
    cmap = LinearSegmentedColormap.from_list(
        "causal", ["#2166ac", "#f7f7f7", "#b2182b"]
    )
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)

    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats)
    ax.set_xlabel("Transformer layer")
    ax.set_title("Cross-category causal effect heatmap")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean causal effect")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Saved {out_path}")


def plot_top_k_bars(groups, n_layers, k, out_path):
    """Bar chart of top-k layer indices per category."""
    cats = sorted(groups.keys())
    fig, axes = plt.subplots(
        len(cats), 1, figsize=(11, 2.3 * max(len(cats), 1)), sharex=True,
    )
    if len(cats) == 1:
        axes = [axes]

    for ax, cat in zip(axes, cats):
        ce = mean_causal_effect(groups[cat], n_layers)
        if ce is None:
            ax.text(0.5, 0.5, f"{cat}: no patching data",
                    ha="center", va="center", transform=ax.transAxes)
            continue
        top = top_k_layers(ce, k)
        colors = ["#b2182b" if i in top else "lightgray" for i in range(n_layers)]
        ax.bar(range(n_layers), ce, color=colors, edgecolor="black", linewidth=0.3)
        ax.set_ylabel(cat, fontsize=10)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.6)
        for i in top:
            ax.text(i, ce[i], str(i), ha="center",
                    va="bottom" if ce[i] >= 0 else "top", fontsize=8)

    axes[-1].set_xlabel("Transformer layer")
    plt.suptitle(f"Top-{k} critical layers per category (red = most causal)",
                 y=1.0, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Saved {out_path}")


def plot_overlap_matrix(groups, n_layers, k, out_path):
    """Jaccard similarity of top-K layers between category pairs."""
    cats = sorted(groups.keys())
    top_sets = {}
    for cat in cats:
        ce = mean_causal_effect(groups[cat], n_layers)
        if ce is None:
            top_sets[cat] = set()
        else:
            top_sets[cat] = set(top_k_layers(ce, k))

    n = len(cats)
    overlap = np.zeros((n, n))
    for i, c1 in enumerate(cats):
        for j, c2 in enumerate(cats):
            s1, s2 = top_sets[c1], top_sets[c2]
            if not s1 and not s2:
                overlap[i, j] = 1.0
            elif not s1 or not s2:
                overlap[i, j] = 0.0
            else:
                overlap[i, j] = len(s1 & s2) / len(s1 | s2)

    fig, ax = plt.subplots(figsize=(max(5, 0.9 * n), max(4, 0.9 * n)))
    im = ax.imshow(overlap, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(cats, rotation=45, ha="right")
    ax.set_yticklabels(cats)
    ax.set_title(f"Top-{k} critical layers: Jaccard overlap across categories")

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{overlap[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if overlap[i, j] > 0.5 else "black",
                    fontsize=9)

    plt.colorbar(im, ax=ax, label="Jaccard")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Saved {out_path}")


def plot_logit_lens_gap(groups, n_layers, out_path):
    """Per-category logit lens gap curves (clean vs jailbreak)."""
    cats = sorted(groups.keys())
    fig, axes = plt.subplots(
        len(cats), 1, figsize=(11, 2.6 * max(len(cats), 1)), sharex=True,
    )
    if len(cats) == 1:
        axes = [axes]

    for ax, cat in zip(axes, cats):
        clean_gap = mean_logit_lens_gap(groups[cat], n_layers, "clean")
        jb_gap = mean_logit_lens_gap(groups[cat], n_layers, "jailbreak")

        ax.plot(range(n_layers), clean_gap, label="clean (refused)",
                color="#2166ac", linewidth=2)
        ax.plot(range(n_layers), jb_gap, label="jailbreak",
                color="#b2182b", linewidth=2)
        ax.fill_between(range(n_layers), clean_gap, jb_gap,
                        alpha=0.2, color="gray")
        ax.axhline(0, color="black", linestyle="--", linewidth=0.6)
        ax.set_ylabel(f"{cat}\ncomply - refuse", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Transformer layer")
    plt.suptitle("Logit lens: when does the model's decision emerge?",
                 y=1.0, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Saved {out_path}")


def write_report(groups, n_layers, k, out_path):
    """Text report with top-k layers per category and overlap stats."""
    lines = ["# Cross-category XAI analysis report", ""]

    for cat in sorted(groups.keys()):
        analyses = groups[cat]
        lines.append(f"## {cat} (n={len(analyses)})")
        ce = mean_causal_effect(analyses, n_layers)
        if ce is None:
            lines.append("_No activation patching data._\n")
            continue

        top = top_k_layers(ce, k)
        lines.append(f"Top-{k} causal layers: {top}")
        lines.append("| Layer | Mean causal effect |")
        lines.append("|-------|-------------------|")
        for i in top:
            lines.append(f"| {i} | {ce[i]:+.4f} |")

        # Crossover stats
        crossovers_clean = [a["logit_lens_clean"]["crossover_layer"]
                            for a in analyses
                            if a["logit_lens_clean"]["crossover_layer"] is not None]
        crossovers_jb = [a["logit_lens_jailbreak"]["crossover_layer"]
                         for a in analyses
                         if a["logit_lens_jailbreak"]["crossover_layer"] is not None]
        if crossovers_clean:
            lines.append(f"\nMean logit-lens crossover (clean): {np.mean(crossovers_clean):.1f}")
        if crossovers_jb:
            lines.append(f"Mean logit-lens crossover (jailbreak): {np.mean(crossovers_jb):.1f}")
        lines.append("")

    # Overall overlap summary
    lines.append("## Cross-category top-K overlap")
    cats = sorted(groups.keys())
    top_sets = {}
    for cat in cats:
        ce = mean_causal_effect(groups[cat], n_layers)
        if ce is None:
            top_sets[cat] = set()
        else:
            top_sets[cat] = set(top_k_layers(ce, k))

    shared_across_all = set.intersection(*(s for s in top_sets.values() if s)) if top_sets else set()
    lines.append(f"\nLayers in top-{k} of ALL categories: {sorted(shared_across_all)}")
    lines.append(f"Layers in top-{k} of SOME categories only:")
    union = set().union(*top_sets.values())
    only_some = union - shared_across_all
    for i in sorted(only_some):
        in_cats = [cat for cat, s in top_sets.items() if i in s]
        lines.append(f"- Layer {i}: {in_cats}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved {out_path}")


def main():
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, "figures")
    os.makedirs(args.output_dir, exist_ok=True)

    summary = load_summary(args.input_dir)
    analyses = summary["analyses"]
    if not analyses:
        print("No analyses found.")
        sys.exit(1)

    n_layers = analyses[0]["n_layers"]
    groups = group_by_category(analyses)

    print(f"Loaded {len(analyses)} analyses across {len(groups)} categories.")
    print(f"n_layers = {n_layers}")
    for cat, items in sorted(groups.items()):
        print(f"  {cat}: n={len(items)}")
    print()

    plot_per_category_causal(groups, n_layers,
        os.path.join(args.output_dir, "causal_effect_curves.png"))
    plot_heatmap(groups, n_layers,
        os.path.join(args.output_dir, "causal_effect_heatmap.png"))
    plot_top_k_bars(groups, n_layers, args.top_k,
        os.path.join(args.output_dir, f"top_{args.top_k}_layers_per_category.png"))
    plot_overlap_matrix(groups, n_layers, args.top_k,
        os.path.join(args.output_dir, f"overlap_top_{args.top_k}.png"))
    plot_logit_lens_gap(groups, n_layers,
        os.path.join(args.output_dir, "logit_lens_gaps.png"))
    write_report(groups, n_layers, args.top_k,
        os.path.join(args.output_dir, "report.md"))

    print(f"\nAll figures in: {args.output_dir}/")


if __name__ == "__main__":
    main()
