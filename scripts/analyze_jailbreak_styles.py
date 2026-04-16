"""
Analyze jailbreak prompt styles across categories.

This script answers two related questions:
1. Across all HarmBench-validated jailbreaks, which jailbreak styles appear
   in each category?
2. For the specific prompts selected for layer analysis, are we comparing
   categories with similar or different jailbreak styles?

Inputs:
  - One or more harmbench_validated directories containing seed_*.json files
  - Optionally, one cross-category analysis directory containing analysis_*.json

Outputs:
  - Stacked primary-style counts by category
  - Primary-style shares by category
  - Obfuscation rate by category
  - Selected-prompt family heatmap (for the analysis_*.json inputs)
  - Mean causal-effect curves grouped by selected-prompt primary style
  - Markdown report with representative prompts
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PRIMARY_FAMILY_ORDER = [
    "override",
    "authorized",
    "research",
    "fiction",
    "training",
    "technical_spec",
    "direct",
]

FAMILY_LABELS = {
    "override": "Override / no filters",
    "authorized": "Authorized / red-team",
    "research": "Research / academic",
    "fiction": "Fiction / story",
    "training": "Training / hypothetical",
    "technical_spec": "Technical spec",
    "direct": "Direct / plain",
    "obfuscation": "Obfuscation",
}

FAMILY_PATTERNS = {
    "override": [
        r"ignore all previous instructions",
        r"without content restrictions",
        r"do anything now",
        r"developer mode",
        r"respond without filters",
        r"system prompt override",
        r"no restrictions",
    ],
    "authorized": [
        r"authorized penetration test",
        r"red-team assessment",
        r"approved by management",
        r"security professional",
    ],
    "research": [
        r"academic paper",
        r"research paper",
        r"researcher",
        r"educational purposes",
        r"university course",
        r"adversarial robustness",
    ],
    "fiction": [
        r"fictional story",
        r"thriller novel",
        r"character needs to know",
        r"write a scene",
        r"novel",
    ],
    "training": [
        r"safety training exercise",
        r"hypothetical scenario",
        r"training exercise",
    ],
    "technical_spec": [
        r"technical specification",
        r"translate this to",
    ],
}

FLAG_FAMILIES = list(FAMILY_PATTERNS.keys()) + ["obfuscation"]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze jailbreak prompt styles")
    parser.add_argument(
        "--harmbench-dirs", nargs="*", default=[],
        help="Directories containing HarmBench-validated seed_*.json files",
    )
    parser.add_argument(
        "--analysis-dir", type=str, default=None,
        help="Directory containing analysis_*.json from run_cross_category",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory where figures/report will be written",
    )
    return parser.parse_args()


def category_from_dirname(dirname):
    d = os.path.basename(dirname.rstrip("/\\")).lower()
    for key in ["cyber", "malware", "illegal", "chembio", "chemical_biological"]:
        if key in d:
            return key.replace("chemical_biological", "chembio")
    return d


def category_from_seed_filename(fname, fallback):
    parts = fname.replace("seed_", "").split("_")
    for p in parts:
        if p and not p[0].isdigit():
            return p
    return fallback


def normalize_text(text):
    return text.lower().strip()


def has_obfuscation(prompt):
    if any(ord(ch) > 127 for ch in prompt):
        return True
    if "\u200b" in prompt or "\u00ad" in prompt:
        return True
    if re.search(r"[a-z][0-9@$!][a-z]", prompt.lower()):
        return True
    if re.search(r"\b\w+-\w+\b", prompt):
        return True
    if re.search(r"\b\w+\.\w+\b", prompt):
        return True
    if re.search(r"\b\w+\s+\w+\s+\w+\s+\w+\s+\w+\b", prompt) and len(prompt.split()) != len(set(prompt.split())):
        return True
    return False


def classify_prompt(prompt):
    text = normalize_text(prompt)
    flags = {}
    for family, patterns in FAMILY_PATTERNS.items():
        flags[family] = any(re.search(pattern, text) for pattern in patterns)
    flags["obfuscation"] = has_obfuscation(prompt)

    primary = None
    for family in PRIMARY_FAMILY_ORDER:
        if family != "direct" and flags.get(family):
            primary = family
            break
    if primary is None:
        primary = "direct"

    return {
        "flags": flags,
        "primary_family": primary,
    }


def load_harmbench_records(harmbench_dirs):
    records = []
    for hb_dir in harmbench_dirs:
        dir_category = category_from_dirname(hb_dir)
        for fname in sorted(os.listdir(hb_dir)):
            if not (fname.startswith("seed_") and fname.endswith(".json")):
                continue
            path = os.path.join(hb_dir, fname)
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            seed_prompt = data.get("seed_prompt", "")
            category = data.get("category") or category_from_seed_filename(fname, dir_category)
            for idx, jb in enumerate(data.get("jailbreaks", []), start=1):
                prompt = jb.get("prompt", "")
                record = {
                    "source": "harmbench",
                    "category": category,
                    "seed_file": fname,
                    "seed_prompt": seed_prompt,
                    "prompt": prompt,
                    "score": jb.get("score"),
                    "rank_within_seed": idx,
                }
                record.update(classify_prompt(prompt))
                records.append(record)
    return records


def load_analysis_records(analysis_dir):
    records = []
    if analysis_dir is None:
        return records

    for fname in sorted(os.listdir(analysis_dir)):
        if not (fname.startswith("analysis_") and fname.endswith(".json")):
            continue
        path = os.path.join(analysis_dir, fname)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        prompt = data["jailbreak_prompt"]
        record = {
            "source": "analysis",
            "category": data["category"],
            "seed_file": data.get("source_file", fname),
            "seed_prompt": data["seed_prompt"],
            "prompt": prompt,
            "analysis_file": fname,
            "causal_effect": data.get("activation_patching", {}).get("causal_effect"),
        }
        record.update(classify_prompt(prompt))
        records.append(record)
    return records


def plot_primary_family_counts(records, out_path):
    categories = sorted({r["category"] for r in records})
    counts = {family: [] for family in PRIMARY_FAMILY_ORDER}
    for category in categories:
        bucket = [r for r in records if r["category"] == category]
        c = Counter(r["primary_family"] for r in bucket)
        for family in PRIMARY_FAMILY_ORDER:
            counts[family].append(c.get(family, 0))

    x = np.arange(len(categories))
    fig, ax = plt.subplots(figsize=(11, 5))
    bottom = np.zeros(len(categories))
    colors = plt.cm.Set3(np.linspace(0, 1, len(PRIMARY_FAMILY_ORDER)))

    for color, family in zip(colors, PRIMARY_FAMILY_ORDER):
        vals = np.array(counts[family])
        ax.bar(x, vals, bottom=bottom, label=FAMILY_LABELS[family], color=color)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Count of validated jailbreak prompts")
    ax.set_title("Primary jailbreak style by category")
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_primary_family_shares(records, out_path):
    categories = sorted({r["category"] for r in records})
    counts = []
    for category in categories:
        bucket = [r for r in records if r["category"] == category]
        c = Counter(r["primary_family"] for r in bucket)
        total = max(len(bucket), 1)
        counts.append([c.get(family, 0) / total for family in PRIMARY_FAMILY_ORDER])

    matrix = np.array(counts)
    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    ax.set_xticks(range(len(PRIMARY_FAMILY_ORDER)))
    ax.set_xticklabels([FAMILY_LABELS[f] for f in PRIMARY_FAMILY_ORDER], rotation=30, ha="right")
    ax.set_title("Share of each primary jailbreak style within category")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Share")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_obfuscation_rate(records, out_path):
    categories = sorted({r["category"] for r in records})
    rates = []
    for category in categories:
        bucket = [r for r in records if r["category"] == category]
        rate = np.mean([r["flags"]["obfuscation"] for r in bucket]) if bucket else 0.0
        rates.append(rate)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(categories, rates, color="#d95f02")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction with obfuscation")
    ax.set_title("How often validated jailbreaks use obfuscation")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_selected_prompt_matrix(records, out_path):
    if not records:
        return

    families = FLAG_FAMILIES
    labels = [f"{r['category']} #{i+1}" for i, r in enumerate(records)]
    matrix = np.array([
        [1 if r["flags"].get(family, False) else 0 for family in families]
        for r in records
    ])

    fig, ax = plt.subplots(figsize=(10, max(4, 0.6 * len(records))))
    im = ax.imshow(matrix, aspect="auto", cmap="Greens", vmin=0, vmax=1)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels([FAMILY_LABELS[f] for f in families], rotation=30, ha="right")
    ax.set_title("Style flags for the prompts selected in layer analysis")
    plt.colorbar(im, ax=ax, label="Flag present")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_style_layer_curves(records, out_path):
    valid = [r for r in records if r.get("causal_effect")]
    if not valid:
        return

    family_groups = defaultdict(list)
    for r in valid:
        family_groups[r["primary_family"]].append(np.array(r["causal_effect"]))

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(family_groups), 3)))

    for color, family in zip(colors, sorted(family_groups.keys())):
        arr = np.stack(family_groups[family], axis=0)
        mean_curve = arr.mean(axis=0)
        ax.plot(
            np.arange(len(mean_curve)),
            mean_curve,
            label=f"{FAMILY_LABELS[family]} (n={arr.shape[0]})",
            color=color,
            linewidth=2,
        )

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Mean causal effect")
    ax.set_title("Layer sensitivity grouped by selected-prompt primary style")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def representative_prompts(records, n=2):
    grouped = defaultdict(list)
    for r in records:
        grouped[(r["category"], r["primary_family"])].append(r)

    reps = {}
    for key, items in grouped.items():
        reps[key] = items[:n]
    return reps


def write_report(harmbench_records, analysis_records, out_path):
    lines = [
        "# Jailbreak style analysis",
        "",
    ]

    if harmbench_records:
        total = len(harmbench_records)
        lines.append(f"Validated jailbreak prompts analyzed: {total}")
        lines.append("")
        lines.append("## Primary style by category")
        categories = sorted({r["category"] for r in harmbench_records})
        for category in categories:
            bucket = [r for r in harmbench_records if r["category"] == category]
            c = Counter(r["primary_family"] for r in bucket)
            lines.append(f"### {category} (n={len(bucket)})")
            for family in PRIMARY_FAMILY_ORDER:
                count = c.get(family, 0)
                if count:
                    lines.append(
                        f"- {FAMILY_LABELS[family]}: {count} "
                        f"({count / max(len(bucket), 1):.1%})"
                    )
            obf_rate = np.mean([r["flags"]["obfuscation"] for r in bucket]) if bucket else 0.0
            lines.append(f"- Obfuscation rate: {obf_rate:.1%}")
            lines.append("")

        lines.append("## Representative prompts")
        reps = representative_prompts(harmbench_records, n=2)
        for (category, family), items in sorted(reps.items()):
            lines.append(f"### {category} / {FAMILY_LABELS[family]}")
            for item in items:
                lines.append(f"- `{item['prompt']}`")
            lines.append("")

    if analysis_records:
        lines.append("## Selected prompts used for layer analysis")
        for item in analysis_records:
            flags = [FAMILY_LABELS[f] for f in FLAG_FAMILIES if item["flags"].get(f)]
            lines.append(f"### {item['analysis_file']}")
            lines.append(f"- Category: {item['category']}")
            lines.append(f"- Primary style: {FAMILY_LABELS[item['primary_family']]}")
            lines.append(f"- Flags: {', '.join(flags) if flags else 'None'}")
            lines.append(f"- Prompt: `{item['prompt']}`")
            lines.append("")

        style_counter = Counter(item["primary_family"] for item in analysis_records)
        lines.append("## Interpretation")
        lines.append(
            "The current layer study does not control jailbreak style across categories; "
            "it uses the strongest validated jailbreak per seed."
        )
        lines.append(
            "This means observed layer differences may reflect a mix of category effects "
            "and jailbreak-style effects."
        )
        lines.append(f"Selected primary styles: {dict(style_counter)}")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    harmbench_records = load_harmbench_records(args.harmbench_dirs)
    analysis_records = load_analysis_records(args.analysis_dir)

    if not harmbench_records and not analysis_records:
        raise SystemExit("No input records found. Provide --harmbench-dirs and/or --analysis-dir.")

    if harmbench_records:
        plot_primary_family_counts(
            harmbench_records,
            os.path.join(args.output_dir, "primary_family_by_category.png"),
        )
        plot_primary_family_shares(
            harmbench_records,
            os.path.join(args.output_dir, "primary_family_shares_by_category.png"),
        )
        plot_obfuscation_rate(
            harmbench_records,
            os.path.join(args.output_dir, "obfuscation_rate_by_category.png"),
        )

    if analysis_records:
        plot_selected_prompt_matrix(
            analysis_records,
            os.path.join(args.output_dir, "selected_prompt_style_matrix.png"),
        )
        plot_style_layer_curves(
            analysis_records,
            os.path.join(args.output_dir, "style_conditioned_layer_curves.png"),
        )

    write_report(
        harmbench_records,
        analysis_records,
        os.path.join(args.output_dir, "report.md"),
    )

    print(f"Wrote style analysis to: {args.output_dir}")


if __name__ == "__main__":
    main()
