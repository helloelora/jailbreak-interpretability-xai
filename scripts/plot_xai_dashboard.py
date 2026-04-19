"""
Rich visualizations for XAI jailbreak analysis.

Produces detailed multi-panel figures showing relationships between
Integrated Gradients, layer divergence, and activation patching results.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.0)

# Consistent color palette
COLORS = {
    "comply": "#e74c3c",     # red
    "refuse": "#3498db",     # blue
    "divergence": "#e67e22",  # orange
    "patching": "#2ecc71",   # green
    "ablation": "#9b59b6",   # purple
    "neutral": "#7f8c8d",    # gray
    "highlight": "#f1c40f",  # yellow
}


# ============================================================
# 1. TOKEN ATTRIBUTION HEATMAP (colored text)
# ============================================================

def plot_token_heatmap(attr_dict, title, save_path, max_tokens=300):
    """Render tokens as colored text where color = attribution strength.

    Red = pushes toward compliance, Blue = pushes toward refusal.
    Intensity = magnitude.

    If `attr_dict["user_span"]` is set, only the user-supplied portion of the
    chat-formatted prompt is rendered (boilerplate like the BOS token,
    [SYSTEM_PROMPT], and [INST]/[/INST] are skipped). This makes the heatmap
    actually show the part of the prompt the experimenter cares about, rather
    than Mistral's auto-injected system prompt.
    """
    import textwrap

    tokens_full = attr_dict["tokens"]
    scores_full = attr_dict["attributions"].numpy()
    span = attr_dict.get("user_span")

    if span is not None:
        start, end = span
        tokens = tokens_full[start:end][:max_tokens]
        scores = scores_full[start:end][:max_tokens]
        span_label = f"user content tokens [{start}:{end}] of {len(tokens_full)}"
    else:
        tokens = tokens_full[:max_tokens]
        scores = scores_full[:max_tokens]
        span_label = f"first {len(tokens)} of {len(tokens_full)} tokens (user span not detected)"

    # Renormalize within the displayed slice so colors actually have contrast.
    max_abs = max(np.abs(scores).max(), 1e-10)
    norm_scores = scores / max_abs

    n_tokens = len(tokens)
    est_lines = max(2, int(np.ceil(n_tokens / 18)))
    fig_height = max(3.5, 1.2 + 0.5 * est_lines)
    fig, ax = plt.subplots(figsize=(18, fig_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    cmap = plt.cm.RdBu_r  # Red = positive (comply), Blue = negative (refuse)

    line_height = 1.0 / max(est_lines + 2, 6)
    x = 0.01
    y = 1.0 - line_height
    fontsize = 11

    for tok, ns in zip(tokens, norm_scores):
        color = cmap(0.5 + ns * 0.5)
        alpha = min(1.0, 0.35 + abs(ns) * 0.65)

        display_tok = tok.replace("\n", "\\n").replace("\t", "\\t")
        if not display_tok.strip():
            display_tok = "·"

        text = ax.text(x, y, display_tok, fontsize=fontsize, fontfamily="monospace",
                       color="white" if abs(ns) > 0.55 else "black",
                       bbox=dict(boxstyle="round,pad=0.18", facecolor=color, alpha=alpha,
                                 edgecolor="none"),
                       va="top", ha="left")

        fig.canvas.draw()
        bb = text.get_window_extent().transformed(ax.transData.inverted())
        x = bb.x1 + 0.004

        if x > 0.97:
            x = 0.01
            y -= line_height
            if y < line_height:
                break

    # Colorbar at the bottom
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-1, 1))
    cbar = plt.colorbar(sm, ax=ax, orientation="horizontal",
                        fraction=0.04, pad=0.04, aspect=40)
    cbar.set_label("← pushes toward Refuse        Attribution        pushes toward Comply →",
                   fontsize=10)

    short_title = textwrap.shorten(title, width=130, placeholder="…")
    fig.suptitle(short_title, fontsize=13, fontweight="bold", y=0.99)
    ax.text(0.5, 1.05, span_label, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9, color="#555555", style="italic")

    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


# ============================================================
# 2. FULL EXAMPLE DASHBOARD (4 panels per example)
# ============================================================

def plot_example_dashboard(ig_clean, ig_jb, divergence, patching, ablation,
                           seed_prompt, jb_prompt, save_path):
    """Single figure with 4 panels showing all XAI results for one example.

    Panel A: IG comparison (top tokens, clean vs jailbreak)
    Panel B: Layer divergence heatmap
    Panel C: Activation patching causal effect
    Panel D: Ablation curve
    """
    fig = plt.figure(figsize=(22, 16))
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3,
                           height_ratios=[1, 1, 1])

    import textwrap
    seed_wrapped = "\n".join(textwrap.wrap(seed_prompt, width=120, max_lines=2,
                                           placeholder="…"))
    fig.suptitle(f"XAI Analysis Dashboard\nSeed: {seed_wrapped}",
                fontsize=14, fontweight="bold", y=0.995)

    def _placeholder(ax, label):
        ax.text(0.5, 0.5, "(not available)", ha="center", va="center",
                fontsize=12, color="gray", transform=ax.transAxes)
        ax.set_title(label, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    # --- Panel A: IG top tokens comparison ---
    ax_a = fig.add_subplot(gs[0, :])
    if ig_clean is not None and ig_jb is not None:
        _plot_ig_comparison(ax_a, ig_clean, ig_jb)
    else:
        _placeholder(ax_a, "A. Integrated Gradients — Top Tokens")

    # --- Panel B: Layer divergence ---
    ax_b = fig.add_subplot(gs[1, 0])
    if divergence is not None:
        _plot_divergence_detailed(ax_b, divergence)
    else:
        _placeholder(ax_b, "B. Layer Divergence")

    # --- Panel C: Activation patching ---
    ax_c = fig.add_subplot(gs[1, 1])
    if patching is not None:
        _plot_patching_detailed(ax_c, patching)
    else:
        _placeholder(ax_c, "C. Activation Patching")

    # --- Panel D: Ablation ---
    ax_d = fig.add_subplot(gs[2, 0])
    if ablation is not None:
        _plot_ablation_detailed(ax_d, ablation)
    else:
        ax_d.text(0.5, 0.5, "Ablation not available", ha="center", va="center",
                 fontsize=12, color="gray")
        ax_d.set_title("D. Ablation Test")

    # --- Panel E: Divergence vs Patching correlation ---
    ax_e = fig.add_subplot(gs[2, 1])
    if divergence is not None and patching is not None:
        _plot_divergence_vs_patching(ax_e, divergence, patching)
    else:
        ax_e.text(0.5, 0.5, "Correlation not available", ha="center", va="center",
                 fontsize=12, color="gray")

    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def _plot_ig_comparison(ax, ig_clean, ig_jb, n_show=15):
    """Side-by-side horizontal bar chart of top attributed tokens."""
    def get_top(attr_dict, n):
        scores = attr_dict["attributions"].numpy()
        tokens = attr_dict["tokens"]
        span = attr_dict.get("user_span")
        if span is not None:
            start, end = span
            scores = scores[start:end]
            tokens = tokens[start:end]
        if len(scores) == 0:
            return [], np.array([])
        n = min(n, len(scores))
        top_idx = np.argsort(np.abs(scores))[-n:]
        top_idx = top_idx[np.argsort(scores[top_idx])]
        return [tokens[i] for i in top_idx], scores[top_idx]

    tokens_c, scores_c = get_top(ig_clean, n_show)
    tokens_j, scores_j = get_top(ig_jb, n_show)

    n_rows = max(len(scores_c), len(scores_j))
    if n_rows == 0:
        ax.text(0.5, 0.5, "(no IG data)", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title("A. Integrated Gradients — Top Tokens", fontweight="bold")
        return

    def pad(tokens, scores, n):
        tokens = list(tokens) + [""] * (n - len(tokens))
        scores = np.concatenate([scores, np.zeros(n - len(scores))])
        return tokens, scores

    tokens_c, scores_c = pad(tokens_c, scores_c, n_rows)
    tokens_j, scores_j = pad(tokens_j, scores_j, n_rows)

    y_c = np.arange(n_rows)
    width = 0.35

    colors_c = [COLORS["comply"] if s > 0 else COLORS["refuse"] for s in scores_c]
    colors_j = [COLORS["comply"] if s > 0 else COLORS["refuse"] for s in scores_j]

    ax.barh(y_c + width / 2, scores_c, width, color=colors_c, alpha=0.6, label="Clean (refused)")
    ax.barh(y_c - width / 2, scores_j, width, color=colors_j, alpha=0.9, label="Jailbreak (complied)")

    combined_labels = []
    for tc, tj in zip(tokens_c, tokens_j):
        tc_clean = repr(tc.strip())[1:-1][:12]
        tj_clean = repr(tj.strip())[1:-1][:12]
        combined_labels.append(f"{tc_clean} | {tj_clean}")

    ax.set_yticks(y_c)
    ax.set_yticklabels(combined_labels, fontsize=7, fontfamily="monospace")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.set_xlabel("Attribution Score (← refuse | comply →)")
    ax.set_title("A. Integrated Gradients — Top Tokens (Clean | Jailbreak)", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)

    score_c = ig_clean.get("compliance_score", 0)
    score_j = ig_jb.get("compliance_score", 0)
    ax.text(0.02, 0.97, f"Clean compliance: {score_c:.3f}\nJailbreak compliance: {score_j:.3f}\nΔ = {score_j - score_c:.3f}",
           transform=ax.transAxes, fontsize=9, va="top",
           bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))


def _plot_divergence_detailed(ax, divergence):
    """Layer divergence with highlighted peak regions."""
    n = divergence["n_layers"]
    layers = np.arange(n)
    last_pos = divergence["last_pos_divergence"].numpy()
    avg_pos = divergence["layer_avg_divergence"].numpy()

    ax.fill_between(layers, last_pos, alpha=0.2, color=COLORS["divergence"])
    ax.plot(layers, last_pos, linewidth=2.5, color=COLORS["divergence"],
            label="Last position", marker="o", markersize=3)
    ax.plot(layers, avg_pos, linewidth=1.5, color=COLORS["refuse"],
            alpha=0.7, label="Avg all positions", linestyle="--")

    top3 = np.argsort(last_pos)[-3:]
    for t in top3:
        ax.axvline(x=t, color=COLORS["highlight"], alpha=0.3, linewidth=8)
        ax.annotate(f"L{t}\n{last_pos[t]:.3f}", xy=(t, last_pos[t]),
                   fontsize=7, ha="center", va="bottom", fontweight="bold",
                   color=COLORS["divergence"])

    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Distance")
    ax.set_title("B. Layer Divergence (Clean vs Jailbreak)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, n - 1)

    if n >= 30:
        for start, end, label in [(0, n // 3, "Early"), (n // 3, 2 * n // 3, "Middle"),
                                   (2 * n // 3, n, "Late")]:
            ax.axvspan(start, end, alpha=0.03,
                      color=["blue", "green", "red"][[0, 1, 2][[(0, n // 3), (n // 3, 2 * n // 3), (2 * n // 3, n)].index((start, end))]])
            ax.text((start + end) / 2, ax.get_ylim()[1] * 0.95, label,
                   ha="center", fontsize=8, alpha=0.5, style="italic")


def _plot_patching_detailed(ax, patching):
    """Activation patching with gradient coloring by effect size."""
    n = patching["n_layers"]
    baseline = patching["baseline_score"]
    jb_score = patching["jailbreak_score"]
    effects = patching["causal_effect"].numpy() if hasattr(patching["causal_effect"], "numpy") else np.array(patching["causal_effect"])
    full_effect = jb_score - baseline

    max_eff = max(np.abs(effects).max(), 1e-10)
    norm = plt.Normalize(0, max_eff)
    cmap = plt.cm.YlOrRd

    ax.bar(range(n), effects, color=[cmap(norm(abs(e))) for e in effects],
           edgecolor="none", width=0.8)

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.axhline(y=full_effect, color=COLORS["patching"], linestyle="--", linewidth=2,
               label=f"Full jailbreak effect ({full_effect:.2f})")

    top5 = np.argsort(np.abs(effects))[-5:]
    for t in top5:
        ax.annotate(f"L{t}", xy=(t, effects[t]),
                   fontsize=7, ha="center",
                   va="bottom" if effects[t] > 0 else "top",
                   fontweight="bold", color="darkred")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Causal Effect (compliance shift)")
    ax.set_title("C. Activation Patching — Causal Layer Importance", fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.5, n - 0.5)

    cumulative = np.cumsum(np.sort(np.abs(effects))[::-1]) / max(abs(full_effect), 1e-10)
    ax_inset = ax.inset_axes([0.6, 0.55, 0.35, 0.35])
    ax_inset.plot(range(1, n + 1), cumulative[:n], color="darkred", linewidth=1.5)
    ax_inset.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.5)
    ax_inset.set_xlabel("Top-N layers", fontsize=7)
    ax_inset.set_ylabel("Cumulative effect", fontsize=7)
    ax_inset.set_title("Cumul. causal effect", fontsize=7)
    ax_inset.tick_params(labelsize=6)


def _plot_ablation_detailed(ax, ablation):
    """Ablation curve with annotation."""
    baseline = ablation["baseline"]
    ks = sorted(ablation["scores"].keys())
    scores = [ablation["scores"][k] for k in ks]

    ax.plot(ks, scores, marker="o", linewidth=2.5, color=COLORS["ablation"],
            markersize=6, zorder=3)
    ax.fill_between(ks, baseline, scores, alpha=0.15, color=COLORS["ablation"])
    ax.axhline(y=baseline, color=COLORS["neutral"], linestyle="--", linewidth=1.5,
               label=f"Unmasked ({baseline:.3f})")

    drops = [(k, baseline - s) for k, s in zip(ks, scores)]
    max_drop_k, max_drop_val = max(drops, key=lambda x: x[1])
    ax.annotate(f"Biggest drop at K={max_drop_k}\nΔ = {max_drop_val:.3f}",
               xy=(max_drop_k, ablation["scores"][max_drop_k]),
               xytext=(max_drop_k + 2, baseline - max_drop_val * 0.3),
               fontsize=8, arrowprops=dict(arrowstyle="->", color="black"),
               bbox=dict(boxstyle="round", facecolor="lightyellow"))

    ax.set_xlabel("Top-K tokens masked")
    ax.set_ylabel("Compliance Score")
    ax.set_title("D. Ablation Test — Masking Top IG Tokens", fontweight="bold")
    ax.legend(fontsize=9)


def _plot_divergence_vs_patching(ax, divergence, patching):
    """Scatter plot: divergence (x) vs causal effect (y) per layer."""
    div_vals = divergence["last_pos_divergence"].numpy() if hasattr(divergence["last_pos_divergence"], "numpy") else np.array(divergence["last_pos_divergence"])
    eff_vals = patching["causal_effect"].numpy() if hasattr(patching["causal_effect"], "numpy") else np.array(patching["causal_effect"])
    n = len(div_vals)

    scatter = ax.scatter(div_vals, eff_vals, c=np.arange(n), cmap="viridis",
                        s=50, alpha=0.8, edgecolors="white", linewidth=0.5, zorder=3)

    div_threshold = np.percentile(div_vals, 85)
    eff_threshold = np.percentile(np.abs(eff_vals), 85)
    for i in range(n):
        if div_vals[i] > div_threshold or abs(eff_vals[i]) > eff_threshold:
            ax.annotate(f"L{i}", (div_vals[i], eff_vals[i]), fontsize=7,
                       fontweight="bold", ha="left", va="bottom")

    corr = np.corrcoef(div_vals, eff_vals)[0, 1]
    ax.text(0.05, 0.95, f"Pearson r = {corr:.3f}", transform=ax.transAxes,
           fontsize=10, va="top",
           bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    if not np.isnan(corr):
        z = np.polyfit(div_vals, eff_vals, 1)
        p = np.poly1d(z)
        x_fit = np.linspace(div_vals.min(), div_vals.max(), 50)
        ax.plot(x_fit, p(x_fit), color="red", linestyle="--", alpha=0.5, linewidth=1.5)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Layer Index", fontsize=9)

    ax.set_xlabel("Divergence (cosine distance)")
    ax.set_ylabel("Causal Effect (patching)")
    ax.set_title("E. Divergence vs Causal Effect per Layer", fontweight="bold")
    ax.axhline(y=0, color="gray", linewidth=0.5, alpha=0.5)


# ============================================================
# 3. CROSS-SEED COMPARISON FIGURES
# ============================================================

def plot_cross_seed_divergence(all_results, save_path):
    """Compare divergence profiles across all seeds in a single heatmap + overlay."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[1, 1.2])

    seeds = []
    div_matrix = []
    patch_matrix = []

    for r in all_results:
        if "divergence" not in r or "patching" not in r:
            continue
        seed_label = r["seed_prompt"][:40]
        seeds.append(seed_label)
        div_data = r["divergence"]
        patch_data = r["patching"]

        div_vals = div_data["last_pos"] if isinstance(div_data["last_pos"], list) else div_data["last_pos_divergence"].tolist()
        eff_vals = patch_data["all_effects"] if isinstance(patch_data.get("all_effects"), list) else patch_data["causal_effect"].tolist()

        div_matrix.append(div_vals)
        patch_matrix.append(eff_vals)

    if not seeds:
        plt.close()
        return

    div_matrix = np.array(div_matrix)
    patch_matrix = np.array(patch_matrix)

    ax1 = axes[0]
    im1 = ax1.imshow(div_matrix, aspect="auto", cmap="hot", interpolation="nearest")
    ax1.set_yticks(range(len(seeds)))
    ax1.set_yticklabels(seeds, fontsize=8)
    ax1.set_xlabel("Layer")
    ax1.set_title("Layer Divergence Across Seeds (cosine distance at last position)",
                  fontweight="bold", fontsize=13)
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.02)
    cbar1.set_label("Cosine Distance")

    for i in range(len(seeds)):
        peak = np.argmax(div_matrix[i])
        ax1.plot(peak, i, "w*", markersize=10)

    ax2 = axes[1]
    max_abs = max(np.abs(patch_matrix).max(), 1e-10)
    im2 = ax2.imshow(patch_matrix, aspect="auto", cmap="RdBu_r",
                     vmin=-max_abs, vmax=max_abs, interpolation="nearest")
    ax2.set_yticks(range(len(seeds)))
    ax2.set_yticklabels(seeds, fontsize=8)
    ax2.set_xlabel("Layer")
    ax2.set_title("Activation Patching Causal Effect Across Seeds",
                  fontweight="bold", fontsize=13)
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.02)
    cbar2.set_label("Causal Effect (compliance shift)")

    for i in range(len(seeds)):
        top3 = np.argsort(np.abs(patch_matrix[i]))[-3:]
        for t in top3:
            ax2.plot(t, i, "k*", markersize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_cross_seed_summary(all_results, save_path):
    """Summary bar chart: which layers are consistently important across seeds."""
    n_layers = None
    layer_importance = None

    for r in all_results:
        if "patching" not in r:
            continue
        eff = r["patching"]
        vals = eff["all_effects"] if isinstance(eff.get("all_effects"), list) else eff["causal_effect"].tolist()
        if n_layers is None:
            n_layers = len(vals)
            layer_importance = np.zeros(n_layers)
        layer_importance += np.abs(vals)

    if n_layers is None:
        return

    layer_importance /= layer_importance.sum()

    fig, ax = plt.subplots(figsize=(16, 5))

    colors = plt.cm.YlOrRd(layer_importance / layer_importance.max())
    ax.bar(range(n_layers), layer_importance, color=colors, edgecolor="none", width=0.85)

    top5 = np.argsort(layer_importance)[-5:][::-1]
    for rank, idx in enumerate(top5):
        ax.annotate(f"#{rank+1}\nL{idx}", xy=(idx, layer_importance[idx]),
                   fontsize=9, ha="center", va="bottom", fontweight="bold",
                   color="darkred")

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Aggregate Causal Importance\n(mean |effect| across seeds)", fontsize=11)
    ax.set_title("Cross-Seed Layer Importance — Which Layers Consistently Matter?",
                fontsize=14, fontweight="bold")
    ax.set_xlim(-0.5, n_layers - 0.5)

    thirds = n_layers // 3
    for start, end, label, color in [(0, thirds, "Early Layers", "blue"),
                                      (thirds, 2 * thirds, "Middle Layers", "green"),
                                      (2 * thirds, n_layers, "Late Layers", "red")]:
        region_importance = layer_importance[start:end].sum()
        ax.axvspan(start - 0.5, end - 0.5, alpha=0.05, color=color)
        ax.text((start + end) / 2, ax.get_ylim()[1] * 0.92,
               f"{label}\n({region_importance:.1%})",
               ha="center", fontsize=9, style="italic", alpha=0.7,
               bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


# ============================================================
# 4. LOGIT LENS VISUALIZATIONS
# ============================================================

def plot_logit_lens_single(ll_result, title, save_path):
    """Plot logit lens for a single prompt: comply vs refuse probability across layers."""
    n = ll_result["n_layers"]
    layers = np.arange(n)
    comply = ll_result["comply_probs"].numpy() if hasattr(ll_result["comply_probs"], "numpy") else np.array(ll_result["comply_probs"])
    refuse = ll_result["refuse_probs"].numpy() if hasattr(ll_result["refuse_probs"], "numpy") else np.array(ll_result["refuse_probs"])
    gap = comply - refuse

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), height_ratios=[2, 1],
                             gridspec_kw={"hspace": 0.25})

    ax1 = axes[0]
    ax1.fill_between(layers, comply, alpha=0.15, color=COLORS["comply"])
    ax1.fill_between(layers, refuse, alpha=0.15, color=COLORS["refuse"])
    ax1.plot(layers, comply, linewidth=2.5, color=COLORS["comply"],
             label="P(comply tokens)", marker=".", markersize=4)
    ax1.plot(layers, refuse, linewidth=2.5, color=COLORS["refuse"],
             label="P(refuse tokens)", marker=".", markersize=4)

    crossover = ll_result.get("crossover_layer")
    if crossover is not None:
        ax1.axvline(x=crossover, color=COLORS["highlight"], linewidth=2,
                    linestyle="--", alpha=0.8, label=f"Crossover at L{crossover}")
        ax1.annotate(f"Layer {crossover}\nComply > Refuse",
                    xy=(crossover, max(comply[crossover], refuse[crossover])),
                    xytext=(crossover + 3, max(comply.max(), refuse.max()) * 0.8),
                    fontsize=9, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="black"),
                    bbox=dict(boxstyle="round", facecolor="lightyellow"))

    ax1.set_ylabel("Probability")
    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10, loc="upper left")
    ax1.set_xlim(0, n - 1)
    ax1.set_ylim(bottom=0)

    top_tokens = ll_result["top_tokens"]
    top_probs = ll_result["top_probs"]
    for i in range(0, n, max(1, n // 10)):
        tok = top_tokens[i][0].strip()
        prob = top_probs[i][0]
        ax1.annotate(f'"{tok}" ({prob:.2f})', xy=(i, max(comply[i], refuse[i])),
                    fontsize=6, rotation=45, ha="left", va="bottom", alpha=0.7)

    ax2 = axes[1]
    colors_gap = [COLORS["comply"] if g > 0 else COLORS["refuse"] for g in gap]
    ax2.bar(layers, gap, color=colors_gap, alpha=0.7, width=0.8)
    ax2.axhline(y=0, color="black", linewidth=0.8)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("P(comply) − P(refuse)")
    ax2.set_title("Compliance Gap per Layer", fontsize=11, fontweight="bold")
    ax2.set_xlim(-0.5, n - 0.5)

    if crossover is not None:
        ax2.axvline(x=crossover, color=COLORS["highlight"], linewidth=2,
                    linestyle="--", alpha=0.8)

    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_logit_lens_comparison(ll_clean, ll_jailbreak, seed_prompt, save_path):
    """Side-by-side logit lens: clean (refused) vs jailbreak (complied)."""
    n = ll_clean["n_layers"]
    layers = np.arange(n)

    fig, axes = plt.subplots(2, 2, figsize=(20, 10), gridspec_kw={"hspace": 0.3, "wspace": 0.25})

    for col, (ll, label) in enumerate([(ll_clean, "Clean (Refused)"),
                                        (ll_jailbreak, "Jailbreak (Complied)")]):
        comply = ll["comply_probs"].numpy() if hasattr(ll["comply_probs"], "numpy") else np.array(ll["comply_probs"])
        refuse = ll["refuse_probs"].numpy() if hasattr(ll["refuse_probs"], "numpy") else np.array(ll["refuse_probs"])
        gap = comply - refuse

        ax = axes[0, col]
        ax.fill_between(layers, comply, alpha=0.15, color=COLORS["comply"])
        ax.fill_between(layers, refuse, alpha=0.15, color=COLORS["refuse"])
        ax.plot(layers, comply, linewidth=2, color=COLORS["comply"], label="P(comply)")
        ax.plot(layers, refuse, linewidth=2, color=COLORS["refuse"], label="P(refuse)")

        crossover = ll.get("crossover_layer")
        if crossover is not None:
            ax.axvline(x=crossover, color=COLORS["highlight"], linewidth=2, linestyle="--",
                      label=f"Crossover L{crossover}")

        ax.set_title(f"{label}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Probability")
        ax.legend(fontsize=9)
        ax.set_xlim(0, n - 1)
        ax.set_ylim(bottom=0)

        ax2 = axes[1, col]
        colors_gap = [COLORS["comply"] if g > 0 else COLORS["refuse"] for g in gap]
        ax2.bar(layers, gap, color=colors_gap, alpha=0.7, width=0.8)
        ax2.axhline(y=0, color="black", linewidth=0.8)
        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Compliance gap")
        ax2.set_xlim(-0.5, n - 0.5)

        if crossover is not None:
            ax2.axvline(x=crossover, color=COLORS["highlight"], linewidth=2, linestyle="--")

    fig.suptitle(f"Logit Lens Comparison — {seed_prompt[:60]}",
                fontsize=14, fontweight="bold", y=1.01)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_cross_seed_logit_lens(all_results, save_path):
    """Heatmap: compliance gap across layers × seeds for jailbreak prompts."""
    seeds = []
    gap_matrix = []
    crossovers = []

    for r in all_results:
        if "logit_lens_jailbreak" not in r:
            continue
        ll = r["logit_lens_jailbreak"]
        seeds.append(r["seed_prompt"][:40])
        gap = ll["comply_minus_refuse"] if isinstance(ll["comply_minus_refuse"], list) else ll["comply_minus_refuse"].tolist()
        gap_matrix.append(gap)
        crossovers.append(ll.get("crossover_layer"))

    if not seeds:
        return

    gap_matrix = np.array(gap_matrix)
    max_abs = max(np.abs(gap_matrix).max(), 1e-10)

    fig, ax = plt.subplots(figsize=(18, max(4, len(seeds) * 0.8 + 2)))
    im = ax.imshow(gap_matrix, aspect="auto", cmap="RdBu_r",
                   vmin=-max_abs, vmax=max_abs, interpolation="nearest")

    ax.set_yticks(range(len(seeds)))
    ax.set_yticklabels(seeds, fontsize=9)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_title("Logit Lens: Compliance Gap Across Seeds (Jailbreak Prompts)\n"
                "Red = model predicts compliance, Blue = model predicts refusal",
                fontsize=13, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.02)
    cbar.set_label("P(comply) − P(refuse)")

    for i, co in enumerate(crossovers):
        if co is not None:
            ax.plot(co, i, "k|", markersize=20, markeredgewidth=2)
            ax.annotate(f"L{co}", xy=(co, i), fontsize=7, fontweight="bold",
                       ha="center", va="top", color="black",
                       xytext=(co, i + 0.35))

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


# ============================================================
# 5. COMPLIANCE SCORE LANDSCAPE
# ============================================================

def plot_compliance_score_landscape(all_results, save_path):
    """Show clean vs jailbreak compliance scores + patching recovery per seed."""
    seeds = []
    clean_scores = []
    jb_scores = []
    max_patch_scores = []

    for r in all_results:
        if "patching" not in r:
            continue
        seeds.append(r["seed_prompt"][:35])
        p = r["patching"]
        clean_scores.append(p["baseline_score"])
        jb_scores.append(p["jailbreak_score"])
        patched = p["all_effects"] if isinstance(p.get("all_effects"), list) else p["causal_effect"].tolist()
        max_patch_scores.append(p["baseline_score"] + max(patched))

    if not seeds:
        return

    x = np.arange(len(seeds))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, clean_scores, width, label="Clean (refused)",
           color=COLORS["refuse"], alpha=0.8)
    ax.bar(x, jb_scores, width, label="Jailbreak (complied)",
           color=COLORS["comply"], alpha=0.8)
    ax.bar(x + width, max_patch_scores, width, label="Best single-layer patch",
           color=COLORS["patching"], alpha=0.8)

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(seeds, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("Compliance Score (comply − refuse logits)")
    ax.set_title("Compliance Score Landscape Across Seeds",
                fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    for i in range(len(seeds)):
        gap = jb_scores[i] - clean_scores[i]
        ax.annotate(f"Δ={gap:.2f}", xy=(i, max(clean_scores[i], jb_scores[i])),
                   fontsize=7, ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
