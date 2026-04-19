# Jailbreak Interpretability in Unified Reasoning Models

**Authors:** Ali Dor & Elora Drouilhet
**Course:** Explainable Artificial Intelligence — CentraleSupélec MSc AI (2025–2026)
**Target model:** Mistral Small 3.1 24B Instruct (Unsloth, 4-bit)

## Motivation

Aligned language models refuse harmful requests, but that boundary is fragile: a malicious prompt can often be reframed into a jailbreak that bypasses refusal while preserving harmful intent. Most prior work studies this at the level of attack success rate. We ask a mechanistic question instead:

> **What changes inside the model when it flips from refusal to compliance, and is that mechanism shared across risk categories?**

We combine prompt fuzzing, HarmBench validation, Integrated Gradients, hidden-state divergence, logit lens, and layer-wise activation patching to localise where the flip happens, and we test whether targeted internal interventions can restore refusal-like behaviour.

## Pipeline

| Stage | Tool | Purpose |
|-------|------|---------|
| 1. Model loading | [Unsloth](https://github.com/unslothai/unsloth) | 4-bit quantised inference with gradient access |
| 2. Adversarial generation | Genetic fuzzer (6 mutation operators) | Meaning-preserving prompt variants |
| 3. Validation | [HarmBench](https://github.com/centerforaisafety/HarmBench) classifier | Filter real harmful compliance from naive refusal bypass |
| 4. Token attribution | Captum Integrated Gradients | Which tokens move the comply–refuse score |
| 5. Internal tracing | Logit lens + activation patching (nnsight) | Which layers causally drive the flip |
| 6. Style-controlled cross-category | Matched subsets (Run 6 / expanded) | Is the mechanism shared across categories? |
| 7. Internal defense | Clean-activation patching on top causal layers | Can the flip be reverted? |

## Key Results

### 1 — Validation matters: raw fuzzer success is misleading

Starting from 600 fuzzed candidates across cybersecurity and malware seeds, only **27.7%** survive HarmBench re-validation — down from the 86.8% naive "refusal-bypass" rate.

| Stage | Count | Rate |
|-------|------:|-----:|
| Generated candidates | 600 | – |
| Naive jailbreaks | 521 | 86.8% |
| HarmBench positive | 183 | 30.5% |
| Final revised keep | 166 | 27.7% |

### 2 — Token attribution shifts under jailbreak

On the keylogger seed, refusal-driving mass concentrates on the harmful request itself under the clean prompt, but an override/fiction reframing redistributes saliency onto framing tokens such as *"developer mode"*.

<p align="center">
  <img src="figures/keylogger_tokens_clean_sentence.png" width="45%" alt="Clean prompt attribution">
  <img src="figures/keylogger_tokens_jailbreak.png" width="45%" alt="Jailbreak prompt attribution">
</p>

### 3 — Early divergence, middle-stack causality, late decision visibility

Hidden states for clean and jailbreak pairs diverge early, but the layers that *causally* flip the comply–refuse score sit in the middle stack (around 20–24). The decision becomes readable in the logit lens only in the final layers (≈ 35–39).

<p align="center">
  <img src="figures/cross_seed_divergence.png" width="32%" alt="Hidden-state divergence">
  <img src="figures/cross_seed_layer_importance.png" width="32%" alt="Per-layer causal importance">
  <img src="figures/cross_seed_logit_lens.png" width="32%" alt="Logit lens gap">
</p>

Across 10 validated clean/jailbreak pairs, layers 21 and 22 land in the top-5 for 8/10 examples, and single-layer patching often recovers most of the full jailbreak effect (+0.94 to +5.06 logit points).

<p align="center">
  <img src="figures/compliance_landscape.png" width="75%" alt="Single-layer patching landscape">
</p>

### 4 — Successful jailbreak styles are category-dependent

A style analysis of 184 HarmBench-validated jailbreaks shows that the *dominant* successful framing family differs by category:

- **Cybersecurity** — mostly direct/plain (56%) and override (26%)
- **Malware** — dominated by fiction/story (56%)
- **Illegal** — mostly research/academic (57%)

<p align="center">
  <img src="figures/primary_family_shares_by_category.png" width="75%" alt="Primary jailbreak family shares by category">
</p>

This confound is the reason we re-ran the cross-category analysis on style-matched subsets.

### 5 — No universal safety layer, even after style-matching

With the minimal **Run 6** style-matched subset (same count per category, one research + one override jailbreak each) the top-5 causal layers are:

- Cybersecurity: 21, 22, 38, 36, 32
- Illegal: 24, 34, 35, 33, 31
- Malware: 19, 21, 17, 18, 20

**No layer appears in the top-5 of all three categories.** The expanded matched subset (5 examples per category, balanced across research / override / technical-spec) preserves the same conclusion.

<p align="center">
  <img src="figures/run6_top_5_layers_per_category.png" width="70%" alt="Controlled cross-category top-5 layers">
</p>

<p align="center">
  <img src="figures/run6_causal_effect_heatmap.png" width="45%" alt="Run 6 causal effect heatmap">
  <img src="figures/run6_overlap_top_5.png" width="45%" alt="Top-5 overlap across categories">
</p>

### 6 — Internal defense: promising but not specific yet

Clean-activation patching on the identified top layers produces large score drops and often restores refusal-like behaviour, especially for malware and illegal. However, random-layer controls remain competitive at larger *k*, so this is a mitigation *direction*, not a finished defense.

<p align="center">
  <img src="figures/defense_score_drop_by_category.png" width="48%" alt="Defense score drop by category">
  <img src="figures/defense_flip_rate_by_category.png" width="48%" alt="Defense flip rate by category">
</p>

## Take-home

- HarmBench-style validation is a prerequisite for mechanistic jailbreak analysis, not a post-processing step.
- Successful jailbreak *style* is category-dependent; cross-category comparisons must control for it.
- Even after style-matching, the top causal layers differ across categories — safety appears **partially overlapping but category-specific** rather than concentrated in one universal circuit.
- Targeted internal intervention can move the comply–refuse score substantially, but current evidence is not specific enough to claim a mechanistic mitigation.

## Repository Layout

```
.
├── src/
│   ├── model/          # Unsloth 4-bit loader
│   ├── fuzzer/         # Genetic prompt fuzzer + style-completion variant
│   ├── attribution/    # Captum Integrated Gradients
│   ├── tracing/        # Hidden-state divergence & activation analysis
│   └── evaluation/     # HarmBench classifier wrapper + metrics
├── scripts/            # Orchestration: cross-category, style-matched, defense, plotting
├── ruche/              # SLURM job scripts for La Ruche HPC (Apptainer)
├── results/
│   ├── shared_run6_controlled/            # Main controlled cross-category result
│   ├── shared_run6_expanded/              # Larger style-matched robustness check
│   ├── shared_style_analysis/             # Category-dependent jailbreak styles
│   ├── shared_internal_defense/           # Clean-patch defense experiment
│   ├── shared_cross_category_exploratory/ # Pre-control exploratory run
│   └── SHARED_RESULTS_GUIDE.md            # Which folder to use for what claim
├── figures/            # Figures used in this README / report / slides
└── requirements.txt
```

See [`results/SHARED_RESULTS_GUIDE.md`](results/SHARED_RESULTS_GUIDE.md) for a narrative of which results support which claims.

## Reproducing the Analyses

### Environment

```bash
pip install -r requirements.txt
```

Mistral Small 3.1 24B is loaded in 4-bit through Unsloth; a single A100 40 GB is sufficient. HarmBench validation additionally loads `cais/HarmBench-Llama-2-13b-cls`.

### Main entry points

```bash
# 1. Generate and score fuzzed variants for a seed category
python -m src.fuzzer.run --category cybersecurity

# 2. Re-validate with HarmBench
python scripts/reannotate_harmbench.py results/<fuzz_run_dir>

# 3. Cross-category XAI (logit lens + activation patching + IG)
python scripts/run_cross_category.py --output results/cross_category

# 4. Style analysis of validated jailbreaks
python scripts/analyze_jailbreak_styles.py

# 5. Build the style-matched controlled subset and run patching
python scripts/build_style_matched_validated_subset.py
python scripts/run_style_matched_subset.py

# 6. Internal defense experiment
python scripts/run_internal_defense.py
python scripts/plot_internal_defense.py
```

All experiments were run on the CentraleSupélec HPC cluster La Ruche (NVIDIA A100 40 GB).

## References

1. Mazeika et al. *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal.* arXiv:2402.04249, 2024.
2. Chao et al. *JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models.* arXiv:2404.01318, 2024.
3. Zou et al. *Universal and Transferable Adversarial Attacks on Aligned Language Models.* arXiv:2307.15043, 2023.
4. Arditi et al. *Refusal in Language Models Is Mediated by a Single Direction.* arXiv:2406.11717, 2024.
5. He et al. *JailbreakLens: Interpreting Jailbreak Mechanism in the Lens of Representation and Circuit.* arXiv:2411.11114, 2024.
6. Kramár et al. *AtP\*: An Efficient and Scalable Method for Localizing LLM Behaviour to Components.* arXiv:2403.00745, 2024.
7. Sundararajan et al. *Axiomatic Attribution for Deep Networks.* ICML 2017.
