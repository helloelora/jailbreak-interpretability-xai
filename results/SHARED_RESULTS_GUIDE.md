# Shared Results Guide

This file summarizes the most relevant results to use for the final slides/report.

## What we tried

### 1. HarmBench validation of fuzzing outputs
- We first fuzzed prompts in `cybersecurity`, `malware`, and `illegal`.
- Then we revalidated all candidates with the HarmBench classifier.
- Key summaries:
  - `Cyber_harmbench/reannotation_summary.json`
  - `Malware_harmbench/reannotation_summary.json`
  - `Illegal_harmbench/reannotation_summary.json`

### 2. Exploratory cross-category mechanistic analysis
- We ran logit lens + activation patching on the strongest validated jailbreak per seed.
- This gave the first cross-category signal, but it mixed:
  - category effects
  - jailbreak-style effects

### 3. Jailbreak style analysis
- We analyzed which framing styles actually succeeded after HarmBench validation.
- This showed that successful jailbreak styles differ by category:
  - cybersecurity: mostly direct / override
  - malware: mostly fiction
  - illegal: mostly research

### 4. Controlled Run 6
- We built a style-matched subset from the existing results.
- Each category contains the same number of examples and the same primary styles:
  - 1 `research`
  - 1 `override`
- This is the cleanest controlled result.

### 5. Expanded controlled run
- We merged the original validated outputs with new `stylefill_*` results.
- Then we built a larger controlled subset using:
  - `research`
  - `override`
  - `technical_spec`
- This is a robustness check with more data.

### 6. Internal defense intervention
- We tested whether patching/ablating the most causal layers can restore refusal-like behavior.
- This gives a mitigation-oriented result.

## Recommended folders for the final presentation

### `results/shared_run6_controlled`
Use this as the main result in the presentation.

Why:
- same number of examples per category
- same primary jailbreak styles across categories
- easiest result to defend scientifically

Main message:
- Even after controlling for style, we still do **not** find one universal safety layer shared by all categories.

Recommended files:
- `causal_effect_heatmap.png`
- `overlap_top_5.png`
- `selection_report.md`
- `report.md`
- `style_matched_quota_heatmap.png`

### `results/shared_run6_expanded`
Use this as a robustness / follow-up result.

Why:
- more data than the minimal controlled Run 6
- same qualitative conclusion

Main message:
- With a larger matched subset, we still do **not** observe a top-5 layer shared by all 3 categories.

Recommended files:
- `causal_effect_heatmap.png`
- `overlap_top_5.png`
- `selection_report.md`
- `report.md`
- `subset_summary.json`

### `results/shared_style_analysis`
Use this to explain the main methodological confound.

Main message:
- Different categories are not typically broken by the same jailbreak styles.
- Therefore the exploratory cross-category comparison needed a controlled follow-up.

Recommended files:
- `primary_family_shares_by_category.png`
- `selected_prompt_style_matrix.png`

### `results/shared_internal_defense`
Use this as an additional contribution / future-defense angle.

Main message:
- Internal intervention often pushes the model back toward refusal-like behavior.
- However, the effect is not yet specific enough to claim a clean mechanistic defense.

Recommended files:
- `defense_flip_rate_by_category.png`
- `defense_score_drop_by_category.png`
- `defense_example_scatter.png`

### `results/shared_cross_category_exploratory`
Use only as backup or to explain the chronology of the project.

Main message:
- This was the first broad cross-category signal before controlling for style.

## Interpretation to keep

### Strongest claim we can make
- HarmBench-validated jailbreak success is category-dependent.
- Successful jailbreak styles are also category-dependent.
- After controlling for jailbreak style, the top causal layers still differ across categories.

### What we should avoid overstating
- We should not claim a single universal “jailbreak circuit”.
- We should not claim that the internal defense is already a fully specific mitigation.
- We should not present the `stylefill_*` runs alone as final controlled evidence; they are mainly useful through the expanded controlled subset.

## Suggested slide narrative

1. Motivation: explain the refusal-to-compliance flip mechanistically.
2. Pipeline: fuzzing -> HarmBench validation -> tracing -> defense.
3. Validation result: category-dependent real jailbreak rates.
4. Style analysis: successful jailbreak families differ by category.
5. Controlled Run 6: same styles + same counts, still different layers.
6. Expanded run: more data, same qualitative conclusion.
7. Internal defense: promising but not yet fully specific.
8. Conclusion: safety appears more distributed / category-specific than fully centralized.
