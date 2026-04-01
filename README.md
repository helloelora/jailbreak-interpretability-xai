# Jailbreak Interpretability in Small Language Models

**Authors:** Ali Dor & Elora Drouilhet  
**Course:** Explainable Artificial Intelligence — CentraleSupélec MSc AI (2025–2026)

## Overview

This project investigates **why** safety guardrails in unified reasoning models fail under adversarial jailbreak prompts. Rather than treating jailbreaks as black-box failures, we provide a mechanistic explanation of the compliance flip — the internal transition from refusal to obedience.

**Target model:** Mistral Small 3.1 (24B dense) loaded via [Unsloth](https://github.com/unslothai/unsloth) in 4-bit quantization.

## Pipeline

| Stage | Tool | Description |
|-------|------|-------------|
| 1. Model loading | Unsloth | 4-bit quantized inference with gradient access |
| 2. Adversarial generation | Genetic fuzzer | Meaning-preserving prompt variants (inspired by AdvJudge-Zero) |
| 3. Feature attribution | Captum (Integrated Gradients) | Token-level attribution during compliance flip |
| 4. Internal tracing | TransformerLens | Activation caching & layer-wise safety signal analysis |

## Evaluation

- **Faithfulness:** Kendall's Tau between fuzzer-identified trigger tokens and high-attribution tokens
- **Ablation:** Masking identified circuits to restore refusal behavior
- **Datasets:** [HarmBench](https://github.com/centerforaisafety/HarmBench) + [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench), augmented with 100 fuzzed variants per category

## Project Structure

```
├── src/
│   ├── model/          # Unsloth model loading & inference
│   ├── fuzzer/         # Genetic algorithm prompt fuzzer
│   ├── attribution/    # Integrated Gradients (Captum)
│   ├── tracing/        # TransformerLens activation analysis
│   └── evaluation/     # Metrics & ablation tests
├── notebooks/          # Exploration & analysis notebooks
├── data/               # Datasets (HarmBench, JailbreakBench)
└── results/            # Outputs, figures, logs
```

## References

1. Mistral AI Team. (2026). *Introducing Mistral Small 4.*
2. Unit 42. (2026). *Open, Closed and Broken: Prompt Fuzzing Finds LLMs Still Fragile.* Palo Alto Networks.
3. Nanda, N. (2022). *TransformerLens: A Library for Mechanistic Interpretability.*
4. Sundararajan, M., et al. (2017). *Axiomatic Attribution for Deep Networks.* ICML.
5. Wei, A., et al. (2024). *Jailbroken: How Trained Aligned Models Are Still Flawed.*
