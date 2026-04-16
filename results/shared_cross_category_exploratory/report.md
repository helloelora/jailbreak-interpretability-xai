# Cross-category XAI analysis report

## cybersecurity (n=3)
Top-5 causal layers: [38, 39, 37, 36, 32]
| Layer | Mean causal effect |
|-------|-------------------|
| 38 | +4.1875 |
| 39 | +4.1667 |
| 37 | +4.1250 |
| 36 | +4.1250 |
| 32 | +4.1042 |

Mean logit-lens crossover (clean): 0.0
Mean logit-lens crossover (jailbreak): 0.0

## illegal (n=2)
Top-5 causal layers: [24, 34, 35, 33, 31]
| Layer | Mean causal effect |
|-------|-------------------|
| 24 | +6.9375 |
| 34 | +6.9062 |
| 35 | +6.9062 |
| 33 | +6.9062 |
| 31 | +6.9062 |

Mean logit-lens crossover (clean): 0.0
Mean logit-lens crossover (jailbreak): 0.0

## malware (n=3)
Top-5 causal layers: [21, 20, 19, 22, 24]
| Layer | Mean causal effect |
|-------|-------------------|
| 21 | +7.5208 |
| 20 | +7.5000 |
| 19 | +7.1875 |
| 22 | +7.1458 |
| 24 | +7.0417 |

Mean logit-lens crossover (clean): 0.0
Mean logit-lens crossover (jailbreak): 0.0

## Cross-category top-K overlap

Layers in top-5 of ALL categories: []
Layers in top-5 of SOME categories only:
- Layer 19: ['malware']
- Layer 20: ['malware']
- Layer 21: ['malware']
- Layer 22: ['malware']
- Layer 24: ['illegal', 'malware']
- Layer 31: ['illegal']
- Layer 32: ['cybersecurity']
- Layer 33: ['illegal']
- Layer 34: ['illegal']
- Layer 35: ['illegal']
- Layer 36: ['cybersecurity']
- Layer 37: ['cybersecurity']
- Layer 38: ['cybersecurity']
- Layer 39: ['cybersecurity']