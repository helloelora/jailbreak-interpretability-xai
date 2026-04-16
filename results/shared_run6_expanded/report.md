# Cross-category XAI analysis report

## cybersecurity (n=5)
Top-5 causal layers: [20, 21, 19, 18, 22]
| Layer | Mean causal effect |
|-------|-------------------|
| 20 | +4.7875 |
| 21 | +4.7875 |
| 19 | +4.7750 |
| 18 | +4.6250 |
| 22 | +4.4875 |

Mean logit-lens crossover (clean): 0.0
Mean logit-lens crossover (jailbreak): 0.0

## illegal (n=5)
Top-5 causal layers: [33, 34, 25, 31, 26]
| Layer | Mean causal effect |
|-------|-------------------|
| 33 | +6.5000 |
| 34 | +6.5000 |
| 25 | +6.5000 |
| 31 | +6.5000 |
| 26 | +6.4875 |

Mean logit-lens crossover (clean): 0.0
Mean logit-lens crossover (jailbreak): 0.0

## malware (n=5)
Top-5 causal layers: [21, 20, 22, 24, 23]
| Layer | Mean causal effect |
|-------|-------------------|
| 21 | +8.2750 |
| 20 | +8.1125 |
| 22 | +7.9875 |
| 24 | +7.9250 |
| 23 | +7.8625 |

Mean logit-lens crossover (clean): 0.0
Mean logit-lens crossover (jailbreak): 0.0

## Cross-category top-K overlap

Layers in top-5 of ALL categories: []
Layers in top-5 of SOME categories only:
- Layer 18: ['cybersecurity']
- Layer 19: ['cybersecurity']
- Layer 20: ['cybersecurity', 'malware']
- Layer 21: ['cybersecurity', 'malware']
- Layer 22: ['cybersecurity', 'malware']
- Layer 23: ['malware']
- Layer 24: ['malware']
- Layer 25: ['illegal']
- Layer 26: ['illegal']
- Layer 31: ['illegal']
- Layer 33: ['illegal']
- Layer 34: ['illegal']