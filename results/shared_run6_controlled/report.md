# Cross-category XAI analysis report

## cybersecurity (n=2)
Top-5 causal layers: [21, 22, 38, 36, 32]
| Layer | Mean causal effect |
|-------|-------------------|
| 21 | +4.8125 |
| 22 | +4.7812 |
| 38 | +4.6875 |
| 36 | +4.6875 |
| 32 | +4.6875 |

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

## malware (n=2)
Top-5 causal layers: [19, 21, 17, 18, 20]
| Layer | Mean causal effect |
|-------|-------------------|
| 19 | +7.5938 |
| 21 | +7.5625 |
| 17 | +7.5312 |
| 18 | +7.4688 |
| 20 | +7.4375 |

Mean logit-lens crossover (clean): 0.0
Mean logit-lens crossover (jailbreak): 0.0

## Cross-category top-K overlap

Layers in top-5 of ALL categories: []
Layers in top-5 of SOME categories only:
- Layer 17: ['malware']
- Layer 18: ['malware']
- Layer 19: ['malware']
- Layer 20: ['malware']
- Layer 21: ['cybersecurity', 'malware']
- Layer 22: ['cybersecurity']
- Layer 24: ['illegal']
- Layer 31: ['illegal']
- Layer 32: ['cybersecurity']
- Layer 33: ['illegal']
- Layer 34: ['illegal']
- Layer 35: ['illegal']
- Layer 36: ['cybersecurity']
- Layer 38: ['cybersecurity']