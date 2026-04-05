# Running on La Ruche (HPC)

## First-time setup

```bash
ssh drouilheel@ruche.mesocentre.universite-paris-saclay.fr

cd $WORKDIR
git clone https://github.com/helloelora/jailbreak-interpretability-slm.git
cd jailbreak-interpretability-slm
bash ruche/setup_ruche.sh
```

This builds the Apptainer container (~15-20 min).

## Submit jobs

```bash
# Run the prompt fuzzer
sbatch ruche/run_fuzzer.sh

# Re-annotate results with HarmBench classifier
sbatch ruche/run_reannotate.sh $WORKDIR/jailbreak_xai_runs/results/fuzz_<JOB_ID>
```

## Monitor

```bash
squeue -u drouilheel                        # Job status
tail -f jailbreak-fuzz.o<JOB_ID>            # Live output
seff <JOB_ID>                               # Efficiency after completion
scancel <JOB_ID>                            # Cancel
ruche-quota                                 # Check disk quota
```

## Retrieve results

From your local machine:
```bash
scp -r drouilheel@ruche.mesocentre.universite-paris-saclay.fr:$WORKDIR/jailbreak_xai_runs/results/ ./results/
```

## Key paths on La Ruche

| Path | Description |
|------|-------------|
| `$WORKDIR/jailbreak-interpretability-slm/` | Project code |
| `$WORKDIR/jailbreak_xai.sif` | Apptainer container |
| `$WORKDIR/jailbreak_xai_runs/results/` | Output directory |

## GPU partition

- **gpua100**: NVIDIA A100, 40GB VRAM — Mistral Small 3.1 24B in 4-bit (~14GB)
- Max wall time: ~24h
- 8 CPUs × 12GB RAM = 96GB system RAM
