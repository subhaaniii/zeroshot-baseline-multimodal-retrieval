# Benchmark Protocol

This document describes how to reproduce the zero-shot retrieval benchmark.

The benchmark evaluates non-trained multimodal retrieval reference baselines across controlled synthetic data settings.

## Install dependencies

```powershell
pip install -r requirements.txt
```

## Generate one dataset

Aligned setting:

```powershell
python src/make_demo_data.py --mode aligned --n-samples 6000 --n-groups 60
```

Shifted setting:

```powershell
python src/make_demo_data.py --mode shifted --n-samples 6000 --n-groups 60
```

Noisy setting:

```powershell
python src/make_demo_data.py --mode noisy --n-samples 6000 --n-groups 60
```

## Evaluate one baseline

Example using PCA projection:

```powershell
python src/evaluate_baselines.py --baseline pca_projected
```

Other available baselines:

```text
random
raw_feature
metadata_only
pca_projected
random_projection
oracle_latent
```

## Run the full benchmark

The full benchmark evaluates:

```text
3 conditions × 3 sample sizes × 6 baselines = 54 runs
```

Run:

```powershell
python src/run_benchmark.py
```

Then collect all results:

```powershell
python src/collect_results.py
```

This writes:

```text
experiments/results_table.csv
```

## Sample sizes

The benchmark uses:

| Sample size | Number of groups |
|---:|---:|
| 6000 | 60 |
| 12000 | 90 |
| 18000 | 120 |

## Output files

Each individual run writes a metrics file under:

```text
outputs/
```

The collected benchmark table is stored as:

```text
experiments/results_table.csv
```

The detailed result interpretation is stored as:

```text
experiments/results_summary.md
```

## Git tracking policy

The repository tracks:

- source code
- small demo data
- documentation
- aggregate benchmark results

The repository does not track generated run folders:

```text
outputs/
results/
```

This keeps the repository clean while preserving the reproducible benchmark summary.