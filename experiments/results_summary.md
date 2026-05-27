# Zeroshot Baseline Multimodal Retrieval

A lightweight benchmark for answering one uncomfortable but necessary question:

```text
Do we really need to train a model, or are simple retrieval baselines already strong?
```

This repository evaluates **no-training multimodal retrieval baselines** on controlled synthetic paired data. It is designed as a pre-training checkpoint: before building contrastive models, transformers, or alignment networks, first measure what simple similarity methods can already do.

---

## The Point of This Repo

Many retrieval projects start with model training too early.

This repo takes the opposite route:

```text
Step 1: create paired multimodal data
Step 2: do not train anything
Step 3: test simple similarity baselines
Step 4: use the results as a reality check
```

If a trained model cannot beat these baselines, the model is not adding enough value.

---

## Baseline Checklist

This project compares six retrieval baselines:

| Baseline | Why it is included |
|---|---|
| Random | Checks what chance-level retrieval looks like |
| Raw feature | Tests whether the original feature space is already useful |
| Metadata only | Tests whether metadata alone can retrieve pairs |
| PCA projected | Tests whether simple compression improves similarity search |
| Random projection | Tests whether approximate geometry is enough |
| Oracle latent | Gives an upper-bound reference using hidden semantic vectors |

The oracle latent baseline is not a real deployment method. It exists only to show the best possible retrieval if the shared hidden structure were directly visible.

---

## Dataset Stress Tests

The benchmark uses three controlled retrieval conditions:

```text
aligned  -> modalities are already close
shifted  -> one modality is transformed away
noisy    -> feature and metadata signals are weakened
```

And three candidate-pool sizes:

```text
6000
12000
18000
```

Full benchmark size:

```text
3 conditions × 3 sample sizes × 6 baselines = 54 runs
```

---

## What Makes This Different

This is not an architecture repo.

This is not a loss-function repo.

This is not a pair-construction repo.

This is a **baseline audit repo**.

Its job is to answer:

```text
What can we get for free before training anything?
```

That makes it useful before starting heavier multimodal experiments.

---

## Files That Matter

```text
src/make_demo_data.py        -> creates synthetic paired multimodal data
src/evaluate_baselines.py    -> evaluates one zero-shot baseline
src/run_benchmark.py         -> runs the full 54-run benchmark
src/collect_results.py       -> collects all metrics into one table
src/metrics.py               -> retrieval metric utilities
```

Results:

```text
experiments/results_table.csv
experiments/results_summary.md
```

---

## Quick Run

Install dependencies:

```powershell
pip install -r requirements.txt
```

Generate one dataset:

```powershell
python src/make_demo_data.py --mode shifted --n-samples 6000 --n-groups 60
```

Evaluate one baseline:

```powershell
python src/evaluate_baselines.py --baseline pca_projected
```

Run everything:

```powershell
python src/run_benchmark.py
python src/collect_results.py
```

---

## Reading the Output

The important metrics are:

| Metric | Meaning |
|---|---|
| Recall@1 | Correct pair ranked first |
| Recall@10 | Correct pair found in top 10 |
| Recall@50 | Correct pair found in top 50 |
| Lift@K | Improvement over random retrieval |
| Positive similarity | Similarity assigned to true pairs |

The full interpretation is here:

- [Result summary](experiments/results_summary.md)
- [Raw result table](experiments/results_table.csv)

---

## Main Lesson

The benchmark shows a simple pattern:

```text
aligned data  -> raw feature and PCA baselines are already very strong
shifted data  -> PCA becomes the strongest practical baseline
noisy data    -> all simple baselines degrade, leaving room for learned alignment
```

So the practical rule is:

```text
Do not celebrate a trained retrieval model until it beats strong no-training baselines.
```

---

## Related Work

This repository is motivated by zero-shot and no-training retrieval evaluation.

The closest conceptual reference is CLIP, which showed that multimodal representations can support zero-shot transfer without task-specific training. This repo does not use CLIP directly, but follows the same evaluation mindset: test what is possible before additional task-specific training.

The PCA baseline is motivated by classical dimensionality reduction. The random projection baseline is motivated by the Johnson-Lindenstrauss idea that random projections can approximately preserve distances.

---

## References

- Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. *Learning Transferable Visual Models From Natural Language Supervision*. ICML, 2021.
- Harold Hotelling. *Analysis of a Complex of Statistical Variables into Principal Components*. Journal of Educational Psychology, 1933.
- Ian T. Jolliffe. *Principal Component Analysis*. Springer, 2002.
- William B. Johnson and Joram Lindenstrauss. *Extensions of Lipschitz mappings into a Hilbert space*. Contemporary Mathematics, 1984.

---

## Scope

This benchmark is intentionally simple.

It does not claim that PCA, random projection, metadata similarity, or raw feature similarity are universally strong retrieval methods. It only shows how these baselines behave under controlled aligned, shifted, and noisy conditions.

The purpose is to create a clean baseline floor before comparing against trained multimodal retrieval models.