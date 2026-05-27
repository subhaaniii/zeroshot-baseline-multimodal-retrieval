# Zero-Shot Multimodal Retrieval Baseline Leaderboard

## Overview

This benchmark evaluates no-training retrieval baselines for paired multimodal data.

The goal is simple:

> Before training a contrastive model, how far can simple similarity baselines go?

Unlike the other repositories in this portfolio, this project does not train a neural retrieval model. It compares retrieval performance using fixed feature spaces and lightweight transformations.

The benchmark is designed as a baseline leaderboard for multimodal retrieval.

---

## Benchmark Setup

The synthetic dataset contains paired modality-A and modality-B samples.

Each sample includes:

| Component | Description |
|---|---|
| Modality A features | Synthetic feature vector for the query side |
| Modality B features | Synthetic feature vector for the candidate side |
| Metadata | Age, severity, binary conditions, and sex |
| Latent vector | Ground-truth hidden semantic vector |
| True pair ID | Correct A-B match |

Three retrieval conditions are tested:

| Condition | Meaning |
|---|---|
| Aligned | A and B feature spaces are relatively well aligned |
| Shifted | B is transformed away from A, making raw feature matching harder |
| Noisy | Both feature and metadata signals are substantially noisier |

Three sample sizes are tested:

| Sample size | Purpose |
|---:|---|
| 6000 | Small benchmark setting |
| 12000 | Medium benchmark setting |
| 18000 | Larger benchmark setting |

Each run evaluates retrieval over the full candidate pool for that sample size.

---

## Baselines Compared

| Baseline | Description |
|---|---|
| Random | Random similarity scores; lower-bound sanity check |
| Raw feature | Cosine similarity between raw modality features |
| Metadata only | Cosine similarity using metadata features only |
| PCA projected | PCA projection before cosine similarity |
| Random projection | Random linear projection before cosine similarity |
| Oracle latent | Similarity using the hidden latent vectors; upper-bound reference |

The oracle latent baseline is not a deployable method. It is included only to show the best-case retrieval behavior when the shared semantic space is directly available.

---

## Metrics

| Metric | Meaning |
|---|---|
| Recall@1 | Correct match is ranked first |
| Recall@5 | Correct match appears in the top 5 |
| Recall@10 | Correct match appears in the top 10 |
| Recall@50 | Correct match appears in the top 50 |
| Lift@K | Improvement over random retrieval |
| Positive similarity | Similarity of true pairs in the evaluated feature space |

The main comparison metric in this summary is Recall@50, because it shows whether a baseline can recover a useful candidate neighborhood even when exact top-1 retrieval is difficult.

---

## Aligned Setting

In the aligned setting, the raw modality features are already close enough for strong retrieval. PCA projection also performs very well, and random projection retains a surprisingly large amount of retrieval structure.

Metadata-only retrieval is useful but no longer acts as an oracle after separating modality-specific metadata signals.

### Aligned results

| Sample size | Baseline | R@1 | R@10 | R@50 | Lift@50 | Pos Sim |
|---:|---|---:|---:|---:|---:|---:|
| 6000 | Random | 0.0000 | 0.0023 | 0.0083 | 1.00x | 0.0010 |
| 6000 | Raw feature | 0.9600 | 0.9995 | 1.0000 | 120.00x | 0.8237 |
| 6000 | Metadata only | 0.0292 | 0.1862 | 0.4022 | 48.26x | 0.7278 |
| 6000 | PCA projected | 0.9870 | 1.0000 | 1.0000 | 120.00x | 0.9038 |
| 6000 | Random projection | 0.5927 | 0.9223 | 0.9923 | 119.08x | 0.8197 |
| 6000 | Oracle latent | 1.0000 | 1.0000 | 1.0000 | 120.00x | 1.0000 |
| 12000 | Random | 0.0002 | 0.0008 | 0.0040 | 0.96x | 0.0046 |
| 12000 | Raw feature | 0.9388 | 0.9996 | 1.0000 | 240.00x | 0.8182 |
| 12000 | Metadata only | 0.0153 | 0.1153 | 0.3005 | 72.12x | 0.7141 |
| 12000 | PCA projected | 0.9810 | 1.0000 | 1.0000 | 240.00x | 0.9001 |
| 12000 | Random projection | 0.4970 | 0.8696 | 0.9801 | 235.22x | 0.8068 |
| 12000 | Oracle latent | 1.0000 | 1.0000 | 1.0000 | 240.00x | 1.0000 |
| 18000 | Random | 0.0001 | 0.0007 | 0.0030 | 1.08x | -0.0039 |
| 18000 | Raw feature | 0.9500 | 0.9994 | 1.0000 | 360.00x | 0.8267 |
| 18000 | Metadata only | 0.0122 | 0.0856 | 0.2497 | 89.90x | 0.7169 |
| 18000 | PCA projected | 0.9860 | 0.9999 | 1.0000 | 360.00x | 0.9058 |
| 18000 | Random projection | 0.5354 | 0.8870 | 0.9816 | 353.38x | 0.8254 |
| 18000 | Oracle latent | 1.0000 | 1.0000 | 1.0000 | 360.00x | 1.0000 |

### Aligned setting takeaway

In aligned conditions, simple feature-space baselines are already strong. PCA projection slightly improves or matches raw feature retrieval. Random projection preserves enough geometry to remain competitive at Recall@50, though it is weaker at Recall@1.

Metadata-only retrieval is useful but much weaker than feature-based retrieval.

---

## Shifted Setting

The shifted setting tests whether baselines remain useful when modality B is transformed away from modality A.

This is where the benchmark becomes more interesting. Raw feature similarity drops substantially, but PCA projection becomes the strongest practical no-training baseline.

### Shifted results

| Sample size | Baseline | R@1 | R@10 | R@50 | Lift@50 | Pos Sim |
|---:|---|---:|---:|---:|---:|---:|
| 6000 | Random | 0.0000 | 0.0023 | 0.0083 | 1.00x | 0.0010 |
| 6000 | Raw feature | 0.2078 | 0.5580 | 0.8193 | 98.32x | 0.4386 |
| 6000 | Metadata only | 0.0030 | 0.0250 | 0.0753 | 9.04x | 0.3521 |
| 6000 | PCA projected | 0.3355 | 0.7250 | 0.9138 | 109.66x | 0.6010 |
| 6000 | Random projection | 0.0348 | 0.1623 | 0.3508 | 42.10x | 0.4352 |
| 6000 | Oracle latent | 1.0000 | 1.0000 | 1.0000 | 120.00x | 1.0000 |
| 12000 | Random | 0.0002 | 0.0008 | 0.0040 | 0.96x | 0.0046 |
| 12000 | Raw feature | 0.1456 | 0.4292 | 0.6866 | 164.78x | 0.4239 |
| 12000 | Metadata only | 0.0019 | 0.0140 | 0.0462 | 11.08x | 0.3283 |
| 12000 | PCA projected | 0.2461 | 0.5967 | 0.8266 | 198.38x | 0.5862 |
| 12000 | Random projection | 0.0201 | 0.0954 | 0.2333 | 56.00x | 0.4135 |
| 12000 | Oracle latent | 1.0000 | 1.0000 | 1.0000 | 240.00x | 1.0000 |
| 18000 | Random | 0.0001 | 0.0007 | 0.0030 | 1.08x | -0.0039 |
| 18000 | Raw feature | 0.1468 | 0.4274 | 0.6790 | 244.44x | 0.4369 |
| 18000 | Metadata only | 0.0008 | 0.0108 | 0.0366 | 13.16x | 0.3429 |
| 18000 | PCA projected | 0.2521 | 0.6029 | 0.8256 | 297.22x | 0.6004 |
| 18000 | Random projection | 0.0182 | 0.0868 | 0.2067 | 74.40x | 0.4335 |
| 18000 | Oracle latent | 1.0000 | 1.0000 | 1.0000 | 360.00x | 1.0000 |

### Shifted setting takeaway

PCA projection is the best practical no-training baseline under cross-modal shift.

Raw feature similarity still retains useful retrieval signal, but PCA gives stronger Recall@1, Recall@10, and Recall@50. Random projection performs better than random but loses much of the retrieval structure.

Metadata-only retrieval becomes weak, showing that metadata is not enough once modality features are shifted.

---

## Noisy Setting

The noisy setting is the hardest condition. Both feature noise and metadata noise increase.

Here, all practical baselines degrade heavily. PCA remains the strongest no-training baseline at Recall@50, but absolute retrieval performance is low.

### Noisy results

| Sample size | Baseline | R@1 | R@10 | R@50 | Lift@50 | Pos Sim |
|---:|---|---:|---:|---:|---:|---:|
| 6000 | Random | 0.0000 | 0.0023 | 0.0083 | 1.00x | 0.0010 |
| 6000 | Raw feature | 0.0048 | 0.0238 | 0.0793 | 9.52x | 0.1348 |
| 6000 | Metadata only | 0.0015 | 0.0090 | 0.0327 | 3.92x | 0.1923 |
| 6000 | PCA projected | 0.0050 | 0.0375 | 0.1195 | 14.34x | 0.2289 |
| 6000 | Random projection | 0.0017 | 0.0098 | 0.0350 | 4.20x | 0.1360 |
| 6000 | Oracle latent | 1.0000 | 1.0000 | 1.0000 | 120.00x | 1.0000 |
| 12000 | Random | 0.0002 | 0.0008 | 0.0040 | 0.96x | 0.0046 |
| 12000 | Raw feature | 0.0018 | 0.0127 | 0.0443 | 10.62x | 0.1247 |
| 12000 | Metadata only | 0.0006 | 0.0056 | 0.0208 | 4.98x | 0.1866 |
| 12000 | PCA projected | 0.0033 | 0.0218 | 0.0710 | 17.04x | 0.2146 |
| 12000 | Random projection | 0.0002 | 0.0031 | 0.0176 | 4.22x | 0.1241 |
| 12000 | Oracle latent | 1.0000 | 1.0000 | 1.0000 | 240.00x | 1.0000 |
| 18000 | Random | 0.0001 | 0.0007 | 0.0030 | 1.08x | -0.0039 |
| 18000 | Raw feature | 0.0013 | 0.0094 | 0.0338 | 12.18x | 0.1310 |
| 18000 | Metadata only | 0.0003 | 0.0039 | 0.0158 | 5.68x | 0.1980 |
| 18000 | PCA projected | 0.0025 | 0.0177 | 0.0566 | 20.38x | 0.2250 |
| 18000 | Random projection | 0.0007 | 0.0038 | 0.0141 | 5.08x | 0.1295 |
| 18000 | Oracle latent | 1.0000 | 1.0000 | 1.0000 | 360.00x | 1.0000 |

### Noisy setting takeaway

In noisy conditions, no practical baseline is strong enough for reliable exact retrieval.

PCA projection remains the best practical no-training baseline, but the large gap between PCA and oracle latent shows that the shared semantic structure is still present but not easily accessible from simple observed feature similarity.

This is the main reason a trained contrastive model may still be needed after baseline evaluation.

---

## Practical Leaderboard

Across the practical no-training methods, the general ranking is:

```text
PCA projected > raw feature > random projection > metadata only > random
```

This ranking is clearest in the shifted and noisy settings.

In the aligned setting, raw feature similarity is already strong, and PCA mainly acts as a denoising or compression step.

---

## What This Benchmark Shows

### 1. Simple baselines can be surprisingly strong

When the modalities are already aligned, raw feature similarity and PCA projection can nearly solve the retrieval task without training.

This is important because it gives a baseline that any trained model should beat.

### 2. PCA is the strongest practical zero-shot baseline

PCA projection performs especially well under shifted and noisy settings.

It appears to preserve shared structure while reducing some modality-specific noise.

### 3. Metadata-only retrieval is not enough

After fixing modality-specific metadata leakage, metadata-only retrieval behaves realistically.

It helps in aligned data but becomes weak under shifted and noisy conditions.

### 4. Random projection is useful but limited

Random projection preserves some neighborhood structure, especially in aligned data. However, it is clearly weaker than PCA and raw feature matching under harder settings.

### 5. Oracle latent exposes the remaining gap

Oracle latent retrieval is perfect because it uses the hidden semantic vector directly.

The gap between oracle latent and practical baselines shows how much shared structure is still inaccessible without a learned alignment model.

---

## Main Conclusion

Zero-shot baselines are essential before training heavier multimodal retrieval models.

This benchmark shows that:

- aligned data may not need complex training
- shifted data benefits strongly from PCA projection
- noisy data remains difficult for all simple baselines
- metadata-only retrieval can be misleading if metadata is copied across modalities
- oracle latent similarity is useful as an upper-bound reference, not as a deployable method

The main lesson is:

> A trained retrieval model should be compared against strong no-training baselines, not only against random retrieval.

---

## Related Work

This repository is motivated by zero-shot and no-training retrieval evaluation.

The closest conceptual reference is CLIP, introduced by Radford et al. in *Learning Transferable Visual Models From Natural Language Supervision*. CLIP showed that multimodal representations can support zero-shot transfer without task-specific training, making it a natural reference point for evaluating retrieval before training a custom model.

This repository does not use CLIP directly. Instead, it builds a controlled synthetic benchmark to compare no-training retrieval baselines such as raw feature similarity, metadata-only similarity, PCA-projected similarity, random projection, and oracle latent similarity.

The PCA baseline is motivated by classical dimensionality reduction, where high-dimensional features are projected into a lower-dimensional space before similarity comparison. The random projection baseline is related to the Johnson-Lindenstrauss idea that random linear projections can approximately preserve distances in lower-dimensional spaces.

---

## References

- Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. *Learning Transferable Visual Models From Natural Language Supervision*. ICML, 2021.
- Harold Hotelling. *Analysis of a Complex of Statistical Variables into Principal Components*. Journal of Educational Psychology, 1933.
- Ian T. Jolliffe. *Principal Component Analysis*. Springer, 2002.
- William B. Johnson and Joram Lindenstrauss. *Extensions of Lipschitz mappings into a Hilbert space*. Contemporary Mathematics, 1984.

---

## Boundary of This Benchmark

This benchmark is a controlled no-training retrieval leaderboard.

It is not designed to prove that PCA, raw features, or metadata similarity are universally strong retrieval methods. Instead, it shows how simple baselines behave under known aligned, shifted, and noisy conditions.

The synthetic setup makes it possible to compare practical baselines against both a random lower bound and an oracle latent upper bound.

A natural next step would be to compare these no-training baselines against trained contrastive models, frozen pretrained encoders, and cross-modal calibration methods on authorized real paired datasets.