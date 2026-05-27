# Zero-Shot Metric Guide

This document explains the retrieval metrics used in the benchmark.

## Recall@K

Recall@K measures whether the correct candidate appears within the top K retrieved results.

For each modality-A query, the system ranks all modality-B candidates.

Examples:

| Metric | Meaning |
|---|---|
| Recall@1 | The correct candidate is ranked first |
| Recall@5 | The correct candidate appears in the top 5 |
| Recall@10 | The correct candidate appears in the top 10 |
| Recall@50 | The correct candidate appears in the top 50 |

Higher Recall@K means stronger retrieval.

## Lift@K

Lift@K compares retrieval performance against random retrieval.

If there are 6000 candidates, random Recall@50 is:

```text
50 / 6000 = 0.00833
```

If a baseline gets Recall@50 = 0.50, then:

```text
Lift@50 = 0.50 / 0.00833 = 60x
```

This means the baseline is 60 times better than random retrieval at K=50.

## Positive-pair similarity

Positive-pair similarity is the average similarity score assigned to true A-B pairs.

A higher value means the baseline gives true pairs stronger similarity.

However, positive similarity alone is not enough. A method can assign high similarity to true pairs but still fail to rank them above other candidates.

For that reason, positive similarity should be interpreted together with Recall@K.

## Random baseline

The random baseline is a lower-bound sanity check.

It confirms what retrieval looks like without useful signal.

A meaningful baseline should perform above random retrieval.

## Oracle latent baseline

The oracle latent baseline uses the hidden semantic vectors from the synthetic data generator.

It is not a deployable method.

It is included to show the upper-bound retrieval performance when the true shared structure is directly available.

## Practical reading rule

Use the metrics like this:

```text
Recall@K tells whether retrieval works.
Lift@K tells how much better it is than random.
Positive similarity tells whether true pairs are close in the evaluated space.
Oracle latent shows the upper-bound gap.
```