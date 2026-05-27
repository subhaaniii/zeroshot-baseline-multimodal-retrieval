# Baseline Validation Notes

This project evaluates non-trained retrieval reference baselines before introducing learned multimodal alignment.

The purpose is not to avoid modeling. The purpose is to establish a reliable reference floor.

A trained retrieval model should not only outperform random retrieval. It should also outperform strong non-trained references such as raw feature similarity, PCA-projected similarity, metadata-only similarity, and random-projection similarity.

## Why baseline validation matters

In multimodal retrieval, a model can look impressive if it is compared only against random chance.

That comparison is incomplete.

Some datasets already contain strong retrieval signal in their raw features. In those cases, a trained model must demonstrate improvement over that existing signal.

Other datasets may be so noisy that all non-trained baselines fail. In that case, the benchmark justifies the need for learned alignment.

This repository helps identify which situation we are in.

## Baselines used

| Baseline | Purpose |
|---|---|
| Random | Lower-bound sanity check |
| Raw feature | Tests whether the original representation space is already useful |
| Metadata only | Tests whether metadata alone carries retrieval signal |
| PCA projected | Tests whether compression improves retrieval |
| Random projection | Tests whether approximate geometry is enough |
| Oracle latent | Upper-bound reference using hidden semantic structure |

## Important interpretation

The oracle latent baseline is not deployable. It exists only to show the maximum possible retrieval performance when the shared semantic structure is directly visible.

The practical comparison is among:

- raw feature
- metadata only
- PCA projected
- random projection

A future trained model should be evaluated against the strongest of these practical baselines.

## Main validation question

The benchmark asks:

```text
Does a trained model add value beyond the retrieval signal already present in the data?