from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from metrics import cosine_similarity_matrix, retrieval_metrics_from_similarity


METADATA_COLUMNS = [
    "age",
    "severity",
    "condition_a",
    "condition_b",
    "condition_c",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate zero-shot multimodal retrieval baselines without training."
    )

    parser.add_argument("--a-csv", type=Path, default=Path("data_demo/modality_a.csv"))
    parser.add_argument("--b-csv", type=Path, default=Path("data_demo/modality_b.csv"))
    parser.add_argument("--true-pairs-csv", type=Path, default=Path("data_demo/true_pairs.csv"))
    parser.add_argument("--metadata-json", type=Path, default=Path("data_demo/demo_metadata.json"))

    parser.add_argument(
        "--baseline",
        choices=[
            "random",
            "raw_feature",
            "metadata_only",
            "pca_projected",
            "random_projection",
            "oracle_latent",
        ],
        default="raw_feature",
    )

    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--random-proj-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=None)

    return parser.parse_args()


def resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def prepare_metadata_features(a_df: pd.DataFrame, b_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    a_meta = a_df[METADATA_COLUMNS + ["sex"]].copy()
    b_meta = b_df[METADATA_COLUMNS + ["sex"]].copy()

    a_meta = pd.get_dummies(a_meta, columns=["sex"], drop_first=False)
    b_meta = pd.get_dummies(b_meta, columns=["sex"], drop_first=False)

    all_cols = sorted(set(a_meta.columns) | set(b_meta.columns))
    a_meta = a_meta.reindex(columns=all_cols, fill_value=0)
    b_meta = b_meta.reindex(columns=all_cols, fill_value=0)

    scaler = StandardScaler()
    combined = pd.concat([a_meta, b_meta], axis=0)
    scaler.fit(combined)

    return scaler.transform(a_meta), scaler.transform(b_meta)


def get_feature_arrays(a_df: pd.DataFrame, b_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    a_cols = sorted([c for c in a_df.columns if c.startswith("a_feat_")])
    b_cols = sorted([c for c in b_df.columns if c.startswith("b_feat_")])

    if len(a_cols) == 0 or len(a_cols) != len(b_cols):
        raise RuntimeError("Feature column mismatch between modality A and B.")

    return (
        a_df[a_cols].astype(np.float32).to_numpy(),
        b_df[b_cols].astype(np.float32).to_numpy(),
    )


def get_latent_arrays(a_df: pd.DataFrame, b_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    latent_cols = sorted([c for c in a_df.columns if c.startswith("latent_")])

    if len(latent_cols) == 0:
        raise RuntimeError("No latent_* columns found.")

    return (
        a_df[latent_cols].astype(np.float32).to_numpy(),
        b_df[latent_cols].astype(np.float32).to_numpy(),
    )


def baseline_similarity(
    baseline: str,
    a_df: pd.DataFrame,
    b_df: pd.DataFrame,
    seed: int,
    pca_dim: int,
    random_proj_dim: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    if baseline == "random":
        return rng.normal(0.0, 1.0, size=(len(a_df), len(b_df))).astype(np.float32)

    if baseline == "raw_feature":
        a_x, b_x = get_feature_arrays(a_df, b_df)
        return cosine_similarity_matrix(a_x, b_x)

    if baseline == "metadata_only":
        a_x, b_x = prepare_metadata_features(a_df, b_df)
        return cosine_similarity_matrix(a_x, b_x)

    if baseline == "pca_projected":
        a_x, b_x = get_feature_arrays(a_df, b_df)
        combined = np.vstack([a_x, b_x])
        dim = min(pca_dim, combined.shape[1])

        pca = PCA(n_components=dim, random_state=seed)
        pca.fit(combined)

        a_pca = pca.transform(a_x)
        b_pca = pca.transform(b_x)
        return cosine_similarity_matrix(a_pca, b_pca)

    if baseline == "random_projection":
        a_x, b_x = get_feature_arrays(a_df, b_df)

        dim = min(random_proj_dim, a_x.shape[1])
        proj = rng.normal(0.0, 1.0, size=(a_x.shape[1], dim)).astype(np.float32)
        proj = proj / np.sqrt(a_x.shape[1])

        a_proj = a_x @ proj
        b_proj = b_x @ proj
        return cosine_similarity_matrix(a_proj, b_proj)

    if baseline == "oracle_latent":
        a_latent, b_latent = get_latent_arrays(a_df, b_df)
        return cosine_similarity_matrix(a_latent, b_latent)

    raise ValueError(f"Unknown baseline: {baseline}")


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    a_path = resolve_path(args.a_csv, repo_root)
    b_path = resolve_path(args.b_csv, repo_root)
    true_pairs_path = resolve_path(args.true_pairs_csv, repo_root)
    metadata_path = resolve_path(args.metadata_json, repo_root)

    a_df = pd.read_csv(a_path)
    b_df = pd.read_csv(b_path)
    true_pairs = pd.read_csv(true_pairs_path)

    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    mode = metadata.get("mode", "unknown")
    n_samples = metadata.get("n_samples", len(a_df))

    sim = baseline_similarity(
        baseline=args.baseline,
        a_df=a_df,
        b_df=b_df,
        seed=args.seed,
        pca_dim=args.pca_dim,
        random_proj_dim=args.random_proj_dim,
    )

    query_ids = a_df["a_id"].astype(int).to_numpy()
    candidate_ids = b_df["b_id"].astype(int).to_numpy()

    true_b_for_a = {
        int(row["a_id"]): int(row["b_id"])
        for _, row in true_pairs.iterrows()
    }

    metrics = retrieval_metrics_from_similarity(
        sim=sim,
        query_ids=query_ids,
        candidate_ids=candidate_ids,
        true_candidate_for_query=true_b_for_a,
        k_values=(1, 5, 10, 50),
    )

    row = {
        "mode": mode,
        "n_samples": int(n_samples),
        "baseline": args.baseline,
        "pca_dim": args.pca_dim if args.baseline == "pca_projected" else "",
        "random_proj_dim": args.random_proj_dim if args.baseline == "random_projection" else "",
        **metrics,
    }

    out_dir = args.out_dir or (repo_root / "outputs" / f"{mode}_{n_samples}_{args.baseline}")
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([row]).to_csv(out_dir / "metrics.csv", index=False)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "baseline": args.baseline,
                "mode": mode,
                "n_samples": int(n_samples),
                "seed": args.seed,
                "pca_dim": args.pca_dim,
                "random_proj_dim": args.random_proj_dim,
            },
            f,
            indent=2,
        )

    print(f"Mode       : {mode}")
    print(f"Samples    : {n_samples}")
    print(f"Baseline   : {args.baseline}")
    print(f"R@1        : {metrics['recall@1']:.4f}")
    print(f"R@10       : {metrics['recall@10']:.4f}")
    print(f"R@50       : {metrics['recall@50']:.4f}")
    print(f"Lift@50    : {metrics['lift@50']:.2f}x")
    print(f"Pos sim    : {metrics['pos_sim_mean']:.4f}")
    print(f"Metrics    : {out_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()