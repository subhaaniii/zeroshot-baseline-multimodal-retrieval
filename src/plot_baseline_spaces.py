from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


MODES = ["aligned", "shifted", "noisy"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create PCA visualizations for zero-shot multimodal baseline feature spaces."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data_demo"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--max-points", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--title", type=str, default="Zero-shot baseline feature space")
    return parser.parse_args()


def resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def feature_columns(df: pd.DataFrame, possible_prefixes: list[str]) -> list[str]:
    for prefix in possible_prefixes:
        cols = sorted([c for c in df.columns if c.startswith(prefix)])
        if cols:
            return cols

    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and c not in {"sample_id", "pair_id", "group_id", "a_id", "b_id"}
    ]

    if numeric_cols:
        return numeric_cols

    raise RuntimeError(f"No feature columns found. Tried prefixes: {possible_prefixes}")


def sample_indices(n: int, max_points: int, seed: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)

    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_points, replace=False))


def get_group_ids(a_df: pd.DataFrame, pairs_df: pd.DataFrame | None, idx: np.ndarray) -> np.ndarray:
    if "group_id" in a_df.columns:
        return a_df.iloc[idx]["group_id"].astype(int).to_numpy()

    if pairs_df is not None and "group_id" in pairs_df.columns:
        return pairs_df.iloc[idx]["group_id"].astype(int).to_numpy()

    return np.arange(len(idx)) % 20


def load_mode_from_metadata(data_dir: Path) -> str:
    metadata_path = data_dir / "demo_metadata.json"
    if not metadata_path.exists():
        return "demo"

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return str(metadata.get("mode", "demo"))


def combined_pca(a_x: np.ndarray, b_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    combined = np.vstack([a_x, b_x])
    z = PCA(n_components=2, random_state=42).fit_transform(combined)
    return z[: len(a_x)], z[len(a_x):]


def plot_feature_space(
    a_df: pd.DataFrame,
    b_df: pd.DataFrame,
    pairs_df: pd.DataFrame | None,
    mode_name: str,
    out_path: Path,
    max_points: int,
    seed: int,
) -> None:
    n = min(len(a_df), len(b_df))
    idx = sample_indices(n=n, max_points=max_points, seed=seed)

    a_plot = a_df.iloc[idx].reset_index(drop=True)
    b_plot = b_df.iloc[idx].reset_index(drop=True)

    a_cols = feature_columns(a_plot, ["a_feat_", "feat_", "modality_a_", "x_a_", "a_"])
    b_cols = feature_columns(b_plot, ["b_feat_", "feat_", "modality_b_", "x_b_", "b_"])

    a_x = a_plot[a_cols].astype(np.float32).to_numpy()
    b_x = b_plot[b_cols].astype(np.float32).to_numpy()

    group_ids = get_group_ids(a_df, pairs_df, idx)

    a_z, b_z = combined_pca(a_x, b_x)

    plt.figure(figsize=(7.5, 6))

    plt.scatter(
        a_z[:, 0],
        a_z[:, 1],
        c=group_ids,
        cmap="tab20",
        s=9,
        alpha=0.70,
        marker="o",
        linewidths=0,
        label="Modality A",
    )

    plt.scatter(
        b_z[:, 0],
        b_z[:, 1],
        c=group_ids,
        cmap="tab20",
        s=10,
        alpha=0.70,
        marker="x",
        linewidths=0.5,
        label="Modality B",
    )

    plt.title(f"{mode_name.title()} modality feature space")
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    data_dir = resolve_path(args.data_dir, repo_root)
    figures_dir = resolve_path(args.figures_dir, repo_root)
    figures_dir.mkdir(parents=True, exist_ok=True)

    a_path = data_dir / "modality_a.csv"
    b_path = data_dir / "modality_b.csv"
    pairs_path = data_dir / "true_pairs.csv"

    if not a_path.exists() or not b_path.exists():
        raise FileNotFoundError(
            "Expected data_demo/modality_a.csv and data_demo/modality_b.csv. "
            "Run src/make_demo_data.py first."
        )

    a_df = pd.read_csv(a_path)
    b_df = pd.read_csv(b_path)
    pairs_df = pd.read_csv(pairs_path) if pairs_path.exists() else None

    mode_name = load_mode_from_metadata(data_dir)

    out_path = figures_dir / f"{mode_name}_feature_space_pca2d.png"
    plot_feature_space(
        a_df=a_df,
        b_df=b_df,
        pairs_df=pairs_df,
        mode_name=mode_name,
        out_path=out_path,
        max_points=args.max_points,
        seed=args.seed,
    )

    # Also create a stable README name.
    readme_path = figures_dir / "baseline_feature_space_pca2d.png"
    plot_feature_space(
        a_df=a_df,
        b_df=b_df,
        pairs_df=pairs_df,
        mode_name=mode_name,
        out_path=readme_path,
        max_points=args.max_points,
        seed=args.seed,
    )

    print(f"Wrote {out_path}")
    print(f"Wrote {readme_path}")
    print(f"Detected mode: {mode_name}")


if __name__ == "__main__":
    main()