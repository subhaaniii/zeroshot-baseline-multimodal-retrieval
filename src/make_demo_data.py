from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic paired multimodal data for zero-shot retrieval baselines."
    )
    parser.add_argument(
        "--mode",
        choices=["aligned", "shifted", "noisy"],
        default="aligned",
        help="Controls how difficult cross-modal retrieval is before training.",
    )
    parser.add_argument("--n-samples", type=int, default=6000)
    parser.add_argument("--n-groups", type=int, default=60)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def make_projection(rng: np.random.Generator, in_dim: int, out_dim: int) -> np.ndarray:
    mat = rng.normal(0.0, 1.0, size=(in_dim, out_dim))
    return mat / np.sqrt(in_dim)


def mode_settings(mode: str) -> dict[str, float]:
    if mode == "aligned":
        return {
            "latent_noise": 0.12,
            "feature_noise_a": 0.08,
            "feature_noise_b": 0.08,
            "metadata_noise": 0.05,
            "cross_modal_shift": 0.05,
        }

    if mode == "shifted":
        return {
            "latent_noise": 0.18,
            "feature_noise_a": 0.12,
            "feature_noise_b": 0.18,
            "metadata_noise": 0.16,
            "cross_modal_shift": 0.35,
        }

    if mode == "noisy":
        return {
            "latent_noise": 0.35,
            "feature_noise_a": 0.25,
            "feature_noise_b": 0.30,
            "metadata_noise": 0.30,
            "cross_modal_shift": 0.60,
        }

    raise ValueError(f"Unknown mode: {mode}")


def generate_data(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    if args.n_samples < args.n_groups:
        raise ValueError("--n-samples must be >= --n-groups")

    rng = np.random.default_rng(args.seed)
    settings = mode_settings(args.mode)

    sample_ids = np.arange(args.n_samples)

    group_ids = np.arange(args.n_samples) % args.n_groups
    rng.shuffle(group_ids)

    group_centers = rng.normal(0.0, 1.0, size=(args.n_groups, args.latent_dim))
    group_centers = normalize_rows(group_centers)

    latent = (
        group_centers[group_ids]
        + rng.normal(0.0, settings["latent_noise"], size=(args.n_samples, args.latent_dim))
    )
    latent = normalize_rows(latent)

    # Modality-specific projections.
    # In aligned mode, B is close to A.
    # In shifted/noisy mode, B gets a stronger independent transformation.
    proj_a = make_projection(rng, args.latent_dim, args.feature_dim)
    proj_shift = make_projection(rng, args.latent_dim, args.feature_dim)

    proj_b = (
        (1.0 - settings["cross_modal_shift"]) * proj_a
        + settings["cross_modal_shift"] * proj_shift
    )

    a_features = latent @ proj_a + rng.normal(
        0.0,
        settings["feature_noise_a"],
        size=(args.n_samples, args.feature_dim),
    )

    b_features = latent @ proj_b + rng.normal(
        0.0,
        settings["feature_noise_b"],
        size=(args.n_samples, args.feature_dim),
    )

    a_features = normalize_rows(a_features)
    b_features = normalize_rows(b_features)

    # Metadata is correlated with the latent group, but not identical to the latent vector.
    age_base = 35 + 35 * (group_ids / max(args.n_groups - 1, 1))
    age = age_base + rng.normal(0, 4 + 12 * settings["metadata_noise"], args.n_samples)
    age = np.clip(age, 18, 90)

    severity = (
        0.45 * latent[:, 0]
        + 0.35 * latent[:, 1]
        + rng.normal(0, settings["metadata_noise"], args.n_samples)
    )
    severity = (severity - severity.min()) / (severity.max() - severity.min() + 1e-8)

    condition_a = (latent[:, 2] + rng.normal(0, settings["metadata_noise"], args.n_samples) > 0).astype(int)
    condition_b = (latent[:, 3] + rng.normal(0, settings["metadata_noise"], args.n_samples) > 0).astype(int)
    condition_c = (latent[:, 4] + rng.normal(0, settings["metadata_noise"], args.n_samples) > 0).astype(int)

    sex = np.where(
        latent[:, 5] + rng.normal(0, settings["metadata_noise"], args.n_samples) > 0,
        "M",
        "F",
    )
        # Modality-specific metadata observations.
    # A and B share the same latent source, but metadata is not copied exactly.
    age_a = np.clip(
        age + rng.normal(0, 1.5 + 8 * settings["metadata_noise"], args.n_samples),
        18,
        90,
    )
    age_b = np.clip(
        age + rng.normal(0, 1.5 + 8 * settings["metadata_noise"], args.n_samples),
        18,
        90,
    )

    severity_a = np.clip(
        severity + rng.normal(0, settings["metadata_noise"], args.n_samples),
        0,
        1,
    )
    severity_b = np.clip(
        severity + rng.normal(0, settings["metadata_noise"], args.n_samples),
        0,
        1,
    )

    condition_a_mod1 = (
        latent[:, 2] + rng.normal(0, settings["metadata_noise"] * 1.5, args.n_samples) > 0
    ).astype(int)
    condition_a_mod2 = (
        latent[:, 2] + rng.normal(0, settings["metadata_noise"] * 1.5, args.n_samples) > 0
    ).astype(int)

    condition_b_mod1 = (
        latent[:, 3] + rng.normal(0, settings["metadata_noise"] * 1.5, args.n_samples) > 0
    ).astype(int)
    condition_b_mod2 = (
        latent[:, 3] + rng.normal(0, settings["metadata_noise"] * 1.5, args.n_samples) > 0
    ).astype(int)

    condition_c_mod1 = (
        latent[:, 4] + rng.normal(0, settings["metadata_noise"] * 1.5, args.n_samples) > 0
    ).astype(int)
    condition_c_mod2 = (
        latent[:, 4] + rng.normal(0, settings["metadata_noise"] * 1.5, args.n_samples) > 0
    ).astype(int)

    sex_a = np.where(
        latent[:, 5] + rng.normal(0, settings["metadata_noise"] * 1.5, args.n_samples) > 0,
        "M",
        "F",
    )
    sex_b = np.where(
        latent[:, 5] + rng.normal(0, settings["metadata_noise"] * 1.5, args.n_samples) > 0,
        "M",
        "F",
    )

    def make_df(
        prefix: str,
        features: np.ndarray,
        age_values: np.ndarray,
        severity_values: np.ndarray,
        condition_a_values: np.ndarray,
        condition_b_values: np.ndarray,
        condition_c_values: np.ndarray,
        sex_values: np.ndarray,
    ) -> pd.DataFrame:
        rows = []
        for i in range(args.n_samples):
            row = {
                f"{prefix}_id": int(sample_ids[i]),
                "true_pair_id": int(sample_ids[i]),
                "group_id": int(group_ids[i]),
                "age": float(age_values[i]),
                "severity": float(severity_values[i]),
                "condition_a": int(condition_a_values[i]),
                "condition_b": int(condition_b_values[i]),
                "condition_c": int(condition_c_values[i]),
                "sex": str(sex_values[i]),
            }

            for j in range(args.feature_dim):
                row[f"{prefix}_feat_{j:03d}"] = float(features[i, j])

            for j in range(args.latent_dim):
                row[f"latent_{j:03d}"] = float(latent[i, j])

            rows.append(row)

        return pd.DataFrame(rows)

    a_df = make_df(
        "a",
        a_features,
        age_a,
        severity_a,
        condition_a_mod1,
        condition_b_mod1,
        condition_c_mod1,
        sex_a,
    )

    b_df = make_df(
        "b",
        b_features,
        age_b,
        severity_b,
        condition_a_mod2,
        condition_b_mod2,
        condition_c_mod2,
        sex_b,
    )

    

    true_pairs = pd.DataFrame(
        {
            "a_id": sample_ids.astype(int),
            "b_id": sample_ids.astype(int),
            "group_id": group_ids.astype(int),
        }
    )

    metadata = {
        "mode": args.mode,
        "n_samples": args.n_samples,
        "n_groups": args.n_groups,
        "latent_dim": args.latent_dim,
        "feature_dim": args.feature_dim,
        "seed": args.seed,
        "settings": settings,
        "description": (
            "Synthetic paired multimodal data for zero-shot retrieval baseline evaluation."
        ),
    }

    return a_df, b_df, true_pairs, metadata


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data_demo"
    data_dir.mkdir(parents=True, exist_ok=True)

    a_df, b_df, true_pairs, metadata = generate_data(args)

    a_path = data_dir / "modality_a.csv"
    b_path = data_dir / "modality_b.csv"
    true_pairs_path = data_dir / "true_pairs.csv"
    metadata_path = data_dir / "demo_metadata.json"

    a_df.to_csv(a_path, index=False)
    b_df.to_csv(b_path, index=False)
    true_pairs.to_csv(true_pairs_path, index=False)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Mode       : {args.mode}")
    print(f"Samples    : {args.n_samples}")
    print(f"Groups     : {args.n_groups}")
    print(f"Modality A : {a_path}")
    print(f"Modality B : {b_path}")
    print(f"True pairs : {true_pairs_path}")
    print(f"Metadata   : {metadata_path}")


if __name__ == "__main__":
    main()