from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect zero-shot retrieval benchmark results.")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--out-csv", type=Path, default=Path("experiments/results_table.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = []

    for metrics_path in sorted(args.outputs_dir.glob("*/metrics.csv")):
        df = pd.read_csv(metrics_path)

        if df.empty:
            continue

        row = df.iloc[-1].to_dict()
        row["run_name"] = metrics_path.parent.name
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No metrics.csv files found under {args.outputs_dir}")

    out_df = pd.DataFrame(rows)

    preferred_cols = [
        "run_name",
        "mode",
        "n_samples",
        "baseline",
        "pca_dim",
        "random_proj_dim",
        "recall@1",
        "recall@5",
        "recall@10",
        "recall@50",
        "lift@1",
        "lift@5",
        "lift@10",
        "lift@50",
        "pos_sim_mean",
        "n_pool",
    ]

    cols = [c for c in preferred_cols if c in out_df.columns]
    other_cols = [c for c in out_df.columns if c not in cols]
    out_df = out_df[cols + other_cols]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    print(f"Wrote {len(out_df)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()