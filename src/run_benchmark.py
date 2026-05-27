from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


MODES = ["aligned", "shifted", "noisy"]
SIZES = [6000, 12000, 18000]
BASELINES = [
    "random",
    "raw_feature",
    "metadata_only",
    "pca_projected",
    "random_projection",
    "oracle_latent",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full zero-shot retrieval benchmark.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--python", type=str, default=sys.executable)
    return parser.parse_args()


def run_command(cmd: list[str]) -> None:
    print("\n" + " ".join(cmd))
    subprocess.run(cmd, check=True)


def groups_for_size(n_samples: int) -> int:
    if n_samples == 6000:
        return 60
    if n_samples == 12000:
        return 90
    if n_samples == 18000:
        return 120
    return max(10, n_samples // 150)


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    for mode in MODES:
        for n_samples in SIZES:
            n_groups = groups_for_size(n_samples)

            run_command([
                args.python,
                str(repo_root / "src" / "make_demo_data.py"),
                "--mode", mode,
                "--n-samples", str(n_samples),
                "--n-groups", str(n_groups),
                "--seed", str(args.seed),
            ])

            for baseline in BASELINES:
                out_dir = repo_root / "outputs" / f"{mode}_{n_samples}_{baseline}"

                run_command([
                    args.python,
                    str(repo_root / "src" / "evaluate_baselines.py"),
                    "--baseline", baseline,
                    "--seed", str(args.seed),
                    "--out-dir", str(out_dir),
                ])

    print("\nFull benchmark complete.")
    print("Now run: python src/collect_results.py")


if __name__ == "__main__":
    main()