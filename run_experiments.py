"""
run_experiments.py
------------------
Runs train.py for multiple seeds sequentially and aggregates CL metrics.

Usage:
    python run_experiments.py                    # seeds 42, 123, 456
    python run_experiments.py --seeds 42 7 99   # custom seeds

After all seeds complete, calls cl_metrics.py to print mean ± std.
Results are written to eval_outputs/multi_seed_report.txt.
"""

import argparse
import os
import subprocess
import sys


DEFAULT_SEEDS = [42, 123, 456]


def main():
    parser = argparse.ArgumentParser(description="Multi-seed experiment runner")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        metavar="SEED", help="Seeds to run (default: 42 123 456)")
    args = parser.parse_args()

    result_paths = []
    for seed in args.seeds:
        print(f"\n{'#'*60}")
        print(f"#  TRAINING  seed={seed}")
        print(f"{'#'*60}\n")
        ret = subprocess.run(
            [sys.executable, "train.py", "--seed", str(seed)],
            check=False
        )
        if ret.returncode != 0:
            print(f"  ERROR: train.py --seed {seed} exited with code {ret.returncode}. Stopping.")
            sys.exit(ret.returncode)

        path = os.path.join("eval_outputs", f"cl_metrics_seed{seed}.json")
        if os.path.exists(path):
            result_paths.append(path)
        else:
            print(f"  WARNING: expected output {path} not found after training seed={seed}.")

    if not result_paths:
        print("No result files found — cannot aggregate.")
        sys.exit(1)

    print(f"\n{'#'*60}")
    print(f"#  AGGREGATING  ({len(result_paths)} seeds)")
    print(f"{'#'*60}\n")
    subprocess.run(
        [sys.executable, "cl_metrics.py", "--results"] + result_paths,
        check=True
    )


if __name__ == "__main__":
    main()
