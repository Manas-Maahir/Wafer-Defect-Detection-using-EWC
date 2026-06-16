"""
cl_metrics.py
-------------
Display and aggregate continual learning metrics produced by train.py.

Single seed:
    python cl_metrics.py --results eval_outputs/cl_metrics_seed42.json

Multiple seeds (reports mean ± std):
    python cl_metrics.py --results eval_outputs/cl_metrics_seed42.json eval_outputs/cl_metrics_seed123.json eval_outputs/cl_metrics_seed456.json

Output is also written to eval_outputs/multi_seed_report.txt when multiple files are provided.
"""

import argparse
import ast
import json
import os
import numpy as np

TASK_NAMES = {1: "none+Center", 2: "Edge-Ring+Edge-Loc", 3: "Scratch+Loc+Random", 4: "Donut+Near-full"}


def load_metrics(path):
    with open(path) as f:
        data = json.load(f)
    # Keys in accuracy_matrix are stored as string repr of tuples, e.g. "(1, 2)"
    R = {}
    for k, v in data["accuracy_matrix"].items():
        R[ast.literal_eval(k)] = v
    data["R"] = R
    return data


def print_accuracy_matrix(R, seed=None):
    header = f"  Accuracy Matrix R[trained_up_to][task]"
    if seed is not None:
        header += f"  (seed={seed})"
    print(header)
    print(f"  {'':18}", end="")
    for j in range(1, 5):
        print(f"  Task{j:1d}", end="")
    print()
    for i in range(1, 5):
        row_entries = [R.get((i, j), None) for j in range(1, 5)]
        row_str = f"  After Task {i} ({TASK_NAMES[i][:12]:<12})"
        for v in row_entries:
            if v is None:
                row_str += "   —  "
            else:
                row_str += f"  {v:.3f}"
        print(row_str)


def print_single(data):
    seed = data.get("seed", "?")
    print(f"\n{'='*60}")
    print(f"  CL METRICS  (seed={seed})")
    print(f"{'='*60}")
    print_accuracy_matrix(data["R"], seed=seed)
    print(f"\n  Average Accuracy (AA)  : {data['average_accuracy']:.4f}")
    bwt = data['bwt']
    print(f"  Backward Transfer(BWT) : {bwt:+.4f}  ({'forgetting' if bwt < 0 else 'improvement'})")
    print(f"  Forward Transfer (FWT) : {data['fwt']:+.4f}")
    print(f"{'='*60}")


def aggregate_and_report(all_data, output_path):
    seeds   = [d["seed"] for d in all_data]
    AAs     = [d["average_accuracy"] for d in all_data]
    BWTs    = [d["bwt"] for d in all_data]
    FWTs    = [d["fwt"] for d in all_data]

    lines = []
    lines.append("=" * 60)
    lines.append(f"  MULTI-SEED CL METRICS  (seeds={seeds})")
    lines.append("=" * 60)
    lines.append(f"  Average Accuracy (AA)  : {np.mean(AAs):.4f} ± {np.std(AAs):.4f}")
    bwt_mean = np.mean(BWTs)
    lines.append(f"  Backward Transfer(BWT) : {bwt_mean:+.4f} ± {np.std(BWTs):.4f}  "
                 f"({'forgetting' if bwt_mean < 0 else 'improvement'})")
    lines.append(f"  Forward Transfer (FWT) : {np.mean(FWTs):+.4f} ± {np.std(FWTs):.4f}")
    lines.append("")

    # Per-seed breakdown
    lines.append("  Per-seed breakdown:")
    for d in all_data:
        lines.append(f"    seed={d['seed']}  AA={d['average_accuracy']:.4f}  "
                     f"BWT={d['bwt']:+.4f}  FWT={d['fwt']:+.4f}")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print("\n" + report)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report + "\n")
    print(f"\n  Report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Display / aggregate continual learning metrics")
    parser.add_argument("--results", nargs="+", required=True,
                        metavar="PATH", help="One or more cl_metrics_seed*.json files")
    args = parser.parse_args()

    all_data = []
    for path in args.results:
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found — skipping")
            continue
        all_data.append(load_metrics(path))

    if not all_data:
        print("No valid result files found.")
        return

    if len(all_data) == 1:
        print_single(all_data[0])
    else:
        for d in all_data:
            print_single(d)
        aggregate_and_report(all_data, os.path.join("eval_outputs", "multi_seed_report.txt"))


if __name__ == "__main__":
    main()
