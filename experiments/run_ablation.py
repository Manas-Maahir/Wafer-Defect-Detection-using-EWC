"""
run_ablation.py
---------------
Drives the EWC / replay ablation by invoking train.py in four regimes across one or more
seeds, then aggregates the resulting cl_metrics_*.json files into a single comparison table.

This produces the EWC-vs-no-EWC evidence that INVENTION_DISCLOSURE.md (§5) flags as missing.
Run from the repository root on a CUDA machine:

    python -m experiments.run_ablation --seeds 42 123 456
"""

import argparse
import ast
import json
import os
import subprocess
import sys

# regime tag -> extra CLI flags passed to train.py
REGIMES = {
    "naive":       ["--no-ewc", "--no-replay"],
    "ewc_only":    ["--no-replay"],
    "replay_only": ["--no-ewc"],
    "ewc_replay":  [],
}

EVAL_DIR = "eval_outputs"


def metrics_path(tag, seed):
    return os.path.join(EVAL_DIR, f"cl_metrics_{tag}_seed{seed}.json")


def run_regime(tag, seed, extra_flags, skip_existing):
    out = metrics_path(tag, seed)
    if skip_existing and os.path.exists(out):
        print(f"  [skip] {out} already exists")
        return
    cmd = [sys.executable, "train.py", "--seed", str(seed), "--tag", f"{tag}", *extra_flags]
    print(f"  [run ] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def load_metrics(tag, seed):
    path = metrics_path(tag, seed)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(description="EWC/replay ablation driver")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--regimes", nargs="+", default=list(REGIMES.keys()),
                    choices=list(REGIMES.keys()))
    ap.add_argument("--skip-existing", action="store_true",
                    help="Do not re-run a regime/seed whose metrics file already exists.")
    ap.add_argument("--no-train", action="store_true",
                    help="Only aggregate existing metrics; do not launch training.")
    args = ap.parse_args()

    if not args.no_train:
        for seed in args.seeds:
            for tag in args.regimes:
                print(f"\n=== regime={tag} seed={seed} ===")
                run_regime(tag, seed, REGIMES[tag], args.skip_existing)

    # ── Aggregate ──
    print("\n" + "=" * 70)
    print(f"  ABLATION SUMMARY  (mean over seeds {args.seeds})")
    print("=" * 70)
    print(f"  {'regime':<14}{'AA':>10}{'BWT':>10}{'FWT':>10}")
    rows = {}
    for tag in args.regimes:
        data = [load_metrics(tag, s) for s in args.seeds]
        aa  = mean([d["average_accuracy"] for d in data if d])
        bwt = mean([d["bwt"] for d in data if d])
        fwt = mean([d["fwt"] for d in data if d])
        rows[tag] = {"AA": aa, "BWT": bwt, "FWT": fwt}
        print(f"  {tag:<14}{aa:>10.4f}{bwt:>10.4f}{fwt:>10.4f}")

    if "naive" in rows and "ewc_replay" in rows:
        d_bwt = rows["ewc_replay"]["BWT"] - rows["naive"]["BWT"]
        d_aa  = rows["ewc_replay"]["AA"] - rows["naive"]["AA"]
        print("-" * 70)
        print(f"  EWC+replay vs naive:  ΔBWT={d_bwt:+.4f}  ΔAA={d_aa:+.4f}  "
              f"({'forgetting reduced' if d_bwt > 0 else 'no improvement'})")

    out = os.path.join(EVAL_DIR, "ablation_summary.json")
    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"seeds": args.seeds, "regimes": rows}, f, indent=2)
    print(f"\n  Summary saved: {out}")


if __name__ == "__main__":
    main()
