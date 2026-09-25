"""Run all FewShotHDC experiments.

``python experiments/run_all.py``        quick defaults (a few minutes)
``python experiments/run_all.py --full`` larger sweeps for the paper runs
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true", help="use the larger paper settings")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--only", nargs="+", default=["exp1", "exp2", "exp3", "exp4"])
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.full:
        common = ["--seeds", "5", "--hd-dim", "10000", "--epochs", "20"]
        exp1 = common + ["--snrs", "0.25", "0.5", "1.0"]
        exp2 = common + ["--seeds", "5"]
        exp3 = ["--seeds", "5", "--hd-dim", "10000", "--oracle-per-class", "5000",
                "--sweep", "--sweep-seeds", "3"]
        exp4 = common + ["--seeds", "5"]
    else:
        common = ["--seeds", "3", "--hd-dim", "4096", "--epochs", "10"]
        exp1 = common + ["--snrs", "0.25", "0.5", "1.0"]
        exp2 = common + ["--seeds", "3"]
        exp3 = ["--seeds", "3", "--hd-dim", "2048", "--oracle-per-class", "2000",
                "--sweep", "--sweep-seeds", "2"]
        exp4 = common + ["--seeds", "3"]

    jobs = []
    if "exp1" in args.only:
        jobs.append(("exp1", ["exp1_fewshot_ability.py"] + exp1))
    if "exp2" in args.only:
        jobs.append(("exp2", ["exp2_breakage.py"] + exp2))
    if "exp3" in args.only:
        jobs.append(("exp3", ["exp3_prototype_samples.py"] + exp3))
    if "exp4" in args.only:
        jobs.append(("exp4", ["exp4_robustness.py"] + exp4))

    for name, cmd in jobs:
        if args.tag:
            cmd = cmd + ["--tag", args.tag]
        print(f"\n########## {name} ##########", flush=True)
        t0 = time.time()
        subprocess.run([sys.executable] + [str(HERE / cmd[0])] + cmd[1:],
                       check=True)
        print(f"########## {name} finished in {time.time() - t0:.0f}s ##########",
              flush=True)


if __name__ == "__main__":
    main()
