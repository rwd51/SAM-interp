"""End-to-end runner: download -> Q1 -> Q2 -> Q3 -> assemble.

Colab / local:
    python run_all.py
    python run_all.py --sweep-blocks 2    # fast mode for Q3

Each step is idempotent: data + checkpoints are cached, figures are overwritten.
"""
from __future__ import annotations

import argparse
import time

from scripts import assemble, bootstrap_ci, download, q1, q1_cka, q2, q3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-blocks", type=int, default=4,
                    help="trailing blocks swept in Q3 (2 = fast, 4 = default)")
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["download", "q1", "q1_cka", "q2", "q3",
                             "bootstrap_ci", "assemble"],
                    help="stages to skip")
    args = ap.parse_args()

    stages = [
        ("download",     download.main,     {}),
        ("q1",           q1.main,           {}),
        ("q1_cka",       q1_cka.main,       {}),
        ("q2",           q2.main,           {}),
        ("q3",           q3.main,           {"sweep_blocks": args.sweep_blocks}),
        ("bootstrap_ci", bootstrap_ci.main, {}),
        ("assemble",     assemble.main,     {}),
    ]

    for name, fn, kwargs in stages:
        if name in args.skip:
            print(f"[run_all] skipping {name}")
            continue
        print(f"\n===== {name} =====")
        t = time.time()
        fn(**kwargs)
        print(f"[run_all] {name} finished in {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
