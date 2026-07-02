"""`python -m sakshi.bench` — run the full benchmark and write reports."""
from __future__ import annotations

import argparse
import json

from .harness import compare, run_suite
from .report import write_reports


def main() -> None:
    ap = argparse.ArgumentParser(description="Run SakshiBench.")
    ap.add_argument("--out", default="runs/bench", help="output directory")
    args = ap.parse_args()

    results = run_suite()
    paths = write_reports(results, args.out)
    cmp = compare(results)

    print(json.dumps(cmp, indent=2))
    print("\nartifacts:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
