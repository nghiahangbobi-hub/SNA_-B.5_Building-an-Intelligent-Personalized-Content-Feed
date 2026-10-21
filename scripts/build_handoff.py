"""Collect the verified seed-0 baseline/E1 artifacts into a new handoff directory."""

import argparse
import json

from src.evaluation.handoff import build_handoff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dirs", nargs="+", default=[
        "outputs/tables/phuc02_mindsmall_seed0_2026-10-19",
        "outputs/tables/phuc03_b3_mindsmall_seed0_2026-10-19",
        "outputs/tables/phuc03_b4_mindsmall_seed0_2026-10-19",
    ])
    parser.add_argument("--e1-dir", default="outputs/tables/phuc04_e1_mindsmall_seed0_2026-10-20")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = build_handoff(baseline_dirs=args.baseline_dirs, e1_dir=args.e1_dir, output_dir=args.output_dir)
    print(json.dumps({"status": result["status"], "metric_rows": result["rows"]}))


if __name__ == "__main__":
    main()
