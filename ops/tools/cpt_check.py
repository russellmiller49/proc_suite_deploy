"""
Find any token pieces that look like split 5-digit numbers still labeled GEO.

Example: ["12", "##345"] => "12345"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("data/ml_training/distilled_phi_labels.jsonl"),
        help="Path to distilled_phi_labels.jsonl",
    )
    parser.add_argument("--limit", type=int, default=20, help="Stop after printing N matches")
    args = parser.parse_args()

    bad = 0
    with args.path.open() as f:
        for line in f:
            r = json.loads(line)
            toks, labs = r["tokens"], r["ner_tags"]
            for i in range(len(toks) - 1):
                if toks[i].isdigit() and toks[i + 1].startswith("##"):
                    combined = toks[i] + toks[i + 1][2:]
                    if re.fullmatch(r"\d{5}", combined) and (
                        "GEO" in labs[i] or "GEO" in labs[i + 1]
                    ):
                        bad += 1
                        print("BAD", combined, labs[i], labs[i + 1], "id=", r.get("id"))
                        if bad >= args.limit:
                            print("done, bad=", bad)
                            return 0

    print("done, bad=", bad)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
