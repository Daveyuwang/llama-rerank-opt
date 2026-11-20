"""Merge rerank_bench.py runs from several sessions into one file, renumbering rounds so keys stay unique.

usage: python3 -I pool_runs.py OUT.jsonl RUN1.jsonl RUN2.jsonl ...
"""
import json
import sys
from pathlib import Path


def main():
    out, runs = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    offset = 0
    with out.open("x", encoding="utf-8") as handle:
        for run in runs:
            rows = [json.loads(line) for line in run.read_text(encoding="utf-8").splitlines() if line.strip()]
            for row in rows:
                row["repeat"] += offset
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            offset += max(row["repeat"] for row in rows) - offset + 1
    print(f"wrote {out} from {len(runs)} runs")


if __name__ == "__main__":
    main()
