"""Compare two rerank_bench.py runs (reference A, candidate B) of the same workload.

Latency: P50, P95 and throughput for each candidate count N, overall and per domain, with a bootstrap 95% interval
for the P95 change that resamples queries (all rounds of a query move together). Throughput is total A time over
total B time, since requests are sent one at a time. Scores: largest and P95 absolute difference, top-1 agreement
and nDCG@10 change.

usage: python3 -I ab_compare.py A.jsonl B.jsonl [--out result.json] [--cases cases.jsonl]
Without --cases the domain is the prefix of the query id (metatool_..., apibank_...).
"""
import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def load(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return {(r["id"], r["n"], r["repeat"]): r for r in rows}


def p95(values):
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * 0.95) - 1]


def top(scores):
    return min(scores, key=lambda k: (-scores[k], k))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--boot", type=int, default=2000)
    args = parser.parse_args()
    domain = {}
    if args.cases:
        for line in args.cases.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                case = json.loads(line)
                domain[case["id"]] = case["domain"]

    def domain_of(query_id):
        return domain.get(query_id) or query_id.split("_", 1)[0]

    a, b = load(args.reference), load(args.candidate)
    if a.keys() != b.keys():
        raise SystemExit("runs differ in workload, sizes or rounds")

    report = {}
    print(f"A = {args.reference.name}\nB = {args.candidate.name}")
    print(f"{'N':>3} {'scope':8} {'A P50':>7} {'B P50':>7} {'A P95':>7} {'B P95':>7} {'P95 chg':>8} "
          f"{'95% CI':>17} {'thru chg':>8} {'max|ds|':>7} {'p95|ds|':>7} {'top1':>5} {'dNDCG':>7}")
    rng = random.Random(48)
    for n in sorted({k[1] for k in a}):
        for scope in ("all", "metatool", "apibank"):
            keys = [k for k in a if k[1] == n and (scope == "all" or domain_of(k[0]) == scope)]
            if not keys:
                continue
            la = [a[k]["ms"] for k in keys]
            lb = [b[k]["ms"] for k in keys]
            change = p95(lb) / p95(la) - 1
            by_query = defaultdict(list)
            for k in keys:
                by_query[k[0]].append(k)
            queries = sorted(by_query)
            boots = []
            for _ in range(args.boot):
                sample = [k for q in (rng.choice(queries) for _ in queries) for k in by_query[q]]
                boots.append(p95([b[k]["ms"] for k in sample]) / p95([a[k]["ms"] for k in sample]) - 1)
            boots.sort()
            low, high = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]
            thru = sum(la) / sum(lb) - 1
            diffs = [abs(a[k]["scores"][d] - b[k]["scores"][d]) for k in keys for d in a[k]["scores"]]
            top1 = sum(top(a[k]["scores"]) == top(b[k]["scores"]) for k in keys) / len(keys)
            ndcg = [b[k]["ndcg"] - a[k]["ndcg"] for k in keys if a[k]["ndcg"] is not None]
            dndcg = statistics.fmean(ndcg) if ndcg else float("nan")
            print(f"{n:>3} {scope:8} {statistics.median(la):7.1f} {statistics.median(lb):7.1f} {p95(la):7.1f} "
                  f"{p95(lb):7.1f} {change:8.1%} [{low:7.1%},{high:7.1%}] {thru:8.1%} {max(diffs):7.4f} "
                  f"{p95(diffs):7.4f} {top1:5.2f} {dndcg:7.4f}")
            report[f"N{n}/{scope}"] = dict(
                samples=len(keys), a_p50=statistics.median(la), b_p50=statistics.median(lb), a_p95=p95(la),
                b_p95=p95(lb), p95_change=change, p95_change_ci=[low, high], throughput_change=thru,
                max_abs_score_diff=max(diffs), p95_abs_score_diff=p95(diffs), top1_agreement=top1,
                ndcg_delta=dndcg)
    if args.out:
        with args.out.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
