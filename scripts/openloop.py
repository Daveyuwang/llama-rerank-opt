"""Open-loop load test for /v1/rerank: Poisson arrivals at fixed rates, latency includes queueing.

Each request is one query with N candidates from the chosen partition, cycling through the queries in a
fixed shuffled order. Latency is measured from the scheduled arrival time to the end of the response.
achieved_rps is the number of successful requests divided by the time from the first arrival to the last
response, i.e. the arrival window plus the time to drain the queue; under overload it approaches the capacity.

usage: python3 -I openloop.py --cases cases.jsonl --partition holdout --n 32 --rates 2 3 4 \
           --seconds 30 --port 8090 --out result.jsonl
"""
import argparse
import http.client
import json
import math
import random
import statistics
import threading
import time
from collections import defaultdict


def rerank(port, query, docs):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    try:
        body = json.dumps(dict(query=query, documents=docs, top_n=len(docs)), ensure_ascii=False).encode("utf-8")
        conn.request("POST", "/v1/rerank", body=body, headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        response.read()
        return response.status
    finally:
        conn.close()


def p95(values):
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * 0.95) - 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--partition", choices=("dev", "holdout"), required=True)
    parser.add_argument("--n", type=int, choices=(8, 32), default=32)
    parser.add_argument("--rates", type=float, nargs="+", required=True)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cases = [json.loads(line) for line in open(args.cases, encoding="utf-8-sig") if line.strip()]
    groups = defaultdict(list)
    for case in cases:
        groups[case["domain"]].append(case)
    parity = 0 if args.partition == "dev" else 1
    cases = [case for group in groups.values() for case in group[parity::2]]
    random.Random(48).shuffle(cases)
    requests = [(c["query"], [c["candidates"][i]["text"] for i in c["orders"][str(args.n)]]) for c in cases]

    with open(args.out, "x", encoding="utf-8") as output:
        for rate in args.rates:
            rng = random.Random(int(rate * 1000))
            arrivals, t = [], 0.0
            while True:
                t += rng.expovariate(rate)
                if t > args.seconds:
                    break
                arrivals.append(t)
            results = [None] * len(arrivals)

            def send(i, scheduled, start):
                query, docs = requests[i % len(requests)]
                try:
                    status = rerank(args.port, query, docs)
                except Exception:  # noqa: BLE001 - count as an error
                    status = -1
                results[i] = (status, (time.perf_counter() - start - scheduled) * 1000)

            threads = []
            start = time.perf_counter()
            for i, scheduled in enumerate(arrivals):
                delay = start + scheduled - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                thread = threading.Thread(target=send, args=(i, scheduled, start))
                thread.start()
                threads.append(thread)
            for thread in threads:
                thread.join()
            elapsed = time.perf_counter() - start
            ok = [ms for status, ms in results if status == 200]
            row = dict(rate=rate, sent=len(arrivals), ok=len(ok), errors=len(arrivals) - len(ok),
                       achieved_rps=len(ok) / elapsed, p50_ms=statistics.median(ok) if ok else None,
                       p95_ms=p95(ok) if ok else None)
            print(json.dumps(row), flush=True)
            output.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
