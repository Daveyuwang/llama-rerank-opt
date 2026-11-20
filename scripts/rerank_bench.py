"""Prepare the ToolRet rerank workload and measure a /v1/rerank endpoint.

prepare: read the metatool and apibank queries of ToolRet from the Hugging Face datasets server and write 20 cases
    per domain. A case holds the labelled positives plus the most similar other tools of the domain (word overlap),
    32 candidates in all, and fixed shuffled candidate orders for N = 1, 8 and 32.
run: send every case at every size once per round, in a shuffled order, one request at a time, and write one JSON
    line per request with the latency, the scores and nDCG@10. The two partitions take alternate cases of each
    domain (dev: even, holdout: odd). With --qwen3-template the client applies the Qwen3-Reranker prompt template,
    which llama-server takes from the GGUF file, for servers that score the query and document texts joined as
    they are (vLLM).

usage: python3 -I rerank_bench.py prepare --out data/cases.jsonl
       python3 -I rerank_bench.py run --cases data/cases.jsonl --partition holdout --rounds 5 --port 8090 \
           --out run.jsonl
"""
import argparse
import hashlib
import http.client
import json
import math
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

# the rerank template of the Qwen3-Reranker GGUF files (tokenizer.chat_template.rerank)
QWEN3_TEMPLATE = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the '
                  'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
                  '<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n'
                  '<Query>: {query}\n<Document>: {document}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')


def fetch_json(url, attempts=8):
    delay = 5
    for attempt in range(1, attempts + 1):
        try:
            with urlopen(url, timeout=90) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError) as error:
            if attempt == attempts:
                raise
            print(f"attempt {attempt} failed ({error}); retrying in {delay}s", file=sys.stderr, flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 60)


def read_queries(config):
    rows = []
    while True:
        params = dict(dataset="mangopy/ToolRet-Queries", config=config, split="queries",
                      offset=len(rows), length=100)
        page = fetch_json("https://datasets-server.huggingface.co/rows?" + urlencode(params))
        if not page["rows"]:
            break
        rows.extend(item["row"] for item in page["rows"])
        if len(rows) >= page["num_rows_total"]:
            break
    return rows


def words(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def prepare(count):
    cases = []
    for domain in ("metatool", "apibank"):
        rows = read_queries(domain)
        docs = {}
        labels = {}
        for row in rows:
            label = {}
            for item in json.loads(row["labels"]):
                doc_id = item["id"]
                docs[doc_id] = json.dumps(item["doc"], ensure_ascii=False, sort_keys=True)
                label[doc_id] = max(label.get(doc_id, 0), item["relevance"])
            labels[row["id"]] = {key: grade for key, grade in label.items() if grade > 0}
        if len(docs) < 32:
            raise ValueError(f"Need 32 distinct tools for {domain}; got {len(docs)}")
        terms = {key: words(text) for key, text in docs.items()}
        eligible = [row for row in rows if 0 < len(labels[row["id"]]) <= 8]
        chosen = random.Random(48).sample(eligible, count)
        for row in chosen:
            gold = labels[row["id"]]
            query_terms = words(row["query"])
            def similarity(key):
                overlap = len(query_terms & terms[key]) / max(1, len(query_terms | terms[key]))
                return (-overlap, key)
            others = sorted((key for key in docs if key not in gold), key=similarity)
            pool = list(sorted(gold)) + others[:32 - len(gold)]
            # Both ranked subsets contain all labelled positives but the model sees
            # them in a shuffled order. The singleton is only a latency control.
            rng = random.Random(row["id"])
            first, rest = pool[:8], pool[8:]
            rng.shuffle(first)
            rng.shuffle(rest)
            pool = first + rest
            orders = {}
            for n in (1, 8, 32):
                order = list(range(n))
                rng.shuffle(order)
                orders[str(n)] = order
            cases.append(dict(id=row["id"], query=row["query"], domain=domain, gold=gold,
                              candidates=[dict(id=k, text=docs[k]) for k in pool], orders=orders))
    return cases


def request_scores(conn, query, docs, template):
    if template:
        head, tail = QWEN3_TEMPLATE.split("{document}")
        query, docs = head.replace("{query}", query), [doc + tail for doc in docs]
    body = json.dumps(dict(query=query, documents=docs, top_n=len(docs)),
                      ensure_ascii=False).encode("utf-8")
    conn.request("POST", "/v1/rerank", body=body, headers={"Content-Type": "application/json"})
    response = conn.getresponse()
    raw = response.read()
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status}: {raw[:480]!r}")
    rows = json.loads(raw)["results"]
    if sorted(row["index"] for row in rows) != list(range(len(docs))):
        raise ValueError("Missing or duplicate candidate indexes")
    scores = [float(row["relevance_score"]) for row in sorted(rows, key=lambda row: row["index"])]
    if any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
        raise ValueError(f"Invalid Qwen3 reranker score: {scores}")
    return scores


def evaluate(conn, case, n, template):
    docs = [case["candidates"][i] for i in case["orders"][str(n)]]
    start = time.perf_counter()
    scores = request_scores(conn, case["query"], [doc["text"] for doc in docs], template)
    ms = (time.perf_counter() - start) * 1000
    by_id = {doc["id"]: score for doc, score in zip(docs, scores)}
    ranked = sorted(by_id, key=lambda k: (-by_id[k], k))
    grades = [case["gold"].get(key, 0) for key in ranked]
    def dcg(values):
        return sum((2 ** grade - 1) / math.log2(i + 2) for i, grade in enumerate(values[:10]))
    ideal = dcg(sorted(grades, reverse=True))
    return dict(id=case["id"], n=n, ms=ms, scores=by_id,
                ndcg=dcg(grades) / ideal if ideal and n > 1 else None)


def run(args):
    workload_bytes = args.cases.read_bytes()
    cases = [json.loads(line) for line in workload_bytes.decode("utf-8-sig").splitlines() if line.strip()]
    if args.partition != "all":
        groups = defaultdict(list)
        for case in cases:
            groups[case["domain"]].append(case)
        parity = 0 if args.partition == "dev" else 1
        cases = [case for group in groups.values() for case in group[parity::2]]
    if not cases or args.rounds < 1:
        raise ValueError("Need at least one case and one round")
    conn = http.client.HTTPConnection(args.host, args.port, timeout=120)
    try:
        latency = defaultdict(list)
        quality = defaultdict(list)
        # the hash of the file with LF line endings, as prepare writes it
        digest = hashlib.sha256(workload_bytes.replace(b"\r\n", b"\n")).hexdigest()
        with args.out.open("x", encoding="utf-8") as output:
            for repeat in range(args.rounds):
                schedule = [(case, n) for case in cases for n in args.sizes]
                random.Random(48 + repeat).shuffle(schedule)
                for case, n in schedule:
                    row = evaluate(conn, case, n, args.qwen3_template)
                    row.update(repeat=repeat, workload_sha256=digest)
                    print(json.dumps(row, separators=(",", ":")), file=output, flush=True)
                    latency[n].append(row["ms"])
                    if row["ndcg"] is not None:
                        quality[n].append(row["ndcg"])
        for n, values in sorted(latency.items()):
            p95 = sorted(values)[math.ceil(len(values) * .95) - 1]
            print(json.dumps(dict(n=n, batches=len(values), p50_ms=statistics.median(values),
                                  p95_ms=p95, candidates_per_active_second=1000*n*len(values)/sum(values),
                                  ndcg10=statistics.mean(quality[n]) if quality[n] else None)))
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = parser.add_subparsers(dest="command", required=True)
    prep = subs.add_parser("prepare")
    prep.add_argument("--per-domain", type=int, default=20)
    prep.add_argument("--out", type=Path, required=True)
    bench = subs.add_parser("run")
    bench.add_argument("--cases", type=Path, required=True)
    bench.add_argument("--qwen3-template", action="store_true")
    bench.add_argument("--port", type=int, default=8090)
    bench.add_argument("--host", default="127.0.0.1")
    bench.add_argument("--rounds", type=int, default=3)
    bench.add_argument("--partition", choices=("all", "dev", "holdout"), default="all")
    bench.add_argument("--sizes", nargs="+", type=int, choices=(1, 8, 32), default=[1, 8, 32])
    bench.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepared = prepare(args.per_domain)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        # LF line endings on every platform, so the file hash does not depend on the OS
        with args.out.open("x", encoding="utf-8", newline="\n") as output:
            for row in prepared:
                print(json.dumps(row, ensure_ascii=False), file=output)
        print(f"Wrote {len(prepared)} cases to {args.out}, sha256 {hashlib.sha256(args.out.read_bytes()).hexdigest()}")
    else:
        run(args)
