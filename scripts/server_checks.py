"""Functional checks of /rerank on the stock and the patched server with the local Qwen3 reranker.

Covers the assertions of upstream tools/server/tests/unit/test_rerank.py (ranking, TEI format, invalid input, usage,
top_n, no idle-slot caching), which upstream runs on a non-causal model, plus cases
that exercise prefix sharing: identical and empty documents, more documents than slots, children that span several
ubatches, split KV streams, and concurrent requests. Every config runs on each binary, then the patch is compared
with stock request by request.

usage: python3 -I server_checks.py --bins stock=PATH patch=PATH --model M --cases cases.jsonl --out DIR
"""
import argparse
import http.client
import json
import os
import subprocess
import sys
import threading
import time

DOCS = [
    "A machine is a physical system that uses power to apply forces and control movement to perform an action. The term is commonly applied to artificial devices, such as those employing engines or motors, but also to natural biological macromolecules, such as molecular machines.",
    "Learning is the process of acquiring new understanding, knowledge, behaviors, skills, values, attitudes, and preferences. The ability to learn is possessed by humans, non-human animals, and some machines; there is also evidence for some kind of learning in certain plants.",
    "Machine learning is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions.",
    "Paris, capitale de la France, est une grande ville européenne et un centre mondial de l'art, de la mode, de la gastronomie et de la culture. Son paysage urbain du XIXe siècle est traversé par de larges boulevards et la Seine.",
]

COMMON = ["--reranking", "-ngl", "99", "--fit", "off", "-fa", "on", "--host", "127.0.0.1"]
CONFIGS = {
    "np1":          ["-c", "4096", "-np", "1", "-b", "512", "-ub", "512"],
    "np4-kvu-ub64": ["-c", "16384", "-np", "4", "-kvu", "-b", "64", "-ub", "64"],
    "np4-split-ub64": ["-c", "16384", "-np", "4", "-no-kvu", "-b", "64", "-ub", "64"],
    "np32-kvu":     ["-c", "32768", "-np", "32", "-kvu", "-b", "2048", "-ub", "2048"],
    "np2-cache-kvu":   ["-c", "8192", "-np", "2", "-kvu", "--cache-ram", "100", "-v"],
    "np2-cache-split": ["-c", "8192", "-np", "2", "-no-kvu", "--cache-ram", "100", "-v"],
}
PORT = 8097


def request(path, data, timeout=300):
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout)
    try:
        conn.request("POST", path, body=json.dumps(data).encode(), headers={"Content-Type": "application/json"})
        res = conn.getresponse()
        body = res.read()
        try:
            body = json.loads(body)
        except ValueError:
            body = body.decode(errors="replace")
        return {"status": res.status, "body": body}
    finally:
        conn.close()


def start_server(binary, model, args, log_path):
    log = open(log_path, "w")
    proc = subprocess.Popen([binary, "-m", model, *COMMON, "--port", str(PORT), *args], stdout=log, stderr=subprocess.STDOUT)
    for _ in range(480):
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}, see {log_path}")
        try:
            conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=2)
            conn.request("GET", "/health")
            if conn.getresponse().status == 200:
                return proc, log
        except OSError:
            pass
        time.sleep(0.5)
    raise RuntimeError("server did not become healthy")


def stop_server(proc, log):
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    log.close()


def suite(cases, cache_only):
    out = {}
    if cache_only:
        for i in range(4):
            out[f"cache-{i}"] = request("/rerank", {"query": f"query {i}", "documents": DOCS[i:]})
        return out

    out["basic"] = request("/rerank", {"query": "Machine learning is", "documents": DOCS})
    out["tei"] = request("/rerank", {"query": "Machine learning is", "texts": DOCS})
    for name, docs in [("[]", []), ("None", None), ("123", 123), ("[1,2,3]", [1, 2, 3])]:
        out[f"invalid-{name}"] = request("/rerank", {"query": "Machine learning is", "documents": docs})
    for i, (q, d1, d2) in enumerate([("Machine learning is", "A machine", "Learning is"),
                                     ("Which city?", "Machine learning is ", "Paris, capitale de la")]):
        out[f"usage-{i}"] = request("/rerank", {"query": q, "documents": [d1, d2]})
    for top_n in [None, 2, 4, 99]:
        for key in ["documents", "texts"]:
            data = {"query": "Machine learning is", key: DOCS}
            if top_n is not None:
                data["top_n"] = top_n
            out[f"topn-{key}-{top_n}"] = request("/rerank", data)
    out["identical"] = request("/rerank", {"query": "Machine learning is", "documents": [DOCS[2]] * 5 + [DOCS[3]]})
    # every prompt equal: a child shares all but its last token
    out["all-identical"] = request("/rerank", {"query": "Machine learning is", "documents": [DOCS[2]] * 4})
    out["empty"] = request("/rerank", {"query": "Machine learning is", "documents": ["", DOCS[0], ""]})

    def toolret(case, n=32):
        return {"query": case["query"], "documents": [case["candidates"][i]["text"] for i in case["orders"][str(n)]],
                "top_n": n}

    by_domain = {}
    for case in cases:
        by_domain.setdefault(case["domain"], []).append(case)
    a, b = by_domain["metatool"][1], by_domain["apibank"][1]
    out["toolret-32-metatool"] = request("/rerank", toolret(a))
    out["toolret-32-apibank"] = request("/rerank", toolret(b))
    data = toolret(a)
    data["documents"] += toolret(b)["documents"]
    data["top_n"] = 64
    out["toolret-64"] = request("/rerank", data)

    burst = [c for group in by_domain.values() for c in group[1::2]][:8]
    for c in burst:
        out[f"seq-{c['id']}"] = request("/rerank", toolret(c))
    results = {}

    def send(c):
        results[c["id"]] = request("/rerank", toolret(c))

    threads = [threading.Thread(target=send, args=(c,)) for c in burst]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for c in burst:
        out[f"burst-{c['id']}"] = results[c["id"]]
    return out


def scores(res):
    body = res["body"]
    items = body["results"] if isinstance(body, dict) else body
    key = "relevance_score" if isinstance(body, dict) else "score"
    return {it["index"]: it[key] for it in items}


def ranking(sc):
    return [i for i, _ in sorted(sc.items(), key=lambda kv: (-kv[1], kv[0]))]


def upstream_asserts(res):
    """The assertions of upstream test_rerank.py that do not depend on the model's tokenizer."""
    fails = []

    def check(cond, what):
        if not cond:
            fails.append(what)

    for key in ["basic", "tei"]:
        r = res[key]
        check(r["status"] == 200, f"{key} status")
        sc = scores(r)
        check(len(sc) == 4, f"{key} 4 results")
        rk = ranking(sc)
        check(rk[0] == 2, f"{key} most relevant is 2 (got {rk[0]})")
        check(rk[-1] == 3, f"{key} least relevant is 3 (got {rk[-1]})")
    for name in ["[]", "None", "123", "[1,2,3]"]:
        r = res[f"invalid-{name}"]
        check(r["status"] == 400 and isinstance(r["body"], dict) and "error" in r["body"], f"invalid {name} -> 400")
    for i in range(2):
        r = res[f"usage-{i}"]
        u = r["body"].get("usage", {}) if r["status"] == 200 and isinstance(r["body"], dict) else {}
        check(u and u["prompt_tokens"] == u["total_tokens"], f"usage-{i} prompt == total")
    for top_n, n in [(None, 4), (2, 2), (4, 4), (99, 4)]:
        for key in ["documents", "texts"]:
            check(len(scores(res[f"topn-{key}-{top_n}"])) == n, f"top_n {key} {top_n} -> {n}")
    return fails


def compare(stock, patch):
    rows = []
    for key in stock:
        s, p = stock[key], patch[key]
        if s["status"] != p["status"]:
            rows.append((key, f"STATUS {s['status']} vs {p['status']}"))
            continue
        if s["status"] != 200:
            rows.append((key, f"both {s['status']}"))
            continue
        ss, ps = scores(s), scores(p)
        same_rank = ranking(ss) == ranking(ps)
        diff = max(abs(ss[i] - ps[i]) for i in ss) if ss.keys() == ps.keys() else float("inf")
        su = s["body"].get("usage") if isinstance(s["body"], dict) else None
        pu = p["body"].get("usage") if isinstance(p["body"], dict) else None
        note = "" if su == pu else f" USAGE {su} vs {pu}"
        rows.append((key, f"{'same ranking' if same_rank else 'RANKING DIFFERS'}, max |score diff| {diff:.4f}{note}"))
    return rows


def burst_consistency(res):
    worst, flips = 0.0, 0
    for key in res:
        if key.startswith("burst-"):
            a, b = scores(res[key]), scores(res["seq-" + key[6:]])
            worst = max(worst, max(abs(a[i] - b[i]) for i in a))
            flips += ranking(a) != ranking(b)
    return worst, flips


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bins", nargs="+", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--configs", nargs="*", default=list(CONFIGS))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    bins = dict(b.split("=", 1) for b in args.bins)
    cases = [json.loads(l) for l in open(args.cases, encoding="utf-8-sig") if l.strip()]

    all_res = {}
    for cfg in args.configs:
        cache_only = cfg.startswith("np2-cache")
        for label, binary in bins.items():
            log_path = os.path.join(args.out, f"{label}-{cfg}.log")
            proc, log = start_server(binary, args.model, CONFIGS[cfg], log_path)
            try:
                res = suite(cases, cache_only)
            finally:
                stop_server(proc, log)
            all_res[(cfg, label)] = res
            with open(os.path.join(args.out, f"{label}-{cfg}.json"), "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=1)
            text = open(log_path, encoding="utf-8", errors="replace").read()
            if cache_only:
                ok = "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__" in text and "__TEST_TAG_CACHE_IDLE_SLOT__" not in text
                statuses = sorted({r["status"] for r in res.values()})
                print(f"[{cfg}] {label}: statuses {statuses}, idle-slot tags as upstream expects: {ok}")
                continue
            try:
                fails = upstream_asserts(res)
                worst, flips = burst_consistency(res)
                ident = scores(res["identical"])
                spread = max(ident[i] for i in range(5)) - min(ident[i] for i in range(5))
                same = scores(res["all-identical"])
                spread_all = max(same.values()) - min(same.values())
                print(f"[{cfg}] {label}: upstream asserts {'all pass' if not fails else 'FAIL ' + '; '.join(fails)}; "
                      f"burst vs sequential max diff {worst:.4f}, ranking flips {flips}; identical-doc score spread "
                      f"{spread:.4f} (5 of 6), {spread_all:.4f} (all 4)")
            except (KeyError, TypeError, ValueError) as e:
                print(f"[{cfg}] {label}: CHECK ERROR {type(e).__name__}: {e}")
        if not cache_only and len(bins) == 2:
            (l1, _), (l2, _) = bins.items()
            for key, verdict in compare(all_res[(cfg, l1)], all_res[(cfg, l2)]):
                print(f"   {key:28} {verdict}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
