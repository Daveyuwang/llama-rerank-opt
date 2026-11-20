#!/usr/bin/env bash
# Mixed open loop per build: held-out N=32 requests arrive at 2 per second while N=8 requests arrive at each rate in
# R8S (default 10 20 30 40 per second), DURATION seconds per point (default 40). With patch 0001 an N=32 request is
# one sharing group that needs 32 free slots at once, so this load shows whether small requests starve it.
# usage: MODEL=model.gguf scripts/mixed_load.sh OUT_DIR NAME=SERVER_BIN ...
# SERVER_ARGS replaces the server configuration (default: the benchmark configuration of the results).
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck source=server.sh
. "$repo/scripts/server.sh"
out=${1:?usage: mixed_load.sh OUT_DIR NAME=SERVER_BIN ...}
shift
cases=${CASES:-$repo/data/cases.jsonl}
duration=${DURATION:-40}
read -r -a r8s <<< "${R8S:-10 20 30 40}"
mkdir -p "$out"
small=
trap '[ -z "$small" ] || kill "$small" 2> /dev/null || true; stop_server' EXIT

load() { python3 -I "$repo/scripts/openloop.py" --cases "$cases" --partition holdout --port "$PORT" "$@"; }
show() {
    python3 -I -c 'import json, sys
r = json.loads(open(sys.argv[1]).readline())
rate, n = sys.argv[2], sys.argv[3]
done, errors = r["achieved_rps"], r["errors"]
p50, p95 = ("-" if v is None else f"{v:.0f}" for v in (r["p50_ms"], r["p95_ms"]))
print(f"  N=8 at {rate:>3}/s  N={n:<2} requests: completed {done:5.2f}/s  p50 {p50:>7} ms  p95 {p95:>7} ms  "
      f"errors {errors}")' "$@"
}
for spec in "$@"; do
    name=${spec%%=*}
    start_server "${spec#*=}" "$out/mixed-$name.server.log" "${bench_args[@]}"
    rm -f "$out/mixed-$name.warmup.jsonl"
    load --n 32 --rates 3 --seconds 10 --out "$out/mixed-$name.warmup.jsonl" > /dev/null
    rm -f "$out/mixed-$name.warmup.jsonl"
    echo "== $name"
    for r8 in "${r8s[@]}"; do
        load --n 8 --rates "$r8" --seconds "$duration" --out "$out/mixed-$name-r$r8-n8.jsonl" > /dev/null &
        small=$!
        load --n 32 --rates 2 --seconds "$duration" --out "$out/mixed-$name-r$r8-n32.jsonl" > /dev/null
        wait "$small"
        small=
        show "$out/mixed-$name-r$r8-n8.jsonl" "$r8" 8
        show "$out/mixed-$name-r$r8-n32.jsonl" "$r8" 32
    done
    stop_server
done
