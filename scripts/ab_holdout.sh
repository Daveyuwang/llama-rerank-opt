#!/usr/bin/env bash
# Held-out A/B of several llama-server builds: SESSIONS sessions (default 2) of ROUNDS rounds (default 5), the builds
# alternating within each session; then the sessions of each build are pooled and every build is compared with the
# first one (paired bootstrap over queries, scripts/ab_compare.py).
# usage: MODEL=model.gguf scripts/ab_holdout.sh OUT_DIR NAME=SERVER_BIN NAME=SERVER_BIN ...
# SERVER_ARGS replaces the server configuration (default: the benchmark configuration of the results).
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
out=${1:?usage: ab_holdout.sh OUT_DIR NAME=SERVER_BIN NAME=SERVER_BIN ...}
shift
[ $# -ge 2 ] || { echo "need at least two builds" >&2; exit 1; }
sessions=${SESSIONS:-2}
rounds=${ROUNDS:-5}
read -r -a server_args <<< "${SERVER_ARGS:--np 32 -kvu -b 8192 -ub 2048}"
mkdir -p "$out"

names=()
for spec in "$@"; do names+=("${spec%%=*}"); done
for ((s = 1; s <= sessions; s++)); do
    for spec in "$@"; do
        name=${spec%%=*}
        echo "== session $s: $name"
        bash "$repo/scripts/bench_server.sh" "${spec#*=}" "$out/$name-s$s.jsonl" "$rounds" holdout -- "${server_args[@]}"
    done
done
for name in "${names[@]}"; do
    runs=()
    for ((s = 1; s <= sessions; s++)); do runs+=("$out/$name-s$s.jsonl"); done
    python3 -I "$repo/scripts/pool_runs.py" "$out/$name.jsonl" "${runs[@]}" > /dev/null
done
for name in "${names[@]:1}"; do
    echo "== ${names[0]} -> $name"
    python3 -I "$repo/scripts/ab_compare.py" "$out/${names[0]}.jsonl" "$out/$name.jsonl" \
        --out "$out/ab-${names[0]}-$name.json"
done
