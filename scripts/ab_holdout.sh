#!/usr/bin/env bash
# Held-out A/B of several servers: SESSIONS sessions (default 2) of ROUNDS rounds (default 5); the order of the
# servers rotates from session to session. Then the sessions of each server are pooled and every server is compared
# with the first one (paired bootstrap over queries, scripts/ab_compare.py).
# usage: MODEL=model.gguf scripts/ab_holdout.sh OUT_DIR NAME=SERVER NAME=SERVER ...
# SERVER is a llama-server binary or vllm:DIR, see bench_server.sh; SERVER_ARGS replaces the llama-server configuration.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
out=${1:?usage: ab_holdout.sh OUT_DIR NAME=SERVER NAME=SERVER ...}
shift
[ $# -ge 2 ] || { echo "need at least two servers" >&2; exit 1; }
specs=("$@")
sessions=${SESSIONS:-2}
mkdir -p "$out"

names=()
for spec in "${specs[@]}"; do names+=("${spec%%=*}"); done
for ((s = 0; s < sessions; s++)); do
    for ((i = 0; i < ${#specs[@]}; i++)); do
        spec=${specs[$(((i + s) % ${#specs[@]}))]}
        name=${spec%%=*}
        echo "== session $((s + 1)): $name"
        bash "$repo/scripts/bench_server.sh" "${spec#*=}" "$out/$name-s$((s + 1)).jsonl" "${ROUNDS:-5}" holdout
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
