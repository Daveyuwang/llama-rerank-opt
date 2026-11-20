#!/usr/bin/env bash
# Open-loop sweep per build: Poisson arrivals of held-out N=32 requests at each rate in RATES (default 2 4 6 8 10 12
# per second) for DURATION seconds each (default 40), after a 10 s warm-up at 3 per second. Latency includes queueing.
# usage: MODEL=model.gguf scripts/openloop.sh OUT_DIR NAME=SERVER_BIN ...
# SERVER_ARGS replaces the server configuration (default: the benchmark configuration of the results).
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck source=server.sh
. "$repo/scripts/server.sh"
out=${1:?usage: openloop.sh OUT_DIR NAME=SERVER_BIN ...}
shift
cases=${CASES:-$repo/data/cases.jsonl}
read -r -a rates <<< "${RATES:-2 4 6 8 10 12}"
read -r -a server_args <<< "${SERVER_ARGS:--np 32 -kvu -b 8192 -ub 2048}"
mkdir -p "$out"
trap stop_server EXIT

load() { python3 -I "$repo/scripts/openloop.py" --cases "$cases" --partition holdout --n 32 --port "$PORT" "$@"; }
for spec in "$@"; do
    name=${spec%%=*}
    start_server "${spec#*=}" "$out/openloop-$name.server.log" "${server_args[@]}"
    rm -f "$out/openloop-$name.warmup.jsonl"
    load --rates 3 --seconds 10 --out "$out/openloop-$name.warmup.jsonl" > /dev/null
    echo "== $name"
    load --rates "${rates[@]}" --seconds "${DURATION:-40}" --out "$out/openloop-$name.jsonl"
    stop_server
done
