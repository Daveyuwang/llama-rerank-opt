#!/usr/bin/env bash
# Start one llama-server build, run a discarded warm-up round and then ROUNDS measured rounds of rerank_bench.py on
# PARTITION, one request at a time, and print how many prompt tokens the server processed in the measured rounds.
# usage: MODEL=model.gguf scripts/bench_server.sh SERVER_BIN OUT.jsonl ROUNDS PARTITION [-- SERVER_ARGS...]
# CASES defaults to data/cases.jsonl, PORT to 8090. The server log goes next to OUT.jsonl.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck source=server.sh
. "$repo/scripts/server.sh"
[ $# -ge 4 ] || { echo "usage: bench_server.sh SERVER_BIN OUT.jsonl ROUNDS PARTITION [-- SERVER_ARGS...]" >&2; exit 1; }
bin=$1 out=$2 rounds=$3 partition=$4
shift 4
[ "${1:-}" != "--" ] || shift
cases=${CASES:-$repo/data/cases.jsonl}
warmup=${out%.jsonl}.warmup.jsonl

trap stop_server EXIT
start_server "$bin" "${out%.jsonl}.server.log" --metrics "$@"
metric() { curl -fs "http://127.0.0.1:$PORT/metrics" | awk -v k="$1" '$1 == k { print $2 }'; }
bench() {
    python3 -I "$repo/scripts/rerank_bench.py" run --cases "$cases" --partition "$partition" --mode batch \
        --port "$PORT" "$@"
}

rm -f "$warmup"
bench --rounds 1 --out "$warmup" > /dev/null
rm -f "$warmup"
before=$(metric llamacpp:prompt_tokens_total)
bench --rounds "$rounds" --out "$out"
after=$(metric llamacpp:prompt_tokens_total)
echo "prompt tokens processed in the measured rounds: $(python3 -c "print(int(float('$after') - float('$before')))")"
