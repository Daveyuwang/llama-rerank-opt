#!/usr/bin/env bash
# Start one server, run a discarded warm-up round and then ROUNDS measured rounds of rerank_bench.py on PARTITION, one
# request at a time. For llama-server it also prints how many prompt tokens the server processed in the measured rounds.
# usage: MODEL=model.gguf scripts/bench_server.sh SERVER OUT.jsonl ROUNDS PARTITION
# SERVER is a llama-server binary (run with the configuration in SERVER_ARGS, see server.sh), or vllm:DIR to serve the
# Hugging Face checkpoint in DIR with vLLM (VLLM_ARGS adds vLLM options). CASES defaults to data/cases.jsonl, PORT to
# 8090, SIZES (candidates per request) to "1 8 32". The server log goes next to OUT.jsonl.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck source=server.sh
. "$repo/scripts/server.sh"
[ $# -eq 4 ] || { echo "usage: bench_server.sh SERVER OUT.jsonl ROUNDS PARTITION" >&2; exit 1; }
server=$1 out=$2 rounds=$3 partition=$4
cases=${CASES:-$repo/data/cases.jsonl}
read -r -a sizes <<< "${SIZES:-1 8 32}"
warmup=${out%.jsonl}.warmup.jsonl

trap stop_server EXIT
client_args=()
if [[ $server == vllm:* ]]; then
    read -r -a vllm_args <<< "${VLLM_ARGS:-}"
    start_vllm "${server#vllm:}" "${out%.jsonl}.server.log" "${vllm_args[@]}"
    client_args=(--qwen3-template)
else
    start_server "$server" "${out%.jsonl}.server.log" --metrics "${bench_args[@]}"
fi
metric() {
    [[ $server == vllm:* ]] && return 0
    curl -fs "http://127.0.0.1:$PORT/metrics" | awk -v k="$1" '$1 == k { print $2 }'
}
bench() {
    python3 -I "$repo/scripts/rerank_bench.py" run --cases "$cases" --partition "$partition" --port "$PORT" \
        --sizes "${sizes[@]}" "${client_args[@]}" "$@"
}

rm -f "$warmup"
bench --rounds 1 --out "$warmup" > /dev/null
rm -f "$warmup"
before=$(metric llamacpp:prompt_tokens_total)
bench --rounds "$rounds" --out "$out"
after=$(metric llamacpp:prompt_tokens_total)
if [ -n "$before" ]; then
    python3 -c "b, a = float('$before'), float('$after'); print('prompt tokens processed in the measured rounds:', round(a - b), '(the server prints the counter with 6 significant digits)' if a >= 1e6 else '')"
fi
