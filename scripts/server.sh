# Sourced by the benchmark scripts. start_server BIN LOG [ARGS...] starts llama-server with the reranker in MODEL,
# the common flags below and ARGS, waits until /health answers and sets SERVER_PID; stop_server stops it.
PORT=${PORT:-8090}
common_args=(--reranking -ngl 99 --fit off -fa on -c 32768 -ctk f16 -ctv f16 --host 127.0.0.1)

start_server() {
    local bin=$1 log=$2
    shift 2
    [ -n "${MODEL:-}" ] || { echo "set MODEL to the reranker GGUF file" >&2; return 1; }
    "$bin" -m "$MODEL" "${common_args[@]}" --port "$PORT" "$@" > "$log" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 240); do
        curl -fs "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && return 0
        kill -0 "$SERVER_PID" 2> /dev/null || { echo "server exited, see $log" >&2; return 1; }
        sleep 0.5
    done
    echo "server not healthy after 120 s, see $log" >&2
    return 1
}

stop_server() {
    [ -n "${SERVER_PID:-}" ] || return 0
    kill "$SERVER_PID" 2> /dev/null
    wait "$SERVER_PID" 2> /dev/null || true
}
