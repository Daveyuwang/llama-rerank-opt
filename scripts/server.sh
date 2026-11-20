# Sourced by the benchmark scripts. start_server BIN LOG [ARGS...] starts llama-server with the reranker in MODEL, the
# common flags below and ARGS; start_vllm DIR LOG [ARGS...] serves the Hugging Face model in DIR with vLLM. Both wait
# until /health answers and set SERVER_PID; stop_server stops the server.
# SERVER_ARGS replaces the llama-server configuration of the results in bench_args. VLLM is the vllm command.
PORT=${PORT:-8090}
common_args=(--reranking -ngl 99 --fit off -fa on -c 32768 -ctk f16 -ctv f16 --host 127.0.0.1)
read -r -a bench_args <<< "${SERVER_ARGS--np 32 -kvu -b 8192 -ub 2048}"

start_cmd() {
    local log=$1
    shift
    if curl -fs "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
        echo "another server already answers on port $PORT" >&2
        return 1
    fi
    "$@" > "$log" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 1200); do
        kill -0 "$SERVER_PID" 2> /dev/null || { echo "server exited, see $log" >&2; SERVER_PID=; return 1; }
        curl -fs "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && return 0
        sleep 0.5
    done
    echo "server not healthy after 600 s, see $log" >&2
    return 1
}

start_server() {
    local bin=$1 log=$2
    shift 2
    [ -n "${MODEL:-}" ] || { echo "set MODEL to the reranker GGUF file" >&2; return 1; }
    start_cmd "$log" "$bin" -m "$MODEL" "${common_args[@]}" --port "$PORT" "$@"
}

# The overrides load the original Qwen3-Reranker checkpoint as a yes/no classifier, as vLLM's documentation describes.
start_vllm() {
    local dir=$1 log=$2
    shift 2
    start_cmd "$log" "${VLLM:-vllm}" serve "$dir" --served-model-name reranker --host 127.0.0.1 --port "$PORT" \
        --hf-overrides '{"architectures": ["Qwen3ForSequenceClassification"], "classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": true}' \
        --max-model-len 8192 --gpu-memory-utilization 0.8 "$@"
}

stop_server() {
    [ -n "${SERVER_PID:-}" ] || return 0
    kill "$SERVER_PID" 2> /dev/null || true
    wait "$SERVER_PID" 2> /dev/null || true
    SERVER_PID=
}
