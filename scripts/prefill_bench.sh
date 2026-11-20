#!/usr/bin/env bash
# Single-sequence prefill A/B with llama-bench: ROUNDS rounds (default 8), the order of the builds rotated every
# round, 3 repetitions per round, at the given ubatch size. Prints per prompt length the change against the first
# build: median over rounds [min, max].
# usage: MODEL=model.gguf UB=2048 scripts/prefill_bench.sh OUT.jsonl NAME=LLAMA_BENCH_BIN NAME=LLAMA_BENCH_BIN ...
set -euo pipefail
log=${1:?usage: prefill_bench.sh OUT.jsonl NAME=LLAMA_BENCH_BIN NAME=LLAMA_BENCH_BIN ...}
shift
[ $# -ge 2 ] || { echo "need at least two builds" >&2; exit 1; }
model=${MODEL:?set MODEL to a GGUF file}
ub=${UB:-2048}
prompts=${PROMPTS:-512,1024,2048,4096,8192,16384,32768}
names=() bins=()
for spec in "$@"; do names+=("${spec%%=*}"); bins+=("${spec#*=}"); done
: > "$log"
for ((round = 0; round < ${ROUNDS:-8}; round++)); do
    for ((i = 0; i < ${#names[@]}; i++)); do
        k=$(((round + i) % ${#names[@]}))
        "${bins[$k]}" -m "$model" -ngl 99 -fa 1 -ub "$ub" -b "$ub" -p "$prompts" -n 0 -r 3 -o jsonl 2> /dev/null |
            python3 -I -c 'import json, sys
for line in sys.stdin:
    if line.strip():
        r = json.loads(line)
        print(json.dumps(dict(build=sys.argv[1], round=int(sys.argv[2]), n_prompt=r["n_prompt"], avg_ts=r["avg_ts"])))' \
            "${names[$k]}" "$round" >> "$log"
    done
done
python3 -I - "$log" "${names[@]}" <<'EOF'
import json
import statistics
import sys
from collections import defaultdict

ts = defaultdict(dict)
for line in open(sys.argv[1]):
    r = json.loads(line)
    ts[r["n_prompt"]][(r["build"], r["round"])] = r["avg_ts"]
ref, others = sys.argv[2], sys.argv[3:]
for n in sorted(ts):
    d = ts[n]
    rounds = sorted({r for (_, r) in d})
    base = statistics.median(d[(ref, r)] for r in rounds if (ref, r) in d)
    cells = []
    for name in others:
        ch = [100 * (d[(name, r)] / d[(ref, r)] - 1) for r in rounds if (name, r) in d and (ref, r) in d]
        cells.append(f"{name} {statistics.median(ch):+5.1f}% [{min(ch):+.1f}, {max(ch):+.1f}]")
    print(f"pp{n:<6} {ref} {base:8.0f} t/s   " + "   ".join(cells))
EOF
