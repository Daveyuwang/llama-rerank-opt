#!/usr/bin/env bash
# Kernel A/B of FLASH_ATTN_EXT with test-backend-ops perf, without order bias: for every case all builds run back to
# back, and the order of the builds rotates with every repetition. Each line of OUT.log is
# "build repetition n_kv n_q n_seq n_prefix us_per_run"; the summary gives, per case and build, the change against
# the first build in the same repetition: median, bootstrap 95% CI of the median, and how often it was slower.
# usage: scripts/fa_kernel_ab.sh OUT.log NAME=TEST_BACKEND_OPS_BIN NAME=TEST_BACKEND_OPS_BIN ...   (REPS, default 6)
# The cases (n_seq_diag) come from the test part of patch 0003; for a stock-kernel build apply only that part:
#   git -C llama.cpp apply --include='tests/*' patches/0003-cuda-fa-skip-masked-kv.patch
set -uo pipefail
log=${1:?usage: fa_kernel_ab.sh OUT.log NAME=TEST_BACKEND_OPS_BIN NAME=TEST_BACKEND_OPS_BIN ...}
shift
[ $# -ge 2 ] || { echo "need at least two builds" >&2; exit 1; }
reps=${REPS:-6}
names=() bins=()
for spec in "$@"; do names+=("${spec%%=*}"); bins+=("${spec#*=}"); done
# K/V length x queries; one causal sequence, nine sequences, nine sequences after a 96-token shared prefix.
# The -p regex sees the case parameters without the closing parenthesis, hence the $ anchors.
cases=('kv=2048,nb=2048,.*n_seq_diag=1,' 'kv=2048,nb=2048,.*n_seq_diag=9,n_prefix_diag=0$'
       'kv=2304,nb=2048,.*n_seq_diag=9,n_prefix_diag=96$' 'kv=4096,nb=1024,.*n_seq_diag=1,'
       'kv=4096,nb=2048,.*n_seq_diag=1,' 'kv=8192,nb=1024,.*n_seq_diag=1,' 'kv=8192,nb=2048,.*n_seq_diag=1,'
       'kv=32768,nb=1024,.*n_seq_diag=1,' 'kv=32768,nb=2048,.*n_seq_diag=1,')
: > "$log"
for ((rep = 0; rep < reps; rep++)); do
    for re in "${cases[@]}"; do
        for ((i = 0; i < ${#names[@]}; i++)); do
            k=$(((i + rep) % ${#names[@]}))
            out=$("${bins[$k]}" perf -o FLASH_ATTN_EXT -b CUDA0 -p "$re" 2>&1 || true)
            rows=$(sed -nE 's/.*kv=([0-9]+),nb=([0-9]+),.*n_seq_diag=([0-9]+),n_prefix_diag=([0-9]+)\):[^0-9]*[0-9]+ runs - +([0-9.]+) us\/run.*/\1 \2 \3 \4 \5/p' <<< "$out")
            if [ -z "$rows" ]; then
                echo "no result from ${names[$k]} for $re:" >&2
                tail -n 5 <<< "$out" >&2
                exit 1
            fi
            awk -v b="${names[$k]}" -v r="$rep" '{print b, r, $0}' <<< "$rows" >> "$log"
        done
    done
done
python3 -I - "$log" "${names[@]}" <<'EOF'
import random
import statistics
import sys
from collections import defaultdict

us = defaultdict(dict)
for line in open(sys.argv[1]):
    b, rep, kv, nq, nseq, npre, t = line.split()
    us[(int(kv), int(nq), int(nseq), int(npre))][(b, int(rep))] = float(t)
ref, others = sys.argv[2], sys.argv[3:]
rng = random.Random(1)
print(f"change against {ref} in the same repetition: median, 95% CI of the median, repetitions slower")
for key in sorted(us):
    d = us[key]
    reps = sorted({r for (_, r) in d})
    base = statistics.median(d[(ref, r)] for r in reps if (ref, r) in d)
    cells = []
    for name in others:
        ch = [100 * (d[(name, r)] / d[(ref, r)] - 1) for r in reps if (name, r) in d and (ref, r) in d]
        boot = sorted(statistics.median(rng.choices(ch, k=len(ch))) for _ in range(4000))
        low, high = boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot)) - 1]
        cells.append(f"{name} {statistics.median(ch):+6.1f}% [{low:+.1f}, {high:+.1f}] {sum(c > 0 for c in ch)}/{len(ch)}")
    print(f"kv={key[0]:<6} nq={key[1]:<5} seqs={key[2]:<2} prefix={key[3]:<3} {ref} {base:8.1f} us   " + "   ".join(cells))
EOF
