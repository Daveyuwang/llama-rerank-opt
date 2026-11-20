#!/usr/bin/env bash
# Configure and build one llama.cpp tree with CUDA into LLAMA_CPP_DIR/build.
# usage: scripts/build.sh LLAMA_CPP_DIR [TARGET...]   (default targets: llama-server llama-bench test-backend-ops)
# CUDA_ARCH sets CMAKE_CUDA_ARCHITECTURES (default 120a-real, RTX 50 series), CMAKE_ARGS adds cmake options, JOBS
# sets the build parallelism.
set -euo pipefail
dir=${1:?usage: build.sh LLAMA_CPP_DIR [TARGET...]}
shift
[ $# -gt 0 ] || set -- llama-server llama-bench test-backend-ops
# shellcheck disable=SC2086 # CMAKE_ARGS holds several options
cmake -S "$dir" -B "$dir/build" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-120a-real}" ${CMAKE_ARGS:-}
cmake --build "$dir/build" --target "$@" -j "${JOBS:-$(nproc)}"
