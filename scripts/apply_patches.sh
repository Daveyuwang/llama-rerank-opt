#!/usr/bin/env bash
# Apply patches, in the order given, to a llama.cpp checkout at the base commit they were made for.
# usage: scripts/apply_patches.sh LLAMA_CPP_DIR [PATCH...]
# Without PATCH arguments it applies all patches; PATCH arguments replace that list, e.g. patches/0001-*.patch alone.
set -euo pipefail
base=b94959a64d011915d77a5381c7d4846dcda99530
repo=$(cd "$(dirname "$0")/.." && pwd)
dir=${1:?usage: apply_patches.sh LLAMA_CPP_DIR [PATCH...]}
shift
head=$(git -C "$dir" rev-parse HEAD)
if [ "$head" != "$base" ]; then
    echo "$dir is at $head, the patches are for $base" >&2
    exit 1
fi
[ $# -gt 0 ] || set -- "$repo"/patches/*.patch
for patch in "$@"; do
    git -C "$dir" apply "$(realpath "$patch")"
    echo "applied $(basename "$patch")"
done
