#!/usr/bin/env bash
# Build the patched llama.cpp the pipeline needs: upstream LLAMA_COMMIT + patches/0*.patch, in order.
#   native/build.sh [SRC_DIR]          default SRC_DIR = native/llama.cpp (cloned if missing)
# env: LLAMA_REPO (clone source, default upstream), TARGETS (cmake targets), CMAKE_EXTRA (extra -D flags)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
commit=$(head -1 "$here/LLAMA_COMMIT")
src=${1:-$here/llama.cpp}
repo=${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp}
stamp="$src/.gguf-refine-applied"
# fingerprint of the base commit and the ordered patch contents: a stamp from another patch set is refused
want="$commit $(cat "$here"/patches/0*.patch | sha256sum | cut -d' ' -f1)"
if [ -e "$stamp" ]; then
    [ "$(cat "$stamp")" = "$want" ] || { echo "$src was patched from a different commit/patch set ($(cat "$stamp")); use a fresh SRC_DIR" >&2; exit 1; }
else
    [ -e "$src" ] || git clone "$repo" "$src"
    git -C "$src" rev-parse --git-dir > /dev/null 2>&1 || { echo "$src is not a git checkout" >&2; exit 1; }
    [ -z "$(git -C "$src" status --porcelain --untracked-files=no)" ] || {
        echo "$src has local changes (an interrupted apply leaves them too); refusing to patch: reset it or use a fresh SRC_DIR" >&2; exit 1; }
    git -C "$src" checkout --quiet --detach "$commit"
    for p in "$here"/patches/0*.patch; do
        echo "applying $(basename "$p")"
        git -C "$src" apply --whitespace=nowarn "$p"
    done
    echo "$want" > "$stamp"
fi
targets=${TARGETS:-llama-kld-swap llama-perplexity-fast llama-perplexity llama-imatrix llama-eval-callback llama-quantize test-quantize-cuda}
cmake -S "$src" -B "$src/build" -DGGML_CUDA=ON -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release ${CMAKE_EXTRA:-}
cmake --build "$src/build" -j"$(nproc)" --target $targets 2>&1 | tee "$src/build/build.log"
echo "built into $src/build/bin"
