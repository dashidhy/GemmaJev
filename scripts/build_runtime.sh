#!/usr/bin/env bash
# Download and build the text scorer; all generated files stay in .runtime/.
set -euo pipefail
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
    printf 'GemmaJev currently supports macOS on Apple Silicon.\n' >&2
    exit 1
fi
project_root="$(cd "$(dirname "$0")/.." && pwd)"
workspace="${GEMMAJEV_WORKSPACE:-$project_root}"
revision="8086439a4cea94c71a5dfb8fe4ad1546aebd640f"
runtime_dir="$workspace/.runtime"
source_dir="$runtime_dir/llama.cpp-text-$revision"
build_dir="$runtime_dir/text-build"
jobs="${GEMMAJEV_BUILD_JOBS:-2}"
if [[ ! "$jobs" =~ ^[1-9][0-9]*$ ]]; then
    printf 'GEMMAJEV_BUILD_JOBS must be a positive integer.\n' >&2
    exit 1
fi
for program in cmake curl tar patch shasum; do
    command -v "$program" >/dev/null || { printf 'Missing build tool: %s\n' "$program" >&2; exit 1; }
done
if [[ ! -f "$source_dir/include/llama.h" ]]; then
    mkdir -p "$runtime_dir"
    unpack_dir="$(mktemp -d "$runtime_dir/unpack.XXXXXX")"
    trap 'rm -rf "$unpack_dir"' EXIT
    if [[ -n "${GEMMAJEV_LLAMA_SOURCE:-}" ]]; then
        actual_revision="$(git -C "$GEMMAJEV_LLAMA_SOURCE" rev-parse HEAD)"
        if [[ "$actual_revision" != "$revision" ]]; then
            printf 'Expected source revision %s, found %s\n' "$revision" "$actual_revision" >&2
            exit 1
        fi
        git -C "$GEMMAJEV_LLAMA_SOURCE" archive "$revision" | tar -x -C "$unpack_dir"
    else
        archive="$runtime_dir/llama.cpp-$revision.tar.gz"
        curl --fail --location --retry 3 --output "$archive" \
            "https://github.com/ggml-org/llama.cpp/archive/$revision.tar.gz"
        tar -xzf "$archive" -C "$unpack_dir" --strip-components=1
    fi
    printf '%s\n' "$revision" > "$unpack_dir/.gemmajev-revision"
    mv "$unpack_dir" "$source_dir"
    trap - EXIT
fi
actual_revision="$(cat "$source_dir/.gemmajev-revision" 2>/dev/null || true)"
if [[ "$actual_revision" != "$revision" ]]; then
    printf 'Invalid private runtime source; remove %s and rebuild.\n' "$source_dir" >&2
    exit 1
fi
patch_file="$project_root/native/patches/gemma4-raw-logits.patch"
patch_hash="$(shasum -a 256 "$patch_file" | awk '{print $1}')"
patch_stamp="$source_dir/.gemmajev-raw-logits.sha256"
if [[ -f "$patch_stamp" && "$(cat "$patch_stamp")" != "$patch_hash" ]]; then
    printf 'Runtime patch changed; remove %s and rebuild.\n' "$source_dir" >&2
    exit 1
fi
if ! (cd "$source_dir" && patch -p1 -f -R --dry-run -i "$patch_file" >/dev/null 2>&1); then
    (cd "$source_dir" && patch -p1 -f --dry-run -i "$patch_file")
    (cd "$source_dir" && patch -p1 -f -i "$patch_file")
fi
printf '%s\n' "$patch_hash" > "$patch_stamp"
cmake -S "$project_root/native" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release -DGEMMAJEV_LLAMA_SOURCE="$source_dir"
cmake --build "$build_dir" --target gemmajev-worker --parallel "$jobs"
# Atomic publication does not overwrite a running worker's executable inode.
mkdir -p "$runtime_dir/build/bin"
install_tmp="$(mktemp "$runtime_dir/build/bin/.gemmajev-worker.XXXXXX")"
trap 'rm -f "$install_tmp"' EXIT
cp "$build_dir/staging/gemmajev-worker" "$install_tmp"
chmod 755 "$install_tmp"
mv -f "$install_tmp" "$runtime_dir/build/bin/gemmajev-worker"
trap - EXIT
printf '\nRuntime built: %s\n' "$runtime_dir/build/bin/gemmajev-worker"
