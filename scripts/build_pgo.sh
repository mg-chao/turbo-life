#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TRAIN_RUNS="${1:-3}"
if [ "$#" -gt 0 ]; then
    shift
fi

if ! [[ "$TRAIN_RUNS" =~ ^[0-9]+$ ]] || [ "$TRAIN_RUNS" -eq 0 ]; then
    echo "usage: $0 [train_runs>0] [main.rs args...]" >&2
    exit 2
fi

HOST_TRIPLE="$(rustc -vV | awk '/^host:/ {print $2}')"
SYSROOT="$(rustc --print sysroot)"
PROFDATA_BIN="$SYSROOT/lib/rustlib/$HOST_TRIPLE/bin/llvm-profdata"

if [ ! -x "$PROFDATA_BIN" ]; then
    if command -v llvm-profdata >/dev/null 2>&1; then
        PROFDATA_BIN="$(command -v llvm-profdata)"
        echo "note: using llvm-profdata from PATH: $PROFDATA_BIN" >&2
    else
        echo "missing llvm-profdata at $PROFDATA_BIN" >&2
        echo "install it with: rustup component add llvm-tools-$HOST_TRIPLE" >&2
        echo "or, on older toolchains: rustup component add llvm-tools-preview" >&2
        exit 1
    fi
fi

PGO_DIR="target/pgo-data"
GEN_TARGET_DIR="target/pgo-gen"
USE_TARGET_DIR="target/pgo-use"
PROFDATA_FILE="$PGO_DIR/turbo-life.profdata"

rm -rf -- "$PGO_DIR" "$GEN_TARGET_DIR" "$USE_TARGET_DIR"
mkdir -p "$PGO_DIR"

echo "==> building instrumented binary"
RUSTFLAGS="-Cprofile-generate=$PGO_DIR -Ctarget-cpu=native" \
    CARGO_TARGET_DIR="$GEN_TARGET_DIR" \
    cargo build --release --bin turbo-life

echo "==> collecting profiles via main.rs harness ($TRAIN_RUNS runs)"
for ((i = 1; i <= TRAIN_RUNS; i++)); do
    echo "  training run $i/$TRAIN_RUNS"
    LLVM_PROFILE_FILE="$PGO_DIR/turbo-life-%p-%m.profraw" \
        "$GEN_TARGET_DIR/release/turbo-life" "$@"
done

echo "==> merging profile data"
"$PROFDATA_BIN" merge -o "$PROFDATA_FILE" "$PGO_DIR"/*.profraw

echo "==> building PGO-optimized binary"
RUSTFLAGS="-Cprofile-use=$PROFDATA_FILE -Ctarget-cpu=native -Cllvm-args=-pgo-warn-missing-function" \
    CARGO_TARGET_DIR="$USE_TARGET_DIR" \
    cargo build --release --bin turbo-life

echo ""
echo "PGO binary ready:"
echo "  $USE_TARGET_DIR/release/turbo-life"
echo ""
echo "Quick benchmark (single run):"
"$USE_TARGET_DIR/release/turbo-life" "$@" | awk '/--- Summary/{in_summary=1} in_summary && /^TurboLife:/{print; exit}'
