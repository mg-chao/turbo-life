#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
    echo "usage: $0 [train_runs>0] [bench_runs>0] [main.rs args...]" >&2
}

TRAIN_RUNS="3"
BENCH_RUNS="9"
if [ "$#" -gt 0 ]; then
    if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
        TRAIN_RUNS="$1"
        shift
    elif [[ "${1:-}" =~ ^-?[0-9]+$ ]] || [[ "${1:-}" != -* ]]; then
        usage
        exit 2
    fi
fi

if [ "$#" -gt 0 ]; then
    if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
        BENCH_RUNS="$1"
        shift
    elif [[ "${1:-}" =~ ^-?[0-9]+$ ]] || [[ "${1:-}" != -* ]]; then
        usage
        exit 2
    fi
fi

if ! [[ "$TRAIN_RUNS" =~ ^[0-9]+$ ]] || [ "$TRAIN_RUNS" -eq 0 ]; then
    usage
    exit 2
fi
if ! [[ "$BENCH_RUNS" =~ ^[0-9]+$ ]] || [ "$BENCH_RUNS" -eq 0 ]; then
    usage
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

PGO_DIR="$ROOT_DIR/target/pgo-data"
GEN_TARGET_DIR="$ROOT_DIR/target/pgo-gen"
USE_TARGET_DIR="$ROOT_DIR/target/pgo-use"
BASELINE_TARGET_DIR="$ROOT_DIR/target/pgo-baseline"
PROFDATA_FILE="$PGO_DIR/turbo-life.profdata"
BASELINE_BIN="$BASELINE_TARGET_DIR/release/turbo-life"
PGO_BIN="$USE_TARGET_DIR/release/turbo-life"

run_benchmark_report() {
    local runs="$1"
    local bin="$2"
    shift 2
    "$ROOT_DIR/scripts/bench_main.sh" "$runs" "$bin" "$@"
}

extract_median_ms() {
    awk '/^median_ms:/{print $2; exit}'
}

echo "==> building baseline release binary"
rm -rf -- "$PGO_DIR" "$GEN_TARGET_DIR" "$USE_TARGET_DIR" "$BASELINE_TARGET_DIR"
mkdir -p "$PGO_DIR"

RUSTFLAGS="-Ctarget-cpu=native" \
    CARGO_TARGET_DIR="$BASELINE_TARGET_DIR" \
    cargo build --release --bin turbo-life

echo "==> baseline benchmark via main.rs harness (${BENCH_RUNS} runs)"
BASELINE_REPORT="$(run_benchmark_report "$BENCH_RUNS" "$BASELINE_BIN" "$@")"
echo "$BASELINE_REPORT"
BASELINE_MEDIAN="$(printf '%s\n' "$BASELINE_REPORT" | extract_median_ms)"
if [ -z "$BASELINE_MEDIAN" ]; then
    echo "failed to parse baseline median" >&2
    exit 1
fi

echo "==> building instrumented binary"
RUSTFLAGS="-Cprofile-generate=$PGO_DIR -Ctarget-cpu=native" \
    CARGO_TARGET_DIR="$GEN_TARGET_DIR" \
    cargo build --release --bin turbo-life

echo "==> collecting PGO profiles via main.rs harness (${TRAIN_RUNS} runs)"
for ((i = 1; i <= TRAIN_RUNS; i++)); do
    echo "  training run $i/$TRAIN_RUNS"
    LLVM_PROFILE_FILE="$PGO_DIR/turbo-life-%p-%m.profraw" \
        "$GEN_TARGET_DIR/release/turbo-life" "$@"
done

echo "==> merging profile data"
shopt -s nullglob
PROFRAW_FILES=("$PGO_DIR"/*.profraw)
shopt -u nullglob
if [ "${#PROFRAW_FILES[@]}" -eq 0 ]; then
    echo "no profile data generated in $PGO_DIR" >&2
    exit 1
fi
"$PROFDATA_BIN" merge -o "$PROFDATA_FILE" "${PROFRAW_FILES[@]}"

echo "==> building PGO-optimized binary"
RUSTFLAGS="-Cprofile-use=$PROFDATA_FILE -Ctarget-cpu=native" \
    CARGO_TARGET_DIR="$USE_TARGET_DIR" \
    cargo build --release --bin turbo-life

echo "==> PGO benchmark via main.rs harness (${BENCH_RUNS} runs)"
PGO_REPORT="$(run_benchmark_report "$BENCH_RUNS" "$PGO_BIN" "$@")"
echo "$PGO_REPORT"
PGO_MEDIAN="$(printf '%s\n' "$PGO_REPORT" | extract_median_ms)"
if [ -z "$PGO_MEDIAN" ]; then
    echo "failed to parse PGO median" >&2
    exit 1
fi

python3 - "$BASELINE_MEDIAN" "$PGO_MEDIAN" <<'PY'
import sys

baseline = float(sys.argv[1])
pgo = float(sys.argv[2])

if baseline <= 0.0 or pgo <= 0.0:
    print("invalid benchmark medians; expected positive milliseconds.", file=sys.stderr)
    sys.exit(1)

improvement = (baseline - pgo) / baseline * 100.0

print("")
print(f"baseline_median_ms: {baseline:.6f}")
print(f"pgo_median_ms:      {pgo:.6f}")
print(f"delta_percent:      {improvement:+.3f}%")

if pgo >= baseline:
    print("PGO regression detected; refusing to accept this build.", file=sys.stderr)
    sys.exit(1)
PY

echo ""
echo "PGO binary ready:"
echo "  $PGO_BIN"
