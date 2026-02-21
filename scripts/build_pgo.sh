#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
    echo "usage: $0 [train_runs>0] [bench_runs>0] [training_mode: synthetic|checked|both] [main.rs args...]" >&2
}

TRAIN_RUNS="3"
BENCH_RUNS="9"
TRAINING_MODE="both"
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

if [ "$#" -gt 0 ]; then
    case "${1:-}" in
        synthetic|checked|both)
            TRAINING_MODE="$1"
            shift
            ;;
        --|-*)
            ;;
        *)
            echo "error: unknown training mode '$1' (expected synthetic, checked, or both)." >&2
            usage
            exit 2
            ;;
    esac
fi

if ! [[ "$TRAIN_RUNS" =~ ^[0-9]+$ ]] || [ "$TRAIN_RUNS" -eq 0 ]; then
    usage
    exit 2
fi
if ! [[ "$BENCH_RUNS" =~ ^[0-9]+$ ]] || [ "$BENCH_RUNS" -eq 0 ]; then
    usage
    exit 2
fi

if [ "$#" -gt 0 ] && [ "${1:-}" = "--" ]; then
    shift
fi

HARNESS_ARGS=("$@")

has_harness_args() {
    [ "${#HARNESS_ARGS[@]}" -gt 0 ]
}

if has_harness_args; then
    for arg in "${HARNESS_ARGS[@]}"; do
        if [ "$arg" = "--pgo-train" ]; then
            echo "error: do not pass --pgo-train explicitly; build_pgo.sh controls training mode." >&2
            exit 2
        fi
    done
fi

HOST_TRIPLE="$(rustc -vV | awk '/^host:/ {print $2}')"
SYSROOT="$(rustc --print sysroot)"
PROFDATA_BIN="$SYSROOT/lib/rustlib/$HOST_TRIPLE/bin/llvm-profdata"
BASE_RUSTFLAGS="-Ctarget-cpu=native -Cllvm-args=-unroll-threshold=300"
case "$HOST_TRIPLE" in
    aarch64-apple-darwin)
        BASE_RUSTFLAGS+=" -Clink-arg=-Wl,-dead_strip -Clink-arg=-Wl,-dead_strip_dylibs"
        ;;
    x86_64-pc-windows-msvc)
        BASE_RUSTFLAGS+=" -Clink-arg=/OPT:REF -Clink-arg=/OPT:ICF"
        ;;
esac

if [ ! -x "$PROFDATA_BIN" ]; then
    if command -v llvm-profdata >/dev/null 2>&1; then
        PROFDATA_BIN="$(command -v llvm-profdata)"
        echo "note: using llvm-profdata from PATH: $PROFDATA_BIN" >&2
    elif command -v xcrun >/dev/null 2>&1 && xcrun --find llvm-profdata >/dev/null 2>&1; then
        PROFDATA_BIN="$(xcrun --find llvm-profdata)"
        echo "note: using llvm-profdata from xcrun: $PROFDATA_BIN" >&2
    else
        echo "missing llvm-profdata at $PROFDATA_BIN" >&2
        echo "install it with: rustup component add llvm-tools-$HOST_TRIPLE" >&2
        echo "or, on older toolchains: rustup component add llvm-tools-preview" >&2
        exit 1
    fi
fi

run_benchmark_report() {
    local runs="$1"
    local bin="$2"
    shift 2
    if [ "$#" -eq 0 ]; then
        "$ROOT_DIR/scripts/bench_main.sh" "$runs" "$bin"
    else
        "$ROOT_DIR/scripts/bench_main.sh" "$runs" "$bin" -- "$@"
    fi
}

run_benchmark_with_harness_args() {
    local runs="$1"
    local bin="$2"
    if has_harness_args; then
        run_benchmark_report "$runs" "$bin" "${HARNESS_ARGS[@]}"
    else
        run_benchmark_report "$runs" "$bin"
    fi
}

extract_median_ms() {
    awk '/^median_ms:/{print $2; exit}'
}

build_release_bin() {
    local target_dir="$1"
    local rustflags="$2"
    local wrapper_profile_use="${3:-}"

    rm -rf -- "$target_dir"
    if [ -n "$wrapper_profile_use" ]; then
        RUSTFLAGS="$rustflags" TURBOLIFE_PROFILE_USE="$wrapper_profile_use" CARGO_TARGET_DIR="$target_dir" cargo build --release --bin turbo-life
    else
        RUSTFLAGS="$rustflags" CARGO_TARGET_DIR="$target_dir" cargo build --release --bin turbo-life
    fi
}

run_training_once() {
    local mode="$1"
    local bin="$2"

    case "$mode" in
        synthetic)
            if has_harness_args; then
                LLVM_PROFILE_FILE="$TRAIN_DATA_DIR/turbo-life-%p-%m.profraw" \
                    "$bin" --pgo-train "${HARNESS_ARGS[@]}"
            else
                LLVM_PROFILE_FILE="$TRAIN_DATA_DIR/turbo-life-%p-%m.profraw" \
                    "$bin" --pgo-train
            fi
            ;;
        checked)
            if has_harness_args; then
                LLVM_PROFILE_FILE="$TRAIN_DATA_DIR/turbo-life-%p-%m.profraw" \
                    "$bin" "${HARNESS_ARGS[@]}"
            else
                LLVM_PROFILE_FILE="$TRAIN_DATA_DIR/turbo-life-%p-%m.profraw" \
                    "$bin"
            fi
            ;;
        *)
            echo "internal error: unknown training mode '$mode'" >&2
            exit 1
            ;;
    esac
}

merge_training_profile() {
    local data_dir="$1"
    local out_file="$2"

    local profraw_files=()
    while IFS= read -r file; do
        profraw_files+=("$file")
    done < <(find "$data_dir" -maxdepth 1 -name '*.profraw' -type f | sort)
    if [ "${#profraw_files[@]}" -eq 0 ]; then
        echo "no profile data generated in $data_dir" >&2
        return 1
    fi
    "$PROFDATA_BIN" merge -o "$out_file" "${profraw_files[@]}"
}

BASELINE_TARGET_DIR="$ROOT_DIR/target/pgo-baseline"
BASELINE_BIN="$BASELINE_TARGET_DIR/release/turbo-life"

echo "==> building baseline release binary"
build_release_bin "$BASELINE_TARGET_DIR" "$BASE_RUSTFLAGS"

echo "==> baseline benchmark via main.rs harness (${BENCH_RUNS} runs)"
BASELINE_REPORT="$(run_benchmark_with_harness_args "$BENCH_RUNS" "$BASELINE_BIN")"
echo "$BASELINE_REPORT"
BASELINE_MEDIAN="$(printf '%s\n' "$BASELINE_REPORT" | extract_median_ms)"
if [ -z "$BASELINE_MEDIAN" ]; then
    echo "failed to parse baseline median" >&2
    exit 1
fi

RESULTS_FILE="$(mktemp)"
trap 'rm -f "$RESULTS_FILE"' EXIT
printf '%s\t%s\t%s\n' "baseline" "$BASELINE_MEDIAN" "$BASELINE_BIN" >> "$RESULTS_FILE"

BEST_NAME="baseline"
BEST_MEDIAN="$BASELINE_MEDIAN"
BEST_BIN="$BASELINE_BIN"

case "$TRAINING_MODE" in
    both)
        TRAINING_MODES=(synthetic checked)
        ;;
    synthetic)
        TRAINING_MODES=(synthetic)
        ;;
    checked)
        TRAINING_MODES=(checked)
        ;;
    *)
        echo "unknown training mode: $TRAINING_MODE" >&2
        exit 2
        ;;
esac

evaluate_mode() {
    local mode="$1"
    local data_dir="$ROOT_DIR/target/pgo-${mode}-data"
    local gen_target_dir="$ROOT_DIR/target/pgo-${mode}-gen"
    local use_target_dir="$ROOT_DIR/target/pgo-${mode}-use"
    local gen_bin="$gen_target_dir/release/turbo-life"
    local use_bin="$use_target_dir/release/turbo-life"
    local profdata_file="$data_dir/turbo-life.profdata"

    echo ""
    echo "==> candidate (${mode} training): build instrumented binary"
    rm -rf -- "$data_dir"
    mkdir -p "$data_dir"
    build_release_bin "$gen_target_dir" "$BASE_RUSTFLAGS -Cprofile-generate=$data_dir"

    echo "==> collecting ${mode} PGO profiles (${TRAIN_RUNS} runs)"
    TRAIN_DATA_DIR="$data_dir"
    for ((i = 1; i <= TRAIN_RUNS; i++)); do
        echo "  training run $i/$TRAIN_RUNS"
        run_training_once "$mode" "$gen_bin"
    done

    echo "==> merging profile data (${mode})"
    merge_training_profile "$data_dir" "$profdata_file"

    echo "==> building PGO-optimized binary (${mode})"
    build_release_bin "$use_target_dir" "$BASE_RUSTFLAGS" "$profdata_file"

    echo "==> ${mode} PGO benchmark via main.rs harness (${BENCH_RUNS} runs)"
    local report
    report="$(run_benchmark_with_harness_args "$BENCH_RUNS" "$use_bin")"
    echo "$report"

    local median
    median="$(printf '%s\n' "$report" | extract_median_ms)"
    if [ -z "$median" ]; then
        echo "failed to parse ${mode} median" >&2
        exit 1
    fi
    printf '%s\t%s\t%s\n' "pgo-${mode}" "$median" "$use_bin" >> "$RESULTS_FILE"

    local better
    better="$(python3 - "$median" "$BEST_MEDIAN" <<'PY'
import sys
cand = float(sys.argv[1])
best = float(sys.argv[2])
print("1" if cand < best else "0")
PY
)"
    if [ "$better" = "1" ]; then
        BEST_NAME="pgo-${mode}"
        BEST_MEDIAN="$median"
        BEST_BIN="$use_bin"
    fi
}

for mode in "${TRAINING_MODES[@]}"; do
    evaluate_mode "$mode"
done

echo ""
echo "==> results (median ms)"
python3 - "$RESULTS_FILE" <<'PY'
import sys
rows = []
with open(sys.argv[1], encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        name, median, bin_path = line.split("\t", 2)
        rows.append((name, float(median), bin_path))
rows.sort(key=lambda row: row[1])
for name, median, _ in rows:
    print(f"{name:16s} {median:9.3f} ms")
PY

python3 - "$BASELINE_MEDIAN" "$BEST_MEDIAN" <<'PY'
import sys
baseline = float(sys.argv[1])
best = float(sys.argv[2])
if baseline <= 0.0 or best <= 0.0:
    print("invalid benchmark medians; expected positive milliseconds.", file=sys.stderr)
    sys.exit(1)
improvement = (baseline - best) / baseline * 100.0
print("")
print(f"baseline_median_ms: {baseline:.6f}")
print(f"best_median_ms:     {best:.6f}")
print(f"delta_percent:      {improvement:+.3f}%")
if best >= baseline:
    print("PGO regression detected; refusing to accept this build.", file=sys.stderr)
    sys.exit(1)
PY

BEST_DEST_DIR="$ROOT_DIR/target/pgo-best"
mkdir -p "$BEST_DEST_DIR"
cp "$BEST_BIN" "$BEST_DEST_DIR/turbo-life"

echo ""
echo "best candidate: $BEST_NAME"
echo "median: ${BEST_MEDIAN} ms"
echo "binary: $BEST_DEST_DIR/turbo-life"
