#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
    echo "usage: $0 [bench_runs>0] [train_runs>0] [main.rs args...]" >&2
}

BENCH_RUNS="7"
TRAIN_RUNS="3"
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
    if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
        TRAIN_RUNS="$1"
        shift
    elif [[ "${1:-}" =~ ^-?[0-9]+$ ]] || [[ "${1:-}" != -* ]]; then
        usage
        exit 2
    fi
fi

if ! [[ "$BENCH_RUNS" =~ ^[0-9]+$ ]] || [ "$BENCH_RUNS" -eq 0 ]; then
    usage
    exit 2
fi
if ! [[ "$TRAIN_RUNS" =~ ^[0-9]+$ ]] || [ "$TRAIN_RUNS" -eq 0 ]; then
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
            echo "error: do not pass --pgo-train explicitly; build_maxperf.sh injects it for PGO training runs." >&2
            exit 2
        fi
    done
fi

HOST_TRIPLE="$(rustc -vV | awk '/^host:/ {print $2}')"
SYSROOT="$(rustc --print sysroot)"
PROFDATA_BIN="$SYSROOT/lib/rustlib/$HOST_TRIPLE/bin/llvm-profdata"
NATIVE_RUSTFLAGS="-Ctarget-cpu=native"
case "$HOST_TRIPLE" in
    aarch64-apple-darwin)
        NATIVE_RUSTFLAGS+=" -Clink-arg=-Wl,-dead_strip -Clink-arg=-Wl,-dead_strip_dylibs"
        ;;
    x86_64-pc-windows-msvc)
        NATIVE_RUSTFLAGS+=" -Clink-arg=/OPT:REF -Clink-arg=/OPT:ICF"
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

RESULTS_FILE="$(mktemp)"
trap 'rm -f "$RESULTS_FILE"' EXIT

bench_binary() {
    local runs="$1"
    local bin="$2"
    shift 2
    if has_harness_args; then
        "$ROOT_DIR/scripts/bench_main.sh" "$runs" "$bin" --quiet -- "${HARNESS_ARGS[@]}" "$@"
    else
        "$ROOT_DIR/scripts/bench_main.sh" "$runs" "$bin" --quiet "$@"
    fi
}

build_bin() {
    local target_dir="$1"
    local rustflags="$2"
    local lto="$3"
    local features="$4"

    rm -rf -- "$target_dir"

    local -a cmd=(cargo build --release --bin turbo-life)
    if [ -n "$features" ]; then
        cmd+=(--features "$features")
    fi

    if [ -n "$rustflags" ] && [ -n "$lto" ]; then
        RUSTFLAGS="$rustflags" CARGO_PROFILE_RELEASE_LTO="$lto" CARGO_TARGET_DIR="$target_dir" "${cmd[@]}"
    elif [ -n "$rustflags" ]; then
        RUSTFLAGS="$rustflags" CARGO_TARGET_DIR="$target_dir" "${cmd[@]}"
    elif [ -n "$lto" ]; then
        CARGO_PROFILE_RELEASE_LTO="$lto" CARGO_TARGET_DIR="$target_dir" "${cmd[@]}"
    else
        CARGO_TARGET_DIR="$target_dir" "${cmd[@]}"
    fi
}

accept_if_faster() {
    local candidate_name="$1"
    local candidate_median="$2"
    local candidate_bin="$3"
    local candidate_rustflags="$4"
    local candidate_lto="$5"
    local candidate_features="$6"

    local faster="0"
    faster="$(python3 - "$candidate_median" "$BEST_MEDIAN" <<'PY'
import sys
cand = float(sys.argv[1])
best = float(sys.argv[2])
print("1" if cand < best else "0")
PY
)"

    if [ "$faster" = "1" ]; then
        BEST_NAME="$candidate_name"
        BEST_MEDIAN="$candidate_median"
        BEST_BIN="$candidate_bin"
        BEST_RUSTFLAGS="$candidate_rustflags"
        BEST_LTO="$candidate_lto"
        BEST_FEATURES="$candidate_features"
        echo "accepted: $candidate_name (${candidate_median} ms)"
    else
        echo "rejected: $candidate_name (${candidate_median} ms; not faster than ${BEST_MEDIAN} ms)"
    fi
}

run_candidate() {
    local name="$1"
    local rustflags="$2"
    local lto="$3"
    local features="$4"

    local target_dir="$ROOT_DIR/target/maxperf/$name"
    local bin="$target_dir/release/turbo-life"

    echo ""
    echo "==> candidate: $name"
    if ! build_bin "$target_dir" "$rustflags" "$lto" "$features"; then
        echo "warning: skipped candidate '$name' (build failed)" >&2
        return 0
    fi
    local median
    if ! median="$(bench_binary "$BENCH_RUNS" "$bin")"; then
        echo "warning: skipped candidate '$name' (benchmark failed)" >&2
        return 0
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$median" "$rustflags" "$lto" "$features" >> "$RESULTS_FILE"
    accept_if_faster "$name" "$median" "$bin" "$rustflags" "$lto" "$features"
}

# Baseline candidate (project defaults)
BASE_NAME="baseline"
BASE_TARGET_DIR="$ROOT_DIR/target/maxperf/$BASE_NAME"
BASE_BIN="$BASE_TARGET_DIR/release/turbo-life"

echo "==> baseline build"
build_bin "$BASE_TARGET_DIR" "" "" ""
BASE_MEDIAN="$(bench_binary "$BENCH_RUNS" "$BASE_BIN")"
printf '%s\t%s\t%s\t%s\t%s\n' "$BASE_NAME" "$BASE_MEDIAN" "" "" "" >> "$RESULTS_FILE"

echo "baseline median: ${BASE_MEDIAN} ms"

BEST_NAME="$BASE_NAME"
BEST_MEDIAN="$BASE_MEDIAN"
BEST_BIN="$BASE_BIN"
BEST_RUSTFLAGS=""
BEST_LTO=""
BEST_FEATURES=""

# Build-config candidates
run_candidate "lto-thin" "" "thin" ""
run_candidate "lto-off" "" "off" ""
run_candidate "llvm-unroll300" "$NATIVE_RUSTFLAGS -Cllvm-args=-unroll-threshold=300" "" ""
run_candidate "llvm-inline275" "$NATIVE_RUSTFLAGS -Cllvm-args=-inline-threshold=275" "" ""

if [[ "$HOST_TRIPLE" == aarch64-* || "$HOST_TRIPLE" == arm64-* ]]; then
    run_candidate "feat-prefetch" "" "" "aggressive-prefetch-aarch64"
fi

# Optional PGO over the current best non-PGO config.
PGO_DATA_DIR="$ROOT_DIR/target/maxperf/pgo-data"
PGO_GEN_DIR="$ROOT_DIR/target/maxperf/pgo-gen"
PGO_USE_DIR="$ROOT_DIR/target/maxperf/pgo-use"
PGO_PROFDATA="$PGO_DATA_DIR/turbo-life.profdata"

rm -rf -- "$PGO_DATA_DIR"
mkdir -p "$PGO_DATA_DIR"

echo ""
echo "==> candidate: pgo-on-${BEST_NAME}"

pgo_gen_rustflags="${BEST_RUSTFLAGS}"
if [ -n "$pgo_gen_rustflags" ]; then
    pgo_gen_rustflags+=" "
fi
pgo_gen_rustflags+="-Cprofile-generate=$PGO_DATA_DIR"

PGO_CANDIDATE_READY="1"
if ! build_bin "$PGO_GEN_DIR" "$pgo_gen_rustflags" "$BEST_LTO" "$BEST_FEATURES"; then
    echo "warning: skipped PGO candidate (instrumented build failed)" >&2
    PGO_CANDIDATE_READY="0"
fi

if [ "$PGO_CANDIDATE_READY" = "1" ]; then
    for ((i = 1; i <= TRAIN_RUNS; i++)); do
        echo "  training run $i/$TRAIN_RUNS"
        if has_harness_args; then
            if ! LLVM_PROFILE_FILE="$PGO_DATA_DIR/turbo-life-%p-%m.profraw" \
                "$PGO_GEN_DIR/release/turbo-life" --pgo-train "${HARNESS_ARGS[@]}" >/dev/null; then
                echo "warning: skipped PGO candidate (training run $i failed)" >&2
                PGO_CANDIDATE_READY="0"
                break
            fi
        else
            if ! LLVM_PROFILE_FILE="$PGO_DATA_DIR/turbo-life-%p-%m.profraw" \
                "$PGO_GEN_DIR/release/turbo-life" --pgo-train >/dev/null; then
                echo "warning: skipped PGO candidate (training run $i failed)" >&2
                PGO_CANDIDATE_READY="0"
                break
            fi
        fi
    done
fi

if [ "$PGO_CANDIDATE_READY" = "1" ]; then
    shopt -s nullglob
    PROFRAW_FILES=("$PGO_DATA_DIR"/*.profraw)
    shopt -u nullglob
    if [ "${#PROFRAW_FILES[@]}" -eq 0 ]; then
        echo "warning: skipped PGO candidate (no profile data generated)"
        PGO_CANDIDATE_READY="0"
    elif ! "$PROFDATA_BIN" merge -o "$PGO_PROFDATA" "${PROFRAW_FILES[@]}"; then
        echo "warning: skipped PGO candidate (profile merge failed)" >&2
        PGO_CANDIDATE_READY="0"
    fi
fi

if [ "$PGO_CANDIDATE_READY" = "1" ]; then
    pgo_use_rustflags="${BEST_RUSTFLAGS}"
    if [ -n "$pgo_use_rustflags" ]; then
        pgo_use_rustflags+=" "
    fi
    pgo_use_rustflags+="-Cprofile-use=$PGO_PROFDATA"

    if ! build_bin "$PGO_USE_DIR" "$pgo_use_rustflags" "$BEST_LTO" "$BEST_FEATURES"; then
        echo "warning: skipped PGO candidate (optimized build failed)" >&2
        PGO_CANDIDATE_READY="0"
    fi
fi

if [ "$PGO_CANDIDATE_READY" = "1" ]; then
    PGO_BIN="$PGO_USE_DIR/release/turbo-life"
    if ! PGO_MEDIAN="$(bench_binary "$BENCH_RUNS" "$PGO_BIN")"; then
        echo "warning: skipped PGO candidate (benchmark failed)" >&2
    else
        printf '%s\t%s\t%s\t%s\t%s\n' "pgo-on-$BEST_NAME" "$PGO_MEDIAN" "$pgo_use_rustflags" "$BEST_LTO" "$BEST_FEATURES" >> "$RESULTS_FILE"
        accept_if_faster "pgo-on-$BEST_NAME" "$PGO_MEDIAN" "$PGO_BIN" "$pgo_use_rustflags" "$BEST_LTO" "$BEST_FEATURES"
    fi
fi

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
        parts = line.split("\t", 4)
        if len(parts) != 5:
            raise ValueError(f"invalid result row: {line!r}")
        name, median, rustflags, lto, features = parts
        rows.append((name, float(median), rustflags, lto, features))
rows.sort(key=lambda r: r[1])
for name, median, rustflags, lto, features in rows:
    extras = []
    if lto:
        extras.append(f"lto={lto}")
    if features:
        extras.append(f"features={features}")
    if rustflags:
        extras.append("rustflags")
    suffix = ""
    if extras:
        suffix = " [" + ", ".join(extras) + "]"
    print(f"{name:28s} {median:9.3f} ms{suffix}")
PY

BEST_DEST_DIR="$ROOT_DIR/target/maxperf/best"
mkdir -p "$BEST_DEST_DIR"
cp "$BEST_BIN" "$BEST_DEST_DIR/turbo-life"

echo ""
echo "best candidate: $BEST_NAME"
echo "median: ${BEST_MEDIAN} ms"
echo "binary: $BEST_DEST_DIR/turbo-life"
