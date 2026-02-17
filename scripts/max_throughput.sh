#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
    cat >&2 <<'USAGE'
usage: ./scripts/max_throughput.sh [bench_runs>0] [train_runs>0] [tune_runs>0] [main.rs args...]

Pipeline:
  1) baseline benchmark on main.rs harness
  2) build-time autotuning + PGO candidate search
  3) runtime autotuning (threads/kernel)
  4) final regression gate vs baseline
USAGE
}

BENCH_RUNS="9"
TRAIN_RUNS="3"
TUNE_RUNS="5"

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

if [ "$#" -gt 0 ]; then
    if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
        TUNE_RUNS="$1"
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
if ! [[ "$TUNE_RUNS" =~ ^[0-9]+$ ]] || [ "$TUNE_RUNS" -eq 0 ]; then
    usage
    exit 2
fi

if [ "$#" -gt 0 ] && [ "${1:-}" = "--" ]; then
    shift
fi

HARNESS_ARGS=("$@")

echo "==> baseline benchmark (${BENCH_RUNS} runs)"
if [ "${#HARNESS_ARGS[@]}" -eq 0 ]; then
    BASELINE_MEDIAN="$("$ROOT_DIR/scripts/bench_main.sh" "$BENCH_RUNS" --quiet)"
else
    BASELINE_MEDIAN="$("$ROOT_DIR/scripts/bench_main.sh" "$BENCH_RUNS" --quiet target/release/turbo-life -- "${HARNESS_ARGS[@]}")"
fi
echo "baseline median: ${BASELINE_MEDIAN} ms"

echo ""
echo "==> build-time autotuning + PGO"
BUILD_LOG="$(mktemp)"
TUNE_LOG="$(mktemp)"
trap 'rm -f "$BUILD_LOG" "$TUNE_LOG"' EXIT

if [ "${#HARNESS_ARGS[@]}" -eq 0 ]; then
    "$ROOT_DIR/scripts/build_maxperf.sh" "$BENCH_RUNS" "$TRAIN_RUNS" | tee "$BUILD_LOG"
else
    "$ROOT_DIR/scripts/build_maxperf.sh" "$BENCH_RUNS" "$TRAIN_RUNS" "${HARNESS_ARGS[@]}" | tee "$BUILD_LOG"
fi

BEST_BUILD_MEDIAN="$(awk '/^median:/{print $2; exit}' "$BUILD_LOG")"
BEST_BUILD_NAME="$(awk -F': ' '/^best candidate:/{print $2; exit}' "$BUILD_LOG")"
BEST_BIN="$ROOT_DIR/target/maxperf/best/turbo-life"

if [ -z "$BEST_BUILD_MEDIAN" ] || [ ! -x "$BEST_BIN" ]; then
    echo "failed to resolve best build candidate from build_maxperf output" >&2
    exit 1
fi

echo ""
echo "==> runtime autotuning on best build"
if [ "${#HARNESS_ARGS[@]}" -eq 0 ]; then
    "$ROOT_DIR/scripts/tune_runtime.sh" "$TUNE_RUNS" "$BEST_BIN" | tee "$TUNE_LOG"
else
    "$ROOT_DIR/scripts/tune_runtime.sh" "$TUNE_RUNS" "$BEST_BIN" -- "${HARNESS_ARGS[@]}" | tee "$TUNE_LOG"
fi

BEST_RUNTIME_ARGS_LINE="$(
    awk '
        /^==> best runtime configuration$/ { in_best = 1; next }
        in_best && /^  args:/ { sub(/^  args: /, "", $0); print; exit }
    ' "$TUNE_LOG"
)"
BEST_RUNTIME_LABEL="$(
    awk '
        /^==> best runtime configuration$/ { in_best = 1; next }
        in_best && /^  [^[:space:]]/ {
            line = $0
            sub(/^  /, "", line)
            sub(/[[:space:]]+[0-9]+([.][0-9]+)?[[:space:]]+ms$/, "", line)
            print line
            exit
        }
    ' "$TUNE_LOG"
)"

TUNED_ARGS=()
if [ -n "$BEST_RUNTIME_ARGS_LINE" ] && [ "$BEST_RUNTIME_ARGS_LINE" != "(none)" ]; then
    read -r -a TUNED_ARGS <<< "$BEST_RUNTIME_ARGS_LINE"
fi

echo ""
echo "==> final benchmark gate (${BENCH_RUNS} runs)"
FINAL_ARGS=()
if [ "${#HARNESS_ARGS[@]}" -gt 0 ]; then
    FINAL_ARGS+=("${HARNESS_ARGS[@]}")
fi
if [ "${#TUNED_ARGS[@]}" -gt 0 ]; then
    FINAL_ARGS+=("${TUNED_ARGS[@]}")
fi
if [ "${#FINAL_ARGS[@]}" -eq 0 ]; then
    FINAL_MEDIAN="$("$ROOT_DIR/scripts/bench_main.sh" "$BENCH_RUNS" --quiet "$BEST_BIN")"
else
    FINAL_MEDIAN="$("$ROOT_DIR/scripts/bench_main.sh" "$BENCH_RUNS" --quiet "$BEST_BIN" -- "${FINAL_ARGS[@]}")"
fi

echo "build best candidate: ${BEST_BUILD_NAME} (${BEST_BUILD_MEDIAN} ms)"
if [ -n "$BEST_RUNTIME_LABEL" ]; then
    echo "runtime best candidate: ${BEST_RUNTIME_LABEL}"
fi
echo "final tuned median: ${FINAL_MEDIAN} ms"

echo ""
python3 - "$BASELINE_MEDIAN" "$FINAL_MEDIAN" <<'PY'
import sys
baseline = float(sys.argv[1])
final = float(sys.argv[2])
if baseline <= 0.0:
    print("invalid baseline median from benchmark output", file=sys.stderr)
    sys.exit(1)
improvement = (baseline - final) / baseline * 100.0
print(f"baseline_median_ms: {baseline:.6f}")
print(f"final_median_ms:    {final:.6f}")
print(f"delta_percent:      {improvement:+.3f}%")
if final >= baseline:
    print("Regression detected in final tuned configuration; refusing to accept.", file=sys.stderr)
    sys.exit(1)
PY

echo ""
echo "accepted max-throughput configuration"
if [ "${#FINAL_ARGS[@]}" -eq 0 ]; then
    echo "  binary: $BEST_BIN"
    echo "  run:    $BEST_BIN"
else
    printf '  binary: %s\n' "$BEST_BIN"
    printf '  run:    %q' "$BEST_BIN"
    for arg in "${FINAL_ARGS[@]}"; do
        printf ' %q' "$arg"
    done
    printf '\n'
fi
