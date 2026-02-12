#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v perf >/dev/null 2>&1; then
  echo "perf not found. Install linux perf tools first."
  exit 1
fi

THREADS=(1 2 4 8 12)
SIZES=(4096 8192 16384)
DENSITY="0.42"
WARMUP=3

iters_for_size() {
  case "$1" in
    4096) echo 80 ;;
    8192) echo 40 ;;
    16384) echo 20 ;;
    *) echo 30 ;;
  esac
}

echo "=== Build ==="
cargo build --release --bin bench_perf

echo
echo "=== Raw time sweep (bench_perf) ==="
for t in "${THREADS[@]}"; do
  export TURBOLIFE_NUM_THREADS="$t"
  export TURBOLIFE_MAX_THREADS="$t"
  for s in "${SIZES[@]}"; do
    iters="$(iters_for_size "$s")"
    cargo run --release --quiet --bin bench_perf -- \
      --size "$s" --density "$DENSITY" --warmup "$WARMUP" --iters "$iters" --json
  done
done

echo
echo "=== perf stat counters (repeat 7x) ==="
EVENTS="task-clock,cycles,instructions,cache-references,cache-misses,context-switches,cpu-migrations"
for t in "${THREADS[@]}"; do
  export TURBOLIFE_NUM_THREADS="$t"
  export TURBOLIFE_MAX_THREADS="$t"
  for s in "${SIZES[@]}"; do
    iters="$(iters_for_size "$s")"
    echo
    echo "--- threads=$t size=$s ---"
    perf stat -r 7 -e "$EVENTS" \
      cargo run --release --quiet --bin bench_perf -- \
      --size "$s" --density "$DENSITY" --warmup "$WARMUP" --iters "$iters" --json
  done
done

echo
echo "Done. Fill docs/perf-report.md with before/after data and derived metrics."
