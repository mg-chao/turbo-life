#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
    cat >&2 <<USAGE
usage: $0 [runs>0] [max_threads>0] [binary_path] [-- main.rs args...]

Examples:
  $0
  $0 5
  $0 5 12
  $0 5 target/release/turbo-life
  $0 5 10 target/release/turbo-life -- --max-threads 8
USAGE
}

RUNS="3"
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

max_threads_default() {
    if command -v sysctl >/dev/null 2>&1; then
        local v
        v="$(sysctl -n hw.logicalcpu 2>/dev/null || true)"
        if [[ "$v" =~ ^[0-9]+$ ]] && [ "$v" -gt 0 ]; then
            echo "$v"
            return
        fi
    fi
    if command -v nproc >/dev/null 2>&1; then
        local v
        v="$(nproc 2>/dev/null || true)"
        if [[ "$v" =~ ^[0-9]+$ ]] && [ "$v" -gt 0 ]; then
            echo "$v"
            return
        fi
    fi
    echo 8
}

MAX_THREADS="$(max_threads_default)"
BIN="target/release/turbo-life"

if [ "$#" -gt 0 ] && [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    RUNS="$1"
    shift
fi

if [ "$#" -gt 0 ] && [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    MAX_THREADS="$1"
    shift
fi

if [ "$#" -gt 0 ] && [ "${1:-}" != "--" ] && [[ "${1:-}" != -* ]]; then
    BIN="$1"
    shift
fi

if [ "$#" -gt 0 ] && [ "${1:-}" = "--" ]; then
    shift
fi

MAIN_ARGS=("$@")
BENCH_SCRIPT="$ROOT_DIR/scripts/bench_main.sh"

if ! [[ "$RUNS" =~ ^[0-9]+$ ]] || [ "$RUNS" -eq 0 ]; then
    usage
    exit 2
fi

if ! [[ "$MAX_THREADS" =~ ^[0-9]+$ ]] || [ "$MAX_THREADS" -eq 0 ]; then
    usage
    exit 2
fi

bench() {
    local kernel="$1"
    local threads="$2"
    if [ "${#MAIN_ARGS[@]}" -eq 0 ]; then
        "$BENCH_SCRIPT" "$RUNS" "$BIN" --quiet -- --threads "$threads" --kernel "$kernel"
    else
        "$BENCH_SCRIPT" "$RUNS" "$BIN" --quiet -- "${MAIN_ARGS[@]}" --threads "$threads" --kernel "$kernel"
    fi
}

arch="$(uname -m)"
kernels=(scalar)
case "$arch" in
    arm64|aarch64)
        kernels=(neon scalar)
        ;;
    x86_64|amd64)
        kernels=(avx2 scalar)
        ;;
esac

echo "==> baseline (current defaults)"
if [ "${#MAIN_ARGS[@]}" -eq 0 ]; then
    baseline="$("$BENCH_SCRIPT" "$RUNS" "$BIN" --quiet)"
else
    baseline="$("$BENCH_SCRIPT" "$RUNS" "$BIN" --quiet -- "${MAIN_ARGS[@]}")"
fi
echo "baseline median: ${baseline} ms"

echo ""
echo "==> sweeping kernels/threads"
best_median="$baseline"
best_kernel="auto"
best_threads="auto"
for kernel in "${kernels[@]}"; do
    for ((threads = 1; threads <= MAX_THREADS; threads++)); do
        median="$(bench "$kernel" "$threads")"
        printf 'kernel=%-6s threads=%2d -> %s ms\n' "$kernel" "$threads" "$median"
        faster="$(python3 - "$median" "$best_median" <<'PY'
import sys
cand = float(sys.argv[1])
best = float(sys.argv[2])
print("1" if cand < best else "0")
PY
)"
        if [ "$faster" = "1" ]; then
            best_median="$median"
            best_kernel="$kernel"
            best_threads="$threads"
        fi
    done
done

echo ""
python3 - "$baseline" "$best_median" <<'PY'
import sys
base = float(sys.argv[1])
best = float(sys.argv[2])
print(f"baseline_ms: {base:.6f}")
print(f"best_ms:     {best:.6f}")
if base == 0.0:
    print("delta_pct:   n/a (baseline is 0)")
else:
    print(f"delta_pct:   {(base - best) / base * 100.0:+.3f}%")
PY

if [ "$best_kernel" = "auto" ]; then
    echo "best runtime config: keep defaults (auto scheduler remains fastest)"
else
    echo "best runtime config: --threads ${best_threads} --kernel ${best_kernel}"
fi
