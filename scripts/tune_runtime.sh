#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
    cat >&2 <<'EOF'
usage: ./scripts/tune_runtime.sh [runs>0] [binary_path] [--] [main.rs args...]

Examples:
  ./scripts/tune_runtime.sh
  ./scripts/tune_runtime.sh 7
  ./scripts/tune_runtime.sh 7 target/release/turbo-life -- --max-threads 4
EOF
}

RUNS="5"
if [ "$#" -gt 0 ] && [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    RUNS="$1"
    shift
fi

BIN="target/release/turbo-life"
if [ "$#" -gt 0 ] && [[ "${1:-}" != -* ]] && [ "${1:-}" != "--" ]; then
    BIN="$1"
    shift
fi

if [ "$#" -gt 0 ] && [ "${1:-}" = "--" ]; then
    shift
fi
PASS_ARGS=("$@")

if ! [[ "$RUNS" =~ ^[0-9]+$ ]] || [ "$RUNS" -eq 0 ]; then
    usage
    exit 2
fi

if [ ! -x "$BIN" ] && [ "$BIN" = "target/release/turbo-life" ]; then
    echo "building release binary: $BIN" >&2
    cargo build --release --bin turbo-life
fi

if [ ! -x "$BIN" ]; then
    echo "binary not found or not executable: $BIN" >&2
    exit 1
fi

bench_candidate() {
    local -a args=("$@")
    if [ "${#PASS_ARGS[@]}" -eq 0 ]; then
        if [ "${#args[@]}" -eq 0 ]; then
            "$ROOT_DIR/scripts/bench_main.sh" "$RUNS" --quiet "$BIN"
        else
            "$ROOT_DIR/scripts/bench_main.sh" "$RUNS" --quiet "$BIN" -- "${args[@]}"
        fi
    else
        if [ "${#args[@]}" -eq 0 ]; then
            "$ROOT_DIR/scripts/bench_main.sh" "$RUNS" --quiet "$BIN" -- "${PASS_ARGS[@]}"
        else
            "$ROOT_DIR/scripts/bench_main.sh" "$RUNS" --quiet "$BIN" -- "${PASS_ARGS[@]}" "${args[@]}"
        fi
    fi
}

declare -a LABELS=()
declare -a MEDIANS=()
declare -a ARG_STRS=()

add_result() {
    local label="$1"
    local median="$2"
    shift 2
    local arg_str="$*"
    LABELS+=("$label")
    MEDIANS+=("$median")
    ARG_STRS+=("$arg_str")
}

best_index() {
    python3 - "$@" <<'PY'
import sys
vals = [float(v) for v in sys.argv[1:]]
best = min(range(len(vals)), key=lambda i: vals[i])
print(best)
PY
}

echo "==> benchmarking runtime candidates ($RUNS runs each)"

default_median="$(bench_candidate)"
add_result "default(auto)" "$default_median"
printf '  %-26s %10s ms\n' "default(auto)" "$default_median"

THREAD_CANDIDATES="${TURBOLIFE_TUNE_THREADS:-2 3 4 5 6 8}"
best_thread_median=""
best_thread=""
for t in $THREAD_CANDIDATES; do
    if ! [[ "$t" =~ ^[0-9]+$ ]] || [ "$t" -eq 0 ]; then
        continue
    fi
    median="$(bench_candidate --threads "$t")"
    add_result "threads=$t" "$median" "--threads $t"
    printf '  %-26s %10s ms\n' "threads=$t" "$median"
    if [ -z "$best_thread_median" ] || \
        python3 - "$median" "$best_thread_median" <<'PY'
import sys
sys.exit(0 if float(sys.argv[1]) < float(sys.argv[2]) else 1)
PY
    then
        best_thread_median="$median"
        best_thread="$t"
    fi
done

ARCH="$(uname -m)"
declare -a KERNELS=()
case "$ARCH" in
    arm64|aarch64)
        KERNELS=(scalar neon)
        ;;
    x86_64|amd64)
        KERNELS=(scalar avx2)
        ;;
    *)
        KERNELS=(scalar)
        ;;
esac

for k in "${KERNELS[@]}"; do
    median="$(bench_candidate --kernel "$k")"
    add_result "kernel=$k" "$median" "--kernel $k"
    printf '  %-26s %10s ms\n' "kernel=$k" "$median"
    if [ -n "$best_thread" ]; then
        median_t="$(bench_candidate --kernel "$k" --threads "$best_thread")"
        add_result "kernel=$k,threads=$best_thread" "$median_t" "--kernel $k --threads $best_thread"
        printf '  %-26s %10s ms\n' "kernel=$k,threads=$best_thread" "$median_t"
    fi
done

best_i="$(best_index "${MEDIANS[@]}")"
echo ""
echo "==> best runtime configuration"
printf '  %-26s %10s ms\n' "${LABELS[$best_i]}" "${MEDIANS[$best_i]}"
if [ -n "${ARG_STRS[$best_i]}" ]; then
    echo "  args: ${ARG_STRS[$best_i]}"
else
    echo "  args: (none)"
fi

echo ""
echo "==> all candidates"
python3 - "${LABELS[@]}" -- "${MEDIANS[@]}" -- "${ARG_STRS[@]}" <<'PY'
import sys

def split_sections(items):
    sections = []
    cur = []
    for item in items:
        if item == "--":
            sections.append(cur)
            cur = []
        else:
            cur.append(item)
    sections.append(cur)
    return sections

labels, medians, args = split_sections(sys.argv[1:])
rows = list(zip(labels, medians, args))
rows.sort(key=lambda row: float(row[1]))
for label, median, arg in rows:
    suffix = f" ({arg})" if arg else ""
    print(f"  {label:26s} {float(median):10.3f} ms{suffix}")
PY
