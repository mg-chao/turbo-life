#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUNS="${1:-10}"
if [ "$#" -gt 0 ]; then
    shift
fi

BIN_DEFAULT="target/release/turbo-life"
BIN="$BIN_DEFAULT"
if [ "$#" -gt 0 ] && [[ "${1:-}" != -* ]]; then
    BIN="$1"
    shift
fi

BIN_ARGS=("$@")

if ! [[ "$RUNS" =~ ^[0-9]+$ ]] || [ "$RUNS" -eq 0 ]; then
    echo "usage: $0 [runs>0] [binary_path] [turbo-life args...]" >&2
    exit 2
fi

if [ ! -x "$BIN" ]; then
    if [ "$BIN" = "$BIN_DEFAULT" ]; then
        echo "building release binary: $BIN_DEFAULT"
        cargo build --release --bin turbo-life
    fi
fi

if [ ! -x "$BIN" ]; then
    echo "binary not found or not executable: $BIN" >&2
    exit 1
fi

tmp_file="$(mktemp)"
trap 'rm -f "$tmp_file"' EXIT

for ((i = 1; i <= RUNS; i++)); do
    ms="$("$BIN" "${BIN_ARGS[@]}" | awk '/--- Summary/{in_summary=1} in_summary && /^TurboLife:/{print $2; exit}')"
    if [ -z "$ms" ]; then
        echo "failed to parse TurboLife total ms from benchmark output" >&2
        exit 1
    fi
    printf '%s\n' "$ms" >> "$tmp_file"
    printf 'run %d/%d: TurboLife %s ms\n' "$i" "$RUNS" "$ms"
done

python3 - "$tmp_file" <<'PY'
import statistics
import sys

values = [float(line.strip()) for line in open(sys.argv[1], "r", encoding="utf-8") if line.strip()]
values_sorted = sorted(values)

print("")
print(f"count:  {len(values)}")
print(f"min:    {values_sorted[0]:.3f} ms")
print(f"median: {statistics.median(values_sorted):.3f} ms")
print(f"mean:   {statistics.mean(values_sorted):.3f} ms")
print(f"max:    {values_sorted[-1]:.3f} ms")
PY
