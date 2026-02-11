#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
    echo "usage: $0 [runs>0] [--quiet] [binary_path] [--] [turbo-life args...]" >&2
}

RUNS="10"
if [ "$#" -gt 0 ] && [[ "${1:-}" =~ ^-[0-9]+$ ]]; then
    usage
    exit 2
fi
if [ "$#" -gt 0 ] && [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    RUNS="$1"
    shift
fi

BIN_DEFAULT="target/release/turbo-life"
BIN="$BIN_DEFAULT"

QUIET="0"
while [ "$#" -gt 0 ]; do
    case "${1:-}" in
        --quiet)
            QUIET="1"
            shift
            ;;
        --)
            shift
            break
            ;;
        -*)
            break
            ;;
        *)
            if [ "$BIN" != "$BIN_DEFAULT" ]; then
                break
            fi
            BIN="$1"
            shift
            ;;
    esac
done

BIN_ARGS=("$@")

if ! [[ "$RUNS" =~ ^[0-9]+$ ]] || [ "$RUNS" -eq 0 ]; then
    usage
    exit 2
fi

NEEDS_BUILD_DEFAULT="0"
if [ "$BIN" = "$BIN_DEFAULT" ]; then
    if [ ! -x "$BIN" ]; then
        NEEDS_BUILD_DEFAULT="1"
    elif [ Cargo.toml -nt "$BIN" ] || [ Cargo.lock -nt "$BIN" ]; then
        NEEDS_BUILD_DEFAULT="1"
    elif [ -f .cargo/config.toml ] && [ .cargo/config.toml -nt "$BIN" ]; then
        NEEDS_BUILD_DEFAULT="1"
    elif find src -type f -newer "$BIN" -print -quit | grep -q .; then
        NEEDS_BUILD_DEFAULT="1"
    fi
fi

if [ "$NEEDS_BUILD_DEFAULT" = "1" ]; then
    echo "building release binary: $BIN_DEFAULT" >&2
    cargo build --release --bin turbo-life
fi

if [ ! -x "$BIN" ]; then
    echo "binary not found or not executable: $BIN" >&2
    exit 1
fi

tmp_file="$(mktemp)"
trap 'rm -f "$tmp_file"' EXIT

run_once() {
    if [ "${#BIN_ARGS[@]}" -eq 0 ]; then
        "$BIN"
    else
        "$BIN" "${BIN_ARGS[@]}"
    fi
}

for ((i = 1; i <= RUNS; i++)); do
    ms="$(run_once | awk '/--- Summary/{in_summary=1} in_summary && /^TurboLife:/{print $2; exit}')"
    if [ -z "$ms" ]; then
        echo "failed to parse TurboLife total ms from benchmark output" >&2
        exit 1
    fi
    printf '%s\n' "$ms" >> "$tmp_file"
    if [ "$QUIET" != "1" ]; then
        printf 'run %d/%d: TurboLife %s ms\n' "$i" "$RUNS" "$ms"
    fi
done

python3 - "$tmp_file" "$QUIET" <<'PY'
import statistics
import sys

values = [float(line.strip()) for line in open(sys.argv[1], "r", encoding="utf-8") if line.strip()]
values_sorted = sorted(values)
median = statistics.median(values_sorted)

if sys.argv[2] == "1":
    print(f"{median:.6f}")
    sys.exit(0)

print("")
print(f"count:     {len(values)}")
print(f"min:       {values_sorted[0]:.3f} ms")
print(f"median:    {median:.3f} ms")
print(f"mean:      {statistics.mean(values_sorted):.3f} ms")
print(f"max:       {values_sorted[-1]:.3f} ms")
print(f"median_ms: {median:.6f}")
PY
