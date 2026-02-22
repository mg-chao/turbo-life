#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -eq 0 ]]; then
    echo "rustc_wrapper.sh: missing rustc path argument" >&2
    exit 2
fi

real_rustc="$1"
shift

profile_override="${TURBOLIFE_PROFILE_USE:-}"
disable_auto_pgo="${TURBOLIFE_DISABLE_WRAPPER_PGO:-}"
host_os="$(uname -s)"
host_arch="$(uname -m)"

is_truthy() {
    local value="${1:-}"
    value="$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')"
    case "$value" in
        1|true|yes|on)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

is_falsey() {
    local value="${1:-}"
    value="$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')"
    case "$value" in
        0|false|no|off|none)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

if [[ "$host_os" != "Darwin" || "$host_arch" != "arm64" ]] && [[ -z "$profile_override" ]]; then
    exec "$real_rustc" "$@"
fi

release_like=0
explicit_pgo=0
crate_name=""
target_triple=""
prev=""
for arg in "$@"; do
    case "$arg" in
        -Copt-level=3)
            release_like=1
            ;;
        -Cprofile-generate=*|-Cprofile-use=*)
            explicit_pgo=1
            ;;
        --crate-name=*)
            crate_name="${arg#--crate-name=}"
            ;;
        --target=*)
            target_triple="${arg#--target=}"
            ;;
        opt-level=3)
            if [[ "$prev" == "-C" ]]; then
                release_like=1
            fi
            ;;
        profile-generate=*|profile-use=*)
            if [[ "$prev" == "-C" ]]; then
                explicit_pgo=1
            fi
            ;;
        *)
            if [[ "$prev" == "--crate-name" ]]; then
                crate_name="$arg"
            elif [[ "$prev" == "--target" ]]; then
                target_triple="$arg"
            fi
            ;;
    esac
    prev="$arg"
done

if [[ "$release_like" -eq 1 && "$explicit_pgo" -eq 0 && "$crate_name" == "turbo_life" ]]; then
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    root_dir="$(cd "$script_dir/.." && pwd)"
    profdata=""

    if [[ -n "$profile_override" ]]; then
        if is_falsey "$profile_override"; then
            exec "$real_rustc" "$@"
        fi
        if [[ "$profile_override" = /* ]]; then
            profdata="$profile_override"
        else
            profdata="$root_dir/$profile_override"
        fi
    elif [[ "$host_os" == "Darwin" && "$host_arch" == "arm64" ]] \
        && [[ -z "$target_triple" || "$target_triple" == "aarch64-apple-darwin" ]] \
        && ! is_truthy "$disable_auto_pgo"; then
        profdata="$root_dir/pgo/turbo-life-main.profdata"
    fi

    if [[ -n "$profdata" && -f "$profdata" ]]; then
        exec "$real_rustc" "$@" \
            "-Cprofile-use=$profdata" \
            "-Cllvm-args=-pgo-warn-missing-function=false"
    fi
fi

exec "$real_rustc" "$@"
