#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -eq 0 ]]; then
    echo "rustc_wrapper.sh: missing rustc path argument" >&2
    exit 2
fi

real_rustc="$1"
shift

if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
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

    if [[ -z "$target_triple" || "$target_triple" == "aarch64-apple-darwin" ]]; then
        use_profile_target=1
    else
        use_profile_target=0
    fi

    if [[ "$release_like" -eq 1 && "$explicit_pgo" -eq 0 && "$crate_name" == "turbo_life" && "$use_profile_target" -eq 1 ]]; then
        script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        root_dir="$(cd "$script_dir/.." && pwd)"
        profdata="$root_dir/pgo/turbo-life-main.profdata"
        if [[ -f "$profdata" ]]; then
            exec "$real_rustc" "$@" \
                "-Cprofile-use=$profdata" \
                "-Cllvm-args=-pgo-warn-missing-function=false"
        fi
    fi
fi

exec "$real_rustc" "$@"
