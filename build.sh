#!/usr/bin/env bash
# Build llama-server: fetch the pinned upstream commit, apply the patch, compile with CUDA.
#
# Usage: ./build.sh [--arch ARCHS] [--src DIR] [--build DIR] [--jobs N]
#   --arch   CMAKE_CUDA_ARCHITECTURES, ';'-separated (default: "86;89"; supported: 86, 89)
#   --src    source checkout to create (default: ./work/llama.cpp); must not exist
#   --build  CMake build directory to create (default: ./work/build); must not exist and must not
#            be the same as, inside, or around --src
#   --jobs   parallel compile jobs (default: CPU count from nproc or sysctl)
#
# AMBIENT_ENGINE_UPSTREAM_URL overrides the repository URL in UPSTREAM (for a mirror).
# Requires bash, git, cmake with a build tool, and a CUDA toolkit with nvcc. No GPU is needed.
set -euo pipefail

usage() { sed -n '2,12p' "$0"; }
die_usage() { echo "$1" >&2; usage >&2; exit 2; }

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
archs="86;89"
src="$root/work/llama.cpp"
build="$root/work/build"
jobs=""

while [ $# -gt 0 ]; do
    case "$1" in
        --arch|--src|--build|--jobs)
            [ $# -ge 2 ] && [ -n "$2" ] || die_usage "$1 needs a value"
            case "$1" in
                --arch)  archs="$2" ;;
                --src)   src="$2" ;;
                --build) build="$2" ;;
                --jobs)  jobs="$2" ;;
            esac
            shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die_usage "unknown argument: $1" ;;
    esac
done

if [ -z "$jobs" ]; then
    jobs="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || true)"
    [ -n "$jobs" ] || die_usage "cannot determine the CPU count; pass --jobs N"
fi
case "$jobs" in
    ''|*[!0-9]*) die_usage "--jobs must be a positive integer, got: $jobs" ;;
esac
jobs="$(printf '%s' "$jobs" | sed 's/^0*//')"
[ -n "$jobs" ] || die_usage "--jobs must be a positive integer, got: 0"

# Only the supported architectures.
IFS=';' read -r -a arch_list <<< "$archs"
[ "${#arch_list[@]}" -gt 0 ] || die_usage "--arch needs at least one architecture"
for arch in "${arch_list[@]}"; do
    case "$arch" in
        86|89) ;;
        *) die_usage "unsupported architecture: '$arch' (supported: 86, 89)" ;;
    esac
done
archs="$(IFS=';'; printf '%s' "${arch_list[*]}")"

# Absolute path with symlinks resolved; components that do not exist yet are kept as given.
resolve() {
    local p="$1" rest="" name
    case "$p" in /*) ;; *) p="$PWD/$p" ;; esac
    while [ ! -d "$p" ]; do
        name="$(basename "$p")"
        case "$name" in .|..) die_usage "'.' or '..' after a missing directory in: $1" ;; esac
        rest="/$name$rest"
        p="$(dirname "$p")"
    done
    printf '%s%s\n' "$(cd "$p" && pwd -P)" "$rest"
}
src="$(resolve "$src")" || exit 2
build="$(resolve "$build")" || exit 2
case "$src/" in "$build"/*) die_usage "--src and --build must be separate: $src, $build" ;; esac
case "$build/" in "$src"/*) die_usage "--src and --build must be separate: $src, $build" ;; esac

upstream_url="$(sed -n 's/^url=//p' "$root/UPSTREAM")"
upstream_commit="$(sed -n 's/^commit=//p' "$root/UPSTREAM")"
upstream_url="${AMBIENT_ENGINE_UPSTREAM_URL:-$upstream_url}"
patch="$root/patches/ambient-engine.patch"

# Both directories must be new: a reused checkout or CMake cache can carry other options.
for dir in "$src" "$build"; do
    if [ -e "$dir" ]; then
        echo "$dir already exists; remove it or pass a new --src / --build" >&2
        exit 1
    fi
done

# 1. Fetch exactly the pinned commit.
mkdir -p "$src"
git -C "$src" init -q
git -C "$src" remote add origin "$upstream_url"
git -C "$src" fetch -q --depth 1 origin "$upstream_commit"
git -C "$src" checkout -q --detach FETCH_HEAD
actual="$(git -C "$src" rev-parse HEAD)"
if [ "$actual" != "$upstream_commit" ]; then
    echo "fetched $actual, expected $upstream_commit" >&2
    exit 1
fi

# 2. Apply the patch.
git -C "$src" apply --check "$patch"
git -C "$src" apply "$patch"

# 3. Configure and build in a fresh directory. Every other option keeps its upstream default.
cmake -S "$src" -B "$build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$archs"
cmake --build "$build" --target llama-server --parallel "$jobs"

# 4. Keep the license terms with the binaries.
cp "$root/LICENSE" "$root/NOTICE" "$build/bin/"

echo "llama-server, its shared libraries, LICENSE and NOTICE are in $build/bin"
