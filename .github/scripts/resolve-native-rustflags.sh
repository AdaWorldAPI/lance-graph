#!/usr/bin/env bash
# Print RUSTFLAGS whose target features equal `.cargo/config.toml`'s
# `-Ctarget-cpu=native` on this machine, spelled out as an explicit list.
# With --cpu, print the name rustc resolves `native` to instead (for logs).
#
# Why: cargo fingerprints the flag TEXT. `-Ctarget-cpu=native` reads the same
# on every machine, so an artifact built on a CPU with AVX-512 is reused on one
# without and dies with SIGILL. Spelled out, the text names the actual
# features: a different CPU gives different flags, and cargo recompiles.
#
# No `-Ctarget-cpu=<name>`: a named CPU enables its model's default features,
# and a `+feature` list cannot take them back. On a VM that hides AVX-512 from
# a CPU still reporting a Skylake-server model, `-Ctarget-cpu=skylake-avx512`
# would compile AVX-512 back in. The target's baseline CPU plus the exact
# native list has no hidden defaults; the cost is tuning, not correctness.
# Features rustc hides on stable (amx, avx10) are not in the list, so they are
# never enabled; native could have enabled them, this cannot.
#
# RUSTFLAGS replaces every rustflags entry in .cargo/config.toml. That file
# sets only target-cpu, so nothing else is lost; extra flags given as
# arguments are appended.
set -euo pipefail
die() { echo "resolve-native-rustflags: $*" >&2; exit 1; }

if [ "${1:-}" = "--cpu" ]; then
  cpu=$(rustc --print target-cpus | sed -n 's/.*native.*(currently \([^)]*\)).*/\1/p')
  [ -n "$cpu" ] || die "rustc did not report the native CPU"
  echo "$cpu"
  exit 0
fi

target_features() { rustc --print cfg "$@" | sed -n 's/^target_feature="\(.*\)"$/\1/p' | sort; }

native=$(target_features -C target-cpu=native)
[ -n "$native" ] || die "no target features reported"
flags="-Ctarget-feature=$(printf '%s\n' "$native" | sed 's/^/+/' | paste -sd, -)"

# The point of the script: these flags must enable exactly what native does.
# shellcheck disable=SC2086
resolved=$(target_features $flags)
[ "$resolved" = "$native" ] || die "flags do not reproduce native:
$(diff <(printf '%s\n' "$native") <(printf '%s\n' "$resolved") || true)"

echo "$flags${*:+ $*}"
