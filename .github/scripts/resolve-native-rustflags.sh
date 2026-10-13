#!/usr/bin/env bash
# Print RUSTFLAGS equivalent to `.cargo/config.toml`'s `-Ctarget-cpu=native`,
# with `native` resolved to this machine's CPU name and exact feature list.
#
# Why: cargo fingerprints the flag TEXT. `-Ctarget-cpu=native` reads the same
# on every machine, so an artifact built on a CPU with AVX-512 is reused on one
# without and dies with SIGILL. Resolved, the text names the actual features:
# a different CPU gives different flags, and cargo recompiles by itself.
#
# The feature list, not just the CPU name, is what matters: a hypervisor can
# hide features from a CPU that still reports the same model name.
#
# RUSTFLAGS replaces every rustflags entry in .cargo/config.toml. That file
# sets only target-cpu, so nothing else is lost; extra flags given as
# arguments are appended.
set -euo pipefail
cpu=$(rustc --print target-cpus | sed -n 's/.*native.*(currently \([^)]*\)).*/\1/p')
[ -n "$cpu" ] || { echo "resolve-native-rustflags: rustc did not report the native CPU" >&2; exit 1; }
features=$(rustc --print cfg -C target-cpu=native \
  | sed -n 's/^target_feature="\(.*\)"$/+\1/p' | sort | paste -sd, -)
[ -n "$features" ] || { echo "resolve-native-rustflags: no target features reported" >&2; exit 1; }
echo "-Ctarget-cpu=$cpu -Ctarget-feature=$features${*:+ $*}"
