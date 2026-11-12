#!/usr/bin/env bash
# Decide whether a commit can affect lance-graph-java's `lgj-abi`.
#
# Usage: lgj-abi-relevant.sh <commit>      (default: HEAD)
# Prints `true` or `false`. Compares <commit> with its FIRST parent, which for
# a pull-request merge commit is the base branch and for a merge on `main` is
# the previous `main`, so the diff is exactly what the change brings in.
#
# The paths are what `lgj-abi` compiles from this repository: the contract,
# mask-RISC, and Quack (a dev-dependency of its test targets), plus the files
# that decide how they are built. Keep this list equal to the path
# dependencies in lance-graph-java `native/lgj-abi/Cargo.toml`.
set -euo pipefail

commit="${1:-HEAD}"

if ! git rev-parse --verify --quiet "${commit}^1" >/dev/null; then
  # No parent to compare with (a shallow or root commit): do not guess.
  echo true
  exit 0
fi

if git diff --name-only "${commit}^1" "${commit}" | grep -qE \
  '^(crates/lance-graph-(contract|mask-risc|quack)/|rust-toolchain\.toml$|\.github/workflows/lgj-abi-compat\.yml$|\.github/scripts/lgj-abi-relevant\.sh$)'; then
  echo true
else
  echo false
fi
