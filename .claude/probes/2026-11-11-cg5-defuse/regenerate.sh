#!/usr/bin/env bash
# Regenerates crates/cg-defuse-probe/tests/data from crates/foundry-parity-lab.
#
#   CHARON=/path/to/charon .claude/probes/2026-11-11-cg5-defuse/regenerate.sh
#
# Works in a scratch git worktree, so the checkout is never edited. m0 is the
# crate as committed; m1, m2, m3 are the three mutants of src/method.rs
# written by mutate.py. Needs the Charon binary (not on PATH by default).
set -euo pipefail
: "${CHARON:?set CHARON to the charon binary}"
here=$(cd "$(dirname "$0")" && pwd)
root=$(git -C "$here" rev-parse --show-toplevel)
out=$root/crates/cg-defuse-probe/tests/data
work=$(mktemp -d)
trap 'git -C "$root" worktree remove --force "$work/wt" 2>/dev/null || true; rm -rf "$work"' EXIT
git -C "$root" worktree add --detach "$work/wt" HEAD >/dev/null
cd "$work/wt/crates/foundry-parity-lab"
export CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$work/target}
for m in m0 m1 m2 m3; do
  if [ "$m" != m0 ]; then
    python3 "$here/mutate.py" "$m"
    mkdir -p "$out/src-$m"
    cp src/method.rs "$out/src-$m/method.rs"
  fi
  "$CHARON" cargo --no-dedup-serialized-ast --dest-file "$work/$m.llbc" -- --lib
  python3 "$here/extract_defuse.py" "$work/$m.llbc" "$out/$m.defuse.tsv"
  git checkout -q src/method.rs
done
