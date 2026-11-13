#!/bin/bash
# Refresh `.codegraph-charon/`, the committed ruff_charon store that fills `.codegraph/calls.tsv`.
#
#   CHARON_BIN=<dir with charon + charon-driver> RUFF_CHARON=<ruff_charon binary> \
#     .claude/tools/charon_store.sh [crate ...]      # no crate = every crate in crates.tsv
#
# For each crate: `charon cargo` in the crate directory (lib target when there is one,
# default features), `ruff_charon import` into a scratch store, then `<lib>.rows.tsv` and
# `<lib>.files.tsv` are copied to `.codegraph-charon/`. The LLBC (tens of MB per crate) is
# deleted after the import. A crate that fails keeps its previous files and is listed in the
# run log this script prints (and writes to $WORK/run.tsv) with the reason. Finish with:
#   python3 .claude/tools/codegraph_index.py --write
#
# Build the two tools first (see `.codegraph-charon/README.md`). The build cache is removed
# when free space drops below FREE_KB_MIN, because one shared nightly target for the whole
# workspace does not fit a small disk.
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
: "${CHARON_BIN:?directory holding charon and charon-driver}"
: "${RUFF_CHARON:?path to the ruff_charon binary}"
WORK=${WORK:-$(mktemp -d)}
FREE_KB_MIN=${FREE_KB_MIN:-2200000}
OUT="$ROOT/.codegraph-charon"
export PATH="$CHARON_BIN:$PATH" CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target/charon}"
unset RUSTFLAGS
mkdir -p "$WORK/store" "$WORK/llbc" "$WORK/out" "$OUT"
cd "$ROOT" || exit 2
want=" $* "
grep -v '^#' .codegraph/crates.tsv | while IFS=$'\t' read -r name path _ws _digest; do
  [ $# -gt 0 ] && [[ "$want" != *" $name "* ]] && continue
  lib=${name//-/_}
  free=$(df --output=avail -k "$ROOT" | tail -1)
  [ "$free" -lt "$FREE_KB_MIN" ] && rm -rf "$CARGO_TARGET_DIR"
  libarg=(); [ -f "$path/src/lib.rs" ] && libarg=(-- --lib)
  t0=$(date +%s)
  ( cd "$path" && timeout "${CRATE_TIMEOUT:-1800}" charon cargo --no-dedup-serialized-ast \
      --dest-file "$WORK/llbc/$lib.llbc" "${libarg[@]}" ) > "$WORK/out/$name.log" 2>&1
  rc=$?; secs=$(( $(date +%s) - t0 ))
  if [ ! -f "$WORK/llbc/$lib.llbc" ]; then
    printf '%s\tCHARON_FAIL\trc=%s\t%ss\t%s\n' "$name" "$rc" "$secs" \
      "$(grep -m1 -E '^error' "$WORK/out/$name.log" | cut -c1-160)"
    continue
  fi
  imp=$("$RUFF_CHARON" import "$WORK/store" "$WORK/llbc/$lib.llbc" "$lib" 2>&1 | tail -1)
  rm -f "$WORK/llbc/$lib.llbc"
  cp "$WORK/store/$lib.rows.tsv" "$WORK/store/$lib.files.tsv" "$OUT/" || continue
  printf '%s\tOK\trc=%s\t%ss\t%s\n' "$name" "$rc" "$secs" "$imp"
done | tee "$WORK/run.tsv"
echo "log: $WORK/run.tsv  (per-crate output in $WORK/out)" >&2
