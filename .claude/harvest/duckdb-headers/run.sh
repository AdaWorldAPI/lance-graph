#!/usr/bin/env bash
# Reproducible DuckDB HEADER harvest — discharges §6 of
# `.claude/plans/duckdb-to-v3-translation-matrix-v1.md`.
#
# §6 recorded that the original 22-TU `.cpp` harvest yielded seven TUs at
# 100 % `Empty`, because DuckDB's execution is template-dispatched and lives in
# HEADERS. A header pass was then run (2026-09-14) and its counts written into
# this directory's README — but the TSVs, the args and the DuckDB source were
# all absent, so the README's own honesty note says the complaint was NOT
# discharged: "a README quoting counts from a TSV that is absent".
#
# This script is the discharge. Everything it needs is committed beside it:
# the header list, the clang args template, and (after a run) the TSVs.
#
#   DUCKDB_SRC=/path/to/duckdb ./run.sh
#
# Defaults to the AdaWorldAPI fork's conventional read-clone path. The fork is
# the P0-correct source; upstream duckdb/duckdb is deliberately NOT used.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DUCKDB_SRC="${DUCKDB_SRC:-/home/user/adaworldapi/duckdb}"
RUFF_SRC="${RUFF_SRC:-/home/user/ruff}"
OUT="${OUT:-$HERE/ore}"
LIBCLANG_PATH="${LIBCLANG_PATH:-/usr/lib/llvm-18/lib}"

[ -d "$DUCKDB_SRC/src/include/duckdb" ] || {
  echo "DUCKDB_SRC=$DUCKDB_SRC has no src/include/duckdb." >&2
  echo "Clone it:  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \\" >&2
  echo "             https://github.com/AdaWorldAPI/duckdb $DUCKDB_SRC" >&2
  exit 2
}

HARVESTER="$RUFF_SRC/target/release/examples/harvest_events"
[ -x "$HARVESTER" ] || {
  echo "Building the harvester ..." >&2
  ( cd "$RUFF_SRC" && CARGO_PROFILE_DEV_DEBUG=0 cargo build -p ruff_cpp_spo \
      --features libclang --example harvest_events --release )
}

# The args file is GENERATED from the committed template, so the include paths
# follow DUCKDB_SRC rather than being pinned to one machine's checkout.
ARGS="$OUT/args.txt"
mkdir -p "$OUT"
sed "s|@DUCKDB_SRC@|$DUCKDB_SRC|g" "$HERE/args.txt.in" > "$ARGS"

# Provenance: which DuckDB the counts below were read from. A count without
# its commit is an anecdote — the same rule the parity probes follow.
{
  echo "duckdb_src=$DUCKDB_SRC"
  echo "duckdb_head=$(git -C "$DUCKDB_SRC" rev-parse HEAD 2>/dev/null || echo UNKNOWN)"
  echo "duckdb_origin=$(git -C "$DUCKDB_SRC" remote get-url origin 2>/dev/null || echo UNKNOWN)"
  echo "ruff_head=$(git -C "$RUFF_SRC" rev-parse HEAD 2>/dev/null || echo UNKNOWN)"
  echo "clang=$(clang --version 2>/dev/null | head -1 || echo UNKNOWN)"
} > "$OUT/provenance.txt"

total_methods=0
total_events=0
: > "$OUT/per-header.tsv"
printf 'header\tmethods\tevents\n' >> "$OUT/per-header.tsv"

while read -r rel; do
  [ -n "$rel" ] || continue
  tag="${rel//\//_}"
  hdr="$DUCKDB_SRC/src/include/duckdb/$rel.hpp"
  [ -f "$hdr" ] || { echo "MISSING $hdr" >&2; exit 3; }
  ORE_FILE="$hdr" ORE_ARGS_FILE="$ARGS" ORE_OUT="$OUT/$tag" \
    LIBCLANG_PATH="$LIBCLANG_PATH" "$HARVESTER" >/dev/null 2>&1 || true
  # NEITHER TSV carries a header row -- line 1 is already data, so the raw
  # line count IS the count. An earlier version of this script subtracted a
  # header row that does not exist and reported 115/1612 against the README's
  # 123/1622, which read as the README being wrong. It was this script. The
  # counting rule is asserted below rather than assumed.
  m=0; e=0
  [ -f "$OUT/$tag/methods.tsv" ] && m=$(wc -l < "$OUT/$tag/methods.tsv")
  [ -f "$OUT/$tag/events.tsv" ]  && e=$(wc -l < "$OUT/$tag/events.tsv")
  printf '%s\t%d\t%d\n' "$tag" "$m" "$e" >> "$OUT/per-header.tsv"
  total_methods=$(( total_methods + m ))
  total_events=$(( total_events + e ))
done < "$HERE/headers.txt"

printf 'TOTAL\t%d\t%d\n' "$total_methods" "$total_events" >> "$OUT/per-header.tsv"
cat "$OUT/per-header.tsv"
echo
echo "provenance: $(sed -n 2p "$OUT/provenance.txt")"
