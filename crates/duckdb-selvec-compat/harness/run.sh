#!/usr/bin/env bash
# Differential run: the same C fixture against the real libduckdb (pinned)
# and against the adapter, in two processes. Usage: harness/run.sh [workdir]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
crate="$(dirname "$here")"
work="${1:-$(mktemp -d)}"
tag=v1.5.6
sha=b845005f5132a7d8180057c35e14a7626632258782f871a90861b19c1c03841b
mkdir -p "$work/duckdb"
if [ ! -f "$work/duckdb/libduckdb.so" ]; then
	curl -sSL --retry 3 -o "$work/duckdb.zip" \
		"https://github.com/duckdb/duckdb/releases/download/$tag/libduckdb-linux-amd64.zip"
	echo "$sha  $work/duckdb.zip" | sha256sum -c -
	unzip -o -q "$work/duckdb.zip" -d "$work/duckdb"
fi
(cd "$crate" && cargo build --release --features duckdb-symbols)
adapter="$crate/target/release"
cc -O1 -Wall -Wextra -I"$work/duckdb" "$here/selvec_parity.c" -L"$work/duckdb" -lduckdb -o "$work/parity_duckdb"
cc -O1 -Wall -Wextra -I"$work/duckdb" "$here/selvec_parity.c" -L"$adapter" -lduckdb_selvec_compat -o "$work/parity_adapter"
LD_LIBRARY_PATH="$work/duckdb" "$work/parity_duckdb" >"$work/out_duckdb.txt"
LD_LIBRARY_PATH="$adapter" "$work/parity_adapter" >"$work/out_adapter.txt"
# The adapter binary must not have picked up the real library, or vice versa.
ldd "$work/parity_adapter" | grep -q duckdb_selvec_compat
! ldd "$work/parity_adapter" | grep -q 'libduckdb\.so'
ldd "$work/parity_duckdb" | grep -q 'libduckdb\.so'
if diff -u "$work/out_duckdb.txt" "$work/out_adapter.txt"; then
	echo "PARITY: $(wc -l <"$work/out_duckdb.txt") lines identical (DuckDB $tag vs adapter)"
else
	echo "PARITY FAILED" >&2
	exit 1
fi
if command -v valgrind >/dev/null; then
	LD_LIBRARY_PATH="$adapter" valgrind -q --error-exitcode=9 --leak-check=full --errors-for-leak-kinds=all \
		"$work/parity_adapter" >/dev/null
	echo "VALGRIND adapter: clean"
fi
