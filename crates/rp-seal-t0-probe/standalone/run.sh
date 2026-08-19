#!/bin/sh
# Fresh-compile + run the E2 economics benches (min-of-3 discipline is
# the READER's job: quote warm minima, never the first cold pass).
set -e
T=$(mktemp -d)
H=$(cd "$(dirname "$0")" && pwd)
for b in bench bench2 bench3 bench4; do rustc -O "$H/$b.rs" -o "$T/$b"; done
echo "── bench (b=1 / b=4 / b=64):"
for f in 1.0 0.25 0.015625; do "$T/bench" 65536 $f; echo; done
echo "── bench4 (fsync):"; "$T/bench4" "$T/fsyncdir"
echo "── bench3 (scatter):"; "$T/bench3"
