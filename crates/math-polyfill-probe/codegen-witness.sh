#!/usr/bin/env bash
# D-MPF-2 LLVM witness for the primitives the native fold delegates to.
# Modelled on ndarray scripts/codegen-witness.sh: build the probe's assembly,
# run its self-check first, then assert per-symbol facts. Pinned observations
# of the current codegen, not targets: a change that moves them should fail
# here and be re-measured on purpose.
#
# Local, not in CI: the k-mask check assumes an AVX-512 build (the default
# config is target-cpu=native), and a CI runner's tier is not pinned.
#
#   bash crates/math-polyfill-probe/codegen-witness.sh
#   CARGO_ARGS='--config ../../.cargo/config-v3.toml' bash ...   # pin a tier
set -euo pipefail
cd "$(dirname "$0")"
export CARGO_PROFILE_RELEASE_DEBUG=0 CARGO_INCREMENTAL=0
# shellcheck disable=SC2086
cargo ${CARGO_ARGS:-} run --release --example codegen_probe
rm -f target/release/examples/codegen_probe-*.s
# shellcheck disable=SC2086
cargo ${CARGO_ARGS:-} rustc --release --example codegen_probe -- --emit=asm -C debuginfo=0
asm=$(ls target/release/examples/codegen_probe-*.s)

# Body of one probe symbol: from its label to its .size directive.
body() { awk -v s="$1" '$0 ~ "^_R[A-Za-z0-9_]*"s":$" {on=1; next} on && /\.size/ {exit} on' "$asm"; }
vec_lines() { grep -cE '%[xyz]mm|%k[0-7]' || true; }
count() { grep -cE "$1" || true; }

fail=0
check() { if eval "$2"; then echo "PASS: $1"; else echo "FAIL: $1"; fail=1; fi; }

ms=$(body probe_masked_sum_i32)
bf=$(body probe_masked_sum_branchless)
gt=$(body probe_gt_i32_to_mask)
[ -n "$ms" ] && [ -n "$bf" ] && [ -n "$gt" ] || { echo "FAIL: probe symbol missing"; exit 1; }

check "masked_sum_i32 is a scalar set-bit walk (tzcnt, no vector register)" \
  '[ "$(echo "$ms" | count "tzcnt|bsf")" -ge 1 ] && [ "$(echo "$ms" | vec_lines)" -eq 0 ]'
check "branch-free sum vectorizes (vector add, no tzcnt)" \
  '[ "$(echo "$bf" | count "vpaddq")" -ge 1 ] && [ "$(echo "$bf" | count "tzcnt|bsf")" -eq 0 ]'
check "gt_i32_to_mask compares 16 lanes into a k-mask" \
  '[ "$(echo "$gt" | count "vpcmp[a-z]*d")" -ge 1 ] && [ "$(echo "$gt" | count "%k[0-7]")" -ge 1 ]'
exit $fail
