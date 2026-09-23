#!/usr/bin/env python3
"""Generate the 256-arm dispatch tables in `src/ternlog_dispatch.rs`.

A runtime `imm: u8` must reach a const-generic `ndarray::simd::mask_ternlog::<IMM>`
(and its in-place sibling `mask_ternlog_assign::<IMM>`, and the two no-mask folds
`mask_ternlog_popcount::<IMM>` / `mask_ternlog_any::<IMM>`) — the const generic can only
ever be instantiated with a literal, so there is no way to route a runtime byte into
it except a 256-arm `match`, one arm per immediate. That match is pure boilerplate —
256 near-identical lines twice over — so it is generated here instead of hand-typed,
which is also what keeps the crate's own "no ISA, one delegation per op" laws
checkable by inspection rather than by trusting a human counted to 256 twice.

Everything between the `// GEN-TERNLOG-DISPATCH-BEGIN` / `// GEN-TERNLOG-DISPATCH-END`
marker LINES in `src/ternlog_dispatch.rs` is this script's output; everything outside
those markers (the module doc, the `use` line, and the `#[cfg(test)]` module) is
hand-written and is preserved untouched — this script only ever splices the region
between the two marker lines, so it never needs to know about, or regenerate, the
tests. Marker detection is line-exact (a line whose trimmed content equals the marker
text) rather than a raw substring search, so the tests' own string literals naming the
markers (needed so they can find them in the committed file) do not get counted as a
second occurrence of the marker itself.

Regenerate (writes the file in place):

    python3 crates/lance-graph-mask-risc/tools/gen_ternlog_dispatch.py

Check without writing — exits 1 if the committed file's generated region would
differ from a fresh run, exits 0 if it already matches (this is the invocation
CI's regenerate-and-diff gate uses):

    python3 crates/lance-graph-mask-risc/tools/gen_ternlog_dispatch.py --check

Paths are resolved relative to this script's own location, so it can be invoked
from any working directory.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

BEGIN_MARKER = "// GEN-TERNLOG-DISPATCH-BEGIN"
END_MARKER = "// GEN-TERNLOG-DISPATCH-END"

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
TARGET = SCRIPT_DIR.parent / "src" / "ternlog_dispatch.rs"

# Rustfmt's default `max_width` (100) — every generated line is checked against
# it so a future change to this generator cannot silently produce a line the
# orchestrator's `cargo fmt --check` would then have to reformat.
MAX_WIDTH = 100


def _arm(facade_fn: str, imm: int, call_args: str) -> str:
    """One match arm: `        N => facade_fn::<N>(call_args),`."""
    line = f"        {imm} => {facade_fn}::<{imm}>({call_args}),"
    assert len(line) <= MAX_WIDTH, f"generated arm exceeds {MAX_WIDTH} cols: {line!r}"
    return line


def generate_region_lines() -> list[str]:
    """The full generated region as a list of lines, markers included."""
    lines: list[str] = [BEGIN_MARKER]

    lines.append(
        "/// `dst = table[imm](a, b, c)` — the runtime immediate routed to the "
        "const-generic facade word."
    )
    lines.append(
        "pub fn ternlog_dispatch(imm: u8, a: &[u64], b: &[u64], c: &[u64], dst: &mut [u64]) {"
    )
    lines.append("    match imm {")
    for imm in range(256):
        lines.append(_arm("mask_ternlog", imm, "a, b, c, dst"))
    lines.append("    }")
    lines.append("}")

    lines.append("")

    lines.append(
        "/// `a = table[imm](a, b, c)` in place — the aliasing form the executor "
        "uses when `dst` is `a`."
    )
    lines.append(
        "pub fn ternlog_dispatch_assign(imm: u8, a: &mut [u64], b: &[u64], c: &[u64]) {"
    )
    lines.append("    match imm {")
    for imm in range(256):
        lines.append(_arm("mask_ternlog_assign", imm, "a, b, c"))
    lines.append("    }")
    lines.append("}")

    lines.append("")

    lines.append(
        "/// `Σ popcount(table[imm](a, b, c))` — the no-mask Count fold, routed "
        "like the others."
    )
    lines.append(
        "pub fn ternlog_popcount_dispatch(imm: u8, a: &[u64], b: &[u64], c: &[u64]) -> u64 {"
    )
    lines.append("    match imm {")
    for imm in range(256):
        lines.append(_arm("mask_ternlog_popcount", imm, "a, b, c"))
    lines.append("    }")
    lines.append("}")

    lines.append("")

    lines.append(
        "/// `table[imm](a, b, c) != 0` anywhere — the no-mask Any fold, routed "
        "like the others."
    )
    lines.append(
        "pub fn ternlog_any_dispatch(imm: u8, a: &[u64], b: &[u64], c: &[u64]) -> bool {"
    )
    lines.append("    match imm {")
    for imm in range(256):
        lines.append(_arm("mask_ternlog_any", imm, "a, b, c"))
    lines.append("    }")
    lines.append("}")

    lines.append(END_MARKER)
    return lines


def marker_line_indices(lines: list[str], marker: str) -> list[int]:
    """Indices of every line whose trimmed content is exactly `marker`."""
    return [i for i, line in enumerate(lines) if line.strip() == marker]


def splice(existing_lines: list[str], region_lines: list[str]) -> list[str]:
    """Replace the marker-line-delimited span of `existing_lines` with `region_lines`."""
    begins = marker_line_indices(existing_lines, BEGIN_MARKER)
    ends = marker_line_indices(existing_lines, END_MARKER)
    if len(begins) != 1:
        raise ValueError(f"expected exactly one bare {BEGIN_MARKER!r} line, found {len(begins)}")
    if len(ends) != 1:
        raise ValueError(f"expected exactly one bare {END_MARKER!r} line, found {len(ends)}")
    if begins[0] >= ends[0]:
        raise ValueError("BEGIN marker line must precede END marker line")
    return existing_lines[: begins[0]] + region_lines + existing_lines[ends[0] + 1 :]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="don't write; exit 1 if the committed region is stale, 0 if it already matches",
    )
    args = parser.parse_args(argv)

    if not TARGET.exists():
        print(
            f"error: {TARGET} does not exist — the hand-written frame (module doc, "
            "the `use` line, both markers, and the test module) must be authored "
            "before this generator has anything to splice into",
            file=sys.stderr,
        )
        return 1

    existing_text = TARGET.read_text()
    existing_lines = existing_text.split("\n")

    try:
        updated_lines = splice(existing_lines, generate_region_lines())
    except ValueError as exc:
        print(f"error: {TARGET}: {exc}", file=sys.stderr)
        return 1

    updated_text = "\n".join(updated_lines)

    if args.check:
        if updated_text != existing_text:
            print(f"{TARGET} is stale — run without --check to regenerate", file=sys.stderr)
            return 1
        return 0

    if updated_text != existing_text:
        TARGET.write_text(updated_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
