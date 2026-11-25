# Presence differential and the Q1 refusal (execution order, step 2)

**Status:** In PR.

## Q1, implemented as decided

`Recipe::admit` refuses an equality, a text equality or a truth test over a field the schema declares optional (`Refusal::AmbiguousPresence { field }`). Aliases are followed. Native presence semantics are unchanged. There is no `VrlCompat` execution mode.

With the presence lanes optional, the gate admits only `presence.rs` program 0, which compares nothing. Program 2 is refused as `AmbiguousPresence { status }`; programs 1, 3 and 4 are refused by upstream with E100.

The 131 S1 deviations (Unknown → Drop 60 + 31, Unknown → Keep 40) stay pinned in `tests/upstream.rs` `DIVERGENCES` as the permanent differential corpus.

## Item 9, measured (`tests/presence_matrix.rs`, both pins)

The matrix covers an integer, a bool and a text field. Each holds a value, its zero (`0`, `false`, `""`), an explicit `null`, or no key, under 8 programs: `==`, `!=` and `>` on the integer; `if .b`, `if !.b` and `== false` on the bool; `== ""` and `!= ""` on the text.

- **Present, including the zeros:** the recipe and upstream agree on every case.
- **`null` versus missing:** upstream reads them alike in all 8 programs.
- **Absent:**
  - The recipe answers Unknown for every program.
  - Upstream agrees on `>`, `if .b` and `if !.b`: each is a runtime error there, which is Unknown.
  - Every `==`/`!=` diverges: `null == x` is false and `null != x` is true. That is Q1, and the gate refuses all of them.

## Investigation: can VRL's meaning be expressed with existing methods?

Yes, with no new opcode. The population program already computes one absent mask per term, `ABSENT = (value == 0) & !presence`, using only `Pred`, `AndNot` and `Or`.

VRL's meaning is a per-term choice among the same ops:
- **`==`:** absent reads as not equal. Drop the `KEPT &= !ABSENT` step for that term.
- **`!=`:** absent reads as not equal, so abort. Use `TERM |= ABSENT` and drop `KEPT &= !ABSENT`.

The lowering already has a precedent: a text compare against code 0 is a decided presence test. So compatibility can be a property of lowering, chosen per term at admission. This is the presence-aware lowering the refusal waits for. Enabling it is a separate decision.

## Disable run

With the ambiguity check off:
- `tests/admission` goes red (2 of 4 tests fail);
- the matrix's gate test goes red;
- the matrix's agreement test stays green, as it should.

OPEN:
- Whether to enable per-term VRL lowering, and for which consumers.
