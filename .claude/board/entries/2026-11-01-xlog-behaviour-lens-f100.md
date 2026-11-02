# D-XLOG-NATIVE-8: XLOG's term built-ins addressed at `0xFFFF :: 0xF100`

**Date 2026-11-01 · ID D-XLOG-NATIVE-8 · Status SHIPPED-IN-PR · Plan `.claude/plans/d-dt64-7-behaviour-addressing-registration-v1.md` (decision O7: `F1__` is XLOG's) · Crate `crates/lance-graph-xlog` (`lens`)**

DECISION (session): XLOG's relational core has words in the shared `F000`
algebra and gets none here. What XLOG adds is term inspection: `functor/3`,
`arg/3`, `=../2`. These are behaviour and are addressed at `XLOG_LENS =
0xFFFF_F100`. They are not interned as data. Position binds; the Prolog name
and XLOG's helper-relation name (`__xlog_meta_functor`, `_arg`, `_univ`) are
aliases. `__xlog_meta_fail` is not a word: it is an empty relation, and as
empty data it already means "never true".

VERIFIED-IN-CODE (`src/lens.rs`): `XLOG_WORDS` has 3 positions. `word_of` /
`position_of` resolve a label only when exactly one position lists it.
`declare` binds the lens in a `BindStore` with each word's column count as
`arity`.

SIDE EFFECT of reusing `declare_algebra`'s `arity` (accepted by the session,
documented in `declare`). For `F000`, `arity` is a stack operand count; here
it is a column count. The readers:

- `BindStore::insert` (`lance-graph-dt64/src/lib.rs:597`): a `MethodBinding`
  effect on an XLOG word must carry `columns` arguments.
- `lance-graph-r2il::binding` never reaches an F1 word. It refuses the lens
  as `NotR2ilReading` (`binding.rs:296`) before reading `arity`.

TEST-PINNED (`tests/lens.rs`, 3): the lens is in the `F1__` range. Each alias
selects its own position, and non-words (`join`, `__xlog_meta_fail`, wrong
case, empty) select none. The declared counts read back. The binding is
idempotent and immutable against a different one.

OPEN: the XLOG fork (`xlog-cube`, xlog#2) still has to route these aliases to
`Refusal::Behaviour` instead of interning them; that is the consumer half.
