# ogar-behavior-probe (LAB)

Probes P2, P3 and P9 from the mandatory-codegraph co-research map
(`.claude/board/entries/2026-10-31-coresearch-mandatory-codegraph.md`). They measure what the
existing OGAR structures can express about a method's behaviour. They add no type outside this
crate and change nothing in OGAR or DT64.

Own `[workspace]`, excluded from the lance-graph workspace. Depends read-only on `ogar-vocab`,
`ogar-loco`, `ogar-from-ruff` (OGAR, by path), `lance-graph-dt64`, `lance-graph-contract`, and
`ruff_python_spo` (git `main`, the coordinate `ogar-from-ruff`'s own test uses).

```
CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0 \
  cargo test --manifest-path crates/ogar-behavior-probe/Cargo.toml -- --nocapture
```

Data, all real: the verbatim Odoo 19 `account_payment_term.py` (`fixtures/`, provenance next to it),
lifted with `ruff_python_spo::extract_from_source` and `ogar_from_ruff::mint::compile_graph_python`
exactly as OGAR's kausal-parity test does; the `ogar-vocab` action tables; and the GCD program from
`ogar-loco/examples/interpret_probe.rs`.

## P2: body revision

`body_rev` hashes a `Program` (FNV-1a 64) after renumbering the functions reachable from the entry
in first-reference order. It covers a caller-supplied vocabulary tag, and per function the lane
shape, the call count and every written call. `raw_body_rev` hashes in stored order.

| Fixture | Result |
|---|---|
| identical rebuild | same revision |
| one constant changed | different |
| bodies stored at swapped indices, references swapped with them | raw differs, canonical equal |
| THEN and ELSE bodies swapped (a different program) | canonical differs |
| an unreachable function appended | raw differs, canonical equal |
| vocabulary tag `rev` changed | different |
| reference past the end | refused (`UnresolvableReference`) |
| guard changed on a real `ActionDef`, program bytes fixed | **`body_rev` unchanged**; a lab `envelope_rev` over the ActionDef differs |

Findings:
- **A body revision is computable from existing data** without a key registry or a body store.
- **`ogar-loco::Vocabulary` has no name and no version** (trait `Vocabulary`, `vocabulary.rs`), so the
  interpretation of the bytes cannot be named from the program alone. The probe takes a `VocabTag`
  from the caller. This is a gap for the OGAR owner.
- **`body_rev` does not cover ON × WHEN × DO.** A changed `KausalSpec`, write set, `on_enter`, guard
  failure policy or timeout leaves it unchanged. **No existing revision detects such a change:**
  `ActionDef` carries no revision; DT64 `SigRev` is set by the registrant, not derived; DR
  `FrameRevisions` cover the source, the relation map and the coordinates, not the action envelope.
  A prepared plan keyed only by `body_rev` could be reused after its guard changed.
- Index assignment is a lowerer choice. Without renumbering, an equivalent re-lowering changes the
  revision; with it, unreachable functions drop out of the revision. Both are stated choices.

## P3: ActionDef → MethodBinding, graded

`project` grades each of the 13 `MethodBinding` fields: Direct, SoundlyDerivable,
OverApproximated, Unknown, NotRepresentable. It never builds a binding and never fills a gap.
Basis: the lifted class's attribute names then association names. The real basis is the
`ClassView` of the classid, but the harvested Odoo models mint classid 0, so there is none to read.

Over all 21 harvested methods:

| Field | Grades |
|---|---|
| id, sig_rev | NotRepresentable 21 (string identity; no revision; MethodId is the registrant's) |
| name | Direct 21 |
| on | Direct 10 (single-field `Depends`/`Constrains`), Unknown 10 (no kausal fact), NotRepresentable 1 (dotted paths) |
| when | Unknown 21 (no declared guard; body guards not represented) |
| reads | Unknown 21 (inferred, never graded complete) |
| effect targets | Direct 8 (source-attested writes), Unknown 13 |
| effect algebra, effect args | NotRepresentable 21 (the value written is not captured) |
| effect order | Unknown 20 (at most one field name; a name is not one write), NotRepresentable 1 |
| population, truth, demand | NotRepresentable 21 |

Also: a `StateGuard` is NotRepresentable as a DT64 `Guard` (string values, no value codebook); a
lifecycle trigger is OverApproximated; an external cause has no field trigger.

**Result: no harvested method projects to a sound `MethodBinding`.** The blockers are on the
producer side (no write values, no algebra word, no guard values as lane values) and in identity
(no revision, MethodId is an owner mint). The hand-authored DT64 fields remain the compatibility
surface; deriving them would mean inventing.

## P9: coverage census

| Producer | ActionDefs | cat 1 (Program + action) | cat 2 (action, no Program) | kausal | body_source | reads | writes |
|---|---|---|---|---|---|---|---|
| ogar-vocab document_actions | 3 | 0 | 3 | 3 | 0 | 3 | 3 |
| ogar-vocab geo_actions | 6 | 0 | 6 | 6 | 0 | 0 | 0 |
| ogar-vocab healthcare_actions | 20 | 0 | 20 | 20 | 0 | 0 | 0 |
| ogar-from-ruff, Odoo fixture | 21 | 0 | 21 | 11 | 0 | 19 | 8 |
| **total** | **50** | **0** | **50** | | | | |

Not counted at run time (read at source):
- `ogar-vocab::ocr_actions` returns `OcrActionSpec`, not `ActionDef`.
- `ogar-from-elixir`: the GenServer, Phoenix and Oban extractors are `todo!()`.
- `ogar-loco::Program` is constructed only in tests and examples (`interpret.rs`, `program.rs`,
  `examples/interpret_probe.rs`). No frontend lowers into it.
- Methods outside OGAR, per category:
  - cat 3, native method with an explicit contract: the Tract ONNX operators
    (`tract-harvest-probe/src/registry.rs`, own registry keyed `MethodKey{lens, ordinal}`).
  - cat 4, extracted footprint only: the eight D-SAG-6 fixture methods (`dt64-admission-probe`,
    evidence rows rebuilt from board entries, not measured).
  - cat 5 for OGAR purposes: the XLOG count method (`xlog-relcomp-probe/tests/dto_binding.rs`, a
    hand-authored DT64 binding with no OGAR representation).

**A mandatory, fully inspectable body is not implemented anywhere today.** Category 1 is empty.

## Disable table

Method: commit first; apply one patch (anchor occurs exactly once, the script asserts the file
changed); run the integration tests; `git checkout` the file and assert it is restored. Runs
against commit `3ccbaa5` (D5 and D5b against `9d6ff3a`, which adds the test D5 needed).

| Guard | Patch | Red tests |
|---|---|---|
| D1 canonical renumbering | keep stored indices | `a_renumbered_body_keeps_the_canonical_revision_but_not_the_raw_one` |
| D2 vocab tag in the revision | drop the tag | `the_vocabulary_tag_is_part_of_the_revision` |
| D3 unreachable functions excluded | hash every function | `an_unreachable_function_is_not_part_of_the_canonical_revision` |
| D4 envelope reads `kausal` | drop it | `a_guard_change_on_the_action_leaves_the_body_revision_unchanged` |
| D5 unresolved ON name refuses | skip the name | `a_single_segment_name_outside_the_basis_is_not_representable` |
| D6 reads never complete | Direct when all resolve | `inferred_reads_are_never_graded_complete` |
| D7 unordered writes | always SoundlyDerivable | `two_writes_without_order_are_not_representable_as_an_ordered_do` |
| D8 no Program link claimed | classify as category 1 | `census_tally`, `no_actiondef_is_counted_as_having_a_program` |

Not load-bearing: the dotted-path check in `resolve_all` (D5b, 0 red). A dotted path is never a
basis name, so the lookup refuses it anyway; the check only improves the message. D5 first ran
green: the dotted fixture never reached the unresolved-name branch, so a test reaching it was
added.

## Kill conditions, and which fired

- P2 "body_rev is a complete semantic revision": **fired** (guard change unseen).
- P3 "DT64 MethodBinding derivable from OGAR today": **fired** (0 of 21).
- P9 "mandatory inspectable body exists": **fired** (category 1 = 0 of 50).

## Minimal seam, if any

No new type is proposed. Each gap has an owner:
1. A vocabulary identity (name, version) on `ogar-loco::Vocabulary` or its registry, so a body
   revision names its interpretation. OGAR.
2. A revision for the action envelope (`kausal`, effects, `on_enter`, policies), or a statement that
   plan validity is checked against those fields directly. OGAR, then DT64 for what a prepared plan
   records.
3. Producer capture of written values (ruff `Function::writes` is name-level) before any DO
   projection can be sound. ruff / ogar-from-ruff.
4. A frontend that lowers into `ogar-loco::Program`, or an explicit ruling that `body_source` text
   plus facts is the body. OGAR.

Until then, the hand-authored DT64 binding stays the compatibility surface, keyed by the reference
the co-research map already proposes. P6 (hot-path independence) is unblocked only for methods
with a body; today that is none, so P6 needs a lab body (the GCD program) rather than a harvested
method.

## Operator correction (2026-10-31)

`ogar-loco::Program` is not the canonical method body; the architecture is Kythe semantic facts +
R2IL executable semantics + DT64/DR binding. So seam item 4 above is withdrawn, and P9's empty
category 1 (no `ActionDef` links a Program) is a measured fact, not an admission gate. P2's
`body_rev` remains a correct digest of a `Program` value, but it is not the method's body revision.
