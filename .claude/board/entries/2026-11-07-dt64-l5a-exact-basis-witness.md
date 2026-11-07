# DT64 L5a: the exact predicate-basis witness at register and resolve (D-DT64-7-L5a)

**Date:** 2026-11-07. **Status:** in PR (stacked on L5b, #1976). Plan: `.claude/plans/d-dt64-7-behaviour-addressing-registration-v1.md` §1f.

## What changed

**Registration reads the basis from the source.** `register` and `register_derived` take the source view. They read the basis at the bound revision through `basis_at`, which takes the source's history first and looks a lens up by its full classid. The basis must have `field_count` fields. The class entry keeps a `BasisDigest`, a 128-bit FNV-1a hash of a length-prefixed encoding of the ordered `predicate_iri`s, with labels excluded. It never keeps the fields themselves.

**`resolve` compares the digest.** A mismatch is refused as `PredicateBasisChanged { source: Historical | Current }`. A changed field count is still `FieldBasisChanged`.

**The hot path does not hash.** `LaneBinding::check` re-checks revision and field count only, through a crate-private `resolve_counted`.

**All callers migrated.** 13 files in 9 crates now register through the view they later resolve through.

## Measured

- Both pinned findings now refuse: N1 is `an_equal_length_reorder_is_refused`, N4 is `an_equal_length_predicate_change_is_refused`. The label-only rename twin stays green.
- Oracle sweep: 10 basis variants, of which `resolve` accepts exactly the 2 whose ordered `predicate_iri`s equal the bound ones.
- A same-revision removal is refused.
- Two bases whose predicates concatenate to the same bytes are kept apart.
- Re-registration is idempotent over the same basis and refused (`BehaviourImmutable`) over a different one.
- When the history lacks the bound revision, the current basis is accepted only if proved equal to it. A rewritten history is refused as `Historical`.
- Six guards were each removed in turn, and each removal turned its named tests red: the digest comparison, hashing IRIs rather than labels, the length prefix, the count check at registration, recording the digest, and reading history.
- Report test `a_disagreeing_field_basis_never_returns_the_wrong_lane` re-pinned. A fresh bind under the reordered view is now refused instead of borrowing the other lane.

## Open

- **L5c:** `admit` still accepts a `MethodRef` without basis identity (`finding_admit_does_not_carry_the_basis_identity`).
- **Not evidenced:** collision resistance against a source built to collide. FNV is not cryptographic, so the digest detects drift but does not authenticate a source (O10 owner binding).
- **Polyfill lenses** (`declare_algebra`) are still checked by field count only.
- **Hot path:** a reorder after bind is caught on the hot path only by a source that maps lanes by predicate. L5d decides whether R2IL re-binds.
