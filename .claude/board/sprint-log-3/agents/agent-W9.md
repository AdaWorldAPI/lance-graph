> **W9-rev2 correction (main thread):** original W9 push went to AdaWorldAPI/ada-consciousness instead of AdaWorldAPI/lance-graph. Content recovered verbatim below and pushed to the correct repo by main thread via pygithub. Original ndarray-style misrouted commit lives at `AdaWorldAPI/ada-consciousness/claude/tier-1-implementation-specs` as harmless residue.

---

# Agent W9 — Sprint-3 Log

**Worker:** W9
**Sprint:** 3
**Branch:** `claude/tier-1-implementation-specs`
**Pattern:** D — Meta-Structure Hydration
**Phase:** DESIGN
**Mode:** pygithub-first (token quotes stripped)

---

## Assignment

Author the first concrete Pattern D specification: **PR-D-1, FMA OWL
hydrator**. Deliverable is a single design-phase markdown spec
(~10 KB) at `.claude/specs/pr-d-1-fma-owl-hydrator.md`. No Rust code is
written in this sprint — design only.

---

## Actions taken

1. **Bootstrapped pygithub.** Read `GITHUB_TOKEN` and `GITHUB_REPO` from
   environment. Both arrived quoted — stripped surrounding quotes before
   handing to `github.Github(...)` per protocol. Auth confirmed against
   `AdaWorldAPI/ada-consciousness`.

2. **Branch state probe.** `claude/tier-1-implementation-specs` does not
   yet exist on origin. No sibling Sprint-3 / tier-1 / pattern-D branches
   present either. `.claude/specs`, `.claude/board/sprint-log-3`, and
   `.claude/plans` directories also do not yet exist on `main` — this PR
   is the first to populate them.

3. **Created branch** `claude/tier-1-implementation-specs` from
   `main@4c72d7a5` via pygithub `create_git_ref`.

4. **Authored spec** `.claude/specs/pr-d-1-fma-owl-hydrator.md` (~11 KB).
   Sections: Goal, Why this matters (Pattern D one-paragraph), Files to
   touch, API sketch (owl.rs + fma.rs + mod.rs), Test plan (4 tests),
   Dependencies (PR-A-1, PR-B-1, oxttl, BioPortal data), Acceptance
   criteria (7 boxes), Effort estimate, Open questions for engineer (5),
   Anatomy-demo unlock chain, Cross-references.

5. **Committed spec + log** to the new branch via pygithub
   `create_file`.

---

## Design decisions logged

- **Generic-over-specific.** `OwlHydrator` is the workhorse; `fma.rs`
  is ~40 LOC of declaration only. Pattern-D-conformant: ontologies are
  data, not crates.

- **INERT bundle for FMA v1.** `consumer_pointer = None`. FMA does not
  need a registered consumer crate — it is queried generically through
  the SPO writer + cascade traversal infrastructure delivered by PR-A-1.

- **Edge whitelist hand-curated for v1.** 10 anatomical relations
  (`rdfs:subClassOf`, `BFO:part_of`/`has_part`, three `FMA:*part_of*`
  variants, `FMA:supplies_blood_to`, `FMA:innervates`, `FMA:adjacent_to`,
  `FMA:continuous_with`). Engineer is free to expand once telemetry
  arrives in v2.

- **Parser pick.** `oxttl 0.1` recommended over `rio_xml`: streaming,
  sans-IO, lower alloc, Apache-2.0, same author as oxigraph. Engineer
  may override if FMA is only available as RDF/XML.

- **TTL storage.** Recommended CDN download with sha256 pin (`build.rs`
  fetch). FMA inflated TTL is ~280 MB — too large for plain git.
  Alternative: git-lfs (engineer's call).

- **Literal handling deferred.** v1 drops literals; PR-D-2 will design a
  proper `LiteralStore`. FMA semantics are entity-graph dominant.

---

## Dependencies surfaced

- **PR-A-1 / W2** (SPO-G u32 slot + bulk writer)
- **PR-B-1 / W3** (ContextBundle, OntologySlot, OntologyRegistry)
- **W1 master** (`OGIT::FMA_V1`, `OGIT::DOLCE_V1` constants)

W9 has flagged the cross-spec links inside the spec file. If any of
PR-A-1 / PR-B-1 land with API drift, PR-D-1 must be re-reviewed before
implementation begins.

---

## Anatomy demo handoff

PR-D-1 == PR-ANATOMY-1. Logged the unlock chain in the spec:

```
PR-D-1 (this) → PR-ANATOMY-2 (DICOM hydrator)
              → PR-ANATOMY-3 (FMA SPO-G edges)
              → PR-ANATOMY-4 (Q2 cockpit 3D voxel render w/ FMA labels)
```

---

## Open items handed forward

1. Engineer must confirm `oxttl 0.1` API matches the sketched parser
   loop; if API drift, swap to `rio_xml` per the open-questions section.
2. BioPortal license string for `LICENSES/FMA.txt` should be captured
   verbatim from the redistribution page on download day.
3. `IRI → u32` mapping persistence (open question 2) needs an explicit
   Lance dataset name reservation — coordinate with W2.

---

## Status

**DESIGN PHASE COMPLETE.** Spec and log pushed via pygithub on
`claude/tier-1-implementation-specs`. Ready for review by W1 (sprint
master) and the implementing engineer.
