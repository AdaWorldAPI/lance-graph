# D-OMR-AMEND-1: the BindStore is the method registry, not a second store (2026-11-18)

**Status:** DECISION recorded; documentation only. Amends
`.claude/plans/ontology-method-registry-v1.md` (§8), merged the same day as
#2133.

DECISION (operator, 2026-11-18): "theres a missed opportunity for bindstore
and DT64 / DR to integrate — bindstore = DT classview / DR widefieldmask".

SCOPE: the D-OMR plan. The `Method` row in the ontology cache and the
hydration step are withdrawn (D-OMR-1, D-OMR-3). Four gaps of the `BindStore`
replace them (D-OMR-8..11): registration by name, a known-but-not-executable
state, the effect class on the binding, durability.

BASIS: `BindStore::resolve` already joins a class schema read through a
`ClassView` with the method schema (`lance-graph-dt64/src/lib.rs:1099`);
`lance-graph-ontology` already implements `ClassView`
(`class_resolver.rs:104`); a method's ON and reads are already
`WideFieldMask`s. The two-store plan was not weighed against this: asked
twelve questions by the HIRO session, its author could not say which of row
and binding is the authority, which wins when they differ, or who may call
`BindStore::register`.

REVISIT WHEN: D-OMR-9 shows that a store which refuses what it cannot bind is
the wrong home for declarations it must hold and not run.

**Correction, same day.** D-DT64-7
(`.claude/plans/d-dt64-7-behaviour-addressing-registration-v1.md`) had already
decided this: Learned = registered in the `BindStore` (D2), DT64 holds the
registration authority (D13'', D15''), Explore lives outside the store, and
"a second authoritative registry" is listed as not introduced. That plan was
not read before D-OMR was written. Operator: "the classid is already the
answer to ON"; "F###-F### for methods includes both ON and DO and when is a
register call of the method". D-OMR-6, -8, -9 and -11 are withdrawn (plan §9);
D-OMR is subordinate to D-DT64-7.

MIRROR
- BLIND SPOT: `.claude/plans/` was not searched for an existing registration plan before a new one was proposed.
