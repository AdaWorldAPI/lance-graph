## 2026-08-23 — E-AN-IMPORT-EDGE-IS-NOT-AN-ARCHITECTURAL-RELATION-1 — auditing "does A import B" and concluding "no architectural connection" erased a lineage documented in this repo's own docs

**Status:** CORRECTION (operator-caught). **Confidence:** High — the erased
lineage is quoted below from `docs/architecture/`, in-repo, verifiable.

**What happened.** Asked to audit the Ghidra/R2IL relation, this session
grepped `lance-graph-java` for imports of `ghidra`/`r2sleigh`/`sleigh`/`pcode`,
found none, cloned `r2sleigh`, found no `lance` dependency and only bit-width
masking, and wrote: *"Settled with evidence, so nobody builds a bridge to it
later."* **That sentence is retracted.** It converted a true dependency-level
negative into a false architecture-level negative, and it did so pre-emptively
— telling future sessions not to look.

**What was actually there, in THIS repo, unread.**
`docs/architecture/ARC-B-OWNERSHIP-AND-ADDRESSING-REASSESSMENT.md` §7, titled
"THE GHIDRA / JAVA ABI STANDARD" (2026-08-19):

> *"The external path is the reference discipline: `binary → Ghidra/SLEIGH/
> P-code → r2sleigh → V4 R2IL`, with the domain oracle keeping semantic
> authority and the ABI exposing address + mask + view."*

That section connects Ghidra, the Java ABI, and the address+mask+view
discipline — the exact three subjects being audited separately — and states a
falsifier and a debt against them. `EPIPHANIES.md:1145`
(`E-V4-IS-THE-100-PERCENT-TIER-V3-UNCHANGED-1`) is the ruling it references.

**The four claims that must not be collapsed into one verdict:**

| Claim | Verdict |
|---|---|
| direct code dependency `lance-graph-java → r2sleigh`/Ghidra | **ABSENT** — verified, and unsurprising: lance-graph-java is the ABI/execution membrane |
| cross-repo architectural lineage | **PRESENT** — ARC-B §7 above; ruff PR #94/#103/#104, all merged |
| directly reusable mask implementation inside r2sleigh | **ABSENT** — every `mask` there is bit-width/taint masking in the SSA lifter |
| reusable substrate pattern | **PRESENT and load-bearing** |

**A dependency edge I also missed by auditing the wrong pair.** ruff PR #94
(merged) ships `crates/ruff_r2il`, workspace-excluded, which **path-deps the
`r2sleigh` sibling checkout** and whose `vocab.rs` *"feeds lance-graph's
`ogar_codebook` read-only"*, with a *"16-byte V3-shaped `VarnodeFacet`"*. So a
real code edge exists — `ruff_r2il → r2sleigh` and `ruff_r2il → lance-graph` —
just not on the `lance-graph-java → r2sleigh` pair this session chose to test.
PR #94 names its own origin as a *"three-repo Phase Zero audit (ruff frontends
/ r2sleigh's typed surface / lance-graph V3 ABI)"*.

**The two laws that transfer (and the honest tier of each):**
- **[SHIPPED] Mask-native execution** — lance-graph-java: descriptors compose
  with zero membrane crossings, one fused terminal, `materializeRows()` the
  only named materialisation, enforced reflectively in test.
- **[SHIPPED] Typed behavioural IR** — ruff_r2il: `FunctionBehavior` over
  `SsaArtifact`, lossless/zero-copy, `harvested = classified + residual`,
  `dropped == 0` by construction, no catch-all slag variant, and SPO demoted to
  *"an optional lossy projection, never the behavioral truth"*.
- **[ARCHITECTURAL TRANSFER]** masking as zero-copy attention conductivity;
  BPE-style compression over behavioural IR.
- **[NOT YET PROVEN]** a production behavioural-BPE learner joining the two.
  Corroborating: PR #103's headline (*zero reconstruction mismatches across
  35,946 matched op sites*) is explicitly scoped by its own body to the
  oracle's `permissive_convention` and proves the reconstruction MECHANISM —
  the shipped `minimal_pass_one` measures `matched = 0` and round-trips
  through accounting, not matching. PR #103 also lists codebook wiring
  (`ogar_codebook`) as still open.

**The lesson, generalised.** "No reusable primitive" and "no architectural
connection" are different findings with different evidence. A grep for import
edges can only ever settle the first. Before writing an absence verdict about
a *relation*, search this repo's own `docs/architecture/` and board for the
relation by NAME — the answer here was two directories away the whole time.

**Reflexive note.** The failure has the same shape as the one this session was
convened to fix: reducing a structured relation to a single scalar test
(one grep, one boolean) and reporting the scalar as the whole answer.

