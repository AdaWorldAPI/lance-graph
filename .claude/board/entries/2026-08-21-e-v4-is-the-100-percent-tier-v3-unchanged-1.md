## 2026-08-21 — E-V4-IS-THE-100-PERCENT-TIER-V3-UNCHANGED-1 — operator ruling: V4 is a SIBLING for lossless/special-need coverage, not a successor

**Status:** RULING `[operator]` — explicit in-session decision, verbatim below.
**Confidence:** the ruling is authoritative; the CONSEQUENCES I draw from it are
marked separately and several are open questions, not inferences to act on.

**The ruling, verbatim (operator, 2026-08-21):**
> *"Meine Auffassung ist, V4 ist für R2IL 100% coverage und andere Special
> needs. V3 wie bisher."*
> ("V4 is for R2IL 100% coverage and other special needs. V3 as before.")

**What it settles.** V4 is **real and additive** — a sibling tier for workloads
requiring lossless / 100 % coverage, NOT a replacement for V3. **V3 is
explicitly unchanged** ("wie bisher"). The two coexist; V4 serves what V3's
deliberately projective design does not.

**Storno — this corrects my own entry from earlier today.**
`E-R2IL-VARNODEFACET-IS-A-G3-CARVING-AND-0xC4-WOULD-BIRTH-A-CLASS-INTO-IT-1`
argued the "V4" label "should not be adopted" on three grounds. Regraded, not
deleted:
1. *"reverses the ratified no-V4 verdict"* — **superseded.** The plan's
   `:112` verdict ("V3 is sufficient. No V4.") was a session verdict; this is
   an operator ruling, and rulings are frozen while verdicts are revisable.
   The plan's §22.5 stop condition is now SUPERSEDED, not violated. The plan
   text needs the correction appended.
2. *"V4 already denotes the dialectic-engine V4 field-search slice"* —
   **STANDS as a hygiene item only**, downgraded from architecture objection.
   `E-DIA-V4-FIELD-SEARCH-LOOP-1` is a plan V-slice ordinal in a different
   subsystem; the two are contextually separable but a future session grepping
   "V4" will hit both. Disambiguation owed (suggest: "V4 tier" vs "V4 slice").
3. *"implies V3-superseded"* — **WRONG, and the ruling says so directly.**
   "V3 wie bisher" is the explicit denial. This was my strongest objection and
   it was the one built on an assumption the operator did not share.

**Consequence for the G3 carving question — the frame changes, and this is the
part worth reading twice.** §3a's *"new classes MUST NOT be born into G1–G3"*
is a **V3** rule. If R2IL is a V4 tenant, §3a does not govern it, and the whole
carving question I posed — re-carve / named-exception / defer — was posed
inside the wrong frame. `VarnodeFacet`'s 3×32 density is then not a V3
violation needing an exception; it is **the reason R2IL is V4 in the first
place**. [MY INFERENCE, not the operator's words — flagged as such.]

**Consequence for `E-ADDRESS-FROM-THE-THING-NOT-THE-ACCIDENT-1` — its own
falsifier is PARTIALLY triggered.** That entry pre-registered: *"if a gate is
legitimately cleared by a fix that leaves the address derived from layout …
this entry is scoped down to the other instance."* The ruling does not say
"G3 is correct for R2IL" — it relocates R2IL to a tier whose carving discipline
is not yet stated. So the spatial instance is **suspended, not refuted**,
pending V4's actual layout. The temporal instance (F-ORD) is untouched and
still stands on its own measurement. Scoping down now would be premature;
scoping down later may be correct.

**What the ruling does NOT say — open, do not invent answers:**
- **V4's layout.** Facet width, carving discipline, whether rails apply at all,
  whether the 512-byte row stride is reused. Unknown.
- **Where V4 lives.** Contract crate? Separate module? A ClassView read-mode?
  Unknown.
- **What "andere Special needs" enumerates.** R2IL is one named tenant; the
  set is open.
- **How a reader routes V3 vs V4.** By classid? By domain byte (0xC4 is
  already the R2IL destination)? By an explicit tier discriminator? Unknown —
  and this is the question that most affects whether existing readers break.
- **Whether the `0xC4` mint is a V3 mint or a V4 mint.** Given the ruling, it
  is presumably V4 — but the domain enum lives in the V3-shaped
  `ogar_codebook` mirror, so the interaction needs stating.

**Cross-refs:** the two entries this corrects/suspends (both 2026-08-21),
`le-contract.md` §3 (L1–L8) + §3a (G1–G3), ruff plan
`r2il-behavioral-ir-v1.md` `:112` + §22.5 (both now superseded, correction
owed in-repo), `ogar_codebook.rs:112-117` (0xC4 BinaryLifting),
`E-DIA-V4-FIELD-SEARCH-LOOP-1` (the name collision),
`E-V3-FACET-4-PLUS-12` (the V3 lock that stays untouched).

**⊘ STORNO 2026-08-21 (operator correction, same day) — objection 2 above is
RETRACTED; there is no name collision and no disambiguation is owed.**
`dialectic-engine-v1.md`'s `V0…V5` are **integration-plan STEPS, like its own
`S1…S11`** — §4 is literally headed "Build order", and `V4` there is one stage
("the 64k SIMT lowering … only after V0–V3 green at small scale"), the fifth of
six. A step ordinal and a substrate tier are not two meanings of one label
competing for a namespace; they are two independent counters that never address
the same kind of thing. `E-DIA-V4-FIELD-SEARCH-LOOP-1` inherits the STEP
meaning — it names a finding from that stage — so it was never evidence of a
tier collision either. **Nothing should be renamed**, and a future session must
not read line 30's "disambiguation owed" as a live item; acting on it would
rewrite a correct plan to resolve a conflict that does not exist. What this
leaves: all three of my objections to the V4 label are now retracted, two as
superseded by the ruling and this one as simply mistaken about what the source
document says. The G3 reframing (below) and the five open questions are
unaffected — they never rested on this point.

