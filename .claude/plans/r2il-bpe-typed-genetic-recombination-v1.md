# R2IL × BPE as typed genetic recombination over the autopoiesis lanes (v1)

**Status: PROPOSAL.** Nothing in this doc is built or probed except where
explicitly marked GROUNDED. Do not cite this as an EPIPHANIES-grade finding —
it earns that only after the falsifiers in §5 run green. Written during a
same-day API outage (4/4 meta-review dispatch attempts failed to 529) so the
architecture doesn't evaporate before it can be probed.

## 1. The split (GROUNDED — matches shipped code + PR #998)

```
 SYSTEM / FROZEN                      AUTOPOIETIC SPACE
 native Rust, instinct                R2IL × BPE, evolvable thought
 immutable execution                  recombinable microcode
      │                                      │
      │                         ┌────────────┴────────────┐
      │                         ▼                         ▼
      │                    LEARNED lane               EXPLORE lane
      │                    retained                    frontier
      │                         ▲                         │
      └─────────────────────────┼─────────────────────────┘
                                 │
                          outcome / falsifier
                                 │
                            recombination
```

- `StyleLane::{Frozen, Learned, Explore}` already exist
  (`lance-graph-contract/src/soa_view.rs:52`, backed by
  `ValueTenant::{FrozenStyle,LearnedStyle,ExploreStyle}` at offsets
  152/164/176 — each lane is `[u8; 12]`, one V4 dock's payload).
- The triangle is **resonance-based, not an RL policy** — corrected this
  session from PR #998's actual commit body. MUL is the real admission
  gate (`GateDecision`, `Homeostasis`); the triangle judges the
  `RungReceipt`, never the raw problem.
- `[u8; 12]` per lane is exactly wide enough to hold an ordered sequence of
  BPE macro-id bytes (up to 12 macros/row at 1-byte ids, fewer at wider
  ids) — the SAME bytes the palette-rail reading (`style_rails_at`,
  `6×(u8:u8)`) already uses. Reading those 12 bytes as a macro-composition
  genome is a NEW `ClassView`-selected reading, not a new field — consistent
  with the "classid selects the reading, V4 never widens" rule.

## 2. Genotype, not atoms (GROUNDED — matches this session's POC finding)

> The genotype is not R2IL. The genotype is the ordered composition of
> R2IL/BPE microcode.

This is exactly what `probe_bpe_r2il_loco_microcode.rs` measured (B2,
committed `77e96863`): R2IL atoms (7, `call/cbranch/copy/int_add/load/
return/store`) are fixed; BPE merges over real def-use chains produce 33
reusable macros; def-use-chain carrier compresses 2.3× denser per domain
slot than a linear token stream (B3). The atoms never mutated — only their
composition did. That IS the "stable chemical language / reusable genes"
split the user names, already measured, just not yet named that way in the
probe's own language.

## 3. Four typed genetic operators (PROPOSED, unbuilt)

Deliberately restricted to four, chosen because each is exactly
reconstructible and falsifiable against the def-use-chain carrier already
proven in B2/B3:

```
splice       A|B            — concatenate two macros' atom sequences
substitute   μx → μy         — swap one macro reference for another
duplicate    μx → μx μx      — repeat a macro reference
delete       μx → ∅          — drop a macro reference
```

No crossover-selection policy, no eviction policy, no fitness function is
specified here — deliberately. Per the user: "noch keine konkrete
Eviction- oder Crossover-Policy erfinden. Die Architektur erlaubt
genetische Operatoren, aber welche davon tatsächlich sinnvoll sind, muss
der R2IL/BPE-Receipt zeigen." The receipt decides, not this doc.

## 4. Typed contracts (PROPOSED, unbuilt — the part that keeps this from being blind DNA shuffling)

> output type of left macro must satisfy input contract of right macro

R2IL already carries typed operands (space/offset/size, per the
`r2il-behavioral-carrier.md` knowledge doc). A `splice`/`substitute` is
only legal if the def-use chain's live-out set at the splice point
satisfies the live-in set of the spliced-in macro — this is a
straightforward extension of B2's existing def-use-chain extraction, not a
new type system. Unbuilt: no contract-checking pass exists yet over the 33
mined macros.

## 5. Selection = falsification, not raw fitness (PROPOSED — pattern already exists in this repo)

```
candidate → execute → observed consequence → truth + provenance
    → falsification / counterfactual → survives?
        no  → revise/drop
        yes → learned candidate
```

This is NOT a new mechanism to build — it's the same shape as PR #998's
`FreeEnergyComparison::minority_wins()` / `RevisionOutcome` (`MajorityHolds`
→ refuse, `Revised` → promote via `promote_family`) and the counterfactual
lane (`deposit_counterfactual`, `RawEdge -6`, never observed truth). A
genetic candidate from §3 is a new INPUT to that existing verdict path, not
a new verdict mechanism. This is the one place where "don't build a second
gate" (this session's earlier correction) stays enforced: recombination
proposes, MUL/the triangle still decides.

## 6. Frozen stays Rust, by design, not a temporary shortcut (GROUNDED reasoning)

`SYSTEM[Frozen]` stays hand-written/compiled Rust — no BPE expansion, no
dynamic routing, because the known system path is already fixed and cheap.
`LEARNED`/`EXPLORE` route through OGAR-loco → BPE macro composition → V4/R2IL
SoA ops, paying interpretation cost only where flexibility is actually
needed. Explicit non-goal (user's own words): **"learned = interpreted
forever" is the wrong invariant to bake in.** A learned macro proven stable
enough could later be AOT/JIT-compiled to native without losing its
"learned" provenance tag — but that compilation path is NOT built and NOT
scoped for this doc; it's named here only so the architecture doesn't
accidentally foreclose it.

## 7. What would need to be probed before any of §3–§5 lands as a finding

- Do the 33 macros mined in the POC actually admit non-trivial splice
  points under a live-out/live-in contract check (§4), or do real def-use
  chains turn out too entangled for splice to ever fire cleanly?
- Does `substitute`/`duplicate`/`delete` on a mined macro round-trip
  byte-exact through the existing B4 gate (it already proves round-trip for
  the UN-recombined macros; recombined ones are new territory)?
- Does routing a recombined candidate through the existing counterfactual
  lane (`deposit_counterfactual`) actually produce a distinguishable
  verdict from a non-recombined one, or does the signal wash out?

None of these are run. This doc is the spec for that probe, not its result.

## Fences

- Does NOT propose a new admission gate — MUL/triangle stays the only one.
- Does NOT propose a new SoA field — reads the existing 12-byte lanes
  differently, gated by `ClassView`, per the V4 "classid selects the
  reading" rule.
- Does NOT commit to a compilation/AOT mechanism — named as a non-goal to
  avoid foreclosing, not as a design.
- Does NOT claim "genetic recombination" is proven to work here — the name
  is apt for what B2 already measured (macro merging over def-use chains);
  the four operators in §3 are new and unmeasured.
