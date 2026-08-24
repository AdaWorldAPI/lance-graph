# R2IL × BPE as typed genetic recombination over the autopoiesis PALETTES (v1)

**Status: PROPOSAL, §7's three falsifiers now RUN (see §7).** The
recombination MECHANISM (§3's four operators as a buildable pass, §4's
contract-checking pass, §5's v3 admission/revision loop) is still
unbuilt/unprobed. Only §7's three narrow questions — do splice points
exist, does recombination round-trip, does the counterfactual lane
distinguish — have real measured answers now. Written during a same-day
API outage (4/4 meta-review dispatch attempts failed to 529) so the
architecture doesn't evaporate before it could be probed.

⚠ **§1 CORRECTED (architecture review, same day).** The original §1 read
`StyleLane`'s 12-byte payload as ONE style's ordered macro-genome (up to
12 macro-id bytes in a single row). That is wrong: `soa_view.rs:41`
documents each lane as **12 palette256-indexed slots, one per
`StyleFamily` ordinal**, each byte selecting one of **256 entries in that
lane's own palette** — a separate 256-wide address space per lane. The
12 bytes are ADDRESSES, not the genome. Corrected below; the rest of this
doc (§2–§7) is updated to match.

## 1. The split (GROUNDED — matches shipped code + PR #998)

```
 SYSTEM / FROZEN 256                  AUTOPOIETIC SPACE
 native Rust, fixed                   R2IL × BPE, evolvable thought
 immutable execution                  recombinable microcode
      │                                      │
      │                         ┌────────────┴────────────┐
      │                         ▼                         ▼
      │                  LEARNED palette [≤256]    EXPLORE palette [≤256]
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
- **Corrected addressing:** each of the 12 bytes in a lane is one
  `StyleFamily` ordinal's entry, and its VALUE is a palette256 index —
  `Frozen byte -> SystemPalette[id] -> native Rust`, `Learned/Explore byte
  -> {Learned,Explore}Palette[id] -> R2IL × BPE microcode`. The 256-wide
  palette per lane is where a composed macro-sequence actually LIVES (the
  genotype, §2); the 12-byte row only says which of up to 12 concurrent
  `StyleFamily` readers currently points at which palette entry. This is
  still a `ClassView`-selected reading of bytes already shipped — no new
  field — it is just not "12 macro-ids in one row."
- `ogar-loco` ROUTES R2IL × BPE microcode into these palettes; it does
  not have to OWN one `FnIndex` per macro to do so (the POC's own 90-slot
  number was demoted from "ceiling" to "one encoding's headroom" the same
  day this section was corrected — see `EPIPHANIES.md`).

## 2. Genotype, not atoms (GROUNDED — matches this session's POC finding)

> The genotype is not R2IL. The genotype is the ordered composition of
> R2IL/BPE microcode.

This is exactly what `probe_bpe_r2il_loco_microcode.rs` measured (B2): R2IL
atoms (7, `call/cbranch/copy/int_add/load/return/store`) are fixed; BPE
merges over real def-use chains produce 33 reusable macros. The atoms
never mutated — only their composition did. **B3's comparison against a
linear control was later relabeled** (architecture review): "saves more
tokens per merge on its own occurrence stream," not proven unconditional
compression superiority — the def-use-chain input overlaps/double-counts
source ops relative to the linear control. That relabeling does not touch
this section's claim (atoms fixed, composition varies) — only the
compression-superiority framing was too strong. The genotype — the
composed sequence, e.g. `[μ12, μ7, μ31, μ4]` — is content that would live
as ONE entry in a `{Learned,Explore}Palette` (§1), addressed by a
`StyleFamily` byte, not stored inline in the 12-byte row itself.

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

**RUN — `PROBE-R2IL-BPE-RECOMBINATION-FALSIFIERS-1`, 4/4 gates green,
`crates/lance-graph-planner/examples/probe_r2il_bpe_recombination_falsifiers.rs`.**
All three named below now have real, measured answers — see
`EPIPHANIES.md` for the full findings. §3–§5 are still PROPOSAL-status
for the parts these three falsifiers don't cover (the four operators as a
*building* mechanism, the contract-checking *pass*, and the v3
admission/revision loop are still unbuilt) — but the three questions
originally posed here are resolved:

- **Do the 33 macros admit non-trivial splice points under a live-out/
  live-in contract check?** YES, selectively: **107 of 1056 ordered pairs
  (10.1%)** admit ≥1 real type-legal splice point (a genuine observed
  def-use edge from A's tail into B's head, same episode); 949 do not.
  Real def-use chains discriminate — neither uniformly entangled nor
  uniformly permissive.
- **Does substitute/duplicate/delete round-trip through the B4-equivalent
  decode machinery?** YES — 10 genuinely distinguishable recombined
  sequences, 5 correctly-silent identity substitutions, plus a
  corrupt-table falsifiability demo (mirroring B4's own) proving the
  check can actually fail, not just pass.
- **Does a recombined candidate produce a distinguishable counterfactual
  verdict?** YES, at the REAL scope (`deposit_counterfactual` +
  `FreeEnergyComparison::minority_wins()` only — the v3
  `CounterfactualMailbox`/`revise_if_minority_wins`/`awareness.revise`
  path is still `todo!()`-stubbed, blocked on D-PERSONA-5, never called).
  An ordinary evidence-matched baseline correctly does NOT win
  (`minority_wins()=false`); the strongest recombined splice pair (45
  pooled disjoint episodes) DOES win (`minority_wins()=true`) against the
  identical weak majority. The verdicts differ — the signal does not wash
  out on this corpus, at the primitive level that is actually shipped.

## Fences

- Does NOT propose a new admission gate — MUL/triangle stays the only one.
- Does NOT propose a new SoA field — reads the existing 12-byte lanes as
  `StyleFamily`-indexed palette256 addresses (per §1's correction), gated
  by `ClassView`, per the V4 "classid selects the reading" rule. Does NOT
  propose that the 12 bytes themselves store an ordered macro genome.
- Does NOT claim the 90-free-`FnIndex`-slots number bounds this
  proposal's capacity — that number is one encoding's headroom, not the
  Learned/Explore palette size (256 each, per §1).
- Does NOT commit to a compilation/AOT mechanism — named as a non-goal to
  avoid foreclosing, not as a design.
- Does NOT claim "genetic recombination" is proven to work here — the name
  is apt for what B2 already measured (macro merging over def-use chains);
  the four operators in §3 are new and unmeasured.
