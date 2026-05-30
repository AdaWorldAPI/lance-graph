# discovery-origin-provenance-reconciliation-v1 — what is ACTUALLY correct vs what every source claims

> **READ BY:** the user (who said "I don't know what is correct"); any session about to implement D-ARM-1 / discovery_origin; the council before ratifying the byte layout.
> **Status:** RECONCILIATION. Documents a conflict; does NOT resolve it in code — resolution is the user's decision.
> **Authored:** 2026-05-30. Grounded in on-disk grep with file:line citations, not memory.
> **Instruction honored:** "only document the details." Nothing here is applied to code or to the plans. This file only records, per source, what cannot be right and what the canonical answer is.

---

## 0. The factual question first: "did all 70 get committed to #435?"

**There are 66, not 70, and yes — all 66 are in #435.** Two senses, both verified:

1. **Entity source is on the branch.** `git grep` on branch HEAD `4a6c2776` (branch `claude/activate-lance-graph-att-k2pHI`) counts **66** `pub const NAME: OdooEntity = OdooEntity` declarations across `l1.rs..l15.rs`. Committed in earlier Waves (`f5702675`, `d30186e5`, `333a1ff2`, `c04adf10`); reachable from the #435 HEAD. I did NOT re-commit them this session — they were already present.
2. **Full index of all 66 is in #435.** Commit `4a6c2776` added `odoo-blueprint-inventory-v1.md`, listing every one of the 66.

**Why "70" was wrong:** it came from `grep -c 'OdooEntity {'`, which also matched nested struct refs. Canonical count = the **66** module-scope `pub const ... : OdooEntity`. EXT-6 COVERAGE.md says "53" because it predates the 13 Wave-3 additions (L11-L15). 53 + 13 = 66. All reconcile. **66 is correct.**

---

## 1. GROUND TRUTH — what is actually in the code (the floor)

### 1.1 The only provenance enum that EXISTS in code

`crates/lance-graph-ontology/src/odoo_blueprint/mod.rs:450`:

```rust
pub enum OdooConfidence {
    Curated,     // human-curated L-docs (D-ODOO-BP-1b)
    Extracted,   // Odoo source via Python ast (D-ODOO-EXT-2)
    Conjecture,  // inferred, not yet validated against either source
}
```

**Three variants: Curated, Extracted, Conjecture.** That is the entire provenance vocabulary that compiles today. No `ArmDiscovered`, no `Ratified`, no `Derived`, no `ProvenanceTier` in any `.rs` file.

### 1.2 `discovery_origin` and `ProvenanceTier` do NOT exist in code

`git grep -lE 'discovery_origin|ProvenanceTier' -- '*.rs'` returns nothing. Both tokens appear ONLY in seven `.claude/` files (3 board, 2 handovers, 1 knowledge, 1 plan).

**This resolves the urgency.** The byte layout is a planning artifact; the WAL has NOT hardened around it because it is not in bytes yet. The canonical core spec's "ISA-ossification trap, live now" is real but the window is OPEN — fixing the width today costs a markdown edit, not a migration. No emergency; just a decision to get right before D-ARM-1 is written.

---

## 2. THE CONFLICT MATRIX — every source, verbatim, with citations

### 2.1 ProvenanceTier / confidence value-set, per source

| Source | File:line | Tier values stated | Count |
|---|---|---|---|
| **CODE (authoritative for "today")** | `mod.rs:450` `OdooConfidence` | Curated, Extracted, Conjecture | **3** |
| **My ARM plan, bit layout** | `streaming-arm-nars-discovery-v1.md` §7.2 | Curated, Extracted, ArmDiscovered, Ratified | **4** |
| **My ARM plan, deliverable D-ARM-1** | same file §8 | Curated, Extracted, ArmDiscovered, Ratified, Conjecture | **5** |
| **My survival dossier** | `odoo-extraction-strategies-v1.md` §5 | order: Curated > Extracted > Ratified > ArmDiscovered > Conjecture | **5** |
| **Canonical core spec (uploaded)** | `cognitive-risc-core.md` | Curated, Extracted, ArmDiscovered, Ratified  `[marked stable]` | **4** |
| **Canonical wikidata spec (uploaded)** | `wikidata-hhtl-load.md` | Curated, Extracted, Derived | **3** |

**Union of all distinct tier names across all sources:** Curated, Extracted, Conjecture, ArmDiscovered, Ratified, Derived = **6 distinct concepts.**

### 2.2 discovery_origin (u8) bit layout, per source

| Source | tier bits | proposer-id bits | reserved | proposer-id capacity |
|---|---|---|---|---|
| **My ARM plan §7.2 (what is committed to #435)** | bits 0-1 (2) | **bits 2-3 (2)** = AstWalker/PairStats/Aerial/Other | bits 4-7 (4) | **4** |
| **#434 cross-session review (my held queue, NOT applied)** | bits 0-1 (2) | **bits 2-4 (3)** = 8 | bits 5-7 (3) | **8** |
| **Canonical core spec (uploaded)** | bits 0-1 (2) | **bits 2-4 (3)** = 8, 6 already named, `[marked GROWS, AT RISK]` | bits 5-7 (3) | **8, declared insufficient** |
| **Canonical core spec, its OWN recommendation** | — | **widen to 6 bits (64) or move whole byte to u16** | — | **64 or more** |

---

## 3. THE ARITHMETIC — why these cannot all be right

Two hard numeric facts settle most of the confusion:

**Fact A — a 2-bit field holds exactly 4 values.** Every source that puts ProvenanceTier in bits 0-1 has 4 slots. But §2.1 shows the corpus names **6** distinct tiers. 6 > 4. Therefore:
- My ARM plan is internally inconsistent: §7.2 says 4 tiers (2 bits), but D-ARM-1 enumerates 5. **5 will not fit in 2 bits.** One of my own two statements is wrong.
- The code's `Conjecture` is dropped by the core spec's 4-set, and the wikidata spec's `Derived` is absent from the core spec's 4-set. So even the canonical specs, taken together, name 5 ({Curated, Extracted, ArmDiscovered, Ratified} ∪ {Derived}) and do not fit their own 2-bit field either.

**Fact B — proposer-id capacity by width:** 2 bits = 4, 3 bits = 8, 6 bits = 64.
- The committed plan (§7.2) gives **4** proposer slots and names 4 (AstWalker/PairStats/Aerial/Other) — already full, zero headroom.
- The review/core layout gives **8** and names 6 — 2 headroom.
- The core spec explicitly argues that "business logic is just another proposer" + dual-IPC means proposers proliferate, so **8 is also insufficient** and recommends 64 (6 bits) or u16.

**Conclusion the user can rely on:** the version currently committed to #435 (2-bit proposer-id, 4 slots, full) is the **most wrong** of all the layouts. Every later source widened it. The newest canonical source widens it furthest (to 64 or u16) and additionally exposes that the tier field is over-subscribed too.

---

## 4. WHAT IS CORRECT — per the now-canonical uploaded specs

The four uploaded `cognitive-risc-*` / `wikidata-hhtl-load` / `faiss-homology` specs are the user's blessed source of truth (they are explicitly "session-boot, invariants only, change only with a stated reason"). Per them, the correct positions are:

1. **proposer-id width: widen to 6 bits (64 proposers) or move `discovery_origin` to `u16`.** Neither the committed 2-bit nor the reviewed 3-bit is correct. This is the headline correction. Reason (quoted from core spec): *"Widen proposer field (steal reserved -> 6 bits/64, or go u16) before surrealkv WAL hardens the LE wire format. Once it's in the byte grammar, widening = migration across every component."*

2. **ProvenanceTier blessed set = {Curated, Extracted, ArmDiscovered, Ratified}, marked `[stable]`** in the core spec. This is the canonical 4. BUT see §6 — it omits both `Conjecture` (which is in code today) and `Derived` (which the wikidata spec uses). So "4 stable" is correct ONLY if Conjecture and Derived are handled outside this field (or the field is widened). That is an open decision, not a settled fact.

3. **The byte does not exist in code yet**, so the correct move is to fix the width in the PLAN before any `.rs` is written. No migration is owed.

4. **The homology is a cheat-sheet, not a dependency** (faiss-homology spec): do not pull FAISS/IVF code; the discovery_origin/provenance machinery is implemented natively. (Stated here because the specs foreground it and a future implementer might otherwise reach for FAISS.)

---

## 5. THE ONE INTERNAL CONTRADICTION I SHIPPED (flagged, NOT fixed)

In `streaming-arm-nars-discovery-v1.md` (committed to #435):

- **§7.2** allocates ProvenanceTier to **2 bits** and enumerates **4** values (Curated/Extracted/ArmDiscovered/Ratified — no Conjecture).
- **§8, D-ARM-1** specifies the enum as **5** values: `ProvenanceTier::{Curated, Extracted, ArmDiscovered, Ratified, Conjecture}`.

**5 values cannot be encoded in a 2-bit field.** This is a genuine bug in my own plan. I am NOT fixing it in this pass (instruction: only document). The fix is one of: (a) drop `Conjecture` from D-ARM-1 to match the 4-slot field and the core spec; (b) keep `Conjecture` and widen the tier field to 3 bits; (c) keep `Conjecture` in the code enum but exclude it from the on-wire `discovery_origin` byte (code enum and wire encoding need not be 1:1). My recommendation is (c) or (b), but the call is yours.

---

## 6. THE GENUINELY OPEN DECISIONS (yours to make)

These are NOT resolvable by citing a source — the sources disagree or are silent:

- **OD-1 — proposer-id final width: 6 bits (64) vs `u16` discovery_origin.** Core spec offers both; does not pick. 6-bit keeps the byte at u8 (cheaper, 64 proposers); u16 gives room for both a wider tier field AND 256+ proposers. Recommendation: **u16** if you expect the tier field to also grow (see OD-2); **6-bit/u8** if tier stays at the stable 4.
- **OD-2 — fate of `Conjecture` and `Derived`.** Six concepts, four canonical slots. Options: (a) cap at the core spec's 4 and map Conjecture->one of them, treat Derived as a separate "reasoning-store provenance" field (the wikidata spec already treats Derived as orthogonal/reasoning-tier, which argues for a separate field); (b) widen tier to 3 bits and admit all 6. Recommendation: **(a)** — Derived is conceptually a different axis (logical inference) from ArmDiscovered (statistical mining) and Curated/Extracted (source); squeezing it into the same 2-bit field is the category error.
- **OD-3 — code/spec divergence on Conjecture.** Code has it; core spec dropped it. Either update the code enum or update the spec. Pick one source of truth.

---

## 7. THE SECOND CONTESTED DETAIL — the Jirak noise-floor math (NOW FULLY VERIFIED against the live file)

The other "original wrong / fix suggested / not applied / unsure what's correct" item. As of this pass it is **verified line-by-line against `streaming-arm-nars-discovery-v1.md` on disk** — no longer a hedge.

### 7.1 The correct formula (settled — three independent confirmations)

The Jirak (2016) weak-dependence Berry-Esseen rate is `n^{-(p/2-1)}` for `p ∈ (2, 3]`, saturating at the classical ceiling `n^{-1/2}` for `p ≥ 3`. Confirmed by:
1. **CLAUDE.md iron rule `I-NOISE-FLOOR-JIRAK`**: *"Rate: `n^(p/2-1)` for `p ∈ (2,3]`, `n^(-1/2)` in L^q for `p ≥ 4`."* (The sign is implied negative — it is a convergence rate.)
2. **The plan's OWN worked examples** (line 381): *"p=4 → `n^{-1/2}`, p=2.5 → `n^{-0.25}`."* Check: `-(2.5/2-1) = -0.25`. ✓ These examples fit `n^{-(p/2-1)}` exactly.
3. **The plan's OWN blockquote** (line 375) states the rate correctly as `n^(p/2-1)`.

### 7.2 The bug — `n^{-1/(p/2-1)}` (reciprocal in the exponent), in 2 of the 3 places

- **Line 381 (prose):** writes the rate as `n^{-1/(p/2-1)}`. WRONG. At p=2.5 this is `n^{-1/0.25} = n^{-4}`, which **contradicts the same line's own example `n^{-0.25}`.**
- **Line 393 (the pseudocode threshold):** `(window_size as f32).powf(-1.0 / (p_moment / 2.0 - 1.0))`. WRONG — same reciprocal bug, in code form. Correct: `.powf(-(p_moment / 2.0 - 1.0))`.
- **Line 375 (blockquote) is RIGHT.** So the plan contradicts itself internally: 375 correct, 381 + 393 wrong.

**Magnitude of the bug at a real window:** n=100,000, p=2.5. Correct threshold `n^{-0.25} ≈ 0.056`. Buggy threshold `n^{-4} = 1e-20`. The buggy floor is effectively **zero**, so *every* candidate passes Stage A — the noise floor the rule calls "not optional" is silently disabled, producing exactly the substrate calcification `I-NOISE-FLOOR-JIRAK` exists to prevent. This is a real correctness bug, not cosmetic.

### 7.3 The default-p problem (separate from the formula bug)

Line 388: `p_moment: f32, // default 3.0`. With the CORRECT formula, p=3.0 gives `n^{-(3/2-1)} = n^{-1/2}` = classical Berry-Esseen exactly. So at the stated default the weak-dependence machinery yields **the same** bound as IID — the line-381 claim "much stricter than the IID `n^{-1/2}`" is false at p=3. To actually sit in the weak-dependence regime, default to **`p ≈ 2.5`** (`n^{-0.25}`, more conservative than classical). Either change the default to 2.5, or keep 3.0 and delete the "stricter than IID" claim. Recommended: **default 2.5.**

### 7.4 The fix, exactly (for whoever applies it — NOT applied here)

1. Line 381: `n^{-1/(p/2-1)}` -> `n^{-(p/2-1)}`; fix the garbled "p ≈ 3.0 ... giving `n^{-1}` decay" (it matches neither formula) to "p ≈ 2.5 ... giving `n^{-0.25}` decay, more conservative than the IID `n^{-1/2}`."
2. Line 393: `.powf(-1.0 / (p_moment / 2.0 - 1.0))` -> `.powf(-(p_moment / 2.0 - 1.0))`.
3. Line 388 + OQ-ARM-2: default `p_moment` 3.0 -> 2.5.
4. Leave line 375 and line 360 (D-ARM-7 spec) untouched — they are already correct.

Not applied in this pass (instruction: document only). The fix is unambiguous and ready when you greenlight it.

---

## 8. WHERE THE CANONICAL SPECS LIVE (and their own disagreement)

The four uploaded specs now live on **this branch** at `.claude/specs/` (verified post-rebase 2026-05-30 onto `origin/main` HEAD `4b00d049`; the specs landed via main commits `d1635dbe`, `93ac0463`, `a16d0f41`, `45276eb3`):
- `.claude/specs/cognitive-risc-core.md` (v0.1) — substrate invariants; the discovery_origin layout.
- `.claude/specs/cognitive-risc-classes.md` (v0.2) — class/shape layer; restates proposer-id width as freeze-time move N2; promotes F1 to "frozen identity under live resolution" and adds N1 (`class_id`/`shape_id` column before freeze) as a paired non-deferrable.
- `.claude/specs/faiss-homology-cam-pq.md` (v0.1) — CAM/CAM_PQ; homology-is-cheat-sheet warning; "Reasoning layer = separate indexed store, Derived tier" line that argues `Derived` is a different axis from `ArmDiscovered/Ratified`.
- `.claude/specs/wikidata-hhtl-load.md` (v0.1) — load pipeline; uses the `Derived` tier name and explicitly calls reasoning "orthogonal=beside, not mixed in."

**Correction of v1 of this doc:** the first version of this section said the specs lived only on `origin/claude/cognitive-risc-core-9PMW8`. That was true when written (pre-rebase) and became wrong when the specs were merged to main. Fixed 2026-05-30.

**The canonical set disagrees with itself on ONE point:** the core spec's tier set is `{Curated, Extracted, ArmDiscovered, Ratified}`; the wikidata spec's is `{Curated, Extracted, Derived}`. `Derived` vs `ArmDiscovered`+`Ratified` is unreconciled IN the canonical corpus. **Council reading (2026-05-30, 4 of 4 reviewers): this is a SPEC-OWNER decision, not a Claude-session decision.** Filed to `.claude/board/ISSUES.md` as OD-1/2/3 + the canonical-self-disagreement; do NOT ship `ProvenanceTier` into code until the spec owner reconciles. The reading the faiss-homology + wikidata specs together support — `Derived` is a separate orthogonal "reasoning provenance" axis, not a tier value, so the core's stable-4 holds — is offered as the council's recommendation if the spec owner wants a default to ratify or reject.

---

## 9. One-paragraph answer to "I don't know what is correct"

The 66 entities are correct and safely in #435. The `discovery_origin` byte exists only in plans, so nothing has hardened and there is no migration debt. Of the byte-layout versions on record, the one committed to #435 (2-bit proposer-id, 4 slots, full) is the most wrong; the cross-session review widened it to 3 bits (8); the canonical core spec says even 8 is too few and to go to 6 bits (64) or u16 — that last is the correct target. The ProvenanceTier field is separately over-subscribed: six tier names exist across the corpus but only four slots, and my own plan contradicts itself (4 in §7.2, 5 in D-ARM-1) — that needs an explicit decision (recommended: keep the canonical stable 4, treat `Derived` as a separate reasoning-provenance axis, decide `Conjecture`'s fate). The Jirak formula fix direction is known (`n^{-(p/2-1)}`, default `p≈2.5`) but the exact in-file target must be re-verified against the live plan before patching. None of this is applied; all of it is documented here for your decision.

---

## 10. Cross-refs

- `odoo-blueprint-inventory-v1.md` — the 66-entity index (answers §0).
- `odoo-extraction-strategies-v1.md` §5 — my 5-tier partial order (one of the conflicting sources, §2.1).
- `.claude/plans/streaming-arm-nars-discovery-v1.md` §7.2 + §8 — the committed byte layout + the internal contradiction (§5).
- `mod.rs:450` `OdooConfidence` — code ground truth (§1.1).
- `remotes/origin/claude/cognitive-risc-core-9PMW8:.claude/specs/*` — the canonical specs (§8).
- The four uploaded specs: cognitive-risc-core/classes, faiss-homology, wikidata-hhtl-load.

End of reconciliation. Documentation only — no code or plan was modified.
