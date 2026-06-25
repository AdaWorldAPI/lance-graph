# SoA Value-Tenant Migration v2 — Operator-Locked Phase Sequencing

> **Status:** OPERATOR-LOCKED SEQUENCING (2026-06-25). Supersedes the v1 BRIEF's
> implicit single-pass framing: the migration is **ONE migration in TWO ordered
> phases** — **Phase 1 identity→V3 (key-side)**, then **Phase 2 V3-shaped value
> tenants (value-side)**. The per-tenant §5 migration body stays gated on the two
> 5+3 panels (unchanged, harvest §7); THIS doc locks only the *ordering* + its
> forcing rationale — which the panels do **not** arbitrate, because it is forced
> by the envelope parser, not chosen.
>
> Reads with: `soa-value-tenant-migration-v1.md` (brief),
> `soa-value-tenant-migration-v1-harvest.md` (the filled §4 inventory = the
> Phase-2 input), `substrate-unification-thesis.md` §8, OGAR #128 (the
> producer-side envelope parser).

## 0. The lock, one line

Identity is the **coordinate system**; the value tenants are expressed in it.
**Migrate the address to V3 first; shape the tenants to V3 second.** Not a
preference — a forcing function (§1).

## 1. Why identity-first is the ONLY correct order (forced, not chosen)

OGAR #128's envelope parser resolves `classid → registry → {tail_variant,
value_schema, edge_codec}`, and **`tail_variant` (the identity/key shape) is
resolved UPSTREAM of `value_schema` (the tenant shape).** You physically cannot
shape value tenants to a V3 geometry the key does not yet express — the parser
reads `tail_variant` *first* to know how to read everything after it.

This mirrors the thesis axes: **identity** (key-as-place) and **structure**
(HHTL = `(part_of:is_a)`) are the **key-side** axes; the value tenants are read
*through* them. A V3 address carries the two orthogonal hierarchies
(where-it-is ⊥ what-it-is); the Phase-2 contained facet's
`helix-place(6) ‖ CAM-PQ(6)` shape is a **reflection of that same part_of/is_a
split**. The facet is a shadow the V3 key casts — so the key casts first.

## 2. Phase 1 — identity → V3 (key-side; NOT in the harvest's value inventory)

The harvest correctly **bracketed this key-side** ("a SEPARATE, zero-dep SoA …
NOT wired into `NodeRow.value`", harvest §4) — so it is *not* covered by the
value-tenant inventory; it is the prerequisite that inventory assumes.

- **New-gen classids carry a leading-`1` generation marker in the HIGH (custom)
  u16, preserving the canon low u16** (e.g. OSINT `0x0000_0700 → 0x1000_0700`; see
  §2.2) → route through the **V3 `tail_variant`** in the OGAR registry; the HHTL
  tiers read as the `(part_of:is_a)` 8:8 tile (`perturbation-sim/src/cascade_key.rs`
  `CascadeKeyV3`, already coded + tested: `v3_two_hierarchies_are_independent`).
- **Coexist-by-classid, NOT rewrite.** Legacy zero-prefix classids keep their
  current `tail_variant` — **V1** (the default `family·identity` tail) or **V2**
  (`new_v2` `leaf·family·identity` 3×u16, `guid-v2-tail`-gated — what q2's
  `osint-bake` already mints). **RESERVE-DON'T-RECLAIM ⇒ zero re-mint of the
  V1/V2 corpus**, layout-preserving: the same 16-byte key, tiers *reinterpreted*
  by `tail_variant`, never re-carved.
- **Mostly an OGAR-registry + envelope-parser wiring job** — exactly the
  producer side OGAR #128's canon (`E-CLASSID-ENVELOPE-PARSER` / `D-ENVPARSE`)
  specced (**doc-only**; the `tail_variant` axis is to-wire on OGAR's side too).
- **Grade `[H]`**, gate **F-update**: the cheap end of the thesis's F-update
  axis — but "zero-cost V2/V3 coexistence" is *precisely* the claim F-update
  must confirm. A V2 tail silently reinterpreted under a V3 reader is the
  `I-LEGACY-API-FEATURE-GATED` failure mode; the same version-gate +
  field-isolation discipline that closed the q2 `new_v2` seam applies to the
  tail.

### 2.1 The reusable pattern — extend the one `ReadMode`, never a public `new_v3`

P-A's mechanism **must BE Phase-2's mechanism, not merely enable it.** Both sides
hang off the **single** `classid → ReadMode` dispatch (`canonical_node.rs:912`
`classid_read_mode`), whose doc-invariant (:810) is "consumers and OGAR read the
identical schema." `ReadMode` (:815) already carries the value reading; identity
is a **third field on the same struct**, resolved by the same lookup — mirroring
OGAR #128's `classid → {tail_variant, value_schema, edge_codec}`:

**Target struct** — the live `ReadMode` (`canonical_node.rs:815`) carries only
`value_schema` + `edge_codec`; **P-A (PR #613) adds the `tail_variant` field**:

```rust
pub struct ReadMode {
    pub tail_variant: TailVariant,   // P-A (#613) ADDS — which KEY shape (resolved first, per #128's parse order)
    pub value_schema: ValueSchema,   // existing — which tenants
    pub edge_codec: EdgeCodecFlavor, // existing — edges
}
```

**The symmetric spine** (the test that it serves Phase 2):

| side | call | shapes |
|---|---|---|
| key (Phase 1) | `mint_for(classid_read_mode(c).tail_variant, …)` | the address |
| value (Phase 2) | `to_node_row(classid_read_mode(c).value_schema, …)` | the tenants |

Same `classid_read_mode(c)`, sibling fields. Consumers **mint by classid**, never
hardcode v1/v2/v3 — exactly as they later shape tenants by classid. Migrating a
class's **identity** to V3 = flip `tail_variant` on its `ReadMode` const; later
migrating its **tenants** = flip `value_schema` on the *same* const. Two
field-flips on one registry entry; no consumer rewrite either time.

**Litmus (guardrail):** anything that is **not** `classid_read_mode(c).<field>` —
a consumer `if classid_is_v3 {…}`, or a public `new_v3` consumers call directly —
is the *layer-not-column* anti-pattern and will not serve Phase 2. Reject it.

**Three locks (cross-session grounding, 2026-06-25):**
- **L1 — `ReadMode::DEFAULT.tail_variant = V1`** (the zero-fallback const at :833).
  Every un-minted classid stays V1 → **zero re-mint of the V1/V2 corpus**
  (RESERVE-DON'T-RECLAIM). V3 is opt-in per-classid via `BUILTIN_READ_MODES`
  (:891), never the default.
- **L2 — keep the POC-`Full` flip separable.** `ReadMode::DEFAULT.value_schema`
  is still the unreverted `Full` POC (:825-833, guarded by the revert test).
  Adding `tail_variant` and flipping `value_schema → Bootstrap` are **two
  independent field migrations on one struct** — document the canonical target
  triple `{V1, Bootstrap, CoarseOnly}`, do not entangle them.
- **L3 — mechanism is NOT blocked on the classid lock (P-C).** Extending
  `ReadMode` + the `TailVariant` enum + the `mint_for` carrier + the
  `guid-v3-tail` gate are additive / default-V1 / feature-gated → they land
  **now**, non-breaking. Only the per-consumer `classid → tail_variant: V3`
  **entries** (the high-u16 gen-marker placement, §2.2) need the P-C operator-lock.

**Parity fuse — structural-against-canon, NOT runtime-vs-struct.** OGAR #128
(`E-CLASSID-ENVELOPE-PARSER` / `D-ENVPARSE`, merged, **doc-only**) pins the target
verbatim — *"the registry entry must gain the `tail_variant` (V2/V3) axis beside
`ReadMode {value_schema, edge_codec}`"* — but `tail_variant` is **to-wire on
OGAR's side too** (the pieces #128 lists as CODED — `classid_read_mode` /
`BUILTIN_READ_MODES` / `ClassView` / `new_v2` / `cascade_key`-V3 — carry none). So
the fuse asserts the contract `ReadMode`'s field set **== the three canon axes**
`{tail_variant, value_schema, edge_codec}` (a compile-time *structural* guard
against the canon), **not** a runtime comparison against a live OGAR struct — that
is the shape the codebook `COUNT_FUSE` has only because its `ogar_vocab`
counterpart already exists. It upgrades to the runtime form once OGAR codes its
registry's `tail_variant`.

### 2.2 P-C classid lock (operator-ratified 2026-06-25)

The classid u32 is **`[ custom (hi u16) : canon (lo u16) ]`** — `classid_concept_domain`
routes on the **low** u16 (the OGAR `0xDDCC` codebook; `canonical_node.rs:43`,
`ogar_codebook.rs:103`). So the **generation marker goes in the HIGH (custom) u16,
leaving the canon low u16 untouched** — "replace the first `0` with `1`" on the
*full* u32: `0x0000_0700 → 0x1000_0700`. (Codex-P1 correction: a low-half form like
the earlier `0x1007` overwrites the domain byte — `0x1007 as u16 >> 8 = 0x10` →
Unassigned — so it is **rejected**.) The live `0xDDCC` consts are kept as-is.

| consumer | live classid | V3 (hi-u16 marker) | domain route (`as u16`) | status |
|---|---|---|---|---|
| OSINT | `0x0000_0700` | **`0x1000_0700`** | `0x0700` → Osint ✓ | **wired (#613)** — test asserts the route |
| FMA | `0x0000_0A01` | **`0x1000_0A01`** | `0x0A01` → Anatomy ✓ | deferred ("rest later") |
| CPIC/Genetics | `0x000C_…` | **`0x1000_0?00`** | `0x0?00` → Genetics | **domain TBD** — `0x0D` is **HR** in the contract (`ogar_codebook.rs:97`); Genetics needs a free slot (`0x03–0x06` or `0x0E`). Deferred. |

`ReadMode::DEFAULT.tail_variant = V1` (L1) keeps every other classid legacy; the V3
entries are `guid-v3-tail`-gated. OSINT-V3 is wired in P-A (#613); FMA-V3 + the
Genetics-domain mint + CPIC's move follow.

## 3. Phase 2 — V3-shaped value tenants (value-side; = the harvest)

This is what `-v1-harvest.md` inventoried. With the address already V3:

- The ClassView *reading* shapes the value to match the now-V3 key: the
  contained `facet_classid(4) | helix-place(6) | CAM-PQ(6)` facet
  (HelixResidue → facet), the **8 KEEP** tenants under `Cognitive`/`Full`, and
  Qualia + the future thinking-style i4-32D **DEFER** (substrate-validation).
- **Gates F-1 + F-code** (unchanged). The per-tenant §5 body is written for real
  only after the two independent 5+3 panels sign off (harvest §7). THIS doc does
  not pre-empt them — it only fixes that Phase 2 runs *after* Phase 1, in V3
  coordinates.

## 4. Two structural payoffs of this order

1. **De-risks Finding A (the two-worlds seam).** The canonical slab and the
   parallel `MailboxSoA` share exactly **one** semantic column:
   `entity_type ≡ class_id` (`soa_view.rs:75`). Identity-first migrates **that
   shared anchor** to V3 *before* the A↔B reconciliation — so Phase 2 reconciles
   the two worlds **already in V3 coordinates**, instead of reconciling them in
   V2 and redoing it once the address moves.
2. **Dissolves the harvest's scope question.** "Is the harvest the migration or
   the inventory?" → it is precisely the **Phase-2 input**; Phase 1 is the
   prerequisite it correctly left key-side. No scope confusion survives once the
   two phases are named.

## 5. The watch — Phase 1's substance IS the OGAR gap

Phase 1 leans almost entirely on OGAR (the registry, the envelope parser, the
high-u16 `0x1000_xxxx` gen-marker mint path, §2.2). The harvest's OGAR sweep was the **casing-miss gap**
(harvest §6.1 — the cross-repo agent searched `/home/user/ogar`; the clone is
`/home/user/OGAR`, and `/home/user/MedCare-rs` likewise). So the corrective
`/home/user/{OGAR,MedCare-rs}` sweep is **not optional polish deferred to
"before §5"** — it is **Phase-1's actual substance** and gates Phase-1 start.
OGAR is where `classid → ReadMode` is *produced* (`canonical_node.rs:888`,
"a minted class's read-mode is layered in one level up"); Phase 1 cannot wire
the V3 `tail_variant` without reading that producer side.

## 6. Sequencing summary

| Phase | Object | Side | Mechanism | Grade | Gate | Prereq |
|---|---|---|---|---|---|---|
| **1** | identity → V3 | key | OGAR registry + envelope parser; high-u16 `0x1000_xxxx` gen-marker → V3 `tail_variant`; coexist-by-classid | `[H]` | F-update | OGAR/MedCare-rs corrective sweep |
| **2** | V3-shaped tenants | value | ClassView reading: contained facet + 8 KEEP + 2 DEFER | `[H]`/`[S]` | F-1 + F-code | Phase 1 **+** 5+3 panels |

## Cross-references
- `soa-value-tenant-migration-v1.md` (brief — §2 where-to-read, §3 gates, §6 5+3).
- `soa-value-tenant-migration-v1-harvest.md` (the filled inventory = Phase-2 input).
- `substrate-unification-thesis.md` §8 (the facet as a reflection of the V3 key).
- `perturbation-sim/src/cascade_key.rs` (`CascadeKeyV3`, `v3_two_hierarchies_are_independent`).
- `canonical_node.rs` (`new_v2:244` gated; `classid_read_mode`; `:888` the OGAR read-mode producer).
- `guid-v2-tail-per-family-codebook-v1.md` (the tail repartition this V3 prefix coexists with).
- OGAR #128 — `E-CLASSID-ENVELOPE-PARSER` / `D-ENVPARSE` (merged, **doc-only**): `classid → registry → {tail_variant, value_schema, edge_codec}`; the `tail_variant` axis is to-wire on both sides.
- Iron rules: `I-LEGACY-API-FEATURE-GATED` (the V2/V3 tail version-gate), `I-VSA-IDENTITIES` (the facet's disjoint carve), `I-SUBSTRATE-MARKOV`.
