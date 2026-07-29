# The A9 witness register is a LANE, not a ninth layout — the sub-byte LE contract

> Status: **EXPERIMENTAL reading — NOT in the operator-locked §3 catalogue**,
> and deliberately not petitioning for entry (see §1). Same honest posture as
> `tekamolo_facet.rs` ("legitimate under slot-purity … but not yet registered
> as a sanctioned §3 reading").
>
> Companion: `.claude/knowledge/le-contract-is-the-tenant.md` (the doctrine and
> the facet-serialization failure signature). Catalogue: `le-contract.md`
> §2 slot purity, §3 L1–L8, §3a G1–G3, §5 code ground truth.
>
> Born 2026-07-28: `CausalWitnessFacet` (A9, 24 signed `i4` loci) shipped
> citing **`le-contract §3 L9 G24N4`** — an entry that does not exist. §3 is
> L1–L8; grep for `G24N4|L9` across `soa_layout/` returns zero.

## §1 Why there is no L9 — the canon already refused this shape

The dangling citation is not merely *missing*; it is **wrong in kind**. Three
canon facts decide it:

1. **§3 is byte-axis by construction.** Its organizing principle is that every
   tier is a *byte* — "so the classview projects real rails and `group_of` is a
   pure shift." Byte accounting is exact: `6×2, 4×3, 3×4, 2×6`. A 24×4-bit
   carving is 96 bits ✓ but **sub-byte**: `group_of` stops being a pure shift
   and becomes shift-**and-mask**. That is precisely the property §3 exists to
   guarantee.
2. **§3a is the *wide* waiting room, not a general escape.** G1–G3 are
   *contiguous carvings wider than a byte*, for classes not yet decomposed
   **into** byte tiles, and the section states flatly that **`CascadeShape`
   gains no variants for them**. A nibble carving is the opposite direction —
   finer, not wider — so §3a does not cover it either.
3. **The canon has already declined a "ninth layout" for a structurally
   identical request.** The residual-ladder entry rules: *"this is **NOT a
   ninth 12-byte layout** — the residual ladder is **out-of-row** … the
   sanctioned in-row refinement budget remains the turbovec **6×4-bit nibble
   lane**."*

Fact 3 is the ruling. **Sub-byte granularity's sanctioned home is a LANE.**
The A9 register therefore lands as a value tenant (`ValueTenant` variant +
`VALUE_TENANTS` descriptor — the two places), exactly like the turbovec nibble
lane and `hi_chain/lo_chain`, and **not** as a §3 payload layout. No catalogue
petition is required, and none should be filed.

Consequence for `CascadeShape`: it gains **no** 24-group variant. It stays
byte-axis-only (`G6D2` / `G4D3` / `G3D4`), which is what §3a already demands.
`G24N4` is a *lane shape name*, never a `CascadeShape`.

## §2 The LE contract (the mandatory shape)

Three laws. Code that already *does* this without *saying* it has an
implementation accident, not a contract — the shipped accessors masked
correctly for months while the law went unwritten.

**Byte/nibble law.** 12 bytes, 24 slots. Slot `k` lives in byte `k / 2`;
**even `k` = low nibble, odd `k` = high nibble** — the nibble-granular
extension of little-endian (low first). Reading slot `k`:

```text
byte  = reg[k / 2]
raw   = if k even { byte & 0x0F } else { byte >> 4 }
value = sign_extend_i4(raw)          // (raw ^ 0x08).wrapping_sub(0x08) as i8
```

**Value law.** Each nibble is a two's-complement `i4`, sign-extended on read,
range **−8..+7**. `0` = **unbound** (zero-fallback: an un-populated slot reads
as *not consulted*, never as offset-zero-meaning-self). Slots `16..24` are
**reserved-zero** (RESERVE-DON'T-RECLAIM — held open, never padded with a
construct to reach 24).

The width is not arbitrary: `i4`'s 16 values **exactly tile** the ±8 reference
horizon, and that horizon is empirically anchored — Manning & Carpenter 1997
Table 7, maximum left-corner stack depth over the entire binarized WSJ Penn
Treebank is **8** (~99.4 % of configurations ≤ 5). Range and semantics are the
same fact.

**Placement law.** The reading exists **only** as a lane under the envelope's
LE contract: `ValueTenant` entry + `VALUE_TENANTS` descriptor + `verify_layout`
coverage + `ENVELOPE_LAYOUT_VERSION` discipline. **No lane, no reading.** A
`CausalWitnessFacet` living in a parallel owned `Vec` beside the real stream is
not a tenant — it is a facet serialization (failure signature: see the
companion knowledge doc).

## §3 Slot purity — already canon, restated for this lane

§2 of the catalogue is binding here and needs no extension: *"Labels and
positions come from the ClassView — NEVER from a slot in the payload… A
proposed facet layout containing a label/position slot is a LAYOUT-BREAK-class
defect."*

For this lane concretely: `Locus` and `LOCUS_LABELS` are **ClassView-side
lookup, outside the SoA**. Permitted direction `enum → &'static str`, indexed
by discriminant, for display only. **Forbidden: `str → slot`.** No name-keyed
lookup may ever exist — it is a serialization-shaped access path even with no
serde in sight. Audited 2026-07-28: no such path exists today (three uses, all
index-to-label); this fence keeps it that way.

## §4 Semantic invariant — loci, not magnitudes (operator-locked)

Every nibble is a **context pointer**, never a strength. Sign is orientation:
`−` = before / antecedent, `+` = after / consequent. The filler's meaning is
**read at the offset** (the event at `self_pos + offset`), never stored here —
`I-VSA-IDENTITIES` clean: identity pointers, not content.

Two consequences that are easy to violate and hard to detect:

- **Cross-tenant *pointers* are legitimate; cross-tenant *values* are not.** A
  locus holding "where my kausal cause was grounded" is provenance and belongs.
  The same locus holding the cause's *content* at lossy `i4` would be a
  re-encode of another tenant's axis and does not.
- **Never re-carve the register to fit one reading.** Narrowing slots bakes a
  reading into the layout and destroys the surface elections project over. The
  doctrine is already stated in-code: *"the rung level occupies ZERO slots:
  escalation is carried by the elected ClassView, which re-interprets which
  loci are live, never by a stored magnitude."* Elect with a mask; never delete
  slots. Masks fail **closed** — a missing or narrow mask elects *nothing*,
  never everything.

## §5 Open — flagged as PROPOSALS, for operator ruling (not law)

These are design opinions and are deliberately **not** baked into the contract
above. They may be ruled on independently of §§1–4, which stand on their own.

- **P1** — which axis the register keeps as its primary reading. Slot 7
  (`relativPronomen → antecedent`) is the only natively window-shaped locus;
  slots 0–3 (TEKAMOLO) and 4–6 (SPO) are provenance pointers into tenants that
  already exist in their own right (`ValueTenant::Tekamolo = 13`, `SpoFacet`).
- **P2** — whether slots 8–10 (`basin_anchor` / `supported_by` / `supports`)
  belong at all: they compress long-lived AriGraph graph relations into a ±8
  *stream* window whose referents are generally not 8 events away. The
  escape hatch (`out_of_horizon` → absolute address) is declared with no
  in-repo consumer.
- **P3** — `WitnessStream`'s fate: projection over the lane (preferred) vs.
  an explicitly test-only harness.
- **P4** — the two coexisting agreement semantics: facet-level `agrees_at`
  compares **bare** offsets (valid only for co-located rows); fabric-level
  `absolute_agreement` compares **absolute** events (`pos_a + o_a == pos_b +
  o_b`). One must become canonical and the other route through it.

## §6 Falsifier for this page

The contract is only real if a violation is detectable. Minimum gates on any
PR touching the lane:

- **Field-isolation matrix** — write each of the 24 slots in turn; assert every
  *other* slot is unchanged (`I-LEGACY-API-FEATURE-GATED`, mandatory whenever a
  layout is reclaimed or a `repr` changes).
- **Round-trip** — `to_register(from_register(b)) == b` for arbitrary `b`, and
  the borrowed view agrees byte-for-byte with the copy path.
- **Sign-extension edges** — slots holding `−8` and `+7` must round-trip;
  `0` must read as *unbound*, distinguishable from a stored zero offset.
- **Reserved-zero** — slots 16..24 read `0` on every fixture; a non-zero read
  there is a layout break, not a value.
