# Classid Canon:Custom Half-Order Flip — Plan v1 (the migration is TRIGGERED)

> **Status:** ACTIVE (2026-07-02, operator trigger — verbatim: *"so yes now is
> the right time for the migration — document it accordingly using
> .claude/board and /plans"*). This is the migration `soa-value-tenant-
> migration-v2.md` §2.3 sequenced as "the atomic Canon:Custom half-order flip
> follows once the V3 set is complete" — the V3 set completed (three classes
> wired, both carves matrix-certified, PR #626), and the operator pulled the
> trigger.
>
> Implemented per the standing directive `E-CLASSID-SPLIT-ORDER-IS-A-FLIP`:
> ONE flippable split-order definition, `flip(flip(x)) == x` probed — never
> scattered per-site byte surgery.

---

## §0 The ruling (operator, 2026-07-02 — the semantic ground truth)

Three verbatim anchors, decoded:

1. *"0700 is OSINT domain, 00 as in applied domainweise"* — `0xDD` names the
   domain; low byte `00` = the domain applied **domain-wise** (the domain
   itself as the class).
2. *"0701 is q2 as the OSINT appid, our consumer"* — within a domain the low
   byte is **APPID space**, not concept vocabulary (q2 = `0x01`). Concept
   vocabularies exist only in domains that have them (project mgmt, ERP,
   Health, …); OSINT has none (OGAR PR #146 removed the two hallucinated
   rows; count 67 → 65, COUNT_FUSE balances with zero mirror changes).
3. *"(0x0701_1000 — or for mentally understanding, the domain appid; and 1000
   was just a temporary V3 substrate marker as a hard reminder for V3
   substrate migration — (0x07:01::1000"* — the **target stored form** puts
   the canon half HIGH and the custom half LOW:

```
human-readable   0x07 : 01 :: 1000
                 domain appid  custom (today: the temporary V3 marker)

stored (target)  0x0701_1000      hi u16 = CANON  (domain:appid)
                                  lo u16 = CUSTOM (marker/render, later the
                                           64k dynamic classviews × bitmask)

stored (legacy)  0x1000_0701      hi u16 = marker, lo u16 = canon
                                  (the pre-flip PR #618/#626 form)
```

The `0x1000` marker is **temporary by declaration** — a "hard reminder" of the
V3 migration, not a permanent format bit. Its retirement is §4 Phase 4 (an
operator checkpoint, NOT part of the flip itself).

Also in the ruling: q2's push gate is **waived** ("temporary precaution to not
break the cockpit while working on so many different domains") — the q2-side
phases below are unblocked.

## §1 What the flip IS (and is not)

- **IS:** a swap of which u16 half of the stored `classid: u32` carries the
  canon (`domain:appid`) and which carries the custom (marker/render) half —
  routed through ONE compose/split definition so the swap is a one-place flag.
- **IS NOT:** a re-carve of the 16-byte key (bytes 0..4 stay the classid; no
  `ENVELOPE_LAYOUT_VERSION` change), a change to HEEL/HIP/TWIG/tails, or a
  change to any concept id value. Only the classid's internal half-order moves.
- **The central risk (name it before it bites):** legacy classids
  (`0x0000_0700`-form, hi = 0) read WRONG under a naive global flip
  (`0x0000_0700` would decode as canon `0x0000`). So the flip is
  **mint-forward with a version boundary**, never a blanket reinterpretation —
  full `I-LEGACY-API-FEATURE-GATED` discipline: the same accessor must never
  silently mean two things; readers route through the split definition, which
  knows the boundary; a field-isolation/round-trip matrix ships with Phase 1.

## §2 Site inventory (what routes through the flip)

| Site | Today | Post-flip |
|---|---|---|
| `contract::canonical_node` `CLASSID_OSINT_V3` | `0x1000_0700` | **`0x0701_1000`** (OSINT:q2 — note the appid normalization: the q2 class is `:01`, not the domain root) |
| `CLASSID_FMA_V3` | `0x1000_0A01` | `0x0A01_1000` (Anatomy:q2 — already `:01`, confirmed by the ruling) |
| `CLASSID_CPIC_V3` | `0x1000_0E00` | `0x0E01_1000` (Genetics:q2 — normalized `:00`→`:01` per "same for cpic also under q2") |
| `BUILTIN_READ_MODES` keys | old-form u32s | both forms during transition (the registry maps concrete u32s, so coexistence is free); old-form keys retire at Phase 3 |
| `ogar_codebook::classid_concept_domain` | routes `classid as u16` (lo) | routes the CANON half via `split_classid` |
| `hhtl::NiblePath::from_guid_prefix{,_v3}` | v1 fold refuses `classid >> 16 != 0`; v3 fold ignores classid | both route the marker/canon test through `split_classid` |
| `ogar_codebook::{render_classid, classid_app_prefix, classid_concept}` (OGAR#95 hi-u16 app-prefix mirror) | `(prefix<<16)\|concept` | RECONCILE: the #95 app-prefix scheme put apps in the HI half; the ruling puts canon (domain:appid) HI and custom LO. These must converge on ONE composition — flagged as the Phase-2 operator checkpoint (the #95 table may become the CUSTOM-half render catalogue, or the appid byte subsumes it) |
| q2 `osint-bake` | mints `CLASSID_OSINT` (`0x0000_0700`) rows via `new_v2` | re-mint as `0x0701_1000` via `mint_for(classid_read_mode(c).tail_variant, …)` |
| q2 `cpic` local mirror | domain `0x000C`, V1 tail (ISS-Q2-CPIC-MIRROR…) | re-mint into `0x0E01_1000` + V3 tail by PULLING the contract (dissolves the divergence issue in the same pass) |
| OGAR vocab emission (Phase B of the one-row registry, D-VCW-4) | n/a | emits new-form classids only |

## §3 The mechanism — one flippable definition

Add to `lance-graph-contract` (the zero-dep single source):

```rust
/// The ONE classid composition. canon = domain:appid (e.g. 0x0701),
/// custom = marker/render half (today the temporary V3 reminder 0x1000).
/// CLASSID_CANON_HIGH selects the half-order; flipping it IS the migration.
pub const CLASSID_CANON_HIGH: bool = /* false = legacy, true = target */;
pub const fn compose_classid(canon: u16, custom: u16) -> u32;
pub const fn split_classid(classid: u32) -> (u16 /*canon*/, u16 /*custom*/);
```

Probes (mandatory, ship WITH the definition): `split(compose(c,x)) == (c,x)`
both orders; `flip(flip(id)) == id`; domain routing invariant under the flag
for all registered classids; the legacy-boundary matrix (every pre-flip wired
classid decodes identically through the split as before the refactor).

## §4 Phases

| Phase | Content | Gate |
|---|---|---|
| **P0** | Land `compose/split/CLASSID_CANON_HIGH=false` + route every §2 lance-graph site through it. ZERO behavior change (probed: all 763+ tests identical). | contract suite green, probes green |
| **P1** | Flip `CLASSID_CANON_HIGH=true` + mint the three new-form classids (`0x0701_1000`/`0x0A01_1000`/`0x0E01_1000`) into the registry ALONGSIDE the old forms (coexistence). | round-trip + boundary matrix + `flip(flip)` green |
| **P2** | Reconcile the OGAR#95 hi-u16 app-prefix mirror with the new order (operator checkpoint — see §2 row); OGAR emits new-form vocab. | operator nod on the #95 reconciliation |
| **P3** | q2 re-mints (osint-bake → `0x0701_1000`; cpic → `0x0E01_1000` via contract pull, dissolving `ISS-Q2-CPIC-MIRROR…`); old-form registry keys retire. | q2 bakes green; fuse green |
| **P4** | `0x1000` marker retirement decision — the custom half opens for the real render catalogue ("later it's 64k dynamic classes and classviews × bitmask"). | **operator checkpoint** |

## §5 What this plan deliberately does NOT do

- No key-layout change (16/16/480 locked; classid bytes 0..4 fixed offsets).
- No blanket reinterpretation of legacy classids (mint-forward only).
- No unilateral retirement of the `0x1000` marker (P4 is yours).
- No touching of concept ids in domains that HAVE vocabularies.

## Cross-references

- Ruling + OGAR execution: OGAR PR #146 + `DISCOVERY-MAP.md`
  `D-OSINT-APPID-NOT-CONCEPT`; lance-graph `ISS-OSINT-SYSTEM-ROOT-SLOT-
  VIOLATION` (RESOLVED by the ruling — sharper than its Options A/B).
- Sequencing parent: `soa-value-tenant-migration-v2.md` §2.3 (this IS that
  flip); `v3-convergence-wiring-v1.md` (D-VCW-4 registry work feeds P2).
- Directives honored: `E-CLASSID-SPLIT-ORDER-IS-A-FLIP` (one definition),
  `E-CLASSID-HUMANREADABLE-REORDER-DEFERRED` (now superseded BY TRIGGER —
  status flip recorded on the board, entry preserved),
  `I-LEGACY-API-FEATURE-GATED` (the boundary discipline).
