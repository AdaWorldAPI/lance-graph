# V3-Migration Technical-Debt Ledger — the §2.3 condition-3 burn-down

> **Purpose.** Make `soa-value-tenant-migration-v2.md` **§2.3 condition 3** ("the
> V3-schema-migration technical debt is resolved AND V3 identity is confirmed
> working") *checkable* rather than a vibe. The operator's Canon:Custom flip is gated
> on this ledger going green. This is the **first Phase-1-closeout deliverable**.
>
> **Status:** OPEN (3 OPEN / 1 PARTIAL-clean). Authored 2026-06-26 from a direct
> audit of the tree (file:line evidence below) — **not** a paraphrase.
>
> **Sign-off model (operator-directed 2026-06-26):** *"I'd let the other sessions
> sign off and add their perspective."* So each item carries an **append-only
> Perspectives** slot, and the doc ends with a global **Sign-off ledger**. A fresh
> session does NOT edit a past entry — it appends a dated perspective / verdict.
> Suggested panel (the arc's `READ BY:` + the brutally-honest reviewers):
> `integration-lead`, `truth-architect`, `core-first-architect`,
> `baton-handoff-auditor`, `brutally-honest-tester`.
>
> **Reads with:** `soa-value-tenant-migration-v2.md` (§2.1 L2, §2.2, §2.3, §5, §7),
> `canonical_node.rs`, `ogar_codebook.rs`, OGAR #128 (`E-CLASSID-ENVELOPE-PARSER`),
> iron rule `I-LEGACY-API-FEATURE-GATED`.

---

## What "confirm identity works" means (the exit bar, per item)

For every V3 tail, before the flip: the `I-LEGACY-API-FEATURE-GATED` discipline —
a **field-isolation matrix** (write each key/value field, assert all others
unchanged) **+ a version gate** on the serialize/deserialize path — proven
end-to-end. "Identity works" = a V3-shaped node round-trips, routes to the right
domain, and reads its tenants, with no v1/v2 bit aliasing. The flip is **not** a
substitute for this proof; it happens *after* it.

---

## DEBT-1 — the POC-`Full` default `value_schema` is unreverted  ·  OPEN

**What.** `ReadMode::DEFAULT.value_schema` is still the temporary POC `Full`, not the
canonical `Bootstrap`. An unconfigured classid therefore materializes the whole
480 B slab instead of the zero-fallback minimum.

**Evidence (verified).**
- `canonical_node.rs:968-970` — `ReadMode::DEFAULT { tail_variant: V1, value_schema:
  ValueSchema::Full, … }`.
- `canonical_node.rs:961-965` — doc-comment: *"**TEMPORARY (2026-06-15 POC):**
  `value_schema = Full` … When the POC ends, flip `value_schema` back to `Bootstrap`."*
- `canonical_node.rs:1044-1048` — the structural parity-fuse const already names the
  canonical target triple `{ V1, Bootstrap, CoarseOnly }`.
- `canonical_node.rs:1877-1905` — the revert/guard test pins the POC-`Full` default
  (asserts `.value_schema == Full` "until the POC ends").

**Done when.** `ReadMode::DEFAULT.value_schema` flips `Full → Bootstrap`; the POC
guard test is retired or inverted (assert `Bootstrap`); the §2.1 **L2** canonical
triple `{V1, Bootstrap, CoarseOnly}` is the live default; and the consumers that
relied on the Full default still read (the slab a class needs is its *own*
`value_schema`, not the default). Per §2.1 L2 this is one of **two independent
field flips** — keep it separable from the `tail_variant` work.

**Gate served.** §2.3 cond. 3 (the default read-mode must be canonical before a
prefix reorg). **Touches shipped `canonical_node.rs` → operator go.**

**Perspectives (append-only):**
- _(2026-06-26, authoring session)_ — flagged + grounded; not fixed. Separable from
  DEBT-2 per §2.1 L2; do them as two commits.
- _…next session: append your verdict here…_

## DEBT-2 — the parity fuse is structural-only; OGAR has not coded `tail_variant`  ·  OPEN

**What.** The contract's `tail_variant` axis is wired (the `ReadMode` field +
`BUILTIN_READ_MODES` + tests), but the **producer (OGAR)** codes `tail_variant` only
in prose, so the cross-system parity guard is a *compile-time field-set* assertion,
not a *runtime-vs-OGAR* comparison. Until OGAR codes it, the envelope-parser contract
(#128 `classid → {tail_variant, value_schema, edge_codec}`) is half-realized.

**Evidence (verified).**
- `canonical_node.rs:1038-1048` — *"Structural parity fuse … OGAR #128 is doc-only
  today, so there is no runtime OGAR struct to compare against yet; this upgrades to a
  runtime fuse when OGAR codes its registry's `tail_variant`."*
- OGAR repo: `tail_variant` appears **only** in `OGAR/.claude/board/EPIPHANIES.md` +
  `OGAR/docs/DISCOVERY-MAP.md` — **zero source files** (grep, 2026-06-26).
- Contract side is live: `canonical_node.rs:943` (`ReadMode.tail_variant`), `:1022`
  (OSINT-V3 `tail_variant: V3`), `:1986-2061` (tail_variant tests).

**Done when.** OGAR's registry/envelope-parser codes `tail_variant` (the `classid →
{tail_variant, value_schema, edge_codec}` resolution), and the contract fuse upgrades
from structural to **runtime-vs-OGAR** (the shape `ogar_codebook`'s `COUNT_FUSE`
already has because its `ogar_vocab` counterpart exists). This is **OGAR-repo work**
(in scope: `/home/user/OGAR`) + a contract-side fuse upgrade.

**Gate served.** §2.3 cond. 3 + §2.1 parity-fuse. **OGAR is in scope; the wiring is
producer-side.**

**Perspectives (append-only):**
- _(2026-06-26, authoring session)_ — confirmed OGAR-side gap is real (docs-only).
  This is the substance of §5's "Phase-1 leans almost entirely on OGAR."
- _…next session: append…_

## DEBT-3 — the §5 casing-miss producer/consumer sweep  ·  PARTIAL (consumers clean)

**What.** §5 names the corrective `/home/user/{OGAR,MedCare-rs}` sweep (the earlier
agent searched lowercase paths and missed the real clones) as Phase-1's *actual
substance*. This audit did the targeted version.

**Evidence (verified, 2026-06-26).**
- **OGAR (producer):** the V3 wiring it owes *is* DEBT-2 (code `tail_variant`). No
  separate OGAR debt surfaced beyond it.
- **MedCare-rs (consumer):** grep for `tail_variant | CLASSID | classid_read_mode |
  ReadMode | guid-v3 | TailVariant | new_v2 | new_v3` → **No files found.** MedCare-rs
  does **not** consume the V3/classid machinery today — it is a *clean* future
  consumer, nothing to rewire now.

**Done when.** DEBT-2 lands (the OGAR producer side); and at each FMA-V3 / CPIC-V3
mint (DEBT-4) the relevant consumer is re-confirmed clean or wired via the pull-don't-
reconstruct rule (`ogar-consumer-preflight.md`: `*Port::class_id` /
`canonical_concept_id`, never a local `*Bridge`/codebook copy). MedCare-rs's current
zero-coupling is the baseline to preserve.

**Gate served.** §2.3 cond. 3 + §5 (gates Phase-1 *start*). **Read-only audit done;
the producer wiring is DEBT-2.**

**Perspectives (append-only):**
- _(2026-06-26, authoring session)_ — MedCare-rs is clean (0 refs); the sweep's weight
  is OGAR-producer (DEBT-2), not consumer rewiring. A consumer-side `baton-handoff-
  auditor` pass at mint time would harden this.
- _…next session: append…_

## DEBT-4 — FMA-V3 + CPIC-V3 are unminted; CPIC needs a Genetics domain slot  ·  OPEN

**What.** Phase 1's remaining identity mints. OSINT-V3 shipped (#613); FMA and CPIC
do not exist in V3, and CPIC has no domain at all.

**Evidence (verified).**
- FMA: `canonical_node.rs:65` `CLASSID_FMA = 0x0000_0A01` (V1); no `CLASSID_FMA_V3`.
  Target per §2.2: `0x1000_0A01` (Anatomy route `0x0A01` intact).
- CPIC/Genetics: no `CLASSID_CPIC`/Genetics classid anywhere. `ogar_codebook.rs`
  `ConceptDomain` (`:88-98`): assigned `0x00`–`0x0D` (note `0x0D = HR`, `:97`); free
  high-byte slots `0x03–0x06`, `0x0E+` (`:77` "Any high-byte slot not yet assigned …
  `0x03XX`–`0x06XX`, `0x0EXX+`"; tests `:441-442` assert `0x0500`/`0x0E00` →
  `Unassigned`).

**Done when.** (1) FMA-V3 minted (`CLASSID_FMA_V3 = 0x1000_0A01` + `ReadMode::FMA_V3`
+ `BUILTIN_READ_MODES` under `guid-v3-tail`, mirroring OSINT-V3). (2) A **Genetics
domain slot** chosen (`0x03`/`0x04`/`0x05`/`0x06`/`0x0E`) + minted in `ogar_codebook`
(mirror OGAR `ogar_vocab`), then `CLASSID_CPIC_V3 = 0x1000_0?00`. (3) Each proven with
the DEBT exit bar (field-isolation + version gate).

**Gate served.** §2.2 + §2.3 cond. 2 (the V3 set) and cond. 3 (proven). **Touches
shipped `canonical_node.rs` + `ogar_codebook.rs` → operator go; the Genetics slot is
an operator/codebook decision.**

**Perspectives (append-only):**
- _(2026-06-26, authoring session)_ — FMA-V3 is the clean first mint; CPIC is blocked
  on the Genetics-slot pick (a codebook decision, not code). Don't mint CPIC before the
  slot is chosen.
- _…next session: append…_

---

## Burn-down summary

| id | item | status | blocks the flip? | needs operator go? |
|---|---|---|---|---|
| DEBT-1 | POC-`Full` → `Bootstrap` default | OPEN | yes (cond. 3) | yes (shipped `canonical_node.rs`) |
| DEBT-2 | OGAR codes `tail_variant`; fuse → runtime | OPEN | yes (cond. 3) | producer-side OGAR work |
| DEBT-3 | casing-miss sweep | PARTIAL — consumers clean; = DEBT-2 producer-side | folds into DEBT-2 | — |
| DEBT-4 | FMA-V3 + CPIC-V3 mints (+ Genetics slot) | OPEN | yes (cond. 2 + 3) | yes + the Genetics-slot pick |

**The flip is unblocked only when DEBT-1, DEBT-2, DEBT-4 are green and each V3 tail
passes the exit bar.** `facet_mint` (brick 2) is independent of all of this — its
`facet_classid` is a parameter.

## Sign-off ledger (APPEND-ONLY — other sessions add rows; never edit a past row)

| date | session / agent | scope reviewed | verdict | note |
|---|---|---|---|---|
| 2026-06-26 | authoring session (`claude/serene-mayer-1a09he`) | DEBT-1..4 grounded from tree | DRAFTED | file:line evidence captured; awaiting independent sign-off |
| _…_ | _…_ | _…_ | CONFIRM / DISPUTE / ADD | _…_ |
