# Ledger update, 2026-10-16 (b): deduplicated against PR #1579

Updates `LEDGER-REVISION-2026-10-16.md` (merged in #1578). That file is not
edited; this one supersedes the rows named below.

**Why.** PR #1579 (another session, branch `ccr-9f3b8075-74u47l-synergy-map`,
board only) mapped 123 fleet merges against the 2026-10-11 SIMD-fleet listing
and opened while #1578 was in review. Several of its loose ends are the same
items as rows in the 2026-10-16 revision under different names. This file
names each overlap once, keeps one id per item, and records what #1579 adds
that the revision did not have. Read from #1579 as merged (`7302f58c`). Its last commit (`97b744ec`) already
reconciled the synergy map against the 2026-10-16 revision (§8 of
`entries/2026-10-16-synergy-map-since-1011.md`) and measured label codebook
sizes (§9). Where that reconciliation names a ledger id, this file uses the
same id.

## 1. Overlaps between the two PRs

| here | #1579 | resolution |
|---|---|---|
| C30 (arc entry for #1575, LATEST_STATE section) | PR_ARC entry for #1575, LATEST_STATE section | Same records, written twice. #1579 resolved its conflict by keeping one #1575 arc entry (`a1b79feb`). Closed. |
| N17 relalg absorption, fruit L/H | D1 "one set algebra, four spellings", H1 (#1579 §8: "use N17") | Id kept: **N17**. **Regraded: not low.** #1579 D1 shows the fix has no agreed home until D-RA-2 decides whether relalg's rules move into the contract set IR (D-CMR-1, still Queued) or relalg is deleted. The two reproducers can still become failing tests now; the fix waits on D-RA-2. |
| N18 version clock not wired into a reader, soak outstanding | arc "WAL": reader wiring, 60-min soak truncated at 27.5 min | Same item. Keep N18; the soak figure (27.5 of 60 min) comes from #1579. |
| N19 layout costs, no A/B/C verdict | D3 read masks, clustering, prefilter; H3 S3 arm; H4 clustered write layout | Same stream. #1579 adds the operational form: bytes fall only when survivors are page-local, so the write layout must cluster by the apertures that read it (H4), and on object storage clustering cuts requests, not only bytes (H3). Keep N19 as the umbrella; H3/H4 are its next steps. |
| N20 qualia CLUT | "Research docs" arc, `ISS-QUALIA-A2B-CLUT-UNMEASURED` | Same issue row. No change. |
| N15 → D-ASM | V2 (D-ASM W1–W3, not authorized) and V1 (label codebooks) | Same item. **New input from #1579 §9 (measured):** no two corpora fit one 256-entry codebook (MeTTa stdlib alone has 266 symbols; per script at most 45), so V1 now reads "one codebook per classid and space, not one shared". D-ASM-1's `LabelCodebook` should be that per-space codebook; decide before W1 builds. |
| K2' one evidence root, O30 width | D4 truth: D-NALP-4 (per-row truth lane vs scalar law per survivor), D-HL-3 evidential base | Adjacent, not identical. O30 decides how wide a root list is; D-NALP-4 decides whether a population carries truth per row. Both must agree before revision runs over populations (D-NALP-5). Linked, kept separate. |
| N16 operand registry (`u16 WordId → u8` operand per sealed epoch) | D5 / V1 label codebooks | **Related, not the same** (#1579 §8). N16 maps word ids to 8:8:8 wire operands per epoch; V1 maps symbols to label ids per classid and space. Shared shape: both are frozen, version-identified codebooks selected by scope, and the 256 bound in #1579 §9 is the same `u8` operand width N16 targets. Kept as two items. |

## 2. Rows from #1579 not in the 2026-10-16 revision

Recorded here by reference so this ledger stays complete; #1579 owns the
detail.

| id | item | where |
|---|---|---|
| D2 | One preimage operation (run set under a monotone map) serves D-CMR-5, D-ASM-2, D-RA-4 and Gremlin `out()`; the D-MTP-1b probe is its oracle | #1579 §4 D2, fruit H2 |
| D6 | D-SAG-5-1..10 are queued against D-SAG-1, which #1524 withdrew; they wait on the D-CMR rebuild | #1579 §4 D6, V4 |
| X11 | `F32x8::mul_add` is documented as fused on every arm; the wasm arm without `relaxed-simd` does two roundings | #1579 §1, L4 (ndarray) |
| X19 | lgj config comment stale; `check_build_cpu` not called in the handshake | #1579 L5 (lance-graph-java) |
| D8 | Doc drift that contradicts build rulings (`.cargo/config.toml:55-56`, tesseract and lichess toolchain prose, TD-POPCNT item (b)) | #1579 L3 |
| L6 | #1544 open only as a reminder of D-NALP-4; close with an ISSUES row | #1579 L6 |
| STATUS sweep | ten rows still read "In PR" after their merges | done in #1579 |
| X18 / D7 | Two `ndarray` package ids | closed in #1579: the root and relalg `[patch]` sections resolve OGAR's git ndarray to the same `../ndarray` |

## 3. Done in this PR

| id | what | evidence |
|---|---|---|
| DOC-WIRE (2) | The three stale lines that still described bits 59..60 as two readings now carry a dated correction pointing at `causal-edge/src/layout.rs`'s `TrustTexture` doc: `.claude/knowledge/causal-plane-inventory.md` (after the two-readings paragraph), `docs/TYPE_DUPLICATION_MAP.md` (row 1), `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` (after the collapse table). The originals are kept, marked superseded. | this PR |
| N21 | The five verification commands from #1570's body, run. All exit 0: `haspelmath` 6/6; `deepnsm-v2 --all-targets` 202 + 14 passed; clippy `-D warnings` and fmt clean; the example prints `SPO=(doctor, gave, letter) ce64=0000000000020100 map=Connected`. | guarded-executor run; the haspelmath count re-run on the main thread |

DOC-WIRE (1) (one global reading of 43..52 in
`docs/architecture/ce64-semantic-upper-half.md`) is not touched: it waits on
the operator's yes.

## 4. Open items after this update

Unchanged from the 2026-10-16 revision unless named above: O25 + K2' (needs
O30), O30, K6 (D-HL-5 H4), O29, D-46, D-MATURATION3-1, ABL, D-ASM (now
with the per-space codebook input), O23, R2–R4, O26, O7/O13/O15, DOC-WIRE
(1), BAND-RETIRE, N16 (related to V1), N18, N19, N20, and N17 (gated on
D-RA-2). Closed by this update: C30 duplicate, DOC-WIRE (2), N21.

Low-hanging left in this ledger's own scope: R3/R4 wording, and the two
absorption reproducers as failing tests (the fix waits on D-RA-2).
