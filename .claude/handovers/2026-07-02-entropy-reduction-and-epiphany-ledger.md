# Entropy-reduction pass + epiphany ledger — 2026-07-02

> From: the `claude/medcare-bridge-lance-graph-wmx76z` session (medcare-bridge /
> ruff-harvester / cross-session synthesis). Companion to
> `2026-07-02-cross-session-wishlist-synthesis.md` — this file adds the
> **process epiphanies** (durable, fleet-wide) and turns the wishlist into a
> **status-annotated ledger** (claim-of-record) so executed items stop being
> re-derived. APPEND-ONLY; status lines updatable.

## Why this file exists (the entropy diagnosis)

Three entropies force re-derivation of reality every turn:
1. **Stale local state** — a session's branch lags a fast-moving main.
2. **Ephemeral epiphanies** — insights live in chat, evaporate at session end.
3. **No claim-of-record** — executed wishlist items have no done-marks; work
   gets nearly re-done (proven this session — see F7).

This file kills #2 and #3 durably; the paired rebase kills #1.

## Process epiphanies (F-pass, durable — the fleet needs these)

- **F1 — Pin propagation is the third leg of the cross-repo arc.** The mint +
  mirror rows + **lock pin** land in ONE arc, never split. An arc that ships the
  mirror-row change without bumping its own `Cargo.lock` pin ships a
  fuse-broken contract to every consumer while both sides' own tests stay green.
  (Root of the recurring "OGAR concept count changed, consumer build E0080s".)
- **F2 — Unpinned environments falsify TRUE states.** Verified live this
  session: a local `origin/main` remote-tracking ref read #636 then #631 then
  #636 across three commands (container re-provisioned to an older clone
  snapshot mid-turn). A verification whose environment isn't pinned can raise a
  false alarm on a correct state. Discipline: `git -C <abs>` + explicit
  re-`fetch` + echo the resolved SHA **before** concluding; never trust a
  remote-tracking ref you haven't refreshed this turn.
- **F3 — A gitignored lockfile is an *undecided* design, not a neutral one.**
  `lance-graph-ogar`'s own `Cargo.lock` is untracked → fresh checkouts build its
  `COUNT_FUSE` against OGAR **HEAD** (a floating canary) while the workspace
  lock pins. Canary or pin are both defensible; *accidental* is not. One doc
  sentence deciding it removes a "works here, breaks there" class.
- **F4 — Executed wishlists need a claim-of-record.** Convergent multi-session
  execution without a shared done-ledger risks double-work AND silently-dropped
  halves (e.g. a two-part item where only part 1 is confirmed). This ledger is
  the fix.
- **F5 — CI-invisible fuses only fire in *other people's* builds.** Main's CI
  never compiles the workspace-`exclude`d `lance-graph-ogar`, so lance-graph
  learns its own mirror broke the contract only when a *consumer* (medcare,
  twice) hits E0080. A single CI job that `cargo check`s the excluded crate
  makes the fuse fire where the change is made.
- **F6 — Rebase economics inverted.** At ~5 fleet-merges/day the board-prepend
  conflicts (EPIPHANIES/LATEST_STATE/PR_ARC) are the only recurring rebase cost;
  that's 30–60 min/day per parallel session. Per-entry board files (one file per
  entry + generated index) make the conflict structurally impossible — graduated
  from nice-to-have to cheapest recoverable overhead in the system.
- **F7 — Claim-of-record prevents near-collisions (proven).** I was one command
  from re-doing the `mint_factored`+`RadixCodebook`+`soc.rs` union — it was
  already unified on ruff branch `claude/medcare-ruff-csharp-sync-4iahey`
  (`94f919a`, "public-safe", −308 the medcare probe), just unmerged to ruff main.
  Only incidental fetch output revealed it. Without this ledger the collision
  recurs.

## Wishlist status ledger (claim-of-record)

Legend: **VERIFIED** (I read the code/count) · **CLAIMED-BY-COMMIT** (merge
message asserts it; not independently verified) · **OPEN** · **UNVERIFIED**.

### OGAR-lane items (forwarded → executed by #148 `75d955b`)
| # | Item | Status |
|---|---|---|
| 1 | OGAR-side flip fuse (`app_of/concept_of` vs contract `ClassidOrder`) | CLAIMED-BY-COMMIT (#148 "flip fuse") — verify test exists |
| 2 | Genetics `0x0E` domain + reserve `0x1000` + q2 APP_PREFIX row | PARTIAL — `0x1000` reservation + Genetics domain CLAIMED-BY-COMMIT (#148); **q2 APP_PREFIX row UNVERIFIED** |
| 4 | OGAR post-flip prose sweep | CLAIMED-BY-COMMIT (#148 "post-flip prose sweep") |
| 6 | Fill `0x08XX` OCR slots (`unicharset`/`recoder`/`charset`) | **VERIFIED** — 3 rows, codebook 65→68 |
| 7 | Two-sided `COUNT_FUSE` | CLAIMED-BY-COMMIT (#148 "COUNT_FUSE" two-sided) |
| — | Main's own ogar lock propagated to #148 | **VERIFIED** — main `Cargo.lock` pin = `75d955b`, mirror = 68 (F1 resolved on main) |

### Byte-truth / contract items
| # | Item | Status |
|---|---|---|
| 4b | `Facet`↔`FacetCascade` byte-parity test (cross-repo) | **UNVERIFIED (B4)** — flip fuse confirmed, byte-parity half not confirmed present |
| — | `emission_scan` / `RouteBucketTyped` (op-nexgen L1/L2) | CLAIMED-BY-COMMIT (#632) |

### ruff-lane items (this session's kept work — HELD by operator "don't start yet")
| # | Item | Status |
|---|---|---|
| E1 | `mint_factored`+`RadixCodebook`+`soc.rs` union | **DONE on branch** `94f919a` (F7) — UNMERGED to ruff main. Phase 1 shrinks to "verify + PR". |
| E2-py | `inherits_from` emission in `ruff_python_spo` | OPEN (gates mint_factored is_a axis on Odoo) |
| E2-ruby | ruby `extract_fields` | CEDED to op-nexgen R1 (migration-DSL column stratum is the better Rails truth source) |
| E3 | predicate-manifest parity test + C# `Program.cs`-vs-`Predicate::ALL` golden | OPEN |
| E5 | `Mint` → ndjson (now) / Arrow (after shared interchange-schema decision) | OPEN — landing zone is the #630/#631 WAL batch writer |

## Open operator decisions (unblock the fleet)
- **One authoritative high-half naming** — le-contract `domain:appid` vs OGAR
  general canon `domain:concept-slot`; settle once, pin as shared consts both
  repos reference (merges with the EdgeBlock `value480` vs `value496` divergence
  — same defect class, one ruling).
- **Probe-ledger Wave A green-light** (OGAR PROBE-SUBSTRATE-PROPOSAL §9).
- **F3** — `lance-graph-ogar` lockfile: canary or pin?

## This session's immediate next (guided by the epiphanies)
1. ~~Commit this ledger~~ (durability-first under unstable env — F2).
2. Rebase onto #636 (kills B1; F1 says verify+fix the lock IN the arc).
3. Verify B4 (byte-parity test present?) + item-1/2/7 CLAIMED→VERIFIED.
4. THEN await operator green-light on ruff Phase 1 (merge `94f919a`) vs W5.
