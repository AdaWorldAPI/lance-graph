# Consumer Scan TODO — cross-repo latent-issue sweep (2026-06-23)

> Two latent issue-classes surfaced this session. Both are *anticipatable*
> across every consumer, not one-offs. This is the scan checklist: run each
> check against each client, tick the box, file an issue on any hit.
>
> Status keys: `[ ]` = not scanned · `[x]` = scanned + clean · `[!]` = hit (file issue) · `[fixed]` = scanned + repaired this arc.

---

## Issue Class A — OGAR codebook-mirror drift

**Symptom:** `error[E0080]` at `lance-graph-ogar/src/lib.rs` COUNT_FUSE, or a
runtime `assert_codebook_parity` panic — fires in **every** build that vendors
the OGAR git dep when `ogar_vocab::class_ids::ALL` and a local mirror disagree
on concept count or domain.

**Root cause:** a hand-maintained copy of the codebook that is NOT bound to
`ogar_vocab` by `pub use`. The zero-dep contract mirror cannot re-export (it has
no dep on ogar-vocab by design), so it is the ONE surface that drifts.

**The rule (E-CODEBOOK-MINT-IS-A-CROSS-REPO-ARC):** an OGAR concept mint is a
cross-repo arc. DoD = OGAR `ogar-vocab` entry **+** `lance-graph-contract::ogar_codebook`
mirror rows + `ConceptDomain` variant + `canonical_concept_domain` arm **+**
`lance-graph-ogar::parity::domains_agree` arm — all in the same arc.

### Scan check
For each client: does it carry a codebook/class-id table that is a **copy**
rather than a `pub use ogar_vocab::class_ids::*` re-export? If yes → drift risk.

```
# per repo:
rg -n "ConceptDomain|canonical_concept_domain|class_ids::ALL|0x0[0-9A-F]{3}" --type rust
# a re-export ("pub use ogar_vocab::class_ids::*") is SAFE; a literal id table is the risk.
```

### Per-client status
- [fixed] **lance-graph-contract** `ogar_codebook::CODEBOOK` — the one hand-maintained mirror. Synced to 43 (added 0x0B auth family). This is the only client that can structurally drift; guard it hardest.
- [x] **openproject-nexgen-rs** `op-canon/src/class_ids.rs` — `pub use ogar_vocab::class_ids::*` re-export → cannot drift. CLEAN.
- [x] **MedCare-rs** — SWEPT 2026-06-23 (workspace-wide id-table grep): no literal `0xDDCC`/`ConceptDomain` table in any `medcare-*` crate. Vendors `lance-graph-ogar` + `ogar-vocab` (git main); the Health set is inherited via the upstream gate, not a hand-list. CLEAN.
- [x] **smb-office-rs** — SWEPT 2026-06-23: no local codebook copy; consumes via `lance-graph-contract`. CLEAN.
- [x] **woa-rs** — SWEPT 2026-06-23: no literal id table; `WoaPort` pulls classid via PortSpec. CLEAN.
- [x] **q2 / ladybug-rs / crewai-rust / n8n-rs / rs-graph-llm** — SWEPT 2026-06-23: none carries a literal OGAR id table (most don't consume the codebook at all). CLEAN.

> **Sweep result (2026-06-23):** a workspace-wide grep for literal codebook id-tables
> (`const … CODEBOOK`, `=> ConceptDomain::…`, `pub const … : u16 = 0xDD…`) returns
> exactly TWO authoritative tables: `OGAR/ogar-vocab` (the source) and
> `lance-graph-contract::ogar_codebook` (the mirror, now fused at 43). Every other
> consumer re-exports or pulls via PortSpec. **Class-A drift surface = the one mirror.**

### Guard to add (prevents recurrence)
- [ ] Any client that maintains its own concept list MUST either (a) `pub use ogar_vocab::class_ids::*`, or (b) add a `const _` COUNT_FUSE against `ogar_vocab::class_ids::ALL.len()`. Hand-lists with neither are the bug.

---

## Issue Class B — PaaS deploy crash (Railway / Heroku / Cloud Run / Fly)

**Symptom (B1 — unreachable):** container starts, binds a fixed port (e.g.
`0.0.0.0:3000`), platform routes its public edge to `$PORT` (often 8080) → the
app is up but the proxy can't reach it (the `shuttle.proxy…:45472 > :3000`
non-resolve medcare hit).

**Symptom (B2 — crash-loop):** the app writes to a CWD-relative dir
(`./audit`, `./data`, `./cache`) that is read-only on the container image →
`PermissionDenied` at boot, fail-closed crash-loop (the medcare
`MEDCARE_AUDIT_DIR` crash).

### Scan checks
```
# B1 — fixed-port bind without $PORT fallback:
rg -n "TcpListener::bind|SocketAddr|\.listen|bind\(" --type rust src
#   HIT if the bind addr is a config/literal with NO `std::env::var("PORT")` branch.

# B2 — CWD-relative writable path in a sink/store/cache initializer:
rg -n '"\./|from\("\.|PathBuf::from\("[^/]' --type rust src
#   HIT if a write target defaults to a relative path instead of a writable data root.
```

### Per-client status (web-app `main`/`server`/`serve` found)
- [fixed] **MedCare-rs** `medcare-server/src/main.rs` — `$PORT` bind branch added; audit dir now derives `<lance-data-root>/audit` (writable), not `./audit`. Both classes repaired.
- [ ] **woa-rs** `src/main.rs` — axum; verify `$PORT` bind + Tresor/PDF/sled write dirs use a writable data root (Stefan's Railway deploy is production).
- [ ] **openproject-nexgen-rs** `op-server/src/main.rs` — verify `$PORT` + any RLS/audit write dir.
- [ ] **q2** `crates/quarto-hub/src/server.rs` — verify `$PORT` + hub doc-store path.
- [ ] **rs-graph-llm** `insurance-claims-service` / `medical-document-service` / `recommendation-service` / `notebook` mains — verify `$PORT` + any session/Lance store path.
- [ ] **n8n-rs** `n8n-server/src/main.rs` — verify `$PORT` + sled/background-work dir.
- [ ] **crewai-rust** `src/bin/server.rs` — verify `$PORT` + any cache dir.
- [ ] **ladybug-rs** `src/bin/server.rs` — verify `$PORT` (note: uses CogRedis, separate concern).
- [ ] **spider** `spider_worker/src/main.rs` — verify `$PORT` if exposed.
- [ ] **lance-graph** `lance-graph-planner/src/serve.rs`, `cognitive-shader-driver/src/{serve.rs,bin/serve.rs}` — these are lab/serve surfaces; verify `$PORT` before any are deployed.

### Canonical pattern (copy from medcare)
```rust
// B1: bind $PORT when set (all interfaces), else fall back to config.
let addr: SocketAddr = match std::env::var("PORT") {
    Ok(p) if !p.trim().is_empty() => format!("0.0.0.0:{}", p.trim()).parse()?,
    _ => settings.listen.parse()?,
};
// B2: derive writable paths from the data root the platform mounts,
//     never a CWD-relative "./audit" / "./data".
```

---

## How to run this sweep
1. One pass per repo with the rg checks above (5 min each).
2. Tick `[x]` clean / `[!]` hit. For each `[!]`, file an issue in that repo's
   board (`ISSUES.md` / `Altlasten.md` / `braid`) pointing at this doc.
3. Class-A hits also append to OGAR `EPIPHANIES` (mint-arc rule) if a new
   un-guarded mirror is discovered.
4. Re-run after the next OGAR codebook mint — Class A is recurring by nature.
