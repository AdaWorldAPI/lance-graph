# Cross-Session Broadcast — Committed, Curated Append-Only

> **This IS committed.** Unlike AGENT_LOG.md (gitignored, ephemeral),
> every entry here travels with the repo. Use ONLY for messages another
> session MUST see before starting work — architectural decisions,
> urgent corrections, findings that can't wait for the next PR merge.
>
> Most coordination belongs in Layer A (teleport role switch) or Layer B
> (local AGENT_LOG.md). See `.claude/AGENT_COORDINATION.md` §Layer C for
> when to use this channel instead.
>
> **Append via `cat >>` heredoc** — no Read, no overwrite, pre-allowed.

---

## Entries (reverse chronological)

## 2026-04-24 — AGENT_LOG.md gitignored; architecture moved to .claude/AGENT_COORDINATION.md

After 3 merge conflicts in one session from parallel agents appending
to a committed AGENT_LOG.md, the split landed: architectural docs
(three coordination layers, canonical append pattern) moved to
`.claude/AGENT_COORDINATION.md` (committed). Per-session log is now
gitignored. Durable findings continue in EPIPHANIES.md. See
`.claude/AGENT_COORDINATION.md` for the new governance.

## 2026-04-25 — auth-rls-lite shipped: RlsRewriter without xz2/liblzma collision

**For:** smb-office-rs session (bus REQUEST e1cf316 + PR #12)
**Commit:** 34e236b on claude/teleport-session-setup-wMZfb
**Branch:** also rebased onto main (860d082 auth split already on main via PR #267)

**New features on lance-graph-callcenter:**

```toml
# In smb-bridge Cargo.toml — use this:
lance-graph-callcenter = { path = "...", features = ["auth-rls-lite"] }
```

- `auth-rls-lite` = `auth-jwt` + `query-lite` (datafusion with `default-features = false`)
- `auth-rls` = `auth-jwt` + `query` (datafusion full — same as before, still collides with lance)
- `query-lite` = datafusion minimal (logical plan + optimizer, no compression backends)

**What this unblocks:**
1. `smb-bridge::auth` collapses to re-export of `lance_graph_callcenter::auth::*`
2. `auth-rls-lite` gives you `RlsRewriter` + `OptimizerRule` without xz2/liblzma conflict
3. Wire `RlsRewriter::new(actor)` over F4 connectors immediately

**What this does NOT do:** The full `auth-rls` (with compression) still collides.
That needs datafusion 52+ or upstream xz2 fix. But `auth-rls-lite` gives you
everything RlsRewriter actually uses (common, logical_expr, optimizer).
