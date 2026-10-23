# Goldstaub — method and efficiency gold that must never dilute (append-only)

> **Append-only.** Prepend new entries. An entry is never rewritten,
> paraphrased or merged into a neighbour. It is superseded only by a new
> dated entry plus a `⊘ SUPERSEDED <date> by <entry>` line on the old one. A
> near-duplicate adds its receipt to the existing entry instead of opening a
> new one; that strengthens the entry rather than diluting it. Added
> 2026-10-23. The name and the shape come from `woa-rs` / `WoA`
> `.claude/board/Goldstaub.md`.

## What belongs here

Small, durable insights about method or efficiency. Each one costs real time
again every time a session forgets it. They fail the EPIPHANIES gate
correctly, because they change no architecture. Two tests admit them instead:

1. **Would a future session pay this cost again without the entry?**
2. **Is there a receipt** (the command, the number, the date)?

An insight already enforced mechanically (a CI gate, a test, a hook) gets a
one-line pointer here, not a second explanation.

The "session traps" sections in repo `CLAUDE.md` files are goldstaub kept in
the wrong place: they load into every session and grow without bound. New
gold lands here. A `CLAUDE.md` keeps at most one line pointing to it.

Each entry: `**Claim:**`, `**Receipt:**`, `**Scope:**`, `**Date:**`.

---

## Entries

### G-2026-10-23-1 — on synthetic data, the world construction can decide a probe's verdict before it runs
**Claim:** A probe whose verdict depends on the data distribution (evidence
LUT vs geometry LUT, shadow-mask salience) decides nothing on synthetic data,
because whoever builds the world sets the outcome. A mechanism probe (does an
operation preserve a property?) stays valid on synthetic data only if it
holds across at least two unlike worlds. Before running, write down which
world would make it pass and which would make it fail. If both are easy to
build, wait for real data.
**Receipt:** P-S2 ran on a clustered world (64 centres × 16) and a uniform
world (1024 rows) and was UNSTABLE in both (agreement at k=2 of 0.025 /
0.043). P-L3 and P-SH1 were deliberately not run, for the reason above.
lance-graph #1726, `entries/2026-10-23-ps2-compose-chain-preregistration.md`.
Linear EXO-133.
**Scope:** every pre-registered lab probe in lance-graph and its lab crates.
**Date:** 2026-10-23

### G-2026-10-23-2 — compile with debug info off; debug info is the disk hog, not the code
**Claim:** Export `CARGO_PROFILE_DEV_DEBUG=0` (and the TEST/RELEASE
equivalents plus `CARGO_INCREMENTAL=0`) as environment variables, never as a
committed profile edit. A profile change invalidates the whole cache, so
delete `target/debug` before switching rather than growing a second copy
beside it.
**Receipt:** ndarray `CLAUDE.md` § Hard Rules, measured 2026-09-16 on the same
tree and the same test run: `target/debug` is 1.9 GB with debug info and
291 MB without it (6.5×); 2319 tests pass either way.
**Scope:** every compile in every AdaWorldAPI workspace.
**Date:** 2026-10-23 (recorded here; measured 2026-09-16)

### G-2026-10-23-3 — a RUSTFLAGS env var replaces every cargo-config rustflags entry
**Claim:** Setting `RUSTFLAGS` (for example to `-D warnings`) silently drops
every `rustflags` entry from `.cargo/config.toml`, including the target-cpu
and any cfg that keeps a backend out of the binary. Before citing a config
flag as in force, check whether the caller sets RUSTFLAGS.
**Receipt:** ndarray `CLAUDE.md` (2026-09-16), measured two-sided on one unit:
with RUSTFLAGS unset, 65 `-Ctarget-cpu` and 65 `poly1305_force_soft` flags;
with `RUSTFLAGS="-D warnings"`, zero of each. ndarray's CI had run without its
crypto-backend cfgs because of this.
**Scope:** any workspace whose `.cargo/config.toml` carries rustflags.
**Date:** 2026-10-23 (recorded here; measured 2026-09-16)
