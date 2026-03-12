# CLAUDE.md — Lance-Graph

> **Updated**: 2026-03-12
> **Status**: Monolith working, crate separation planned

---

## What This Is

"The Face." A graph database engine on LanceDB + DataFusion. Cypher + SQL in one engine.
From outside: boring fast graph database. Inside: BlasGraph semiring on bitpacked Hamming SPO.

## ⚠ READ BEFORE WRITING CODE

### 1. MONOLITH STATE

Everything is in one crate: `crates/lance-graph/` (19,262 lines).
Plan for 8-crate separation: `CRATE_SEPARATION_PLAN.md` in repo root.
DO NOT start separation without reading that plan.

### 2. graph/spo/ IS STALE

`crates/lance-graph/src/graph/spo/` is a DIVERGED COPY of ladybug-rs `src/graph/spo/`.
ladybug-rs version is MORE COMPLETE (TruthGate from PR 170, MerkleEpoch, SparseContainer).
During crate separation, this gets REPLACED with extended versions from ladybug-rs + holograph.

**DO NOT** extend lance-graph's graph/spo/. Extend ladybug-rs's, then import.

### 3. parser.rs = ladybug-rs lance_parser

`src/parser.rs` is identical (12 diff lines) to `ladybug-rs/src/query/lance_parser/parser.rs`.
The ladybug-rs copy is being DELETED (it's an orphaned duplicate). This repo keeps the original.

### 4. DataFusion Planner Is The Most Valuable Code

`src/datafusion_planner/` (5,633 lines) is the execution engine. Treat it carefully.
The join_builder.rs, expression.rs, scan_ops.rs are production-grade DataFusion integration.
DO NOT rewrite them. They're correct.

### 5. Tests Are Comprehensive

12 test files, ~9,300 lines. Especially `test_datafusion_pipeline.rs` (5,152 lines).
Run them. Don't break them.

## Build

```bash
cargo test  # runs all tests in workspace
cargo test -p lance-graph  # just the main crate
```

## Role in Four-Repo Architecture

```
rustynum     = The Muscle    (SIMD substrate)
ladybug-rs   = The Brain     (BindSpace, server)
staunen      = The Bet       (6 instructions, no GPU)
lance-graph  = The Face      ← THIS REPO (query surface)
```

## Key Files (by importance)

```
CRITICAL (don't break):
  src/datafusion_planner/         5633 lines  The execution engine
  src/parser.rs                   1931 lines  The Cypher parser (nom)
  src/logical_plan.rs             1417 lines  Logical plan algebra
  src/query.rs                    2375 lines  Query builder + executor

IMPORT TARGETS (will be enriched during crate separation):
  src/graph/spo/                  ~1000 lines  STALE — will be replaced
  src/graph/fingerprint.rs         144 lines  Basic fingerprint ops

CLEAN UTILITIES:
  src/error.rs                     233 lines  snafu errors with location
  src/config.rs                    465 lines  GraphConfig builder
  src/semantic.rs                 1719 lines  Semantic validation
```

## What NOT To Do

```
× Don't extend graph/spo/ here (extend in ladybug-rs, import)
× Don't add BindSpace/SpineCache code (that's ladybug-rs)
× Don't add qualia/awareness code (that's ladybug-rs)
× Don't create a Redis protocol handler yet (wait for crate separation)
× Don't use the parser.rs copy in ladybug-rs (it's being deleted there)
```

## Session Context

```
.claude/HANDOVER.md           Session handover for this repo
CRATE_SEPARATION_PLAN.md      Full 8-crate plan (prompt 21)
```
