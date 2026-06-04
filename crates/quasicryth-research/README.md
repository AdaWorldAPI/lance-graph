# quasicryth-research

Direct Rust transcode of **Quasicryth** (Tacconelli 2026,
[arxiv 2603.14999](https://arxiv.org/abs/2603.14999), upstream
[github.com/robtacconelli/quasicryth](https://github.com/robtacconelli/quasicryth)
v5.6.0).

**Purpose:** research and testing. Two goals:

1. Validate the workspace's φ-substrate decisions (bgz17's `17φ/11`, helix's
   golden-spiral hemisphere) against the reference algebra — without
   depending on the upstream C build.
2. Demonstrate the **codebook architecture in two variants** behind one
   trait: the original flat storage shape from the C reference, and a
   COW Adaptive Radix Tree variant that fits the workspace's append-only
   substrate doctrine.

## What's transcoded

| Phase | Upstream C | Rust module | Status | Tests |
|---|---|---|---|---|
| 0 | `fib.c` + types from `qtc.h` | `tiling`, `hierarchy`, `constants`, `types` | shipped | 19 unit + 9 paper-theorem integration |
| 1 | `md5.c` + `tok.c` (partial) | `md5`, `tok` | shipped | 8 RFC-1321 vectors + 12 tokenizer round-trips |
| 2 | `cb.c` (algorithmic shape) | `codebook` (FlatCodebook + CowRadixCodebook) | shipped | 8 codebook tests, cross-variant validation |
| 3 | `ac.c` | `arith_coder` (Model256, VModel, Encoder, Decoder) | shipped | 9 round-trip tests at multiple scales |
| 4 | `compress.c` + `decompress.c` (simplified) | `pipeline` (compress/decompress) | shipped | 11 round-trip tests covering both variants |
| 5 | integration | `tests/round_trip.rs` + `tests/paper_theorems.rs` | shipped | 7 cross-variant integration tests |
| 6 | `main.c` | `bin/qresearch.rs` | shipped | CLI: compress / decompress / round-trip |

**Total tests: 83 passing.** Zero dependencies. No `unsafe`. Stable Rust.

## What this is NOT

The Rust pipeline is **NOT byte-compatible** with the upstream `.qm56`
format. The full v5.6 production compressor ships:

- multi-level adaptive arithmetic coding with 144 specialised level
  context models + 12 per-index models + recency caches + two-tier
  unigram model + word-level LZ77,
- 36-tiling greedy selection per block,
- LZMA escape stream for OOV words,
- frequency-counter pruning for memory bounds on large inputs,
- multi-level n-gram codebooks (3-gram through 144-gram).

The Rust pipeline here is **simplified to a single-tier (unigram)
encoding** so the codebook trait abstraction is the clean variation
point and the COW radix trie variant is exercised end-to-end. Multi-tier
n-gram encoding is a phase 5+ extension. The Fibonacci tiling +
substitution hierarchy + deep-position detection are verified to
satisfy the paper's five core theorems (see
`tests/paper_theorems.rs`), but the bit-stream itself only encodes
word-ID symbols at the unigram tier.

The Rust pipeline **round-trips with itself**: `decompress(compress(x))
== x` for every test input under both `Variant::Flat` and
`Variant::CowRadix`. This is the property the integration tests verify.

## Verification

```bash
# All 83 tests:
cargo test --manifest-path crates/quasicryth-research/Cargo.toml

# Paper-theorem suite only (the original 9 algebraic claims):
cargo test --manifest-path crates/quasicryth-research/Cargo.toml \
  --test paper_theorems

# Cross-variant integration suite (Phase 5):
cargo test --manifest-path crates/quasicryth-research/Cargo.toml \
  --test round_trip
```

Lint discipline:

```bash
cargo clippy --manifest-path crates/quasicryth-research/Cargo.toml \
  --all-targets -- -D warnings
cargo fmt --manifest-path crates/quasicryth-research/Cargo.toml --check
```

Both clean.

## CLI

```bash
cargo build --release --manifest-path crates/quasicryth-research/Cargo.toml \
  --bin qresearch
./crates/quasicryth-research/target/release/qresearch round-trip /path/to/file.txt
./crates/quasicryth-research/target/release/qresearch round-trip -v cow /path/to/file.txt
./crates/quasicryth-research/target/release/qresearch compress -v flat in.txt out.qrs1
./crates/quasicryth-research/target/release/qresearch decompress out.qrs1 in.txt.recovered
```

The `round-trip` subcommand compresses to memory, decompresses, and
verifies identity — useful for quick fuzz-style validation on real text.

## Paper-theorem verification (algebraic substrate)

Tests in `tests/paper_theorems.rs` verify, on synthetic L/S sequences:

- **Thm 2** Fibonacci hierarchy never collapses (both L and S supertiles persist).
- **Cor 4** Period-5 collapses by level 4 or 5 (vs Fibonacci's unbounded depth).
- **Thm 9** Golden Compensation: L:S ratio = φ at every level.
- **Thm 13/Cor 15** Aperiodic advantage grows with scale.
- **Sturmian** Factor complexity ≤ n+1 (the minimality property that
  gives maximal codebook efficiency, Thm 7).

Plus algebraic and structural invariants: PV-property (φ² = φ+1),
`HIER_WORD_LENS = F_3..F_12`, no-adjacent-S on all 36 canonical tilings.

## Codebook variants

| Property | `FlatCodebook` | `CowRadixCodebook` |
|---|---|---|
| Storage | flat `Vec<u32>` per tier + `HashMap` for lookup | Adaptive Radix Tree (Node4 / Node16 / Node256) per tier |
| Build cost | O(n log n) — frequency sort | O(n log n) sort + O(n · key_len) trie inserts |
| Lookup | O(1) average (HashMap) | O(key_len) tree walk |
| Memory | dense | sparse, shared across versions (Arc) |
| Versioning | no | **path-copy COW** — every insert returns a new root, prior roots stay valid |
| Append-only | no | **yes** — fits the workspace's substrate doctrine |
| Threading | `Send + Sync` (immutable post-build) | `Send + Sync` (Arc-shared subtrees, immutable) |

Both implement the `Codebook` trait. The `pipeline::Variant` enum
picks between them. Tests validate equivalence: under identical
inputs, both produce the same `compress → decompress` output.

The COW property is **explicitly exercised** in
`codebook::tests::cow_art_path_copy_preserves_old_root` — `art_v0`
stays empty after `art_v1.insert(...)` and `art_v2.insert(...)`;
each version sees only its own inserts. This is the architectural
property the workspace's append-only doctrine requires.

## Compressed stream format (v1)

The Rust pipeline writes a self-contained format with the magic
`QRS1` ("Quasicryth Research Simplified v1"):

```
magic       : [4]  "QRS1"
orig_size   : u64  little-endian
n_tokens    : u32
n_words     : u32
n_unique    : u32
lowered_size: u32
lowered     : [u8; lowered_size]   the lowered byte stream
spans       : [(u32 offset, u32 len, u8 case_flag); n_tokens]
case_size   : u32
case_data   : [u8; case_size]      AC over Model256 (token case flags)
word_size   : u32
word_data   : [u8; word_size]      AC over VModel (codebook indices)
```

This is **not** the upstream `.qm56` format — by design, see the
"What this is NOT" section above.

## Relationship to workspace crates

- **bgz17** — uses `17φ/11 ≈ 5/2` (major tenth) as octave-stacking
  constant for codebook hierarchy depth; this crate verifies the
  non-collapse theorem that justifies φ over rational stacking
  approximations.
- **helix** — uses pure φ for golden-angle azimuth and `√u` for
  equal-area hemisphere placement; this crate verifies the Sturmian
  minimality that makes φ optimal among irrational slopes.
- **jc::weyl** — proves 1-D `{k·φ⁻¹ mod 1}` star-discrepancy is
  minimal at N=144 and N=1000; this crate's `qc_word_tiling`
  exercises the same φ-stride at hierarchy scale.

## Upstream

`https://github.com/robtacconelli/quasicryth` — v5.6.0 as of the
transcode date. The upstream is the canonical reference; this crate
tracks its algebraic surface and pipeline shape only and does NOT
attempt byte-for-byte compatibility with its compressed output.

## Crate policy

Standalone, zero-dependency, `exclude`d from the lance-graph
workspace — same convention as `bgz17`, `deepnsm`, `helix`,
`bgz-tensor`. Verified via `cargo test --manifest-path`.
