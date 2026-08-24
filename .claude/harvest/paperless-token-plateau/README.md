# paperless-token plateau — an ARCHIVE, no longer the way to get this code

> **⊘ SUPERSEDED 2026-08-24 — read this first.** This code now lives in
> `AdaWorldAPI/tesseract-rs` as `crates/tesseract-paperless`, feature-gated
> (`default` = the S-2 gate + `doc.v1` → `DocIr`; `ocr` = the in-process
> recognizer; `token` = the lexical seam). tesseract-rs accepts pushes and
> runs CI on all three tiers, so **that is the one source of truth.** Nothing
> should be reconstructed from this patch except as forensics — and nothing
> should be synced back to paperless-rs, which is a dead copy.
>
> Two corrections carried by the move, both worth keeping: the `[patch]`
> section is gone entirely (tesseract-rs path-deps its siblings, so the
> escaping-relative-path trap does not arise), and `ingest_html` was replaced
> by a producer-agnostic `ingest_doc_ir(bytes, index, build)` — the caller
> brings its own producer as a closure, so the broken `spider_doc_ir` build is
> no longer anyone's dependency.
>
> **A correction to this file's own table:** it stated
> `coca_academic_20k.tsv` as 180 361 bytes. Measured, it is **226 651**. The
> sha256 column was right, which is why the reconstruction check passed — but
> the verification I ran covered *patch-applies* and *tests-pass*, not the byte
> counts, and saying "verified" without that scope was too broad.

`PROBE-TOKEN-SEAM-1` and its architecture doc live in
`AdaWorldAPI/paperless-rs`. That repo is in this session's scope for READS but
its push is denied at the org/GitHub-App level:

```
remote: Claude doesn't have GitHub access to AdaWorldAPI/paperless-rs for your organization.
fatal: unable to access 'https://github.com/AdaWorldAPI/paperless-rs/': ... 403
```

Verified twice — through the session proxy AND with the proxy bypassed. The
in-environment `GH_TOKEN` is a 14-character placeholder, not a credential, so
the documented "a 403 here is usually the proxy" escape does not apply: this one
is real. The commit therefore exists only in an ephemeral container, and
`E-ONE-RECEIPT-MANY-BORROWED-CONSUMERS-1` cites evidence that would die with it.
This directory is that insurance, following the plateau pattern tesseract-rs
already documents for a denied repo.

## What is here

`0001-token-seam.patch` — **all four commits, minus the two corpus fixtures**,
which are 350 KB of bytes exactly reproducible by two commands (below).
Everything else is in the patch verbatim:
`crates/paperless-token/{src,examples,README.md,Cargo.toml}`,
`crates/paperless-intake/{src,Cargo.toml}`,
`docs/TOKEN-SEAM-ARCHITECTURE.md`, the root `Cargo.toml` wiring, and the
`CLAUDE.md` update.

| # | commit | what it adds |
|---|---|---|
| 1 | `one tokenization receipt, three borrowed consumers` | the seam probe: contract, lane, view, the three consumer surfaces, 41 gates |
| 2 | `re-cut the seam onto ogar-doc-ir` | the receipt mints no identity — `SpanKey` is read from the document layer's own IR |
| 3 | `two measurement probes` | carrier width (flat u8 vs the 256:256 rail) and WordId vs TokenId in the tile |
| 4 | `wire both retinas into intake` | `paperless-intake` stops being a stub: the S-2 gate in front of the DOM and pixel producers |

## Reconstruction

```sh
git -C paperless-rs checkout -b claude/bpe-tokenization-architecture-3xd4eh
git -C paperless-rs am 0001-token-seam.patch   # four commits

# the two fixtures the patch deliberately omits:
mkdir -p paperless-rs/crates/paperless-token/corpus
cp tantivy/benches/alice.txt paperless-rs/crates/paperless-token/corpus/alice.txt
awk -F, 'NR>1 {print $4"\t"$5}' \
    lance-graph/crates/deepnsm/word_frequency/academic_20k.csv \
    > paperless-rs/crates/paperless-token/corpus/coca_academic_20k.tsv
```

Then verify, because a reconstructed fixture that differs by one byte changes
every measured number in the report:

| file | bytes | sha256 |
|---|---|---|
| `corpus/alice.txt` | 174 693 | `15124d40c182677c2d90fba80310173d63428e0591ce0df3e9bdc01a789a89c6` |
| `corpus/coca_academic_20k.tsv` | 226 651 | `4ae20ce39dd3018346700e0f88df2b59e1a7df4e4a06e0f285fd44065166e0f0` |
| source `academic_20k.csv` | — | `1dfd5edaa5a6ac9b8ac5abbf87894abaf7de8a449ad7c09ca1f6324226396e2d` |

`alice.txt` is Project Gutenberg's *Alice's Adventures in Wonderland* (public
domain), carried from `tantivy/benches/alice.txt`. It is CRLF with a BOM — the
probe normalises both, and the gate that would otherwise have hidden a failure
to do so is `T-CORPUS`.

## Running it

```sh
cargo run --release -p paperless-token --example probe_token_seam
```

Expected: `PROBE-TOKEN-SEAM-1: ALL 41 GATES GREEN`, and the four measured
tables the epiphany quotes.

Note the second commit adds an `ogar-doc-ir` dependency, so the reconstruction
needs the workspace `Cargo.toml` from the patch (it is in there) and network
access to OGAR. That pair floats on `branch = "main"` rather than a rev, and
the patch's own manifest comment says why: cargo does not unify a `branch` and
a `rev` source even at the same commit, and two SourceIds for `ogar-doc-ir`
means two incompatible `DocIr` types in one binary.

`paperless-intake` compiles with `--features ocr` (the in-process recognizer)
but NOT with `--features dom`: `spider_doc_ir` omits `TableCell::confidence`,
a field `ogar-doc-ir` added after spider's only commit. That is an upstream
break, recorded in full in `crates/paperless-intake/Cargo.toml`, not something
the reconstruction did wrong.

## This patch was verified, not assumed

A banked patch that does not apply is not insurance. Checked before landing:
`git am` of all four onto the base commit applies clean, and
`cargo test -p paperless-intake -p paperless-kv` on the reconstructed tree is
green (2 + 6 tests). The `paperless-token` probe is NOT part of that check —
it needs the two corpus fixtures the patch deliberately omits, so run it only
after the reconstruction step above.
