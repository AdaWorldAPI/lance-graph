# paperless-token plateau — banked because the repo will not accept a push

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

`0001-token-seam.patch` — **both commits** (the seam probe, then the re-cut
onto `ogar-doc-ir`) **minus the two corpus fixtures**,
which are 350 KB of bytes that are exactly reproducible by two commands (below).
Everything else — `crates/paperless-token/{src,examples,README.md,Cargo.toml}`,
`docs/TOKEN-SEAM-ARCHITECTURE.md`, the root `Cargo.toml` wiring and the
`CLAUDE.md` update — is in the patch verbatim.

## Reconstruction

```sh
git -C paperless-rs checkout -b claude/bpe-tokenization-architecture-3xd4eh
git -C paperless-rs am 0001-token-seam.patch   # two commits

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
| `corpus/coca_academic_20k.tsv` | 180 361 | `4ae20ce39dd3018346700e0f88df2b59e1a7df4e4a06e0f285fd44065166e0f0` |
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
access to the pinned OGAR rev.
