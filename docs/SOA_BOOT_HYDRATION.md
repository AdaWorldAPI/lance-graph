# SoA boot hydration — config, wiring, and the re-bake procedure

> Companions: `S3_LAYOUT.md` (where things live) · `SOA_BAKE_DEPLOYMENT.md`
> (the walkthrough + hydration cost model) · `ALIGNMENT_PRIMER.md` (why the
> bytes land where they do — **read this before reasoning about offsets**).
> Parser: `crates/lance-graph/src/soa_config.rs`.
> Tools: `examples/soa_to_lance.rs`, `examples/validate_soa_config.rs`,
> `examples/soa_readback_witness.rs`.

## The shape of the thing

A **bake** is a slab of fixed-stride 512-byte rows written into a Lance table
as one `FixedSizeBinary(512)` column named `row`. A **boot config** names the
tables a deployment should know about and which of them to pull local at
startup. Nothing else is required for a deployment to find its data:

```text
   .obo / source            slab (.soa)              Lance table            reader
  ───────────────  bake  ─────────────  soa_to_lance  ───────────  hydrate  ────────
   text            ────►  512·N bytes   ──────────►   one column   ──────►  mmap
                          64-aligned                  + header              or remote
```

The store is a **local mmap-capable filesystem**; object storage is the
*hydration path*, never the store. A volume only decides how often hydration
happens. This is why a laptop and a container converge: both pull the same
bytes from the same place with the same credentials, so "works locally but not
deployed" stops being a category of bug.

## The config

Lives at `.config/<app>/config.yaml` in the bucket. No credentials — endpoint
and keys come from the `AWS_*` environment a deployment already sets.

```yaml
version: 1
ledger_prefix: "MedCare-rs/ledger"      # the app owns this prefix, and only this
bakes:
  - name: obo-mondo                     # stable handle
    table: obo.mondo.lance              # under ledger_prefix
    classid: "0x03010000"               # exactly ONE classid per bake
    slab_digest: "sha256:cf61e8b9…"     # full 64 hex, never truncated
    hydrate: true                       # pull local at boot, or read remote
on_existing: refuse                     # refuse | new_version
```

Validate it with the **real parser** rather than by eye — the example exists
precisely so a config is never "probably fine":

```sh
cargo run --release -p lance-graph --example validate_soa_config -- config.yaml
```

It exits non-zero on rejection and prints the resolved `classid_u32()` per
bake, so a typo'd hex string fails at the desk instead of at boot.

### One classid per bake — why it is a schema rule

The header is what tells a reader which `ClassView` applies. A table carrying
several classids cannot answer that without inspecting rows, which is the exact
thing a header exists to prevent. So a multi-classid bake is split, one table
per classid.

That split is not free, and the cost is measured rather than assumed
(`SOA_BAKE_DEPLOYMENT.md` §4b): hydration is **≈2.63 s fixed + ≈0.021 s/MB**.
The fixed term dominates at these sizes, so five tables pay it five times —
≈13 s to hydrate all five against ≈3.3 s for one. That is the whole reason
`hydrate` is a per-bake flag and not a global switch: the small, rarely-read
lanes stay remote.

## Writing a bake

```sh
soa_to_lance <slab.soa> <uri> <table> <classid-hex> <slab-digest-hex> [--overwrite]
```

The tool re-opens what it wrote and verifies the header against what it was
told, so a write that lands in the wrong place fails at write time rather than
at read time. It resolves S3 credentials **once, up front**, and refuses a
remote URI with no credentials — an earlier shape could write to an
uncontrolled default endpoint and only panic afterwards.

`--overwrite` is **opt-in**. The default `WriteMode::Create` refuses an
occupied table, which is right: a re-bake changes the bytes under an existing
name and doing that by accident is the failure a default should prevent. With
the flag, Lance's `Overwrite` writes a **new version** and the prior one stays
readable — so a re-bake is reversible, not destructive. Unrecognised arguments
are rejected rather than ignored, so a typo cannot fall through to the
safe-but-wrong mode.

## Verifying a bake

Three checks, each answering a different question. Passing one does not imply
the others.

| check | question | tool |
|---|---|---|
| header | did the writer put it where it said? | `soa_to_lance` (self-verifies) |
| **readback** | does a *reader* get those bytes back? | `soa_readback_witness` |
| physical | is it one unbroken aligned run? | `tests/soa_verbatim.rs` |

The header digest only proves the **writer saw** the right bytes. It says
nothing about what a reader gets. That is the readback witness:

```sh
soa_readback_witness <uri> <table> <expected-slab.soa>
```

It scans the table and asserts **byte equality** against the slab. Byte
equality rather than a field-by-field decode, deliberately: a second decoder
living in `lance-graph` would be a worse problem than the one being checked,
and two decoders that disagree is a failure mode with no upside. Byte equality
preserves *every* reading by construction — keys, the degree histogram, value
tenants, edge lanes, and readings that do not exist yet. On a mismatch it
reports row index **and** byte offset, naming the region (`key` / `edge block` /
`value slab`), because those are three different bugs.

### Why the bytes survive verbatim

`NODE_ROW_STRIDE = 512` exceeds `MINIBLOCK_MAX_BYTE_LENGTH_PER_VALUE = 256`, so
the column is not "narrow" and takes Lance's **full-zip** path, whose
`create_per_value` returns the default value encoder unconditionally for
`FixedWidth`. The `lance-encoding:compression = "none"` metadata is **inert
here** — spelled correctly, parsed, and unable to reach a 512-byte column. It
is kept as a documented backstop in case that threshold ever rises, not as the
cause. `ALIGNMENT_PRIMER.md` §4 measures the rest: the data run starts at
**offset 0**, with only a small footer *after* it — there is no header in front
of the data.

## Reproducibility

The bake is deterministic. Re-running it on the same sources reproduces every
slab **bit for bit** — verified by regenerating all five OBO lanes and matching
the `sha256` each config entry carries. So a published artifact is checkable by
re-running the producer, and `slab_digest` is a real cross-check rather than a
label that travels with the file it describes.

## Re-bake procedure (the order matters)

A re-bake changes digests, so the config and the tables disagree for a window.
The order below keeps that window **loud** rather than silent:

1. Bake → slab; record `sha256`.
2. `soa_to_lance … --overwrite` for each table. Each self-verifies its header.
3. `soa_readback_witness` per table — byte equality against the slab.
4. `validate_soa_config` on the patched config.
5. **Upload the config last**, keeping the previous one as a dated `.bak`.

Between (2) and (5) the published config points at digests that no longer
exist, so a hydrating reader fails a checksum instead of quietly reading
different bytes. That is the safe direction: the failure is visible and the fix
is one upload. Doing (5) first would invert it — the config would describe
bytes that were not there yet.

## Anti-patterns

- **Reporting a wrapper's exit status as the build's.** `cmd > log; echo $? > rc`
  inside `( … )` makes the compound exit 0 regardless. Read the `.rc` file, not
  the surrounding notification. This produced a confident "build green" against
  an `exit 101` in this repo's own history.
- **A test run that compiled nothing.** A crate whose module sits behind a
  non-default feature will `cargo test` clean with **0 tests** — a green exit
  over an empty set is indistinguishable from a green exit over a real one.
  Check the test count, not the status.
- **Treating the header digest as a readback.** See the table above.
- **Truncating `slab_digest`.** The config carries the full 64 hex. A shortened
  digest still *looks* like a checksum and no longer is one.
- **Silent overwrite.** Covered by making `--overwrite` opt-in; the reason it
  is opt-in belongs in review, not in a flag description.

## Open

- `hydrate: true` is currently a static choice per bake. Nothing measures
  whether the hot/remote split matches real read patterns, so the flags encode
  an expectation rather than an observation.
- The readback witness is a manual step. It is not wired into a CI gate, so a
  table can be published without one being run.
- `on_existing: new_version` is accepted by the schema but the timestamped-table
  path it describes is not exercised by any tool here; `--overwrite` covers the
  re-bake case instead.
