# SoA bake deployment — from `.soa` slab to a queryable Lance table

This is an operator-facing walkthrough of the one-time write-back, the
physical-layout guarantee it depends on, the two deployment patterns it
enables, and the environment contract a deployment must satisfy. Every
measured claim below cites the source or plan section that measured it;
every convention is labelled as a convention, not a measurement.

## 1. What a bake is

The canonical node row is **512 bytes**: `key(16) | edges(16) | value(480)`
(`CLAUDE.md`, "CANON — Minimal SoA node"). `crates/lance-graph-contract`
exposes this as `canonical_node::NODE_ROW_STRIDE = 512`. A `.soa` slab file
is nothing more than a whole number of these 512-byte rows, back to back —
`soa_to_lance` refuses to even open a file whose length is not a multiple of
`NODE_ROW_STRIDE`, treating that as a truncated bake rather than something to
tolerate.

The 512-byte stride is load-bearing for everything that follows: it is what
keeps the row column uncompressed inside Lance (§3), which is what makes the
mmap-serving deployment pattern possible at all (§4b).

## 2. The one-time write-back — `soa_to_lance`

```text
soa_to_lance <slab.soa> <uri> <table> <classid-hex> <slab-digest-hex>
```

`uri` is either a local directory or an `s3://bucket/prefix` — the same call
either way. There is no separate export step: writing to the object store
**is** the write (`crates/lance-graph/examples/soa_to_lance.rs`, module doc).

The binary:

1. Reads the slab file into one `Vec<u8>`, asserts its length is a multiple
   of `NODE_ROW_STRIDE`.
2. Wraps that `Vec` in an Arrow `FixedSizeBinaryArray` via `Buffer::from_vec`
   — which **adopts** the allocation rather than copying it. The code asserts
   this at runtime by comparing the buffer's pointer against the `Vec`'s
   original pointer; `FixedSizeBinaryArray::try_from_iter` would have copied
   the data chunk-by-chunk instead, which was measured on this branch's
   history (commit `a27b06a`).
3. Writes one `RecordBatch` (a single `row` column) via `Dataset::write`,
   with the SoA contract carried as Arrow schema metadata — which Lance
   persists into its own manifest.
4. Re-opens what it just wrote and asserts the persisted header matches the
   **compiled** contract, the same shape as `SoaEnvelope::verify_layout`.

### The header keys

Every value is imported from `lance-graph-contract` rather than restated as
a literal — a restated constant is a second source of truth that drifts.

| key | value | source |
|---|---|---|
| `soa:envelope_layout_version` | `2` | `soa_envelope::ENVELOPE_LAYOUT_VERSION` |
| `soa:row_stride` | `512` | `canonical_node::NODE_ROW_STRIDE` |
| `soa:row_carving` | `key:0..16\|edges:16..32\|value:32..512` | canon, locked 2026-06-13 |
| `soa:endianness` | `le` | the LE contract |
| `soa:classid` | per bake (CLI arg) | the bake's own report |
| `soa:slab_digest` | per bake (CLI arg) | the bake's own report — pairs the table with its `.books` sidecar |
| `soa:source` | filename + row count | provenance |

A reader is expected to verify `soa:envelope_layout_version` against its own
compiled `ENVELOPE_LAYOUT_VERSION` before casting anything read back — the
same discipline `soa_to_lance` applies to itself on re-open.

## 3. Why the row column lands verbatim — the STRIDE, not the metadata

This is the part most likely to be misunderstood, because an earlier version
of this module's own doc got it wrong and was corrected in place rather than
silently edited (`soa_to_lance.rs` module doc; mirrored in
`tests/soa_verbatim.rs`).

**What is false:** that the `lance-encoding:compression = "none"` field
metadata is what keeps the row column byte-for-byte. It is not — removing
the key leaves the written file byte-identical, and so does setting it to
`"zstd"`. The key is spelled correctly and genuinely parsed
(`lance-encoding-9.0.0` `compression.rs:576`); it simply never reaches this
column.

**What is true**, read from lance 9 source:

- `is_narrow` (`encodings/logical/primitive.rs:3861`) classifies a column as
  narrow when its value length is below `MINIBLOCK_MAX_BYTE_LENGTH_PER_VALUE
  = 256` bytes. `NODE_ROW_STRIDE` is 512, so a canonical row is **not**
  narrow and the column takes the **full-zip** path, not mini-block.
- Full-zip's `create_per_value` returns `ValueEncoder::default()`
  unconditionally for `DataBlock::FixedWidth` (`compression.rs:753`) — it
  computes the merged field params one line earlier and then ignores them on
  that branch.
- The only branch that honours the compression metadata,
  `build_fixed_width_compressor` (`compression.rs:624`), lives on the
  mini-block path — which a 512-byte value never reaches.

So it is `NODE_ROW_STRIDE = 512 > 256` (the mini-block cutoff) that buys the
verbatim write. The `lance-encoding:compression = "none"` metadata is kept in
the writer only as a **defensive pin**: if that 256-byte threshold ever rose
above 512 in a future Lance version, the mini-block path would start
honouring the metadata and would still refuse compression — but today it does
no work.

`tests/soa_verbatim.rs` pins both halves of this claim:

- The physical-layout assertions (`a_slab_is_written_verbatim_and_contiguously`
  and its S3 twin) — the slab's bytes are found as one contiguous run in the
  data file, at a 64-byte-aligned offset, with every sampled row at its
  computed address, and total file overhead bounded under 64 KiB (so it is
  not the slab plus a second encoded copy).
- `the_narrow_column_falsifier` — the sensitivity proof that the byte search
  used above can actually **see** compression at all: at a 64-byte stride
  (below the mini-block cutoff) the same metadata key *is* honoured, and
  `"zstd"` makes the rows vanish from the file while `"none"` still leaves
  them verbatim.

**Why it matters:** this is what makes the mmap-serving deployment pattern
(§4b) sound. If the row column were chunked or compressed, `mmap(file)[off ..
off + rows*512]` would not be the slab at all.

## 4. The two deployment patterns

### (a) Read from S3 directly

A deployment opens the Lance dataset straight from `s3://…` and lets Lance's
S3 object-store client serve reads. No local copy exists. Simplest to
operate; every read after the first for a given byte range costs whatever
Lance's own caching does (see §7 — this is one of the open questions).

### (b) Hydrate to local disk once, then serve locally

A deployment copies the dataset's objects to local disk once and serves
subsequent reads from there — the pattern the mmap-serving argument in §3
targets, and the one `.claude/plans/idle-flush-dataset-eviction-v1.md`
proposes eviction over.

**Measured hydration cost** (`examples/hydration_probe.rs`; plan §8a,
measured 2026-08-07, against the configured endpoint):

| MB | hydrate | implied MB/s |
|---|---|---|
| 0.3 | 2.64 s | 0.1 |
| 6.7 | 2.83 s | 2.4 |
| 33.5 | 3.33 s | 10.1 |

A linear fit over these three points gives **≈2.63 s fixed cost + ≈0.021
s/MB** (≈48 MB/s marginal). The dominant term is size-independent: roughly
2.6 s of every hydration is fixed overhead, not bytes moved.

**Stated scope of these numbers, do not over-read them:** one endpoint, one
day, single-fragment datasets, 0.3–33.5 MB, no concurrency, and eviction
itself is not implemented — this measured only the hydration primitive a
future eviction policy would call. The plan explicitly retires the earlier
"~1.4 s" figure as a *constant* (it was a single, unre-run observation on a
different endpoint) while keeping the *decomposition* (fixed cost + a small
marginal rate) as the finding that survives.

Hydration here means a raw byte copy of every object under the dataset's
root — not a scan-and-rewrite through `Dataset::write`. The probe verified
this (`T10`, plan's own acceptance-criterion name) by comparing every copied
object's raw bytes against its local original, which is strictly stronger
than a column checksum: it would catch a dropped `.txn` file, a deletion
vector, or any other non-row artifact that a logical re-export would silently
lose.

## 5. Environment contract

Both the write-back binary and its verification arms read S3 credentials
through one shared reader, `dev_s3_env.rs`, specifically to avoid the write
path and the verification path independently reading two different option
maps:

| variable | required | purpose |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | yes | credential |
| `AWS_SECRET_ACCESS_KEY` | yes | credential |
| `AWS_ENDPOINT_URL` | yes | the S3-compatible endpoint |
| `AWS_DEFAULT_REGION` | no (defaults to `"auto"`) | region |
| `AWS_S3_BUCKET_NAME` | yes, for the probes | bucket |

These are the **same variable names a Railway deployment already sets**
(`dev_s3_env.rs` module doc) — no separate deployment-specific naming exists.

**The `AWS_ENDPOINT` vs `AWS_ENDPOINT_URL` trap:** `object_store`'s own
environment discovery reads `AWS_ENDPOINT`, which this environment does not
set. `dev_s3_env::s3_options()` explicitly maps `AWS_ENDPOINT_URL` (the
variable this deployment actually sets) into the `aws_endpoint` option Lance
consumes. Relying on `object_store`'s own discovery instead — by passing
`store_params: None` and letting it fall back to default credential
resolution — silently addresses AWS proper (or whatever the default resolves
to) rather than the configured S3-compatible endpoint. `s3_options()`
returns `None` if any *required* variable is missing, and callers on a path
that has already committed to being remote are expected to treat `None` as a
hard error, never a silent fallback to default discovery — a prior defect on
this branch had `soa_to_lance` write with `store_params: None` (silently
falling back) while its own re-open used `s3_options().expect(...)`, which
could panic only *after* the dataset had already been written to the wrong
place.

`dev_s3_env::env()` also strips wrapping quotes from an environment value —
this sandbox's exporter can wrap a value in literal `"…"`, and a quoted
credential authenticates as garbage while pointing the resulting error at the
credential rather than at the quoting.

## 6. Boot config

A deployment reads `.config/<repo-name>/config.yaml` from the same bucket,
declaring which bakes exist and which get hydrated at boot. The schema itself
is defined in `crates/lance-graph/examples/soa-config.example.yaml` — refer
there for the full shape rather than duplicating it here.

**Doctrine, not a measurement:** an existing table is never silently
overwritten. A refresh writes a **new, timestamped table** and flips the
config's pointer to it, because S3 has no atomic rename — an in-place
overwrite of an existing table's objects is exactly the kind of
half-written-directory hazard §5a of the idle-flush plan warns about for
local hydration, and the same reasoning applies to a refresh in place on the
object store.

## 7. What is NOT measured

Stated plainly rather than presented as risks that are probably fine
(`.claude/knowledge/lance-cache-surface.md`; plan §8a, §9):

- **Request count per hydration is still open.** The plan originally claimed
  this was closed at "3 remote objects per dataset," but that number came
  from `dir_stats()` — a **local** pre-upload file count, never wired to any
  object-store request instrumentation. How many actual requests a hydration
  issues, and what that costs against a real provider's pricing (request
  count, retrieval class, egress), remains unmeasured. Closing it needs a
  `WrappingObjectStore` request counter or equivalent instrumentation, not a
  local `read_dir`.

- **P-CACHE-1 — whether decoded data enters `LanceCache` at all.** `moka` is
  an unconditional dependency of `lance-core` (no feature gate) and its
  `MokaCacheBackend` is byte-weighed (`.max_capacity` + a weigher over key and
  entry size), with a `no_cache()` constructor available. But **it is not yet
  known what the cache actually holds** on the data path — decoded column
  pages / record batches, versus only manifests, schemas, and index metadata.
  This is the load-bearing open question: if decoded data does *not* go
  through `LanceCache`, capping its capacity bounds only metadata, and the RAM
  that shows up on a bill lives in whatever the *caller* holds — a failure
  mode already caught once on this branch (a `read_batch` that collected and
  concat-copied a whole table, commit `a27b06a`). If decoded data *does* go
  through the cache, `with_capacity(n)` is a hard ceiling on exactly the
  memory being billed. Two further companion questions are open alongside it
  and equally unrun: whether the capacity knob (or `no_cache`) is reachable
  from public API (`lancedb::connect`, `DatasetBuilder`, env, session object)
  at all, and whether measured RSS actually tracks capacity in practice. None
  of the three has a probe run yet.

Neither gap blocks the write-back path described in §2–3, which is measured
and pinned by tests. Both gaps block any claim about steady-state RAM cost
under deployment pattern (b), and about the true dollar cost of pattern (a)
under request-metered billing.
