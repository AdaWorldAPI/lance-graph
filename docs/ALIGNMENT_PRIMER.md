# Alignment primer — bits, bytes, lanes, and why the 512-byte row never splits

> **Audience:** technical product level. No Rust knowledge assumed; every claim
> is labelled **[MEASURED]** (we ran it), **[SOURCE]** (read from Lance 9.0.0
> source, with file:line), or **[ARITHMETIC]** (checkable on paper).
>
> **Why this doc exists:** during the SoA→Lance work the same three questions
> came back repeatedly, and three separate AI research assistants each answered
> them partially wrong in a *different* way. This doc settles the questions and
> names the specific errors, so the next person consulting the same tools can
> tell signal from noise. Companions: `docs/SOA_BAKE_DEPLOYMENT.md` (the
> deployment story), `crates/lance-graph/tests/soa_verbatim.rs` (the pin that
> keeps all of this true).

## 1. The three numbers that keep getting confused

Three different "64"s and "512"s circulate in this domain. They are different
quantities at different layers:

| Number | Equals | What it actually is | Layer |
|---|---|---|---|
| **64 bit** | 8 bytes | Lance's *metadata* word size — offsets in footers and manifests | file-format envelope |
| **64 byte** | 512 bits | One x86 cache line; one AVX-512 register; `align_of::<NodeRow>()` | CPU / Rust type |
| **512 byte** | 4096 bits | One whole canonical node row (`key 16 \| edges 16 \| value 480`) | our data model |

When a document says "Lance uses 64-bit alignment", that is 8 bytes and about
the serialization envelope — it says nothing about where data pages land.
When our code says the row type requires 64, that is 64 **bytes**. And what our
test pins is 512 **bytes** — natural alignment, the strongest of the three.

## 2. Alignment is about the START address, never about fragmentation

The most common misreading: "aligned to 64 means written in 64-sized pieces."
**No.** Alignment constrains only where a block *begins*. The 512 bytes of a
row are one contiguous, indivisible run — **[MEASURED]**: the physical-layout
test finds the entire slab as a single unbroken byte run inside the Lance data
file and checks sampled rows at their computed addresses
(`tests/soa_verbatim.rs::a_slab_is_written_verbatim_and_contiguously`, run
against a real 335 MB bake: 335,302,144 slab bytes → 335,302,663 file bytes,
identical from offset 0).

Nothing ever writes half a row, half a lane, or "64 versions of anything."

## 3. The straddle arithmetic — why natural alignment ends the discussion

A 4096-bit row is consumed as 32 × 128-bit SIMD lanes. The worry: could a lane,
a cache line, or a disk sector boundary fall *inside* a row?

**[ARITHMETIC]** If a 512-byte row starts at a multiple of 512, it cannot
straddle any boundary whose size divides into or is divided by 512:

- **128-bit (16 B) lane:** 512 / 16 = 32 whole lanes per row. A row starting at
  a 512-multiple starts at a 16-multiple; every lane is whole.
- **64 B cache line (x86):** 8 whole lines per row.
- **128 B cache line (Apple M-series):** 4 whole lines per row.
- **4 KiB page / NVMe sector:** a 512-aligned 512-byte row either ends exactly
  on a 4096 boundary or lies wholly inside one — 8 rows per page, zero padding.

And because the stride *is* 512, alignment of row 0 propagates to every row
forever: row *i* sits at `start + i·512`, which is a 512-multiple whenever
`start` is. One aligned start ⇒ an eternally aligned array.

This is why the test asserts `off % 512 == 0` (natural alignment) rather than
the weaker `off % 64 == 0` the Rust cast minimally needs: natural alignment is
the one condition that excludes **every** straddle at **every** layer at once,
and the measurement (offset 0) already supports it.

## 4. What Lance actually does with our column — measured, not assumed

**[SOURCE]** Lance 9 chooses an encoding per column. Values wider than 256
bytes are "not narrow" (`lance-encoding-9.0.0`
`encodings/logical/primitive.rs:3861`, `MINIBLOCK_MAX_BYTE_LENGTH_PER_VALUE =
256`) and take the **full-zip** path, whose fixed-width branch writes values
verbatim unconditionally (`compression.rs:753`). Our row is 512 > 256, so the
column lands as raw bytes. The `compression = "none"` metadata we also set is
a *backstop*, not the cause — removing it changes nothing **[MEASURED]**; only
the mini-block path (which a 512-byte value never reaches) would read it.

**[MEASURED]** File layout of the written data object: the data run starts at
**offset 0**; the only other content is a 519-byte footer *after* the run.
There is no header in front of the data.

## 5. "Versioning will shift your bytes" — why that cannot happen

One AI dump warned that a Lance version update could "force a 64-byte offset
shift at the file header," splitting rows across pages. This is structurally
impossible, and we have the measurements that show why:

- **[MEASURED]** A Lance dataset is separate objects: the hydration probe
  listed exactly three per dataset — a `.txn`, a `.manifest`, and the data
  file. Manifests are **not** prepended to data files.
- **Data files are immutable.** A new version writes a *new* manifest and
  *new* data files; existing files are never edited or shifted in place. That
  immutability is the entire basis of Lance time-travel — and also of our
  config-as-pointer refresh (`docs/S3_LAYOUT.md`): old bytes never move, a
  refresh only changes which table the config names.
- Lance's 64-bit "row address" (fragment id + row offset) is a **logical row
  index**, not a byte pointer. The reader computes `index × stride`; there is
  no bit-stitching path that misalignment could trigger.

## 6. Cost of guaranteeing alignment: effectively zero

Even in the worst case, forcing a run to natural alignment costs at most 511
padding bytes **per file** — not per row — because §3's propagation means only
the start needs fixing. On the 335 MB bake that is < 0.0002 %. There is no
disk-load tradeoff to weigh; the only question is whether the property holds,
and the test answers that on every run.

## 7. FAQ — the actual questions, with the actual errors named

**Q: "Is it 64-bit or 64-byte alignment?"**
Both exist, at different layers (§1). Our data-run guarantees are all in
bytes; the "64-bit" figure belongs to Lance's metadata envelope.

**Q: "Does `align(64)` mean the 4096-bit row is stored as 64×64-bit pieces
with 64 versions?"**
No. Alignment ≠ granularity (§2). The row is one contiguous 512-byte unit,
proven byte-for-byte on real data.

**Q: "Shouldn't it be at least 128 so lanes don't split?"**
The instinct is right, the fix is stronger: natural alignment (512) is pinned,
which makes lane/line/sector splits arithmetically impossible (§3), not merely
unlikely.

**Q: "Will S3 chunking or multipart uploads fragment the rows?"**
No. S3 stores an object as an opaque byte sequence; HTTP chunking is
transport, invisible in the stored bytes. **[MEASURED]** the S3 arm of the
verbatim test reads the object back and finds the identical unbroken run —
and the byte-copy hydration path (`examples/hydration_probe.rs`) round-trips
every object byte-identically.

**Q: "Do we need 4096-bit SIMD registers?"**
They don't exist in current hardware. The widest common register is AVX-512 =
64 bytes; a row is consumed as 8 such loads (or 32 NEON loads). The relevant
hardware alignment is therefore 64 bytes — which natural alignment satisfies
with room to spare.

## 8. What keeps this true tomorrow

None of the above is trusted as doctrine. Every load-bearing property is a
red-turning assertion in `tests/soa_verbatim.rs`, run against both a local
write and the real object store:

| Property | Pinned by |
|---|---|
| Whole slab is one unbroken run | assertion (1), byte search |
| Every row at `off + i·512` | assertion (2), sampled addresses |
| Run at natural alignment | `off % 512 == 0` |
| No second/encoded copy | assertion (3), bounded footer |
| Encoder can't silently compress | `the_narrow_column_falsifier` (two-sided) |
| Same properties through S3 | `a_slab_is_written_verbatim_to_s3_too` |

A future Lance version that changes any of this turns the suite red — at
which point the answer is *re-measure and re-decide*, never *loosen the
assertion*. If a claim in this document and a red test ever disagree, the test
is right and this document is stale.
