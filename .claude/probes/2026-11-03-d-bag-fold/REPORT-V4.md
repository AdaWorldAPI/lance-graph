# D-BAG-FOLD v4: attested pointer lens (A) and candidate-gated execution (B)

**Status:** MEASURED, lab only. Arm L is not promoted. EXO-140 / EXO-46 are
unchanged. No DT64 route. No production change: mask-risc's IR is used as is,
Quack is unchanged, and `MaskOp::Gather` runs only in `probe-v4/`.

Source: lance-graph `d651b70d4` + #1938. rustc 1.99.0, release, 4 vCPU Xeon @ 2.10 GHz.

```sh
cd .claude/probes/2026-11-03-d-bag-fold/probe-v4
CARGO_PROFILE_DEV_DEBUG=0 cargo +1.99.0 run --release -q   # exits 1 on any failed check
```

Result: **21 passed, 0 failed.**

## A. Attested pointer lens

Two lab witnesses, minted at construction, built on production types:

- **Presence witness.** Minted from the source facts (the edge sets), never from
  the planes. For each plane it records `ordered_lane::digest_of` (production) over
  the sorted child keys, and the edge count. Verification recomputes the digest over
  the keys at the plane's set bits. The witness therefore binds a bit to a key, not to
  a row. A plane stamped at the wrong rows names other keys and fails, however its
  bits are arranged.

  The witness also carries the `OrderedLaneWitness`, a `FrameStore` frame and the
  presence generation.
- **Pointer lens.** `$i` is minted by witnessed `seek_from` (one cursor per ancestor
  depth). At mint time it is checked once by recomputation: origin (address-derived),
  bounds and class scope. It is then sealed with a table digest, and carries the
  `OrderedLaneWitness`, the frame and the lens.

| # | control | result | enforced by |
|---|---|---|---|
| A1 | aligned store | both verify; gated answer 27 = oracle | — |
| A2 | plane **misaligned at construction** (every bit one row late) | refused (`PlaneNotAligned`) | lab presence witness on production `digest_of` |
| A2′ | same store, presence witness skipped | silent 0 vs oracle 27 | (disable) |
| A3 | plane permuted after construction | refused; also caught a bit past `n_rows` | lab |
| A4 | deletion through the API (generation + frame advance) | presence and pointer witnesses both refused (`Frame(SourceChanged)`) | **production `FrameStore`** |
| A5 | deletion bypassing generation and frame | refused (`PlaneNotAligned`) | lab key digest |
| A6 | duplicate key | production `seal` accepts; both mints refuse | lab |
| A7a | pointer set to `n_rows` after mint | refused (table digest) | lab |
| A7b | the same table offered as a fresh mint | refused (`PointerOutOfBounds`) | lab |
| A7c | value-chosen `ib`, digest re-sealed by a forger | **verify passes**; the mint-time origin check refuses | lab, mint only |
| A7d | pointer into another ClassId with identical tiers | refused at mint (`PointerOrigin`) | lab |
| A8a | foreign frame | refused | **production `FrameStore`** |
| A8b | lane resealed at a new version | refused (`VersionMismatch`) | **production `OrderedLaneWitness`** |
| A8c | lens mismatch | not constructible: one `SemanticLens` variant | — (gap) |

One defect was found and fixed during A3: reversing a plane's words moves tail bits
past `n_rows` into range, and the first version of the verifier indexed past the
lane (a panic). A set bit beyond the last row is now a refusal.

## B. Candidate-gated execution

Hand-built mask-risc `Program`, frozen IR used unchanged:

```
s0 = Pred Range{candidate range} under Plane(T_in)    -- candidates ∧ T
s1 = Gather(ib → R_in) under Scratch(s0)
s0 = Gather(ic → S_in) under Scratch(s1)
Count(s0)
```

Both Gathers are gated by an in-program register (asserted on the program). Planes
are read in place, nothing is copied, and the program has one terminal, with no mask
handed to another program.

- **B1:** parity with Quack's Arm L and with the oracle on every root of a 984-row
  lane with fanouts 2–8 and 30 % edge loss.
- **B2 disables:** dropping either Gather turns the answer wrong against the oracle
  (18 → 27 and 24 → 27).

### Scaling (µs, medians of 101; ≥ 1 ms prints FAILED-GATE)

| lane rows | candidates | resident witnesses | Arm L (Quack) | gated | ratio |
|---|---|---|---|---|---|
| 585 | 512 | 3 | 1.6 | 1.4 | 1.2× |
| 4,680 | 512 | 3 | 11.5 | 1.5 | 7.6× |
| 18,720 | 512 | 3 | 45.4 | 2.2 | 20.9× |
| 13,792 | 8 | 3 | 36.0 | 0.8 | 46.6× |
| 13,792 | 64 | 3 | 36.2 | 0.9 | 39.8× |
| 13,792 | 512 | 3 | 35.7 | 2.0 | 17.6× |
| 13,792 | 4,096 | 3 | 48.3 | 10.2 | 4.7× |
| 4,680 | 512 | 5 | 22.7 | 2.5 | 9.0× |
| 4,680 | 512 | 7 | 33.9 | 3.4 | 9.9× |

- **Full lane, candidates fixed at 512:** Arm L grows with the lane (1.6 → 45.4 µs);
  gated stays nearly flat (1.4 → 2.2 µs). The remaining lane term is the `Range`
  predicate and the `Count`, which are word-wide passes.
- **Candidates, lane fixed at 13,792:** gated tracks the candidate population
  (0.8 → 10.2 µs for 8 → 4,096); Arm L does not (36–48 µs).
- **Resident witnesses:** each extra pair of Gathers adds about 0.9–1 µs gated and
  11 µs under Arm L.

### Witness cost (same runs, µs)

| lane rows | pointer mint | presence mint | pointer verify | presence verify |
|---|---|---|---|---|
| 585 | 28.4 | 23.4 | 1.6 | 9.9 |
| 4,680 | 228.0 | 254.0 | 12.9 | 79.6 |
| 18,720 | 927.9 | FAILED-GATE | 51.6 | 317.6 |
| 13,792 (4,096 candidates) | 702.7 | FAILED-GATE | 38.3 | 246.5 |

What these say:

- **Presence verification costs more than the gated execution it protects** (80–318 µs
  vs 1.5–2.2 µs). It is O(popcount), reading the key at every set bit. Run per query,
  it would erase the gain. It is only proportionate if run once per (generation,
  frame) and cached under that pair. That cache is a design decision this experiment
  did not make.
- **Presence mint sorts the source facts** to feed the order-sensitive production
  `digest_of`. It has to run again on every presence mutation, and it crosses 1 ms at
  ~14k–19k rows. An order-independent (multiset) key digest would make a mutation an
  O(1) update. No production type provides one.
- **Pointer mint is once per key version.** It does not need to run again on a
  presence change: A4′ shows the frame retires it anyway, which is conservative.

## Decision memo

**1. Which missing witnesses are now experimentally specified?**

All six from v3 now have a construction-time lab witness that refuses on a
non-trivial input, and each has a disable or a negative control:

- presence-plane alignment, **including misalignment at construction** (A2);
- presence generation, both through the API (A4) and bypassing it (A5);
- key uniqueness (A6);
- pointer bounds (A7a, A7b);
- pointer origin (A7c′, A7d);
- class scope (A7d).

Frame and lane version use production types (A4, A8a, A8b).

**2. Can candidate gating avoid the unnecessary full-lane work?**

Yes, with existing mask-risc IR in one program. Gated execution scales with the
candidate population and is nearly flat in lane size: 1.2× faster than Quack's Arm L on
the 585-row lane and 4.7–46.6× on every larger one, with identical answers.

Quack cannot express it. The restriction is precise: `gate_walk` (`quack/src/lib.rs`
around line 2108) gates the conjuncts of an `And` by one resident plane
(`Node::Gather { under: Option<Mask> }`, "one gate per comparison is all `under` can
carry"). It never threads the accumulated in-program conjunction, a `Scratch`
operand, as the gate. mask-risc's own `MaskOp::Gather { under: Option<Operand> }`
already accepts one. Owner: Quack's lowering ("consumers own execution strategy",
D-RA-2 = A). No production workaround was built.

**3. What remains impossible under current contracts?**

- **Pointer origin cannot be attested after mint.** A table digest proves the table
  is unchanged, not how it was produced: a forger who re-seals passes verification
  (A7c). Origin holds only if the minting authority is trusted, or if verification
  recomputes it (O(n)). No contract type carries an origin claim.
- **Lens identity:** one `SemanticLens` variant, and `seek_from` takes no lens.
- **Presence alignment, uniqueness and generation** exist only as lab types.
  `OrderedLaneWitness` covers the key lane alone, and `seal` accepts duplicates.
- **Incremental presence attestation:** production `digest_of` is order-sensitive,
  so every mutation re-sorts the source facts.
- **Gated lowering in Quack** is not expressible (above).

**4. What exact EXO-140 / EXO-46 operator decision is required?**

- **EXO-140:** whether `MaskOp::Gather` whose index lane is a minted, origin-checked,
  same-table `$i` — gated by an in-program register, reading resident planes — is a
  co-addressed read under ADMISSION B. And who owns that call-site classification.
- **EXO-46:** whether the counted global→focus pointer table may be used in
  production execution. And whether its trust rests on the minting authority, or on
  per-use recomputation of origin.

Both stay open. Neither was changed.

**5. Is Arm L ready for a narrowly scoped admission proposal?**

**Still NO-GO for admission.** It is ready to be written up as a narrowly scoped
**proposal**, because every invariant now has a specified, falsified witness and the
execution cost question is answered. Admission still needs four things:

- the EXO-140 / EXO-46 ruling;
- an owner-authorized contract extension for the presence, uniqueness, pointer-origin
  and lens witnesses;
- the Quack lowering owner's decision on register-gated `Gather`;
- a decision on where presence verification runs (per generation, not per query).

## Limitations

- One hierarchical layout, depth 4; candidates are depth-4 rows under one root.
- The witnesses and the gated program are lab code.
- `FrameStore` is advanced by the probe's own writer.
- Single-machine medians on a shared host.
- The `Range` and `Count` passes are word-wide; at much larger lanes they will
  re-emerge as the lane term.
