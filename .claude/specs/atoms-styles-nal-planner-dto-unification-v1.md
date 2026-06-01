# Unifying atoms + thinking-styles + NAL into the planner DTO across ractor ↔ surrealdb(vart) ↔ lance-graph-planner (v1)

**Status:** DESIGN (3-agent council C1 ractor / C2 surreal / C3 atoms-styles, full-file reads per `E-READ-NOT-GREP`; + vart keystone). Branch claude/jolly-cori-clnf9. The capstone `causal-edge::syllogism` (PR #450) is the reasoning kernel this wires up.

## 0. The one-sentence thesis

The 33 **atoms are the field-decomposition of a reasoning step**, not a separate thing to bolt on: a planner step is the 14-bit triple **(style, rung, rule)** + the emitted little-endian connectome edge; that one artifact is *emitted* by the planner (white-matter thinking), *dispatched* by ractor (grey-matter compute), and *persisted + scheduled* by surrealdb's **vart** versioned radix trie (white-matter memory). The **little-endian `CausalEdge64`/`EpisodicEdges64` bytes are the shared key** binding all three.

## 1. The streamline — atoms ▸ hardwiring ▸ NAL ▸ grammar-styles are ONE basis

The atom families ARE the step's fields (`atoms.rs:135`):
- **Pearl (3: see/do/imagine)** = the SPO 2³ causal mask already on `CausalEdge64` (`CausalMask`). `do`=Intervention, `imagine`=Counterfactual.
- **Rung (9: r1..r9)** = `cognitive_shader::RungLevel` (0..9; Counterfactual=6).
- **Operation (8: abduct/deduce/induce/synthesize/…)** = the NARS rules. **`Figure::rule()` (the shipped hardwiring) maps SPO term-sharing → `InferenceType` → an Operation atom**: Chain/ChainRev→`deduce` (dim18), SharedSubject→`induce` (dim19), SharedObject→`abduct` (dim17).
- **Sigma (5) / Presence (4) / Meta (4)** = modulation knobs (`FieldModulation`).

So: **atom = one i4 lane; style = an i4×33 weighting over atoms (the molecule, identity = `ThinkingStyle`); persona = styles + thresholds** (`atoms.rs:10-16`, `recipe.rs`). The NAL rule is just the Operation slice of the style vector. grammar `GrammarStyleConfig` is the *parse policy* keyed by the same `ThinkingStyle` — keep it separate (it is not an i4 vector).

### Duplication to collapse (C3)
- **5 style representations** → pick ONE basis: `atoms::I4x32` (33-lane vector) + `ThinkingStyle` (36 identities) + grammar-policy (separate). RETIRE: the 8-wide `nars_engine::StyleVector` (make it a Pearl-slice *view*) and the stale "34 styles" doc claim.
- **4 `InferenceType`/`NarsInference` enums** → ONE canonical: `causal-edge::edge::InferenceType` (8-variant, signed i4 mantissa, carries Intervention/Counterfactual = the Pearl-3 atoms). `grammar::inference` already has `.core()`; route `nars.rs` → causal-edge. Keep `cognitive_codebook::NarsInference.notation()` strings as the NAL-notation reference only.
- **Missing bridge:** `ThinkingStyle → I4x32` resolver (today `ThinkingStyleProvider::style_vector` returns a 23D crewai vec; there is no map to the 33 atoms). This single absence forks the style space 3 ways (23D/8D/33D).
- **The missing half of the hardwiring (downward bias):** `figure()` is detect-then-label (integer palette equality, no style — keep that firewall). The complement is a **style-biased try-order**: a style's `deduce/induce/abduct` lane signs reorder which figure `figure()` attempts first when several terms match. New, small, firewall-safe (it only reorders attempts; it never enters the truth math).

### The genuine BLOCKER (the user's decision)
`atoms::I4x32` is **32 lanes**; the basis is **33** (`atoms.rs:39-43`, BLOCKED). `I4x32::pack/unpack` are `todo!()`, which blocks all of `recipe.rs`. Must be decided, not guessed. C3's surfaced option: fold the 33rd lane (`verbosity`, dim 32, a Meta knob) out-of-band so the carrier splits 3+9+5+8+4+3.

## 2. The unified step word (the planner DTO field-set)

```
style : ThinkingStyle   6 bits  → MetaWord.thinking ; resolves to I4x32 over 33 atoms
rung  : RungLevel       4 bits  (0..9)               [Rung atom family]
rule  : InferenceType   i4 mantissa (signed)         [Operation atom family; = Figure::rule()]
        + emitted CausalEdge64 / EpisodicEdges64 (the connectome edge, LE bytes)
```
= 14 bits of identity + the edge. Carriers already exist: `MetaWord(u32)` (`thinking6+nars_f8+nars_c8+…`), `ShaderDispatch{style,rung}`, `plan::ThinkingContext{style,inference_type,…}`, `UnifiedStep`/`OrchestrationBridge`/`StepDomain` (the canonical cross-domain DTO; `PlannerAwareness` implements it). **Gaps:** `ThinkingContext` lacks `rung`; `MetaWord` lacks a rule field; `PlanResult` has **no edge-emission surface** to hand the connectome edge downstream.

## 3. The three-system unification (one artifact, the LE-byte key)

```
 lance-graph-planner (WHITE: thinking)            ractor (GREY: compute)                 surrealdb/vart (WHITE: memory)
 ───────────────────────────────────             ──────────────────────                 ──────────────────────────────
 emit UnifiedStep{style,rung,rule}        ──►   ConsumerEnvelope::Plan(step)      ──►   persist edge → vart versioned ART
 + CausalEdge64/EW64 (LE bytes)                 handle(&self, &mut State)               keyed by LE edge bytes (radix)
        ▲                                        emit CollapseGateEmission                each commit = a vart VERSION
        │                                        = Vec<(u16 target, u64 edge)>                   │
        │  next step (try_advance_phase)         (the Baton; white matter as PAYLOAD)            ▼ DatasetVersion tick
        └──────────────  KanbanMove  ◄──── VersionScheduler::on_version(view, version, exec) ◄── LIVE / LanceVersionWatcher
                         "propose, don't dispose" (scheduler.rs; surreal proposes, ractor owner mutates)
```

- **ractor (C1):** grey-matter dispatch + supervision; it **moves batons, it does not weave them** — `CausalEdge64`/`EW64` ride as *payload* inside the envelope/emission, never as ractor state (BBB membrane bans `Vsa16kF32`). `handle(&self, …, &mut State)` structurally enforces the "no &mut self during compute" rule. **The one missing wire: the supervisor→child forward** (stub `supervisor.rs:198`); the DTO is one new closed `ConsumerEnvelope::Plan` arm (auto-`Message` via blanket impl); emission is `CollapseGateEmission` (zero-dep, Send, `wire_cost=13+10n`). Risks: `Send+'static`, bounded mailbox → `MailboxFull` backpressure at 32k–64k, one-for-one supervision must be overridden.
- **surrealdb/vart (C2 + keystone):** the IN-direction reactive seam `E-SUBSTRATE-IS-THE-SCHEDULER` is fully specified, zero shipped impl. `VersionScheduler::on_version(&V view, DatasetVersion, exec) -> Option<KanbanMove>` proposes (never `&mut`); `DemotionSink::demote` is the hot→cold exit; `ExecTarget::{Native,Jit,SurrealQl,Elixir}` routes the backend. Working primitive: `LanceVersionWatcher` (std-sync, no tokio). **Gaps:** `PlanResult` edge-emission surface (gap 1); a `LanceVersionScheduler` impl; the `WatchReceiver→DatasetVersion→on_version→try_advance_phase` adapter; a vart/surreal `DemotionSink` (OQ-11.6); unblock the surreal kv-lance fork dep (BLOCKED(C)).

## 4. vart — the keystone

`vart = "0.9.3"` is surrealdb's **versioned adaptive radix trie** = its `kv-mem` engine; AdaWorldAPI's `kvs/lance/` backend layers Lance (durable append-only) under a vart memtable (`memtable.rs`/`wal.rs`/`tx_buffer.rs`). vart is simultaneously, with no extra machinery:
1. **the cold/warm connectome store** — radix-keyed by the **LE `CausalEdge64`/`EpisodicEdges64` bytes** (the "append-never-renumber *gives* you a radix" of `E-EPISODIC-CLOSURE`, made concrete);
2. **the MVCC version source** — each transaction commit = a version = the `DatasetVersion` tick that drives the scheduler;
3. **the `DemotionSink` target** — demoted EW64 edges append into the vart cold tier, re-prefetchable (`E-EW64-IS-PREDICTIVE-PREFETCH`);
4. **surreal-native** — it unblocks the BLOCKED(C) kv-lance skeleton instead of inventing a store.

The "32k reasoning-chained mailboxes with little-endian contract" is *literally vart's keyspace*: the LE edge bytes are the radix key; the MVCC versions are the reasoning chain's epochs; the snapshot reads are the lock-free grey-matter actor reads.

## 5. Prioritized actions

**A. Offline / unblocked NOW (testable like the syllogism PR):**
1. Add `rung: RungLevel` to `plan::ThinkingContext` → the step carries the full (style, rung, rule) triple. (Additive, no blocker.)
2. Collapse the 4 `InferenceType` enums onto `causal-edge::edge::InferenceType` via `.core()`-style maps; retire the 8-wide `StyleVector` to a Pearl-slice view + delete the "34 styles" doc.
3. Add the **style-biased figure try-order** (Operation-lane weights reorder `figure()`'s attempts; truth math untouched — firewall held).
4. Add a `PlanResult` edge-emission surface (expose emitted `CausalEdge64`/`EW64` as raw u64/LE bytes) so the DTO has something to persist.

**B. Gated on the 32-vs-33 DECISION (user's call):**
5. Implement `I4x32::pack/unpack` + the `ThinkingStyle → I4x32` resolver → unblocks all of `recipe.rs` (style/persona composition).

**C. Cross-repo seams (each a self-contained slice):**
6. **ractor:** the `ConsumerEnvelope::Plan` arm + the supervisor→child forward (the one stub).
7. **vart/surreal:** the vart-backed `DemotionSink` + `LanceVersionScheduler` + the watcher→scheduler adapter; prerequisite = unblock the surreal kv-lance fork dep (BLOCKED(C)).

## 6. Firewall (held throughout)

Figure detection = integer palette equality (PROPOSE); truth-function = deterministic (ADDRESS); style-bias = try-order only, never truth math. vart keys on integer LE bytes — no float, no language, on the hot path. The LE byte contract is the single shared grammar (ractor batons, vart radix keys, planner emissions, surreal WAL).

Cross-ref: `E-NARS-FIGURE-CAPSTONE`, `E-EPISODIC-CLOSURE`, `E-SUBSTRATE-IS-THE-SCHEDULER`, `E-PLANNING-IS-WHITE-MATTER`, `E-EW64-IS-PREDICTIVE-PREFETCH`, the PROPOSE/ADDRESS firewall.

## 7. vart — vendored + concrete API (added after the zipball grab)

**Vendored now** at `/home/user/vart` (AdaWorldAPI/vart fork, `main`, v0.9.2; surrealdb pins crates.io `0.9.3`). "zipball now, Cargo path-dep vendor-import later, same as `/home/user/ndarray`." Cargo wiring is deferred to the gated surreal/vart slice (action C7).

Concrete API (`vart/src/lib.rs`; the `Tree` + insert/get/snapshot lives in `art.rs`):
- **`Key` trait + `FixedSizeKey<const SIZE>`** with `From<u64>/u16/u8` (via `to_be_bytes`) and `From<&[u8]>`. ⇒ a connectome word is a vart key *directly*: `FixedSizeKey::<8>::from(edge.0)` or `from_slice(&edge_key_bytes)`.
- **`KeyTrait = Key + Clone + Ord + Debug + From<&[u8]>`** — the trie is byte-ordered (lexicographic).
- **`TrieError::{VersionIsOld, SnapshotOlderThanRoot, RootIsNotUniquelyOwned}`** — versioned MVCC with monotonic version ordering + snapshot generations.
- **"immutable versioned"** = persistent / copy-on-write / structural-sharing.

How it lands the architecture concretely:
- **Key = the addressing prefix (BE), value/WAL = LE.** vart orders lexicographically, so encode the *radix key* big-endian over the addressing fields (CausalEdge64 S→P→O, or EW64 family→local) so edges sharing a Subject/family **share a trie prefix** (range-scannable basins). Keep the connectome's frozen `to_le_bytes` as the stored **value**/WAL format. The BE-key vs LE-wire split is a deliberate, citable choice — decide the exact key projection when wiring C7.
- **vart version = DatasetVersion tick.** Each transaction commit = a monotonic vart version (`VersionIsOld` guards forward-only). The surreal `LIVE`/`LanceVersionWatcher` reads the new version → `VersionScheduler::on_version(snapshot_view, DatasetVersion(n), exec)` → next `KanbanMove`. `E-SUBSTRATE-IS-THE-SCHEDULER` is *literally* vart's version counter.
- **Immutable snapshot = lock-free grey-matter read.** A ractor mailbox actor reads a vart snapshot of the connectome with zero locking (structural sharing); the single white-matter writer commits a new version. This satisfies the data-flow rule ("no `&mut self` during compute; caches built once / interior-mutable") for free.
- **`DemotionSink` → `vart.insert(key, edge)`.** A demoted EW64 `EdgeRef` resolves to its connectome word and inserts into the vart cold tier under a new version; re-prefetch = a prefix/range scan of the basin (`E-EW64-IS-PREDICTIVE-PREFETCH`).
- Under surrealdb this rides the existing `kv-mem`/`kvs/lance` path (vart memtable over a Lance append-only WAL) — so C7 unblocks the BLOCKED(C) skeleton by adopting the surreal-native store rather than inventing one.
