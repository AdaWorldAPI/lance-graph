# AriGraph as an LLM-OSINT pattern, episodic arc via the rig seam — v1

> **Status:** PROPOSED (feasibility verified; net-new wiring scoped)
> **Branch:** `claude/cognitive-compilation-lance-graph-h8sgym`
> **Operator:** jan@exo.red
> **Date:** 2026-06-21
> Companion to `.claude/plans/cognitive-compilation-v1.md` — this points the
> Cognitive Compilation loop at OSINT/online-research, with AriGraph as the
> memory organ.

---

## 0. Verdict

**Feasible, ~70% already built.** This is a *wiring* job, not a port. The one
genuinely net-new piece is the **episodic arc in rs-graph-llm + rig** (the
"training wheels"). Everything below it already exists and — as of this session
— compiles on a full checkout.

---

## 1. Verified baseline (was [H], now [G])

`cargo check -p lance-graph` (default features, full checkout, protoc installed)
→ **exit 0, 4m02s, 480 crates, no errors** (8 pre-existing `CausalEdge64::temporal`
deprecation warnings only). This verifies:

- `graph/arigraph/markov_soa.rs` — **AriGraph promoted to the ractor-owned SoA**
  hot path (`SoaWavePrimer::project` over `MailboxSoaView` → addressable
  `SpoRanks` triples + replayable provenance; integer-deterministic;
  register-preserving; `best_guess_match` over an injected distance — no float
  cosine, no learned embed). Its only dependency is the zero-dep contract.
- `graph/arigraph/episodic.rs` — the **episodic arc** (`Episode{observation,
  triplets, fingerprint, step, truth}`, hardness-gated unbundle/rebundle,
  Hamming retrieval).
- `graph/arigraph/{triplet_graph,witness_corpus,retrieval,orchestrator,spo_bridge,xai_client}.rs`.

> The "provisional / unverified-offline" note atop `markov_soa.rs` is now stale —
> update it to cite this green check when next touching that file (kept out of
> PR #571 to avoid scope creep).

---

## 2. Corrected architecture — the rig seam (operator nudge)

rs-graph-llm wires rig; **rig** is the single integration point for the LLM
angle AND both storage backends. There is no direct kv-lance call from
rs-graph-llm.

```
rs-graph-llm (graph-flow orchestration)
        └─► rig ─┬─ rig-core       LLM angle (teacher / critic)
                 ├─ rig-lancedb    vector store → lancedb 0.30  (episodic similarity/retrieval)
                 └─ rig-surrealdb  → kv-lance fork              (semantic SPO graph + provenance/timeline)
                         │  same lance substrate, same bytes
                         ▼
        lance-graph AriGraph-on-SoA (ractor-owned mailbox)  ⇄  surrealdb kv-lance VIEW
```

Confirmed: rig has `rig-lancedb` (→ `lancedb 0.30`) and `rig-surrealdb` (→ our
kv-lance fork); rs-graph-llm already depends on `rig-core` (feature `rig`).

**The episodic/semantic split maps onto the two rig adapters:**

| AriGraph concept | rig adapter | lance-graph runtime mirror |
|---|---|---|
| episodic vertices (recalled experiences) | `rig-lancedb` (embedding/NN retrieval) | `arigraph/episodic.rs` (fingerprint/Hamming) |
| semantic SPO knowledge graph | `rig-surrealdb` (kv-lance) | `arigraph/triplet_graph.rs` |
| episodic **arc** (commit-per-step timeline) | `rig-surrealdb` versioned writes | surrealdb kv-lance version history |
| witness provenance (OSINT citations) | written alongside SPO | `arigraph/witness_corpus.rs` |

What rig persists through these adapters **is** the AriGraph tenant SoA —
transparently the surrealdb kv-lance view (per `E-S6`: one `FixedSizeBinary(512)`
SoA dataset, surrealdb is the VIEW). No bridge, no copy.

---

## 3. Net-new work (the only additive code)

1. **`episodic-arc` graph-flow Task** in rs-graph-llm (sibling of the merged
   `template-task`, cherry-pickable). During a rig-driven online-research run it:
   - drives `rig-core` for the LLM angle (extract claims/facts from fetched text);
   - writes each observation as an episodic vertex via `rig-lancedb`;
   - writes each fact as SPO + witness provenance via `rig-surrealdb` (kv-lance),
     one versioned commit per step → the episodic arc;
   - reads back via `rig-lancedb`/`rig-surrealdb` for AriGraph retrieval.
2. **SoA tenant mapping** — map AriGraph's `SpoRanks` (3×u16) + `truth` + `step`
   onto the SoA **value tenants** (the #511 `SoaMemberSpec` slots) so the
   AriGraph row IS a NodeRow tenant slice. (Schema task; the row shape exists.)
3. **OSINT fetch** reuses the existing `lance-graph-osint` crate
   (`r.jina.ai/URL → DeepNSM → triplets → TripletGraph`); rig-core supplies the
   LLM angle where NARS-only extraction is insufficient.

---

## 4. The loop, OSINT-flavored (rides the cognitive-compilation gates)

```
public claim + N online articles
  → rs-graph-llm orchestrates:
       rig-core (LLM teacher) extracts claims/facts ─┐
       lance-graph-osint fetch+triples ──────────────┤→ episodic-arc Task
       write episodic (rig-lancedb) + SPO/witness (rig-surrealdb, versioned) ─┘
  → AriGraph retrieval answers the research question
  → cognitive-compiler transcribes the run into an Elixir OSINT template
  → template-equivalence replays vs the recorded AriGraph state (AS-OF)
       (fail-closed: claims set-equal, witness provenance preserved, ranking set preserved)
  → review-gates (ethics/adversarial) + §14 OsintGuardrail
  → promote → future runs use the template + AriGraph retrieval, LLM OFF
```

"Training wheels off" = once a compiled template + AriGraph retrieval reproduce
the research result and `template-equivalence` passes, the LLM drops out of the
hot path — the same gate as `source_ranking_v1`, now for OSINT facts/stories.

---

## 5. Ethics + provenance (non-negotiable)

- Every online-research template carries the §14 `OsintGuardrail`
  (`public_interest_reason` / `scope_boundary` / `source_provenance_required` /
  `harm_minimization_checked`) — already in `elixir-template`, round-trip
  preserved. **Public claims / officials / institutions / media only.**
- `witness_corpus` + source spans enforce "no source span → no claim"; the
  fail-closed verifier rejects any uncited fact (provenance must survive replay).
- Brutal reviewers (`Adversarial` / `Ethics`) gate promotion (poisoning,
  propaganda, manipulation).

---

## 6. Dependencies / risks

- **surrealdb #50 (transparent versioning)** — the episodic arc IS commit-per-step
  + AS-OF replay; needs the corrected version→snapshot mapping. Load-bearing.
- **lance-7 lockstep** — `rig-lancedb` (lancedb 0.30) + `rig-surrealdb` (kv-lance)
  + lance-graph + surrealdb must move together (the pin we landed).
- **Python AriGraph is reference, not runtime** — it's a TextWorld/QA agent
  (contriever + GPT prompts). Runtime = the Rust port + rig; don't re-run python.
- **markov_soa**: now compiles ([G]); still update its stale status note.

---

## 7. First slice (mirrors `source_ranking_v1`)

"Build a sourced fact-graph for one public claim from N online articles":
the episodic-arc Task drives rig (core+lancedb+surrealdb) to fetch, extract,
record (episodic + SPO + witness, versioned), and retrieve; transcribe to an
Elixir OSINT template; replay-and-compare fail-closed; promote.

---

## 8. Deliverables

- **D-CC-ARI-1** — verify AriGraph-on-SoA compiles on full checkout — **DONE [G]**
- **D-CC-ARI-2** — `episodic-arc` graph-flow Task (rs-graph-llm, via rig seam) — Queued
- **D-CC-ARI-3** — SoA tenant mapping (`SpoRanks`+truth+step → value tenants) — Queued
- **D-CC-ARI-4** — OSINT first-slice end-to-end (fetch→episodic→template→verify) — Queued
- **D-CC-ARI-5** — refresh `markov_soa.rs` stale status note — Queued (separate small PR)
