# Scientific-KG substrate — crawl → OCR → terms → reason → MUL (v1, scoping)

> **Status (§0):** PROPOSED — **scoping doc**, no code. Records the operator's
> direction (2026-07-23): *"wire spider-rs / tesseract-rs to reason about a
> scientific basis, find adjacent thinking, and build a scientific knowledge
> graph based on MUL and reasoning."* The **outward-facing crawl is BLOCKED**
> until its open decisions (§4) are resolved; the local, non-crawl deliverables
> (D-SCI-1) are buildable on a further "Go". Nothing here supersedes a shipped
> decision.
>
> **The one-line frame:** this wave is mostly a **composition of seams that
> already exist** — tesseract-rs `doc.v1`, the OGAR adapter path,
> `deepnsm-v2` (shape/ancestry/reason, shipped), the D-SRS-3 basin self-codes
> (planned), and MUL (`lance-graph-planner`). The genuinely NEW pieces are two:
> a **term/entity extractor** (the gate the finding below demands) and the
> **spider-rs crawl** (outward-facing).

---

## 1. The pipeline (what composes what)

```text
 spider-rs            tesseract-rs           deepnsm-v2                 lance-graph-planner
 (crawl, NEW)   →     doc.v1 (OCR,     →     term extract (NEW) →       MUL
 [BLOCKED §4]         EXISTS) ──OGAR──┐      FSM → SPO → shape/          (self-uncertainty,
                                      │      ancestry/reason (SHIPPED)   EXISTS)
                                      └──────────────────────────┐      ▲
                                                                 ▼      │
                                                        scientific KG ──┘ D-SRS-3 basin
                                                        (deepnsm-v2)      self-codes (PLANNED)
```

Seam-by-seam, with honest provenance:

- **Crawl → text (NEW, outward-facing).** `spider-rs` fetches scientific
  sources. This is the only genuinely external step; it is BLOCKED on §4
  (fork coordinates, robots/ToS, a licensed source whitelist). Digital-native
  papers (arXiv OA HTML/LaTeX) skip OCR entirely.
- **PDF/scan → text (EXISTS).** `tesseract-rs` already emits **`doc.v1`** (rich
  recognition: words/boxes/regions/tables) via its `OcrExecutor` /
  `decode_image` consumer surface, and its own docs route `doc.v1` **through
  OGAR adapters to `lance-graph-arm-discovery` / DeepNSM** (tesseract-rs
  `CLAUDE.md` "doc.v1 is the OPTIONAL seed a consumer feeds via OGAR to
  lance-graph-arm-discovery / DeepNSM"). We CONSUME that seam; we do not build
  an OCR path.
- **Text → entities (NEW — THE GATE).** `E-ACADEMIC-VOCAB-COLORBLIND-1`
  (2026-07-23): an academic-word vocab is structurally blind to the proper
  nouns that carry structure (0/7 KJV genealogy carriers present; the
  `begat`-forest dissolved). Scientific entities — named methods, authors,
  compounds, gene/theorem/dataset names — are exactly that omitted class. So
  the KG's quality **gates on term/entity extraction, not an academic lens.**
  This is D-SCI-1.
- **Entities → KG + reasoning (SHIPPED).** `deepnsm-v2`: FSM → SPO
  (`fsm`/`spo`), the shape detector routes each relation to its carrier
  (`shape`, `E-SHAPE-DETECTOR-MEASURED-1`), ancestry/taxonomy relocates to the
  DN radix-trie (`ancestry`), and the derivation-pointer fabric reasons over it
  (`reason`, `E-SELF-REASONING-FABRIC-1`). "is_a"/"part_of"/"cites"/"derives-from"
  scientific relations are precisely the per-predicate shapes the detector
  already classifies.
- **KG → self-uncertainty (PLANNED — D-SRS-3).** MUL wants a *"where am I
  uncertain"* signal; the D-SRS-3 basin self-codes (`self-reasoning-substrate-v1`)
  ARE that signal (basin distribution width = uncertainty). **D-SCI-4 depends on
  D-SRS-3** — the self-reasoning ladder and this wave converge at MUL.
- **Reasoning (EXISTS).** MUL (`lance-graph-planner/mul/`: Dunning-Kruger,
  trust qualia, compass, homeostasis, gate) consumes the uncertainty signal and
  drives *"adjacent thinking"* = concept adjacency (shape-similar predicates +
  basin proximity in the KG).

**Wiring direction (non-circular, confirm at §4):** `deepnsm-v2` SUPPLIES the KG
+ shape/uncertainty signals; `lance-graph-planner`/MUL CONSUMES them — the same
non-circular direction the workspace already uses (planner depends on the
substrate, never the reverse). `deepnsm-v2` must NOT gain a planner dependency.

---

## 2. Why the finding makes this tractable, not naive

The color-blindness finding is not a caveat — it is the wave's **first gate and
its cheapest de-risking**. It says: before any crawl or OCR, prove that
term/entity extraction recovers the structure an academic vocab destroys. If
D-SCI-1 fails (terms don't recover forest/dag structure that academic vocab
flattens), the whole wave is built on sand and we stop. If it passes, the
downstream seams are all already-shipped or already-designed — the risk
collapses onto the two new pieces, and one of them (the crawl) is deferred.

---

## 3. Deliverables — the ladder (probe-first, each KILL-gated)

> Gate numbers are registered pre-run per deliverable (git-ordering anti-tuning,
> as in D-SRS-1/2); the values below are the SHAPE of each gate, `X` = registered
> at authorization.

### D-SCI-1 — Term/entity extraction (LOCAL, the gate; buildable now)

Extract domain terms (capitalized/multi-word noun phrases, non-function tokens)
from a scientific corpus (arXiv OA abstracts / a public-domain science text,
never committed), build the KG with the TERM vocab, and compare to the same
corpus under the academic-20k vocab.

- **PASS gate (the inverse of `E-ACADEMIC-VOCAB-COLORBLIND-1`):** the
  term-vocab KG exposes **non-trivial forest/dag structure** (≥ `X`% of
  predicates route to RadixTrie/TriePlusEscalate/MaterializedFabric with
  amortization ≥ `2×`) that the academic-vocab KG **flattens** (routes to
  EdgeTable/Cyclic). The graph's shape must track the corpus's real entities,
  not the lens.
- **KILL:** term-vocab and academic-vocab KGs have statistically
  indistinguishable shape distributions — entity extraction bought nothing, the
  substrate can't see scientific structure, STOP the wave.

### D-SCI-2 — OCR ingest via the tesseract-rs `doc.v1` seam (cross-repo)

A scanned/PDF paper → tesseract-rs `doc.v1` → (OGAR adapter) → term extract →
`deepnsm-v2` KG. Consumes the EXISTING seam; builds no OCR.

- **PASS gate:** a paper round-trips image → `doc.v1` → terms → KG with ≥ `X`%
  of the digital-native (HTML/LaTeX) KG's entities recovered on the same paper
  (OCR noise floor registered against the digital source as oracle).
- **KILL:** OCR noise destroys entity recovery below the floor — route those
  sources to digital-native only.

### D-SCI-3 — The crawl (spider-rs) — ⚠ OUTWARD-FACING, BLOCKED on §4

Fetch a **licensed, whitelisted** scientific source set (e.g. arXiv OA / PMC OA
subset) under robots/ToS compliance, rate-limited, into the D-SCI-1/2 pipeline.

- **PASS gate:** N papers crawled within the licensed whitelist, 0 robots/ToS
  violations, feeding a KG whose shape census matches D-SCI-1's per-domain
  expectation.
- **KILL / STOP:** any source outside the whitelist, any robots/ToS breach —
  hard stop, not a warning.
- **Not authorized by this doc.** Requires the §4 decisions + explicit operator
  green-light. Never crawl on inference.

### D-SCI-4 — MUL reasoning + adjacent thinking (depends on D-SRS-3)

Feed the KG's D-SRS-3 basin self-codes into MUL; surface *"where the graph is
uncertain"* and *"adjacent concepts"* (shape-similar predicates + basin
proximity).

- **PASS gate:** on a held-out split, MUL's uncertainty ranking correlates
  (≥ `X`) with an independent measure of the corpus's under-supported regions,
  AND the adjacency surfacing recovers ≥ `X`% of a human-curated "related
  concepts" set for sampled seed terms.
- **KILL:** uncertainty ranking uncorrelated (the graph doesn't know what it
  doesn't know) OR adjacency is noise (no better than frequency).
- **Prerequisite:** D-SRS-3 shipped. Non-circular MUL wiring confirmed (§4).

---

## 4. Open decisions — BLOCKERS before the outward-facing build

1. **spider-rs coordinates + status.** Is there an `AdaWorldAPI/spider-rs`
   fork? (Per the workspace P0, a forked crate MUST be wired via the fork, never
   crates.io — and if coordinates are unknown, STOP and ask.) Unknown ⇒ this is
   a hard blocker, not a default.
2. **Crawl scope + compliance.** Which sources, under which license (arXiv OA /
   PMC OA have explicit terms)? robots.txt honored, rate-limited, redirects off,
   size caps — the SSRF/politeness posture tesseract-ocr-web already models.
   Crawling is outward-facing and hard to reverse; it needs explicit operator
   scope, not agent inference.
3. **Corpus licensing for committed artifacts.** Like the AVL vocab and the KJV
   text, crawled corpora are **local-only, never committed**; only derived
   graph statistics/codes ship. Confirm no corpus text enters git.
4. **MUL wiring direction.** Confirm `deepnsm-v2` supplies, `lance-graph-planner`
   consumes (non-circular) — via a contract seam or the p64 convergence pattern,
   never a `deepnsm-v2 → planner` dependency.
5. **Where the scientific KG persists.** The SoA SPOG tenant follow-on
   (`Representation::graph_id()` = the G lane, `E-SHAPE-DETECTOR-MEASURED-1`) —
   or in-memory only for the probes.

---

## 5. Cross-refs

- `E-ACADEMIC-VOCAB-COLORBLIND-1` — the entity-extraction gate (D-SCI-1's basis).
- `E-SHAPE-DETECTOR-MEASURED-1` / `E-SELF-REASONING-FABRIC-1` — the shipped
  substrate (shape/ancestry/reason).
- `self-reasoning-substrate-v1` D-SRS-3 — the basin self-codes D-SCI-4 depends on.
- `literature-probe-ladder-v1` — the sibling genre-falsifier program;
  `genre_shapes.rs` is shared apparatus.
- tesseract-rs `CLAUDE.md` (doc.v1 / OcrExecutor / the OGAR → arm-discovery path),
  OGAR consumer-preflight — the OCR seam this wave consumes.
- `lance-graph-planner/mul/` — the MUL consumer.
