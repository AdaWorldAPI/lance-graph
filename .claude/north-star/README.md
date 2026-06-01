# North Star — the two reference diagrams (aspirational, 2026-06-01)

> **Drop the PNGs here** as `business-viewangle-odoo-ogit-dolce-gobd.png` and
> `semantic-viewangle-cognitive-substrate.png` (the harness didn't persist the
> uploads, so this README is the faithful textual capture until the binaries land).
>
> **READ BY:** anyone wiring the atoms+styles+NAL → planner-DTO unification, the
> ractor/surreal/vart seams, or the Odoo-in-Rust business layer. This is the
> *destination*; `.claude/specs/atoms-styles-nal-planner-dto-unification-v1.md`
> (§0–14) is the design; `.claude/plans/north-star-integration-v1.md` is the path.

## The meta-point: the two diagrams ARE the dual grammar (spec §14)

- **Picture 1 = the BUSINESS ViewAngle** — *"ODoo in Rust × OGIT + DOLCE + GoBD → Deterministische Business-Logik in Elixir (OTP/BEAM)"*. The product framing.
- **Picture 2 = the SEMANTIC ViewAngle** — the cognitive substrate, a 1:1 render of spec §9–14. The technical framing.

The same system, resolved under two grammars — the pair **demonstrates the §14 dual-view thesis it documents**. head2head between them is the integration.

## The killer thesis (why this beats Odoo *and* every LLM-agent business tool)

**GoBD compliance is a *consequence* of the firewall, not a feature bolted on.**
No LLM on the hot path → deterministic → **replayable + auditable** →
Nachvollziehbarkeit / Vollständigkeit / Unveränderbarkeit / Ordnung. LLM-based
automation **structurally cannot** be GoBD-auditable (LLM reasoning isn't
replayable). NARS-micro-truth → syllogism reasoning over an **immutable vart/Lance
versioned store** *can*. The "similarity PROPOSES / CAM ADDRESSES" firewall IS the
compliance guarantee.

---

## Picture 1 — BUSINESS ViewAngle (Odoo-in-Rust)

Header: *32.000+ Aktoren · Goal/Belief State Updates · Kanban-Orchestrierung · Lance Versioning · SurrealDB Live Actions.*

- **GRAMMATIK – SEMANTIK – BUSINESS** (3-layer): Grammatik (Syntax) → Bedeutung (Semantik, **NSM/DeepNSM**) → **Business Grammatik (Pragmatik)** [Rollen, Regeln, Kontext & Ziele].
- **OGIT VERERBTE KLASSEN & OPERATIONEN**: OGIT Base Class → Vererbte (inherited) Klassen → Schema → **Maskierung (Bitmask)** → Reasoning (Regelwerk). *"Alle Klassen, Attribute, Operationen, Constraints vererbbar & komposierbar."*
- **DOLCE ONTOLOGIE**: Endurants (Dinge) · Perdurants (Prozesse) · Qualitäten/Rollen · Relationen.
- **GoBD COMPLIANCE** (6): Nachvollziehbarkeit · Vollständigkeit · Richtigkeit · Zeitgerechte Erfassung · Unveränderbarkeit · Ordnung & Aufbewahrung.
- **DETERMINISTISCHE BUSINESS-LOGIK (ODoo in Rust)** — *Kompilierbar · Testbar · Replaybar · Auditierbar*:
  - INPUT: UI/API · Importe · Events · IoT/Geräte · Externe Dienste.
  - **RActor Mailbox System (OTP/BEAM) — 32.000+ parallele Aktoren**: RActor A1..An → Mailbox → **Belief State (Weltmodell) + Goal State (Ziele)** → **Reasoning Engine (deterministisch)**.
  - **OGIT Reasoning Pipeline**: Pattern Matching → Regel-Ableitung → Constraint Solving → Prozess-Ableitung (100%) → Aktion/Transition.
  - **SELBSTORCHESTRIERUNG – KANBAN BOARD**: BACKLOG · NEXT · DOING · VERIFY · DONE — *Board Updates via Lance Versioning.*
- **SURREALDB (Live) — Event Store & Actions**: Daten → Live Queries → **Live Actions (Trigger/Funktionen)**.
- **LANCE VERSIONING (Delta-Lake-Style)**: Immutable Snapshots (Versioned Tables) → Delta Logs (Changes) → Time Travel (Replay) → Branching (Scenarios).
- **SUBSCRIBERS**: Boards · Aktoren · External Services · Dashboards.
- **RUST CORE ENGINE** (Hochperformant · Speichersicher · WASM-Ready · Kompilierte Rules) + **ELIXIR (OTP/BEAM)** (Fault Tolerant · Supervision Trees · RActor Concurrency · Hot Code Upgrade).
- **PROZESS ABLAUF (100% deterministisch)**: Ereignis → Interpretation → OGIT Regelwerk → Reasoning → Prozess-Schritt → Aktion → State Update → **Audit Log**.
- **AUDIT & COMPLIANCE (GoBD)**: Unveränderbare Logs · Vollständige Historie · Nachvollziehbare Prozesse · Export/Archiv (GoBD-konform).
- Footer: *OGIT (Struktur) + DOLCE (Ontologie) + Grammatik/Semantik (Bedeutung) → deterministische Business-Prozesse wie ODOO. Skalierbar auf 32.000+ Aktoren · Echtzeit · Konsistent · Auditierbar · Zukunftssicher.*

## Picture 2 — SEMANTIC ViewAngle (the cognitive substrate; render of spec §9–14)

Example input: *"How can we reduce customer churn for our SaaS product while improving LTV?"*

1. **DUAL GRAMMAR (§14)** — two ViewAngles, head2head-resolved: **SEMANTIC** (DeepNSM → meaning: parse & roles, core NSM, concept lattice) **VS BUSINESS** (OGIT → stakes: **Objectives, Gaps, Impacts, Tradeoffs** = the OGIT lens). HEAD2HEAD RESOLUTION (aligned meaning ↔ stakes; tensions/complements/priorities).
2. **RESOLVER (§9-10)** — **I4x32D → OGIT class → best-practice template + attention bitmask (for free)**: I4x32D encoding (*"integrated 4-view 32D latent"*) → OGIT classifier (multi-label, problem archetype) → best-practice template (playbooks/patterns/anti-patterns) → attention bitmask (what's relevant).
3. **REASONING (§12.2) — CAPSTONE ✓**: NARS micro-truth (f, c) → **`Figure::syllogize`** (*"temporal & eternal syllogistic forms, 64 moods"*) → DERIVED **(style=mood, rung=abstraction, rule=inference)** → **LE EDGE** (S→P Linked-Expectation; strength/mood/time/context).
4. **REPRESENTATION (§12.3)** — edge → **vart radix** (the connectome): LE edge → VART RADIX addressing (`α.β.γ.δ.ε.ζ.η.θ` = 8-position Base-16 surreal address) → CONNECTOME STORE (hypergraph of LE edges + attributes) → INDEXES (by concept · by rung · by time/**vart version** · by style/rule).
5. **ORCHESTRATION (§13)** — ractor mailboxes ↔ surreal/**vart ACTIVE Rubicon kanban** (vart versions = the clock): RACTORS (isolated concurrent workers) ↔ MAILBOXES (async, backpressure) ↔ RUBICON KANBAN (BACKLOG/NEXT/DOING/VERIFY/DONE) ↔ SURREAL/VART CLOCK (vart versions advance time = immutable ticks).
6. **META-AWARE (§12.4) — A2 ✓**: awareness chain over episodic basins, **addressable by `rung`**: episodic basins → awareness chain (r0→r1→…→rn, increasing abstraction) → `get_awareness(r=n)` (jump to the right altitude) → uses (select perspective · control reasoning depth · meta-decide/monitor).
7. **OUTPUT (§13)** — **epiphanies · facts · high-signal separation**: epiphanies (non-obvious high-utility connections) → facts (justified, sourced) → high-signal separation (rank/filter/dedupe/explain/stress-test) → deliverables (recommendations/plans/experiments/metrics) → **feedback loop** (outcomes → new episodic basins).

NOTES (from the diagram): *vart versions = clock (immutable total order); every edge time-stamped in vart; all components composable & replaceable.*

## Shipped (✓) vs target, against Picture 2

- **§3 CAPSTONE ✓** = `causal-edge::syllogism` (`Figure`/`figure()`/`syllogize()`) — merged #450.
- **§6 A2 ✓** = `rung: RungLevel` on `ThinkingContext` — merged #450.
- **§3 `rule`** = A1 `InferenceType::{to,from}_mantissa` + `From<grammar::NarsInference>` — merged #450.
- **§4 entry arrow** = A6 `PlanResult.emitted_edges` (the LE edge → vart) — branch.
- **TARGET (the run):** §2 resolver = A3 (I4x32D carrier) + A4 (OGIT resolver); §3 moods = A5; §4 store/§5 kanban = C7 (vart/surreal); §5 actors = C6 (ractor). Plus the business layer (OGIT classes, GoBD audit, Elixir/Rust split).
