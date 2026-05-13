Status: F1 parity machinery shipped 2026-04-30 (see §3 F1 + §7 for the
as-shipped architecture). F1 latency benchmark not yet started. F2 is a
posture, not a delivery (see §7 next-deliverable).

# MedCare Foundry: Architectural Vision

Client-facing draft. Tone: brutally honest. No hype. No marketing claims that have not been
benchmarked.

---

## 1. Why a Foundry layer

A clinic's data substrate has three properties that a generic database does not give you for
free:

- **Data sovereignty.** The clinic must be able to point at every byte of patient data and say,
  with evidence, who has read it and who has written it. Not "in principle". With a log entry.
  Multi-tenant SaaS approaches that store everyone's data in one Postgres instance and rely on
  application-level filters do not give the clinic that evidence.
- **Audit-grade traceability.** Every query that touches a patient record should produce an
  audit-log entry by construction, not by convention. If audit is implemented at the application
  layer, then any developer who forgets to call the audit helper has a silent gap. We want audit
  to be enforced by the optimizer, so that "forgot to log it" is structurally impossible.
- **Unified record substrate.** A clinic's data model spans structured records (appointments,
  prescriptions), semi-structured documents (chart notes, intake forms), and references
  (lab results, imaging). A Foundry layer is the single substrate where all three live, with
  consistent access control, consistent audit, and consistent backup.

These three together are what we mean by "Foundry layer". It is not a buzzword. It is the
layer between the storage engine and the clinical application that enforces sovereignty, audit,
and unification.

---

## 2. Comparison table

*Anchor: as of 2026-04-30 (post-F1 parity ship). This table will age.
We commit to updating it at the end of every phase, F1 through F5, and
re-dating the anchor.*

| Property            | MedCare Foundry (target)                       | Palantir Gotham                                 | OpenEMR                                       | Raw Postgres + JSONB                                  |
|---------------------|------------------------------------------------|-------------------------------------------------|-----------------------------------------------|--------------------------------------------------------|
| Data sovereignty    | Per-clinic substrate; bring-your-own-storage   | Vendor-managed; sovereignty contractual         | Self-hosted; sovereignty by deployment        | Self-hosted; sovereignty by deployment                  |
| Audit               | Optimizer-enforced; append-only Lance log      | Built-in, vendor-defined                        | Application-level audit hooks                 | DIY (triggers or app code); easy to bypass              |
| Latency             | Designed to match C# direct-MySQL; benchmark pending | Vendor-published; not directly comparable | Bound by application stack on top of MySQL    | Application-stack-dependent                             |
| Cost                | Self-hosted; storage cost dominates            | Enterprise license; high                        | License-free; ops cost dominates              | License-free; ops cost dominates                        |

Caveats on this table:

- "Designed to match" in the latency row is a posture, not a benchmark result. F1's parity
  machinery has shipped (it verifies result equality query-by-query — see §7), but the
  separately-scoped F1 latency benchmark on a fixed corpus has not been started. Until that
  report is published, do not quote latency claims from this table in a customer setting.
- Palantir Gotham is a different product category. The comparison row is included because clinics
  ask about it; it is not a like-for-like.
- OpenEMR and raw Postgres+JSONB are the two realistic alternatives a clinic would actually
  consider. Treat those rows as the load-bearing comparison.

---

## 3. Five-phase transition F1 through F5

Each phase is described in one paragraph: what it is, and what benefit it delivers. We will not
ship a phase whose benefit cannot be stated in one sentence.

### F1 — Oracle parity

**Shipped 2026-04-30.** A MedCare Foundry instance's read path is now compared, query-by-query,
against the existing C# direct-MySQL groundtruth on a hand-curated subset of clinic data. Any
divergence is detected by the harness, not by an unhappy clinician. The full as-shipped data
flow (LanceProbe → ParityWitness → DriftSink → `/api/__parity/csharp`) is documented in §7.
F1 latency benchmarking on the same corpus is a separately-scoped follow-up that has not started.

### F2 — RBAC and audit

We turn on the row-level-security rewriter (lance-graph PR #278's
`lance_graph_callcenter::rls::RlsRewriter`, fortified by the round-2 sealed-registry fixes in
PR #280 and the `PolicyRewriter` trait that landed in PR #284) and the audit log (lance-graph
PR #278's append-only ring buffer, with the `LanceAuditSink` persistent writer in PR #302).
Every read of a patient record goes through the rewriter, which means every read either
matches an explicit role grant or is rejected. Audit entries are append-only and tenant-scoped.
The benefit is that the clinic can answer "who read this record" with evidence, end-to-end,
without trusting application code. **Status today**: lance-graph side is in production;
medcare-rs has not yet adopted the rewriter into its read path. F2 is therefore a posture
until the medcare-rs adoption PR lands — see §7.

### F3 — PostgREST shape

We expose the Foundry over an HTTP surface that follows PostgREST's URL and header grammar
(lance-graph PR #278's `postgrest::parse_path` + `EchoHandler` dispatcher, with the
URL-decoding + table-validation hardening from PR #280's `A-fix-postgrest`). Existing
PostgREST-aware tooling can be pointed at the MedCare Foundry with minimal code change. The
benefit is that integration with third-party clinical tools and dashboards is reduced to a
configuration change, not an SDK port. **Status today**: the URL parser + dispatcher stub is
on lance-graph main; the medcare-rs adopter (which decomposes ParsedQuery into a DataFusion
plan) is the round-2-and-beyond work.

### F4 — Dataflow

We light up the dataflow features that allow clinic-defined transformations (e.g. chart-note
summarization, lab-result normalization) to live as first-class objects with the same audit and
sovereignty guarantees as raw records. The benefit is that derived data — the part that is most
likely to grow over time — does not escape the Foundry into ad-hoc spreadsheets and notebooks.

### F5 — Federation

We allow controlled cross-clinic queries for explicitly federated cohorts (e.g. multi-site
research collaborations). Federation is opt-in, audited at both ends, and never enabled by
default. The benefit is that multi-clinic research stops requiring a manual data-extraction
ritual; the manual extraction was the part that was actually leaking sovereignty.

---

## 4. Performance posture

The Foundry is **designed to match** the C# direct-MySQL baseline on identical workload. F1
parity (correctness) shipped 2026-04-30; F1 latency benchmarking on the fixed corpus has not
been started. The two are separately-scoped F1 sub-deliverables.

> Footnote: this claim is "designed to match" until the F1 latency benchmark report is
> published. We will not market "faster than" on unmeasured ground. Any latency or throughput
> number that appears in client material must cite a specific F-phase benchmark report. If you
> cannot cite the report, do not quote the number.

What we will measure when the F1 latency benchmark runs, in this order:

1. End-to-end query latency, p50 and p95, on the fixed query corpus, against the MySQL
   groundtruth. Equal-or-better on p50 is the bar. Equal-or-worse-by-no-more-than-X% on p95 is
   acceptable for F1; F2 may regress this temporarily as RBAC rewriting is added.
2. Throughput at fixed concurrency, on the same corpus.
3. Cold-start time of a fresh Foundry instance.

We will not measure or claim anything about workloads we have not run. If a customer asks about
a workload we have not benchmarked, the answer is "we have not measured that workload yet; here
is what we have measured". This is the posture, full stop.

---

## 5. Risk posture

We list risks here so that any clinic considering a deployment knows what they are accepting.
Hiding risks would be a worse business strategy than naming them.

- **Calibration uncertainty.** The RLS rewriter uses a confidence-based model to decide when to
  apply column masking vs. row redaction. The confidence model is, at deployment, not yet
  calibrated against any specific clinic's data shape. We mitigate by deploying with conservative
  thresholds (favouring over-redaction). Per-clinic recalibration is a known follow-up.
- **Threshold tuning per clinic.** Different clinics have different sensitivities about different
  fields. A general default cannot satisfy every clinic. We will provide a threshold-tuning
  workflow; we will not pretend that one default is right for everyone.
- **NER tail for unstructured chart text.** Chart notes contain free-text references to people,
  places, and conditions that a named-entity recognizer will sometimes miss. The Foundry's
  guarantees on structured fields are strong; on unstructured chart text, they are weaker, and we
  state this plainly. F4 invests in this; F1 does not.
- **Fallback path to MySQL groundtruth at any phase.** At every phase, the clinic can fall back
  to the MySQL groundtruth in the cold path. This is by design, not by accident. We do not want
  a clinic locked in to a system whose maturity is still being proven.

---

## 6. What we are NOT promising

This list is here to prevent disappointment. If a clinic is shopping for any of these in F1, we
are the wrong vendor today.

- **Full EHR replacement.** Foundry is a substrate, not an EHR. We integrate with EHRs; we do not
  replace one.
- **FHIR conformance in F1.** FHIR conformance is a defensible goal for a later phase. F1 is
  oracle-parity-against-MySQL, which is a different target. Do not conflate.
- **Real-time streaming.** Foundry is request-response in F1 through F4. Streaming is an open
  question for after F5; treat it as research, not roadmap.

---

## 7. Next deliverable

**F1 has shipped.** The MySQL ↔ SPO oracle-parity machinery is now in place across all three
repositories. What is in production today:

- **C# read path (MedCareV2).** A sample-gated `LanceProbe` calls both the existing
  direct-MySQL path and the Foundry path on the same query, hands both result sets to
  `ParityWitness` for canonicalisation, and emits the witness into `DriftSink`'s in-process
  queue. `DriftSink` flushes asynchronously to an HTTP ingest endpoint. Landed in MedCareV2
  PRs **#1**, **#2**, **#3**.
- **medcare-rs ingest.** `POST /api/__parity/csharp` accepts the JSON document `DriftSink`
  emits and writes it into an in-memory ring buffer (capacity 1024, drops oldest). Landed in
  medcare-rs PR **#71**. The ring buffer is deliberately bounded; nothing here is durable
  yet, and that is a known limitation, not a missing feature.
- **lance-graph contract DTO.** The wire format both sides serialise to is defined by
  `lance-graph-callcenter::transcode::parallelbetrieb::DriftEvent`, with the discriminant
  `DriftKind` (`Match` / `ValueDrift` / `ShapeDrift` / `MissingMysql` / `MissingLance`) and
  the `Reconciler` trait that any future runner implements. Landed in lance-graph PR **#309**.

What F1 does **not** yet give us:

- No latency numbers are quoted here. The harness records per-query timing, but the §4 rule
  still applies: until a benchmark report on the fixed corpus is published, treat performance
  as "designed to match", not "matches".
- The drift queue is in-memory on both sides. Persistence and replay are out of scope for F1.
- The corpus is hand-curated. Generalisation to a clinic's full query mix is not claimed.

**What is actually next: F2 RBAC + audit wiring on the medcare-rs read path.** The
lance-graph side already exposes `lance_graph_callcenter::rls::RlsRewriter` and the
`AuditSink` trait (see `crates/lance-graph-callcenter/src/postgrest.rs` and
`crates/lance-graph-callcenter/src/lance_membrane.rs`). The C# direct-MySQL path is the
groundtruth oracle, so its access-control story is unchanged. The blocker is a medcare-rs
PR that wires `RlsRewriter` into the SPO read path so that every read either matches an
explicit role grant or is rejected, and routes the resulting decision through an
`AuditSink` implementation. That PR has not been opened yet. Until it is, F2 is a posture,
not a delivery, and we will not quote F2 guarantees in client material.
---

End of draft.
