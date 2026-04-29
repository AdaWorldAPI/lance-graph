DRAFT — pending review (2026-04-28)

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

*Anchor: as of 2026-04-28. This table will age. We commit to updating it at the end of every
phase, F1 through F5, and re-dating the anchor.*

| Property            | MedCare Foundry (target)                       | Palantir Gotham                                 | OpenEMR                                       | Raw Postgres + JSONB                                  |
|---------------------|------------------------------------------------|-------------------------------------------------|-----------------------------------------------|--------------------------------------------------------|
| Data sovereignty    | Per-clinic substrate; bring-your-own-storage   | Vendor-managed; sovereignty contractual         | Self-hosted; sovereignty by deployment        | Self-hosted; sovereignty by deployment                  |
| Audit               | Optimizer-enforced; append-only Lance log      | Built-in, vendor-defined                        | Application-level audit hooks                 | DIY (triggers or app code); easy to bypass              |
| Latency             | Designed to match C# direct-MySQL; F1 numbers  | Vendor-published; not directly comparable       | Bound by application stack on top of MySQL    | Application-stack-dependent                             |
| Cost                | Self-hosted; storage cost dominates            | Enterprise license; high                        | License-free; ops cost dominates              | License-free; ops cost dominates                        |

Caveats on this table:

- "Designed to match" in the latency row is a posture, not a benchmark result. F1 publishes the
  first numbers. Until then, do not quote latency claims from this table in a customer setting.
- Palantir Gotham is a different product category. The comparison row is included because clinics
  ask about it; it is not a like-for-like.
- OpenEMR and raw Postgres+JSONB are the two realistic alternatives a clinic would actually
  consider. Treat those rows as the load-bearing comparison.

---

## 3. Five-phase transition F1 through F5

Each phase is described in one paragraph: what it is, and what benefit it delivers. We will not
ship a phase whose benefit cannot be stated in one sentence.

### F1 — Oracle parity

We stand up a MedCare Foundry instance whose read path produces results identical to the existing
C# direct-MySQL groundtruth on a fixed query corpus, on a hand-curated subset of clinic data.
Cold path only — no caching tricks, no warm-up. The benefit is a fully automated, query-by-query
parity check: any future divergence between Foundry and the MySQL groundtruth is detected by the
harness, not by an unhappy clinician.

### F2 — RBAC and audit

We turn on the row-level-security rewriter (gated upstream by lance-graph PR-1) and the audit log
(gated upstream by lance-graph PR-2). Every read of a patient record now goes through the
rewriter, which means every read either matches an explicit role grant or is rejected. Audit
entries are append-only and tenant-scoped. The benefit is that the clinic can answer "who read
this record" with evidence, end-to-end, without trusting application code.

### F3 — PostgREST shape

We expose the Foundry over an HTTP surface that follows PostgREST's URL and header grammar
(gated upstream by lance-graph PR-4). Existing PostgREST-aware tooling can be pointed at the
MedCare Foundry with minimal code change. The benefit is that integration with third-party
clinical tools and dashboards is reduced to a configuration change, not an SDK port.

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

The Foundry is **designed to match** the C# direct-MySQL baseline on identical workload. The
benchmark harness lands as part of F1.

> Footnote: this claim is "designed to match" until F1 numbers are published. We will not market
> "faster than" on unmeasured ground. Any latency or throughput number that appears in client
> material must cite a specific F-phase benchmark report. If you cannot cite the report, do not
> quote the number.

What we will measure in F1, in this order:

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

The next concrete deliverable is the **F1 oracle-parity demo** on a hand-curated 1,000-record
subset of clinic data. The demo will show:

- The fixed query corpus running against both MySQL groundtruth and the MedCare Foundry.
- Per-query result equality, automatically checked.
- Per-query latency comparison, recorded.
- A short written summary of where Foundry matches, where it diverges, and (if any) where it is
  faster — with the caveat that F1 numbers on a 1k-record subset do not generalize.

We will publish the F1 numbers exactly as measured. If they are unfavourable, we publish them
unfavourable. The whole point of an oracle-parity phase is to find out where we are, not to
confirm where we wish we were.

---

End of draft.
