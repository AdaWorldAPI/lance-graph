# 2026-10-23 — L1 (CC2 Law Frontier): a certified reassociation reused on a fresh frame (D-DRMC-5 + X8) — pre-registration

**Status:** PRE-REGISTERED. This entry is committed before any code. Lab only:
`crates/dr-math-binding-probe`, no production crate touched.
**Authorization:** operator, 2026-10-23 ("CC2: L1 starten … nur im Lab, keine Produktionsänderungen"), coordinated
in EXO-50 (`20261023-cc2-dual-frontier-05`, `GPT:gpt-ada-01/20261023-cc1-cc2-ownership-sync-13`).

## Hypothesis

A `RecipeCertificate` that `certify_reassociation` issues on frame F1 can be executed against a chain bound on a
second, independently issued frame F2 (other `FrameStore`, other data, same law shape). The run calls egglog zero
times, and its result equals an independent pair-enumeration oracle. Every incompatible request is refused rather
than silently recomputed.

## What already exists (not rebuilt)

D-DRMC-5 (#1707) supplies:
- the typed `Chain` (frame, focus, scope and domain guards);
- `admit_composition_assoc`;
- the egglog proof;
- the `lower`/`raise` round trip;
- `certify_reassociation`, which binds the certificate to `lowering_digest(lower(to))`.

## Additions (minimal, in `dr-math-binding-probe`)

1. **The certificate records which query it answers.** The certificate gains `source_digest` =
   `lowering_digest(lower(from))`. Today it proves `from = to` but does not record `from`. A reuse could then run
   `to` for a request that the certificate never covered.
2. **`execute_certified(cert, chain, requested, runtime_abi)`.** It refuses in three cases:
   - `WrongTarget` when `cert.compiled().target_abi != runtime_abi`. Today `target_abi` is stored but never
     checked.
   - `NotCertifiedQuery` when `lowering_digest(lower(requested)) != source_digest`.
   - `NotThisChain` when `requested` does not cover the chain's positions.

   Otherwise it executes the certificate's step list (the compiled artifact) on a stack with `compose`. It never
   calls the prover.
3. **A thread-local egglog invocation counter.** It is incremented inside `prove_equation`, the only egglog call
   site, and exposed read-only. The hit path asserts the delta is 0.

## Tests (new file `tests/x8_fresh_frame.rs`)

**Positive.**
- Five lanes propose five association orders of one 5-chain (left-deep, right-deep, three balanced).
- Each proposed order is certified on F1 against the cheapest-by-`Work` target.
- On F2 (new store, new random data at three densities), each certificate executes:
  - egglog delta = 0;
  - result == `oracle_eval` of the requested order;
  - result == `Chain::eval` of the requested order.

**Adversarial.** A lane proposes commutation `r0∘r1 → r1∘r0`. Certification is refused, and no certificate
exists to reuse.

**Negative (must refuse, never fall back).**
- `WrongTarget`: runtime ABI ≠ certificate ABI.
- `NotCertifiedQuery`: a different order with the same leaves.
- `NotThisChain`: a 4-chain on F2 against a 5-chain certificate.
- Chain construction refuses on F2 for Bag, three-valued and domain-mismatch operands. That guard lives in
  `Chain::new` (D-DRMC-5) and is cited, not re-claimed.

**Silent twin.** Same shape, different data, same requested order: it MUST execute and MUST NOT refuse. A refusal
or a baseline recompute counts as failure, never as reuse.

**Disable runs** (commit first; each patch's anchor asserted before applying):
- D1: drop the ABI check. The `WrongTarget` test goes red.
- D2: drop the source-digest check. The `NotCertifiedQuery` test goes red.
- D3: route execution through `prove_reassociation`. The zero-egglog assertion goes red.

**Measured, reported, not gates.**
- Cold `certify_reassociation`, in µs.
- Hot `execute_certified` on a 5-chain, in ns per call.
- Certificate bytes: proof text length plus fixed references.
- Units: ns hot, µs cold, never ms.

## Kill conditions

L1 fails if any of the following happens:
- a requested order the certificate does not cover executes;
- an ABI mismatch executes;
- the hit path calls egglog;
- any F2 result differs from the oracle;
- the silent twin refuses.

## Out of scope

- Choosing the cheapest order (specialization selection).
- R2IL/Quack lowering.
- An independent proof checker (S2).
- The Thought-Git side. CC1 owns the passive reference; the shared type is to be agreed as option B before any
  contract PR.

Note from the operator: a compiled-program digest is not the proof's identity. The certificate keeps the law,
`source_digest`, `compiled_digest` and the proof as distinct fields.
