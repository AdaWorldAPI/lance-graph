## 2026-08-19 — E-CONTRACT-INFERENCETYPE-INVERTS-THE-COUNTERFACTUAL-1

**Status:** FINDING (rung-3 runtime audit; sharpened during plan authoring).
Plan: `.claude/plans/counterfactual-rung3-closure-v1.md`.

**The finding:** rung-3 (counterfactual) is fully specified in the substrate
and fully absent — and mis-decoded — at runtime, because the canonical
contract enum is two variants narrower than the causal-edge enum.
`contract::nars::InferenceType` has 5 variants; `causal_edge` has 8, and
the missing pair (Intervention=+6, Counterfactual=−6) is exactly Pearl
rung 3, so the semantics cannot cross the contract boundary. Every dispatch
degrades: `nars/inference.rs:72` + `orchestration_impl.rs:207` map
Counterfactual ⇒ Abduction ("follow-up PR" comments); no
abduce→intervene→predict chain runs anywhere. **Worse than a lossy
narrowing:** `contract::nars::from_mantissa(−6)` decodes to `Synthesis`
(own mantissa **+5**) — a silent DIRECTION INVERSION (a backward-chain
counterfactual read back as forward-chain synthesis). And enum widening
alone cannot surface the fix sites: the three rung-3 routing points
(`orchestration_impl.rs:144/149/176`) are `_ =>` wildcards that silently
absorb new variants. Adjacent fragmentation, same audit: three multi-hop
truth paths share no code (`belief.rs:279` close_transitive is real;
`truth_propagation.rs` is a no-op; `network.rs` forward_chain declares
NarsTables and never reads it); ≥3 reimplementations of the revision
formula across 19 `fn revise` sites; and "Pearl 2³" names three distinct
structures (CausalMask powerset / CausalAmbiguity permutations /
SEE-DO-IMAGINE). Doc-drift note owed to
`triangle-tenants-gestalt-separation-v1.md` §2a: it cites
`world/counterfactual.rs::intervene()` — no such function exists.

Refs: `layout.rs:16-32` (signed mantissa), M20 / le-contract "let go of the
cramped 64-bit register" (the fix is enum + round-trip, never new CE64 bits),
D-TRI-6 (the plane half is already wired and probe-green).

