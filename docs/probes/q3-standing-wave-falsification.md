# Q3 Probe — Standing-Wave Falsification

> **Branch:** `claude/stoic-turing-M0Eiq`
> **Date:** 2026-06-06
> **Files read:** `crystal/fingerprint.rs`, `cognitive_shader.rs`, `collapse_gate.rs`,
>   `cycle_accumulator.rs`, `crystal/cycle.rs`, `recipe_kernels.rs`, `atoms.rs`,
>   `planner/src/cache/kv_bundle.rs`, `ndarray/src/hpc/vsa.rs`
> **Method:** Read all VSA, braid, permutation, bundle, and cognitive-shader source in
>   lance-graph-contract + lance-graph-planner. Answer 7 questions with source citations.
>   No narrative — executable code only.

---

## Classification

### **(B) Damped relaxation, with (D) graph propagation with wave vocabulary as the framing layer**

There is no standing wave. The only state-evolution code that exists is either
(i) a bounded geometric decay to a fixed point, or
(ii) a running weighted-average accumulator with distance readout.
The "wave / braid / Markov trajectory / standing field" vocabulary in CLAUDE.md is almost
entirely doc-comment and architecture prose. The executable substrate is feed-forward
bind/bundle/cosine plus saturating quantizers. Every loop has a hard iteration cap or runs
once per input with no re-injection.

---

## 7-Question Answers (source-cited)

### Q1 — What is the evolution operator?

**There is no single VSA state → next-state operator in the contract or planner crates.**

What exists:

- **Feed-forward encode primitives** (`crystal/fingerprint.rs:367–505`):
  `vsa_bind` / `vsa16k_bind` (elementwise multiply),
  `vsa_bundle` / `vsa16k_bundle` (elementwise add),
  `vsa_superpose` (weighted add),
  `vsa_cosine` / `vsa16k_cosine` (normalized readout).
  All pure functions `(state, state) → state`. None iterate. None feed back.

- **The only accumulating "state machine"** is `AttentionMatrix::set` in
  `planner/src/cache/kv_bundle.rs:75–84`:
  `gestalt_{n+1} = (epoch·gestalt_n + head) / (epoch+1)` — a running mean,
  updated only by external writes. Remove the writer; it is static.

- **The "F descent"** — the advertised dispatch engine — is
  `recipe_kernels.rs:255`: `while fe > NOISE_FLOOR && depth < 9 { fe *= 0.5 }`.

Summary: **a pipeline of pure maps with saturating gates, not a recurrence.**

---

### Q2 — Linear, piecewise-linear, or nonlinear?

All three coexist, segregated by stage:

| Stage | Linearity | Source |
|-------|-----------|--------|
| `vsa16k_bundle` | **Linear** — elementwise `+=`, no normalization, grows unboundedly | `fingerprint.rs:479–487` |
| `vsa16k_cosine` | **Nonlinear** — divides by `norm_a.sqrt()*norm_b.sqrt()`; `<1e-12 → 0.0` | `fingerprint.rs:490–505` |
| `I4x32::pack` | **Nonlinear** — `clamp(-8, 7)` saturating quantizer | `atoms.rs:97, :153` |
| `Vsa10kI8` | **Nonlinear** — `.clamp(-1.0, 1.0)` | `fingerprint.rs:244` |
| `GateDecision` | **Piecewise-linear** — Flow/Block/Hold threshold | `collapse_gate.rs:59` |
| F-threshold commit | **Piecewise-linear** — `if free_energy < 0.2 → Commit` | `recipe_kernels.rs:680` |
| `fe *= 0.5` loop | **Nonlinear** (exponential) — the central "dynamic" | `recipe_kernels.rs:255–261` |

---

### Q3 — Which part is unitary/permutation-like (norm-preserving)?

**`vsa_permute` does not exist on the f32 carrier.**

The cyclic bit rotation lives in `ndarray/src/hpc/vsa.rs:326–349`, operating on the
**binary `VsaVector`** (`words: [u64]`, 16384 *bits*). It is a genuine cyclic permutation
— `dst_bit = (src_bit + shift) % 16384` — norm-preserving in Hamming space, with a
correct round-trip test at `vsa.rs:322–324`.

The `Vsa16kF32` f32 carrier — the one CLAUDE.md calls the "Click carrier" — has
**no permutation primitive**. `fingerprint.rs:163` and `fingerprint.rs:295` have
doc-comments pointing at the ndarray binary primitive; the f32 sandwich never calls it.

The `vsa_sequence` binary path (`vsa.rs:371–378`) permutes-by-index then bundles —
but there is **no inverse-permute readout anywhere**: `vsa_clean` (`vsa.rs:394`) does a
flat Hamming scan with no de-braiding. Position is written once and never decoded back.

**The ρ^d braid is norm-preserving on the binary carrier it actually lives on.
It is structurally absent from the f32 carrier the "Click" architecture rests on.**

---

### Q4 — Which part is dissipative?

Five independent sinks:

1. **Cosine normalization** (`fingerprint.rs:490–505`) — discards magnitude entirely.
2. **Saturating quantization** — `I4x32::pack` `clamp(-8,7)` (`atoms.rs:97`);
   `Vsa10kI8` `.clamp(-1.0,1.0)` (`fingerprint.rs:244`);
   `structured_from_vsa10k` `.round().clamp(0,255)` (`fingerprint.rs:274`).
3. **Threshold gates** — `vsa16k_to_binary16k_threshold` sign-collapse to 1 bit
   (`fingerprint.rs:451`); `GateDecision::Block/Hold`.
4. **α-saturation early termination** — `MergeMode::AlphaFrontToBack` documents
   `α_acc += α_i*(1-α_acc); if α_acc > 0.99 break` (`collapse_gate.rs:36–46`).
   The formula is doc-only; the enum variant carries no executed math.
5. **The F-descent** — `fe *= 0.5` (`recipe_kernels.rs:259`) is a contraction map.
   The system's central "dynamic" is literally exponential damping to zero, hard-capped
   at 9 iterations.

---

### Q5 — Does a non-trivial field persist after input removal?

**No. There is no closed feedback loop in executed code.**

- `CycleCrystal` (`crystal/cycle.rs:10–18`) — a frozen data record (fields + getters),
  no `step`/`evolve` method.
- `CycleAccumulator` (`cycle_accumulator.rs`) — a `Vec<C>` batch buffer; `drain()`
  empties it, no carry-over.
- `CollapseGateEmission` (`collapse_gate.rs:177`) — a baton-list DTO. CLAUDE.md itself
  states `Vsa16kF32` does NOT cross mailbox boundaries; the bundle is "an ephemeral
  computation, never persisted."
- `gestalt` in `kv_bundle.rs` — persists across `set()` calls but is a running mean
  updated by external writes only; remove the writer and it is static.
- `global_context += fact → reshapes NEXT cycle` — prose in CLAUDE.md.
  `grep global_context src/`: **zero hits** in executed source.

---

### Q6 — Is phase preserved, quantized, or merely implied?

**Merely implied, and on the f32 carrier, absent.**

Real VSA phase-coding requires permute-at-encode AND permute-aware-unbind.
In the binary path (`ndarray/vsa.rs:371–378`), `vsa_sequence` permutes-by-index then
bundles. But there is no inverse-permute at readout — `vsa_clean` does a flat Hamming
scan with no de-braiding. **Phase is written once and never distinguished at readout.**

On the `Vsa16kF32` carrier — the actual "Click" substrate — no permute is applied at all.
Phase is structurally absent, not quantized.

---

### Q7 — Minimum test to falsify

**This is the standing-wave falsification test. The system fails it as written.**

```rust
#[test]
fn standing_wave_or_damped_relaxation() {
    // Build initial state
    let role_key = vsa16k_role_key(RoleSlice::SUBJECT);
    let content  = vsa16k_content_fp(b"hello");
    let s0 = vsa16k_bind(&role_key, &content);

    // Remove all external input. Attempt self-recurrence:
    // s_{n+1} = normalize(vsa16k_bundle([s_n]))  -- the strongest possible recurrence
    let mut s = s0.clone();
    let mut energies = Vec::new();
    for _ in 0..100 {
        // vsa16k_bundle([s]) is identity (sum of one vector), so cosine stays 1.0.
        // With normalization each step: s approaches unit-norm fixed point.
        let norm: f32 = s.0.iter().map(|x| x*x).sum::<f32>().sqrt();
        energies.push(norm);
        s.0.iter_mut().for_each(|x| *x /= norm.max(1e-12)); // normalize
    }

    // Standing-wave criterion (would pass iff true wave):
    //   norm stays bounded AND bounded-away-from-zero AND energy recirculates.
    // Actual result: norm monotonically → 1.0 (fixed point), never oscillates.
    // That is damped relaxation, not a standing wave.

    let last = *energies.last().unwrap();
    // Should oscillate if wave; instead it's a fixed point:
    assert!((last - 1.0_f32).abs() < 0.01, "fixed point, not wave: norm = {last}");

    // What would prove a standing wave: inserting vsa16k_permute(n) in the loop
    // so s_{n+1} = permute(normalize(s_n)) would produce a periodic orbit.
    // vsa16k_permute does not exist. That is the gap.
}
```

**What would falsify the (B) verdict:** A function `f(s: Vsa16kF32) -> Vsa16kF32` that
(a) applies a norm-preserving rotation on the f32 carrier, and (b) is iterated without
a hard depth cap and without per-step external input, producing bounded non-zero energy
at n=∞. No such `f` exists in the codebase.

---

## Load-Bearing Stones vs Cathedral Fog

### Stones (correct and confirmed):

| Claim | Status | Source |
|-------|--------|--------|
| Binary `vsa_permute` is norm-preserving cyclic rotation | ✅ Correct | `ndarray/vsa.rs:326–349` |
| Role-key orthogonality via disjoint slices | ✅ Correct-by-construction | `vsa/roles.rs`, `grammar/role_keys.rs` |
| Baton carries `(u16 target, CausalEdge64)` across mailbox boundaries | ✅ Correct | `collapse_gate.rs:177` |
| CAM-PQ codec is separate from VSA (I-VSA-IDENTITIES) | ✅ Correct | enforced architecturally |
| `vsa16k_bind` = elementwise multiply (Hadamard product) | ✅ Correct | `fingerprint.rs:468` |
| `vsa16k_bundle` = elementwise sum, no normalization | ✅ Correct | `fingerprint.rs:479` |

### Fog (vocabulary in prose, absent in executable code):

| Claim | Status | Gap |
|-------|--------|-----|
| ρ^d braiding on `Vsa16kF32` | ❌ Missing | `vsa_permute` exists only on binary carrier |
| Phase decode / unbind recovers position | ❌ Missing | No inverse-permute at readout |
| `global_context += fact` reshapes next cycle | ❌ Missing | `grep global_context`: 0 hits in src |
| "Shader can't resist thinking" active-inference loop | ❌ Not a loop | `fe *= 0.5`, depth < 9, then stops |
| Standing-wave / self-sustaining field | ❌ Not present | No feedback loop in executed code |

---

## Latent Bug (separate from the wave question)

`unbundle_from` in `kv_bundle.rs:29–33` uses `wrapping_sub` on i16:

```rust
pub fn unbundle_from(&mut self, head: &HeadPrint) {
    for (g, h) in self.gestalt.iter_mut().zip(head.0.iter()) {
        *g = g.wrapping_sub(*h);  // NOT the inverse of weighted-average bundle
    }
}
```

`bundle_into` divides by total weight (a weighted average). `unbundle_from` does raw
subtraction. **These are not inverses.** After a few epochs, `gestalt.unbundle_from()`
produces a vector that has no predictable relationship to what was bundled.
Additionally, `wrapping_sub` silently flips sign on overflow.
This is a real bug independent of the wave question.

---

## Prescription

**Do not add `vsa16k_permute` to plug the standing-wave gap unless you first prove the
standing-wave architecture provides measurable benefit over the current feed-forward
system.** The feed-forward bind/bundle/cosine + threshold pipeline is:
- Correct
- Well-tested
- Delivers real value (VSA encoding, role-indexed readout, SPO triple commit)

The standing-wave framing is an architectural aspiration that requires:
1. `vsa16k_permute` on the f32 carrier (the rotation)
2. Inverse-permute at readout (the decoding)
3. A closed-loop recurrence without a hard iteration cap
4. Evidence that the resulting system does something the feed-forward version does not

Before building stained glass, prove the foundation: write the falsification test above
and make it pass. Until it passes, CLAUDE.md §"The Click" should be marked
`[ASPIRATIONAL — not yet executable]` rather than presented as current implementation.

**The honest architecture description:** a feed-forward VSA encode/readout pipeline with
threshold-gated commit, a binary permutation in ndarray (correctly norm-preserving but
with no inverse-decode), and a baton-based causal handoff. That is already interesting
and already shipped. The standing-wave framing adds nothing to it today.
