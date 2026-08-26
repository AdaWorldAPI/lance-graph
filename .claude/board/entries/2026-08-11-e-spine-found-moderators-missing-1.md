## 2026-08-11 — E-SPINE-FOUND-MODERATORS-MISSING-1

**Status:** OPERATOR RULING (framing) + FINDING `[H]` — report §9, commit
paired with the PR opening. Reframes the whole comet-tail chain's verdict.

**Operator:** *"Wir haben ein Spine gefunden — die Stellschrauben müssen noch
mit den Variablen der bekannten Modelle moduliert werden. Uns fehlen die
Moderatoren; aber wir haben bereits das Gerüst, um das Zentrum und die
Dynamik zu modellieren. Außerdem haben wir Feuchtigkeit und Abregnen im
Aufwind an der Kollision zwischen den Gebieten nicht modelliert — das ist
eine Art Entropie bei Verdunstung und Abregnen."*

**Why this reading is defensible — stated at the strength the evidence
actually supports:** a 0.68–0.73 directional main effect whose residual were
random would be a dying claim — but this chain's residual runs MONOTONICALLY
with a measured variable (the 92–102° height ladder, 3–5× apparatus noise).
*Main effect + structured residual + identified covariate* is **consistent
with a missing moderator and requires independent validation**. It does NOT by
itself exclude model misspecification, centre/label error, selection effects,
or chance.

> **Correction (CodeRabbit on PR #926, 2026-08-11).** This paragraph
> originally read "is the signature of a missing moderator, NOT of a null. A
> null does not produce a ladder." That overstated what a monotonic residual
> can establish — it supports the hypothesis, it does not discriminate it from
> the alternatives above. Corrected in place per the append-only rule's
> allowance for regrading; the directional predictor stays SUGGESTIVE and
> unpromoted either way, which is what the PR objective already said.

**The three-part decomposition now on record (report §9):**
1. **Spine `[G]`** — center + ~12 ring means + 1 wn-1 dipole = **90.9–94.3 %**
   of in-disk variance, unshaken across 3 independent samples / 41+ storms /
   1980–2021. **~14 logical model values** (12 ring means + a 2-value
   dipole) plus a center address. *(Corrected 2026-08-12, CodeRabbit PR #926:
   this read "~14 bytes + an address", conflating the MODEL size with a
   CARRIER budget. The measured encoding is a 12-byte `6×(8:8)` L4 facet —
   see report §6.1; the byte budget belongs in the encoding section, not
   in the spine statement.)* *(Regraded in place per the same
   allowance used at line 67 above: this line printed 93–97 %, which
   `E-THE-HEADLINE-NUMBER-MEASURED-A-MODEL-NOBODY-CLAIMED-1` — the entry
   directly above — corrects to the 14-value model's real figure. Flagged by
   CodeRabbit on PR #926 as an internal inconsistency with that entry, and it
   was one.)*
2. **Dry moderators `[H]`** — measured in this chain, not yet wired:
   steering level (THE ladder; CT-F16 = score the dipole against
   steering-level motion instead of 6h surface displacement), displacement/
   label noise, friction/surface type, latitude/regime.
3. **Moist sector `[S]`** — not modeled at all, and "entropy" is technically
   the right word: rain-out in the collision-zone updraft is irreversible
   moist entropy production (θe the state variable, precipitation the sink —
   Emanuel/Pauluis frame). Tractable NOW: the WB2 store carries
   specific_humidity/temperature (θe), TCWV, total_precipitation_6hr,
   vertical_velocity — and θe/TCWV are scalar fields, so the SAME ring/wn-1
   decomposition applies verbatim. CT-M1..M3 named as falsifiers; the July
   failures (wn1_frac 0.19–0.36) are plausibly the diabatically-dominated
   storms, making diabatic dominance itself a computable intake gate.

**The brutal step (operator-directed, `[S]`):** learn the moderator matrix on
the substrate's own proven machinery — the spine as board state, moderators
as `W` in domino.rs' symbiont `C = A·W` tile-GEMM (stencil-as-GEMM already
byte-proven on real WB2 in ndarray `geostrophic_stencil.rs`), recurrence over
6h spine states via the workspace's byte-parity int8 LSTM (E-OCR-LSTM-1).
Explicit physics as spine, learned weights as moderators — the NeuralGCM-
shaped hybrid at 512 B/storm, gated by disjoint-decade train/test + the
plan-§8 audit.

