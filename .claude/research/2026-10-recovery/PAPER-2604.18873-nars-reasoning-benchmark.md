# arXiv 2604.18873 — "From Natural Language to Executable Narsese" (Gabriel & Wang, Temple)

**Grade:** READ-IN-FULL. **Status:** OPEN — candidate external probe, nothing adopted.

## What it is
NL -> FOL -> (deterministic compiler) -> Narsese -> ONA -> {True, False, Uncertain}.
Benchmark NARS-Reasoning-v0.1: 1,000 ProverQA-derived instances, depth tiers
easy 1-2 / medium 3-5 / hard 6-9 steps (400/300/300; split 800/200).
Dataset `MinaGabriel/NARS-Reasoning-v0.1` on Hugging Face, CC-BY-SA-4.0.

## Weak points (read before citing)
1. No quantitative results. The Phi-2 LoRA baseline is a label classifier,
   reported only as "encouraging"; executable generation is never measured.
2. Selection bias: an instance is kept only if the compiled program reproduced
   the gold label in ONA. Scoring that compiler on this set proves nothing.
3. The compiler strengthens some forms (`A -> (B v C)` becomes two rules) and
   answers composite conclusions by querying their first atom.
4. Labels drop confidence: f >= 0.50 -> True, f <= 0.05 -> False, else
   Uncertain. (f=0.51, c=0.01) reads as True.
5. Verdict depends on a fixed 20-cycle budget, so "Uncertain" also means
   "not derived in budget".

## Relation to this recovery
- Its three labels are a lossy projection of (f, c) + EpistemicState5. Only
  ever project at an explicit output boundary.
- "Uncertain is not a fallback" supports keeping CounterfactualVerdict::NotRun
  distinct from False (ledger O23).
- Depth tiers give labelled multi-step chains that directly exercise the
  `chain_replay::replay_step` evidence double-counting defect.

## Proposed probe (not built)
Fetch the 200 test instances at run time (never committed: CC-BY-SA and the
fetch-don't-commit fixture rule). Run premises through the existing NARS
revision/deduction path. Falsifiers:
- confidence must be non-increasing with depth tier (rising c on hard
  instances = double-counted evidence);
- gold-Uncertain instances must stay uncertain, not drift to True/False.
Gold labels came through an unsound compiler, so agreement measures
consistency with their labels, not correctness.

Do NOT port their compiler (a second statement encoder beside deepnsm-v2) or
the Phi-2 adapter.
