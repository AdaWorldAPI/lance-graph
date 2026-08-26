# Proposal — One Substrate, Many Products: The Lowering Engine

**Status:** proposal for direction & investment · **Date:** 2026-08-26
**Prepared by:** the r2il / r2conc / ogar-loco arc

---

## Executive summary

We propose consolidating a family of apparently-different products —
reverse-engineering tools, code sandboxes, emulators, legacy-runtime
accelerators, even AI coding assistants — onto **one engine**. The engine
does three things: it **lowers** any code to a single instruction language,
**addresses** each piece by a compact number, and **executes** it with the
data never leaving where it already sits. Each product is then a thin
*policy or presentation layer* — a "glove" — over that shared engine.

The load-bearing parts are already built and independently verified. What we
are proposing is to (a) finish two small measurements, (b) ship the first
customer-facing glove, and (c) fund the one research bet — a learned decoder —
that makes the whole stack self-contained and portable.

## The problem with the status quo

The reverse-engineering / binary-analysis / sandboxing market is fragmented
because every vendor rebuilt the *same* core — decode a binary, model it,
run or analyze it — from scratch, and then bolted a single product on top.
The core is expensive; the products are shallow. Nobody amortizes the core
across products because their cores are entangled with their UIs.

## Our thesis

**Addressing a thing and running a thing are the same operation.** We proved
this: walking to an address in our model and executing a program are literally
one algebra. Combined with a second proven property — the data never has to be
copied — this means the *engine* is genuinely reusable, and the *marginal cost
of a new product is a UI and a policy file*, not a new engine.

## What is already proven (not promised)

- **Correct execution, zero copy.** A real 1980s CPU program, translated by
  Ghidra (an industry-standard tool), runs through our engine and matches an
  *independently written* reference emulator on every register and flag — 18
  of 18 checks. It even surfaced a two-decade-old bug in Ghidra's own
  definition, which only a genuinely correct engine could expose.
- **Addressing beats carrying, measured.** One billion operations produced a
  *fixed* 960 bytes of overhead — effectively zero per operation — where the
  conventional object-carrying approach would churn tens of gigabytes.
- **Navigation = execution.** Proven bit-for-bit: the same masking algebra
  serves a file browser, a game editor, and a malware scanner.

We are equally clear on what is **not** yet proven: a headline throughput
number (our own experiment declined to bank one; the timing was too noisy),
and the "learned decoder" that would remove the last third-party dependency.

## The eight products, one engine

1. **Bring-your-own-code platform** — drop in any binary or service, get an
   addressable, queryable model. (Palantir-Foundry-style, for code.)
2. **Reverse-engineering / security workbench** — a Ghidra-class tool with no
   JVM, CLI-native and scriptable. The downstream analysis stages already
   exist in Rust.
3. **Zero-trust code sandbox** — every binary runs *only if whitelisted*, and
   is pattern-scanned for malware **before it can execute**, with autonomous
   alerting. This is an architectural guarantee, not after-the-fact behavior
   monitoring — most sandboxes cannot honestly make it.
4. **Retro-game studio** — "Mario-Maker for the Commodore 64": load a classic
   game, edit its levels and sprites visually, re-run instantly.
5. **No-friction legacy-Java runtime** — run stone-age Java (EDI processors,
   graph libraries) at near-native cost with no rewrite, by addressing lanes
   instead of carrying objects.
6. **Scientific / deterministic emulation** — byte-exact replay of legacy
   numerical code, and thousands of parameter variants run in parallel over
   the same code.
7. **Transcode-with-a-GUI** — point at legacy code, get a structured,
   editable representation you can retarget to a new platform.
8. **Coding-agent code graphs** — an AI agent lowers a codebase into an
   addressable graph and reasons over it as *runnable* structure, not just
   text: it can ask "what does this actually do" by executing, not guessing.

## The security wedge (recommended first market)

Product 3 is the sharpest near-term entry. "Every piece of code runs only if
whitelisted, and is scanned for known-bad shapes *before* it can act" is a
**zero-trust** promise most sandboxes cannot make, because they observe
behavior *after* code runs. Our engine refuses by default and scans the
*lowered* code first. For regulated, air-gapped, or supply-chain-anxious
buyers, that architectural guarantee is the product.

## What we are asking for

1. **Direction:** agreement to treat this as *one engine, many gloves* rather
   than funding point products separately.
2. **Two short measurements** (weeks): the missing throughput benchmark, and
   an exhaustive conformance test of our executor against Ghidra's own
   reference — done now, while we still depend on it, so the result outlives
   the dependency.
3. **First glove** (1–3 months): recommend the zero-trust sandbox or the RE
   workbench — whichever reuses the most already-built code.
4. **The research bet** (3–6 months): the learned decoder that removes the
   last piece of third-party C++ from the runtime, making the stack pure-Rust,
   portable (WASM / edge / air-gapped), and fully self-contained.

## The one honest risk

The final third-party component — Ghidra's instruction decoder — is still in
the runtime path. Removing it is a research bet, not a certainty. Every
product above works *today* with that decoder in place; the learned-decoder
work is what makes it disappear. We recommend funding the products on the
proven engine now, and the decoder research in parallel — not gating the
products on the bet.

---

*Technical detail: `TECH-SPEC-AGENTS.md` (mechanics + falsifier ledger) and
`TECH-SPEC-PRODUCT.md` (per-product maturity) in this folder.*
