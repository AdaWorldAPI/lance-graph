# Tech Spec — The Lowering Substrate (for the product lead)

> **Audience:** product & strategy. No Rust required. This explains *what
> the thing is*, *why it is one thing and not eight*, *what is actually
> proven today*, and *what each product would cost to reach*. Claims are
> tagged **[SHIPPED]** (running code), **[PROVEN]** (a test demonstrates
> it), or **[CONCEPT]** (design, not yet built).

## The one idea

Almost every "reverse-engineering", "sandbox", "emulator", "code-analysis",
or "bring-your-own-code" product is secretly the **same machine** wearing a
different face:

1. **Lower** whatever code you have — an old game, a Windows binary, a legacy
   Java service, a business rule — into **one common instruction language**.
2. **Address** each piece by a compact number (not a heavy object).
3. **Run** it, extremely cheaply, with the data staying exactly where it
   already is (no copying, no per-operation memory churn).

Everything a *customer* sees — the security dashboard, the game editor, the
compliance report — is a **glove** on top of that one machine. We build the
machine once; each product is a glove.

## Why this is credible and not a slide

We have already built and *proven* the load-bearing parts:

- **The executor runs real code, correctly, with zero copying.** A genuine
  1980s CPU program, translated by an industry-standard tool (Ghidra), runs
  through our engine and matches an independently-written reference on every
  register and flag. **[PROVEN]** — and in doing so it *found a 20-year-old
  bug in Ghidra's own definition*, which is the kind of result you only get
  when the machinery is actually correct.
- **Addressing-by-number beats carrying-objects, measured.** On the Java
  side, one billion operations produced a *fixed* 960 bytes of memory
  overhead — essentially zero per operation — versus tens of gigabytes for
  the conventional object-carrying approach. **[PROVEN]**
- **Navigation and execution are the same operation.** We proved that
  "walking to an address" and "running a program" are literally the same
  algebra. That is what lets one engine serve a file browser, a game editor,
  and a malware scanner without three separate codebases. **[PROVEN]**

We are honest about what is *not* proven: a headline "operations per second"
number is **not** banked yet (our own experiment refused to pin it because
the timing was too noisy), and the "learned decoder" that would remove the
last dependency on Ghidra is **[CONCEPT]**, not measured.

## The product surface — eight gloves, one machine

Each row is one product. "Lift" = a lot of the machine is shared; "Glove" =
the product-specific work on top.

| # | Product | What the customer gets | Maturity | Glove work |
|---|---|---|---|---|
| 1 | **Bring-your-own-code platform** (Foundry-style) | drop in any binary/service, get an addressable, queryable model of it | machine **[PROVEN]**; product **[CONCEPT]** | ingest UI + ontology |
| 2 | **Reverse-engineering / security workbench** | a Ghidra-class RE tool with no JVM, CLI-native, scriptable | downstream analysis arms **[SHIPPED]** in Rust | analyst UI |
| 3 | **Zero-trust code sandbox** | every binary whitelisted, scanned for malware **before it runs**, autonomous alerting | scan-before-run invariant **[SHIPPED]** as a rule; engine **[PROVEN]** | policy masks + alert routing |
| 4 | **Retro-game studio** ("Mario-Maker for the C64") | load a classic game, edit levels/sprites visually, re-run | tile-render skin **[SHIPPED]** (a2ui-paint); game-lift **[CONCEPT]** | editor UI |
| 5 | **No-friction legacy-Java runtime** | run stone-age Java (EDI, graph libraries) at near-native cost, no rewrite | zero-alloc lane execution **[PROVEN]** | packaging |
| 6 | **Scientific / deterministic emulation** | byte-exact replay of legacy numerical code; run thousands of parameter variants in parallel | executor **[PROVEN]**; the parallel sweep **[CONCEPT]** | domain harness |
| 7 | **Transcode-with-a-GUI** | point at legacy code, get a structured, editable representation you can retarget | the structured-output pattern **[SHIPPED]** elsewhere (OCR doc model) | mapping UI |
| 8 | **Coding-agent code graphs** | an AI agent lowers a codebase into an addressable graph and reasons over it as *runnable* structure | addressing = execution **[PROVEN]**; the agent loop **[CONCEPT]** | agent tooling |

## Why "one machine" is the moat

Competitors ship these as *separate* products because they built separate
engines. Our bet: because addressing and execution are the same algebra
(proven), and because the data never moves (proven zero-copy), **the marginal
cost of the Nth glove is a UI and a policy file, not a new engine.** A malware
scanner and a game editor share 90% of their code.

The security framing is the sharpest near-term wedge: "every piece of code
runs only if whitelisted, and is pattern-scanned for malware *before* it can
do anything" is a zero-trust posture most sandboxes cannot honestly promise,
because they scan *behavior after the fact*. Our engine refuses-by-default and
scans the lowered code *before* execution — that is an architectural
guarantee, not a heuristic.

## What it would take (honest roadmap)

- **Now → 1 month:** the throughput measurement that is currently missing
  (the "N programs at once" benchmark); an exhaustive conformance test of our
  executor against Ghidra's own reference *while we still depend on it*.
- **1 → 3 months:** the first end-to-end glove (recommend #3 zero-trust or #2
  RE workbench — both reuse the most that is already built).
- **3 → 6 months:** the "learned decoder" that removes the last Ghidra
  dependency at runtime, making the whole stack pure-Rust and portable
  (WASM/edge/air-gapped). This is the biggest technical bet and the biggest
  differentiator; it is **[CONCEPT]** today.

## The one risk to name

The last piece of someone else's C++ (Ghidra's instruction decoder) is still
in the runtime path. Removing it is a research bet, not a certainty. Until it
lands, "pure Rust, zero external engine" is a *destination*, not a current
claim — every product above works *with* the Ghidra decoder today; the
learned-decoder work is what makes it disappear.

## Proof pointers (for a technical reviewer)

`live_regfile.rs` (executor vs independent oracle, 18/18), `R7` (zero-alloc,
960 B / 10⁹ ops), `R12` (ordinals flatten, objects don't), `zipper_hop_parity`
(navigation = execution). Full mechanics in the sibling `TECH-SPEC-AGENTS.md`.
