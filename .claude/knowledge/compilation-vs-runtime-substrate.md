# Compilation vs runtime substrate — contracts compile types; the engine thinks alone

> READ BY: layer-boundary-warden, codegen-flow-cartographer, integration-lead,
> baton-handoff-auditor, core-first-architect, any session filing a cross-repo
> issue or writing a seam/lowering design that mentions a consumer repo
> (a2ui-rs / medcare-rs / smb-office-rs / woa-rs / q2) together with any of:
> SoA, temporal.rs, NARS, RBAC, mailbox, write-on-behalf, lance-graph-planner.
>
> Born 2026-07-14 from a real failure: three consecutive drafts of one issue
> (OGAR #208, closed as hallucinated) gated a *compile-time lowering* behind
> *runtime* ownership ceremony. Operator ruling is the canon here; OGAR #208's
> repurposed body is the short form; #209 is the corrected work item.

## The two phases, kept apart

**Compile time.** ruff harvests source → OGAR mints IR (`Class`/`ActionDef`,
classids, facets) → codegen emits Rust **into the consumer repo**. The
consumer depends on `ogar-vocab` (re-exported via `lance-graph-ogar`) and/or
`lance-graph-contract` as **plain type dependencies** — the compile-time
handshake. cargo builds the repo: on a laptop, in CI, on Railway pulling from
GitHub. A binary comes out. **That is the entire compilation story.**

**Run time.** The SoA, the `temporal.rs` standing wave / version-range reads,
NARS, RBAC, mailbox write-on-behalf, lance-graph-planner strategies — the
running engine's *internal* mechanics when the cognitive substrate executes.
They are real, and they are nobody's checkpoint: they never gate a consumer's
build, and they never gate a consumer *constructing a typed value* it
compiled against.

## The door-knocking compiler (the canonical absurdity — operator's image)

> "No compiler goes to lance-graph knocking on the door of every SoA
> little-endian contract politely asking: *please, may I first download the
> code into the SoA, so my second pass can compile the same binary
> interlacing temporal.rs and NARS, if I'm allowed to compile on behalf of
> the SoA using RBAC and lance-graph-planner?*"

If a design, issue, or seam doc implies that sentence anywhere, it is wrong
**at the frame** — do not fix the details, refile the frame. Symmetrically:
"contracts compile types" (medcare-rs commitment #7) means the contract crate
is a compile-time handshake — it never serializes, never negotiates, never
"receives".

## Consequences (each one was violated at least once before this doc existed)

1. **A lowering function is a plain Rust function.** `X → ogar_vocab::Y` is
   code in a repo, unit-tested by `cargo test` with nothing running. It needs
   type-level decisions (field semantics, identity resolution), never
   permission decisions.
2. **Emitting a typed value is not a write ceremony.** Constructing an
   `ActionInvocation` is `struct` literal syntax. Where values are *recorded*
   when an engine runs is a runtime/storage concern that belongs to the
   engine — a separate design, a separate doc, never a gate on the
   compile-time seam.
3. **Build infrastructure sees only cargo.** Railway/CI pull a Git repo and
   run `cargo build`. If a plan requires the build to touch the substrate,
   the plan is wrong.
4. **Runtime doctrine stays in runtime designs.** Mailbox ownership /
   write-on-behalf / membrane crossing are V3 *engine* rules. Citing them in
   a codegen or type-seam issue is the tell that phases got mixed.

## The 30-second checklist (run it BEFORE filing/designing, not after)

- **Who executes this step?** cargo/rustc → compile time. A running engine →
  runtime. "Both" → split the design into two documents.
- **Could this be unit-tested with nothing running?** Yes → it is
  compile-time; strip every runtime word from the design.
- **Does the design make a build or a constructor ask permission?** Then it
  is the door-knocking compiler. Refile.
- **Is a contract crate described as receiving/ingesting/serializing?**
  Contracts compile types. Refile.

Cross-refs: `assembler-vs-storage-substrate.md` (the sibling distinction),
OGAR #208 (repurposed) / #209 (corrected work item), medcare-rs CLAUDE.md
commitment #7, `ogar-consumer-preflight.md` (the classid consumer spellbook).
