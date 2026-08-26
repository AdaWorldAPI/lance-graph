## 2026-08-21 — E-R2IL-VARNODEFACET-IS-A-G3-CARVING-AND-0xC4-WOULD-BIRTH-A-CLASS-INTO-IT-1

**Status:** FINDING (measured at file:line this session, after rebasing
`claude/ruff-r2il-lancegraph-3tdt8d` onto `main` @ `8a93423`).
**Confidence:** High for the layout arithmetic and the §3a rule text (both
read from source); the CONSEQUENCE is a gate question for PR3, deliberately
not pre-decided here.

**The finding.** `ruff_r2il::facet::VarnodeFacet` (`facet.rs:36-48`) carves its
16 bytes as `classid: u32` + `offset_lo: u32` + `offset_hi: u32` + `size: u32`.
The 12-byte payload is therefore **3 × 32-bit contiguous = exactly `le-contract.md`
§3a's G3 "wide-quad"** (`:82`, `96 ✓`) — the axis-LESS grace carving, not the
axis-grouped `3×(8:8:8:8)` L6/`CascadeShape::G3D4` shape. It carries no rail:
there is no `X:Y` byte pair anywhere in it.

Today this is invisible and legal: `PROVISIONAL_R2IL_VARNODE = 0x0000`
(`facet.rs:21`) puts every varnode in the **default class** — the CANON
zero-fallback ladder's "no prefix routing (dormant)" state, which is not a
minted class at all.

**Why the mint changes its status.** `lance_graph_contract::ogar_codebook`
already names the destination in source (`ogar_codebook.rs:112-117`): *"`0xC4XX`
— Binary lifting … the R2IL container concepts mint here in the ruff PR3 arc,
**replacing `PROVISIONAL_R2IL_VARNODE = 0x0000`**"*. The moment that mint lands,
the facet stops being the dormant default and becomes a **real, addressed class
whose payload is G3** — and §3a is explicit (`le-contract.md:104`): *"New classes
MUST NOT be born into G1–G3; the waiting room is not a destination."*

So PR3 as currently specified would, without a deliberate decision, do the one
thing §3a names. Three legible resolutions, none prescribed here:
(a) re-carve the payload onto a rail shape before minting (e.g. the L4
`6×(u8:u8)` reading, which would make space/offset/size rail-addressable);
(b) mint and take a **named ratified exception**, the way
`ISS-IDENTITY-QUAD-WIDE-CARVING-HOME` did for `identity_quad` in G2 (operator,
2026-08-17) — that precedent exists precisely for a permanent, honest resident;
(c) keep `0x0000` and defer the mint until the carving question is settled.
The honest-cost accounting §3a already demands applies either way: a 64-bit
varnode offset genuinely needs 64 bits, so (a) is a real design question about
whether offset belongs in the facet at all, not a mechanical re-slice.

**Two things this session checked that are NOT defects, recorded so they are
not re-investigated:**

1. **The ruff SPO intake arms do not use the R2IL/r2sleigh format — correctly.**
   `ruff_{cpp,csharp,python,ruby,sqlalchemy}_spo` depend on `ruff_spo_triplet`
   only; none depends on `r2il`/`r2ssa`/`ruff_r2il`. That is the plan's two-arm
   split (`ruff/.claude/plans/r2il-behavioral-ir-v1.md:320`): structural
   (AST → `ModelGraph` → `expand()` → `Vec<Triple>`) beside behavioral
   (machine code → R2IL/SSA), converging only at OGAR/ClassView. Feeding R2IL
   into an AST harvest would be the "backdoor opcode vocabulary" the plan
   forbids at `:335`. Consistent with this board's own line 9347.
2. **`ruff_spo_address` having zero in-ruff dependents is not dead code.** It is
   consumed CROSS-REPO by OGAR `ogar-from-ruff` (`Cargo.toml:35`, a git dep on
   ruff `main`; `mint.rs:35` imports `{Facet, Mint, mint_with_classid}`). The
   in-ruff dependent count is the wrong measurement for a producer crate.

**Correction of this session's own prior statement (storno).** Earlier this
session I reported PR3's home as an OPEN question — "slot in `ogar_codebook`
versus a new `ogar-r2il` crate, your call" — and leaned slot. That was wrong on
the facts available: the decision was already made and landed. `BinaryLifting`
(`0xC4`) is in the mirrored `ConceptDomain` enum on both sides, is pinned by
`reserved_empty_domains_agree_across_the_mirror`, and
`E-OGAR-CODEBOOK-MIRROR-DOMAIN-DRIFT-SYNCED-1` (2026-08-18) already states *"the
R2IL container-concept mints under 0xC4 arrive with the ruff arc's PR3 and will
rebase trivially on this."* No `ogar-r2il` crate was ever proposed anywhere
(zero hits across `ruff/` and `OGAR/`). **The lesson is the process one:** the
finding was reachable only by reading the board AFTER the rebase — I formed a
conclusion from the ruff tree alone and presented a settled decision as open.
Rebase, then read EPIPHANIES/plans, THEN conclude.

**⚠ NAMING — a sibling session is calling `ruff_r2il` "V4"; the INSIGHT is
right and the LABEL is wrong, and they must not be collapsed into each other.**
Reported mid-session: *"another session called `ruff_r2il` V4, just because the
IR format is way denser from the perspective of a compiler substrate."* The
density observation is **correct and is the same fact as this entry's finding**
— R2IL's address space genuinely is denser than a rail shape can express (a
varnode needs a 64-bit offset plus a size; `6×(u8:u8)` cannot hold that), which
is exactly WHY `VarnodeFacet` ended up carved as G3. Two sessions reached one
fact from opposite sides. Preserve that.

The LABEL collides three ways and should not be adopted:

1. **It reverses a ratified verdict without falsifying it.** The r2il plan's own
   `:112` reads *"Verdict: V3 is sufficient. **No V4.** The hypothesis stands
   un-falsified"*, restated as stop condition §22.5 (`:397`, "no V4; variable
   arity already routed upstream"). Nothing measured since has falsified it —
   PR2's oracle went the other way, reconstructing 35,946 op sites with zero
   mismatches *through V3 routes*. A version bump needs a falsifier, not a
   density impression.
2. **"V4" already denotes something else here.** `E-DIA-V4-FIELD-SEARCH-LOOP-1`
   (2026-07-23) is the dialectic-engine's **V4 foveated field-search slice** — a
   plan V-slice number in `dialectic-engine-v1.md`, an unrelated subsystem. A
   second meaning on the same token divides the search surface for every future
   session, which is the duplicate-id failure this board already fights.
3. **It implies V3-superseded, which no ruling supports.** V3 is a LAYOUT canon
   (4+12 content-blind facet, 512-byte row, classid canon-high). Density is not
   a property of the canon; it is a property of ONE CLASS's carving choice
   *inside* it. That is precisely what §3a's G-carvings and the `identity_quad`
   named exception exist to express.

**The honest reframing:** R2IL is not a new substrate version — it is a **dense
class whose carving question is open** (the three options above). If the density
is genuinely permanent and irreducible, the sanctioned vocabulary for saying so
is a **named ratified G3 exception** in the `ISS-IDENTITY-QUAD-WIDE-CARVING-HOME`
mould, not a version number. Route the disagreement to the operator as a carving
ruling; do not let either session settle it by naming.

**Cross-refs:** `le-contract.md` §3a (G1–G3 + the `identity_quad` named
exception), `E-IDENTITY-QUAD-4X24-RATIFIED-PERMANENT-1` (the exception
precedent), `E-OGAR-CODEBOOK-MIRROR-DOMAIN-DRIFT-SYNCED-1` (the 0xC4 sync),
`E-V3-FACET-4-PLUS-12`, `ogar_codebook.rs:112-117`, ruff
`crates/ruff_r2il/src/facet.rs:21,36-48`, ruff plan `r2il-behavioral-ir-v1.md`
PR3 + O5.

