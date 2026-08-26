## 2026-08-18 — E-OGAR-CODEBOOK-MIRROR-DOMAIN-DRIFT-SYNCED-1

**Status:** FINDING + fix (cross-session: found by the lance-graph-java
session, verified independently by the ruff/R2IL session at `db488f5`,
ownership handed to lance-graph-java's session; this is that sync).

**The drift.** `lance_graph_contract::ogar_codebook` documents itself as a
wire-compatible mirror of OGAR `ogar_vocab::ConceptDomain` under an
explicit both-sides-update-together drift guard — yet its enum ended at
`Geo` (0x0F) while OGAR carries `Ontology` (0x03, POPULATED since the
DisMech 0x0333 mints, OGAR #275), `Blocks` (0x17, since 2026-08-04), and
the C-band `JavaRuntime`/`Analytics`/`BinaryLifting` (0xC0/0xC1/0xC4,
OGAR #276+#277 — the altitude ruling: the domain byte is stratified by
layer; 0xC0 is **Panama FFM alone**, Valhalla being a property of the C0
vocabulary, not an addressable concept).

**Why the guard never fired — the real lesson.** `lance-graph-ogar`'s
`domains_agree` + `assert_codebook_parity` only walk ids that carry
CONCEPT ROWS. A reserved-EMPTY domain added to one enum but not the other
is invisible to a codebook-content walk — the exact class of drift a
DOMAIN mirror exists to catch. Proven live: the first disable-run
(dropping the new `BinaryLifting` pair from `domains_agree`) stayed GREEN
— the pairing was vacuously guarded until this PR added
`reserved_empty_domains_agree_across_the_mirror`, which pins one id per
new/reserved domain, the populated 0x0333, the deliberate 0xC2–0xC3 gap,
the band edges, and the 0x0C/0xC0 digit-swap two-sided. Both disable-runs
(bridge pair dropped; contract arm dropped) now go red on exactly that
test; the contract's own `domain_routes_on_high_byte` independently
catches the arm removal.

**Scope, stated:** DOMAIN-level sync only. Codebook CONTENT parity was
re-run and is green (`assert_codebook_parity`); the R2IL container-concept
mints under 0xC4 arrive with the ruff arc's PR3 and will rebase trivially
on this. Pre-existing `lance-graph-ogar` clippy warnings (11, measured
identical with this diff stashed) left untouched.

