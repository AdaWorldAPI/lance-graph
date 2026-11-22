# 2026-11-22 — Synergy and fruit revision after #1806–#2157 (OGAR #363–#415, ndarray #378–#437, ruff #127–#150, report-rs #96–#182)

**Status:** MAP. Supersedes the 2026-10-26 synergy revision (`2026-10-26-d-tflp-v2-synergy-revision.md`) as the latest detailed mapping. Ledger changes landed with it are listed in the commit; everything else here is a map, not a decision.

**Method.** Grindwork by Haiku cards: three quote-harvesters checked the 40 FRUITS rows against the ledgers, the PR lists and `entries/`; two substrate-mappers read the 190 entries added since 2026-10-26 (188 exist; 2 were later renamed); one mapper listed the 16 dated ledger rows (all carried a Fruit line). Drafting by one Sonnet agent from those maps. Every CLOSE and every promoted row below was re-read by the orchestrator in code or in the cited file; items marked "(inferred)" were not. Haiku ledger: 6 runs; one miss (the report-rs import map scanned a stale worktree instead of report-rs `origin/main`, so its counts were discarded and re-measured with `git grep origin/main`); one quote mis-split a wrapped line (Kythe row), caught on re-read.


Status: DRAFT. Nothing here was compiled or run. Facts come from the maps, the PR lists and verdicts.md; anything I inferred is marked "(inferred)". Entry files live under `/home/user/lance-graph/.claude/board/entries/`. "verdict row N" means the row number in verdicts.md. Entry names are given without the directory.

Scope notes:
- The 40 fruit rows carry the verdicts of verdicts.md unchanged (section B). verdicts.md numbers 38 open rows. FRUITS.md in the local checkout also lists the D-NAM-2 TFLite row, which verdicts.md does not mention; I list it as row 39 with "no verdict supplied". The other two of the 40 are already in the closed section (T-2026-10-23-1 killed; ISS-ACTIVATION-IGNORES-APP-HALF resolved 2026-11-12), per FRUITS.md.
- The two ms-class numbers in the entries (codegraph intake 2.5 ms; Cypher bench) are reported as failed gates, never as results, per the lance-graph no-millisecond rule.
- The maps did not carry entry text for the D-VAS (R9-R14), D-QSP, D-FOUNDRY and OGAR PR bodies. Where a loose end comes only from a PR title, it is marked "(PR title only)".

---

## A. ARCS

Counts are PRs seen in the PR lists inside the stated range, lance-graph unless named. PR ranges overlap in time because arcs interleave.

### Arc 1. DT64 bind store, polyfill lens, exact basis (D-DT64-6, D-DT64-7, D-DRDT)
- PRs: #1794-#1799, #1832, #1834-#1836, #1869, #1871, #1882, #1898-#1905, #1975-#1982 (about 35). OGAR #365, #394, #404, #409.
- What it established: DR is HHTL16 Cartesian direct routing; DT is a ClassId binding store (`2026-10-28-d-drdt-dr-dt-ownership-amendment.md`, proof D-DRCA-1 #1834). The bind store became production code as `crates/lance-graph-dt64` (`2026-10-30-d-dt64-6-bs-bind-store-crate.md`). The polyfill lens `0xFFFF :: 0xF000` binds by position and R2IL, SQL and Quack names are aliases (`2026-10-31-polyfill-positional-lens.md`, #1899). Behaviour lenses read their basis by the full classid (#1899/#1903, `2026-10-31-behaviour-lens-full-classid-l1.md`). The class basis is exact within a SchemaRev (Decision J, #1975), with a source-owned historical basis (#1976) and an exact basis witness at register and resolve (#1977).
- Closed ends: DT64-6 bind store (VERIFIED-IN-CODE, `d-dt64-6-bs-bind-store-crate`); DR addressing module (#1871, `d-dt64-6-dr-addressing-module`); L1 and L4 alias views (`polyfill-alias-views-l4`); MapTable versioned table (`maptable-versioned-local-map`, #1905); MapTable made durable through the cycle writer (`2026-11-10-dt64-map-table-lance.md`); method-authority census with two-way ratchet (`d-dt64-auth-census-shipped`, #1927); L5a/L5b (#1976, #1977); `MethodRef` stamped with issuing store and binding digest (#1979, PR title only).
- Loose ends:
  - `d-dt64-6-dr-addressing-module`: "OPEN: the probe owner switches to dt64::dr; bind_dr64 waits on the MethodSignature vs MethodRef identity (EXO-50); the reservation as a global range."
  - `polyfill-positional-lens`: "## Not covered (OPEN) ... The lens is still declared at runtime through BindStore::declare_algebra, and DT64 does not yet act as the authority for 0xFFFF (D-DT64-7-4, O8)."
  - `dt64-l5a-exact-basis-witness`: "L5c: admit still accepts a MethodRef without basis identity (finding_admit_does_not_carry_the_basis_identity)."
  - `maptable-versioned-local-map`: "OPEN: stored selections (D7'), which need O4."
  - `d-dt64-6-bs-bind-store-crate`: "OPEN: OGAR CapabilityAuthority wiring (D6); persistence (EXO-55); a DR #4 router carrying MethodRef; the immutable unit (§6 Q1)."
  - `d-dt64-6-ogit-predicates-impl` / `d-dt64-6-routing-bindstore-behavior`: root-namespace form in OgitUri; who owns the allowed table; WHEN has no attested source (## Open).

### Arc 2. DR / DT split, lane socket, class governance, reader scope (D-DRDT-SPLIT-1, D-LANESOCK, D-DRDT-CLASS-GOVERNANCE)
- PRs: #1966, #1967, #1969, #1970, #1971, #2013-#2031, #2044 (about 24).
- What it established: the property-to-lane socket and joint route/frame witness (D-LANESOCK-1 #1967); AbiBatch lanes read through the socket (#1969); one field basis and a test-only scope admission (#1970); the seam inventory for reader scope (`reader-scope-seam-inventory`). The 2026-11-12 ruling "DR routes, DT64 gives meaning" (plan v3 §2e, #2031) resolved ISS-ACTIVATION-IGNORES-APP-HALF. D-DRDT-SPLIT-1 says "version the mapping, not the content" (#2016) and "a refusal means not witnessed for this version" (#2026). Scoped lane binding tests (#2024, #2025) pinned that a copied scope id still binds another reader's fields.
- Closed ends: lane socket 1-3 (#1967, #1969, #1970); activation app half (RESOLVED, FRUITS closed section); slab planes as 256 borrowed selections and OrderIndex persist/reopen (#2019, #2020); lane binding refuses a predicate change at a bound position (#2021); W12 consumer class activated through OGAR reaches a binding (#2029).
- Loose ends:
  - `coresearch-drdt-split`: "OPEN: probe P5 (binding across a restart); L5d for the reorder gap."
  - `abibatch-lane-socket`: "Open, unchanged: value ingress, durable writer, codebook, exact decimal, D4/S-INGRESS."
  - `dt64-l5b-historical-basis`: "OPEN: until L5a lands, that proof is the field count only" (now stale: L5a is #1977; inferred).
  - ISS-DT64-SCOPE-ID-IS-NOT-A-GRANT stays open (verdict row 33; council plan v3 W11).

### Arc 3. XLOG native, relcomp, litho (D-XLOG-NATIVE, D-XLOG-DR64, D-LITHO, D-DYNBAG)
- PRs: #1857, #1859-#1862, #1864, #1866, #1884, #1896, #1897, #1906, #1907, #1909-#1913 (about 22). ndarray #407.
- What it established: XLOG's compiled RIR lowers onto the DR64 relation register (`d-xlog-dr64-0-xlog-rir-over-dr64-relcomp`). XLOG runs through Quack to mask-risc gated by the bind store (`dynbag-0-xlog-through-quack-mask-risc`, SHIPPED, #1906) and on `ndarray::simd` (#1884, ndarray #407). `crates/lance-graph-xlog` shipped (`2026-11-01-lance-graph-xlog-crate.md`), then the resident hypercube (#1911), derived relations (#1912) and the behaviour lens `0xFFFF::0xF100` (#1913). The nat-compose for `p(X, Z)` is exact, and `matmul_i8_to_i32` is the wrong carrier on this host as called (`dynbag-1-n-compose`).
- Closed ends: DYNBAG-0 (SHIPPED), XLOG-NATIVE-4/-8 (SHIPPED), XLOG-NATIVE-5/-6 (IN PR then merged, #1884), the Gate B register steps (D-LITHO-1, #1861).
- Loose ends:
  - `d-xlog-native-5-bindstore-epi5`: "OPEN: method identity (MethodSignature / MethodRef / MethodKey, EXO-50); frame catalogue extent (D8); production classids and the XLOG Epi5 declaration (OGAR); many-to-many join home (D-XLOG-NATIVE-3)."
  - `xlog-behaviour-lens-f100`: "OPEN: the XLOG fork (xlog-cube, xlog#2) still has to route these aliases to Refusal::Behaviour instead of interning them."
  - `xlog-resident-hypercube-context`: "OPEN: recursion ..., union and difference as views; more than one tile; n-ary relations; a CudaBuffer::View variant ... per-node cost vs the per-call path."
  - `dynbag-1-n-compose`: "OPEN: A bit-packed i1 x i1 -> i32 word ... in ndarray::simd (D-DYNBAG-3 candidate); AMX tier not on this host; cross-tile compose (MISSING-7)."
  - `d-litho-photolithography-contract`: "## Open: Q1: does the rod belong to DR's address or to DT's field law?" (Q2-Q4 follow).
  - `d-litho-1-xlog-register-steps-and-bag-fixture`: "OPEN: counts need a per-Y count width with defined overflow." This is verdict row 16 (TD-XLOG-NATIVE-COUNT-OVERFLOW-UNTESTED).

### Arc 4. XLOG cube, coverage census, W5/W7, add_pm (D-XLOG-COV, D-XLOG-W5/W7, D-XLOG-PM, D-XLOG-C9)
- PRs: #2082-#2087, #2090-#2096, #2109-#2115, #2119, #2120, #2123-#2125, #2131, #2136, #2137, #2139, #2146, #2149 (about 40). xlog fork #7, #8, #9. ndarray #430-#434.
- What it established: Cube column counts and join size (#2082, #2083 i128 checked); coverage reconciliation (`d-xlog-cov-0`), parity (`d-xlog-cov-1`) and census over xlog `007bb19` (`d-xlog-cov-2-census-007bb19`); Datafrog as an independent comparator (`d-xlog-knowledge-g1-datafrog`, 87/87 constants #2096); positive witnesses for 15 empty-only agreements (G1c, #2110) and expected answers for refused rules (G1b, #2109); stored one-column sets and `Cube::reach` (#2112), tiled reach (#2115, W7 parked); conjunctive bodies answered as count, support mask per variable and edge-only witness rows (`d-xlog-w7-bodies`, #2124); cube refuses stale or foreign closures (#2086); PmLedger on `abakus::pm` (`pm-ledger-abakus-and-2086-review-exception`, #2090, #2092); the cube against PostgreSQL (#2149).
- Closed ends: cube safety P0 (`xlog-cube-safety-p0`, STATUS fixed); P1 add_pm degree ledger (#2087); W5-A analysis (#2111), W5-B (#2112), W5-D (#2115); 2086 retrospective review and checked i64 escape (#2092, inferred from title); W7 steps 1-4B (#2126, #2127, #2131, #2136, #2137).
- Loose ends:
  - `d-xlog-w7-2-cube-planes`: "Open: presence-writer lifetime (#1941 D3) ... whether an owner's write must instead issue a new frame is the operator's to rule."
  - `d-xlog-w7-2-cube-lanes`: "Open: bodies, reach and closure do not read through the socket; this lends, it does not replace a terminal."
  - `d-xlog-w7-2-method-over-cube-lanes`: "Open: the guard's constants are coordinates of one frame ... nothing here refuses it after the map changes a coordinate."
  - `d-xlog-w5d-tiled-reach-w7-park`: "Open: the xlog fork's one-column route builds single-tile cubes, so no census row exercises the tiled path yet."
  - `d-xlog-cov-2-census-007bb19`: "W7: the remaining refusals ...; reach on a tiled cube is refused" (## Open).
  - `pm-ledger-abakus-and-2086-review-exception`: "OPEN: retrospective review of #2086 at ece628bc4; P4 economics ...; wasm tiers of tests/add_pm.rs not in ndarray CI."

### Arc 5. Staunen, epistemic state, derivation ledger, counterfactual flow (D-STAUNEN, D-EPI-COT, D-CFA, D-EAW, A10)
- PRs: #1813-#1830 (docs, about 18), #1840-#1844, #1846-#1848, #1885-#1889, #1896, #1914-#1916, #1936 (about 40).
- What it established: derivation ledger and load-bearing ablation (`d-epi-cot-c0-c2`, `support.rs`); tactics run only on grounded Epi5-eligible markers (C3, #1843); scent ranks mediators by testability (C3b, #1842); learned ordering over Codec9/lut9 (C3c, #1844); NSL neighbour selection against a frozen oracle (`d-nsl-w0-w1`, #1846). D-STAUNEN-0..5 (#1885-#1889, #1914). The source-root ruling (#1916: origin is neither observer nor report). Epistemic writers, `pearl::retract`, writer census (#1936, `a10-writers-retract-xlog-pr`). A10 found no CE64 change justified; NNUE parked for the triangle optimizer (`a10-epistemic-admission-and-nnue-compiler`, `a10-probes-s6-magic-nnue-baselines`).
- Closed ends: C0-C3c (TEST-PINNED); NNUE: PARK ("Verdict: PARK NNUE for this optimizer"); source root ruled (verdict row 21); `ogar-behavior-probe` kills fired ("body_rev is a semantic revision", "derivable today", "mandatory inspectable body exists").
- Loose ends:
  - `d-epi-cot-c3c-learn`: "The label should become 'the ablation discriminated', using the ledger's sensitivity."
  - `d-epi-cot-c3-warrant`: "Effect size, sample size, uncertainty and multiple comparisons are still unreported by CertificationModel."
  - `d-epi-cot-c3b-scent`: "Maturation bits 50..52 are unwritten" (= verdict row 25 WIRE50).
  - `d-epi-cot-c0-c2`: "evidence horizon k not declared; no memoization (three refusal bounds)."
  - `a10-epistemic-admission-and-nnue-compiler`: "OPEN: admission step design; rule identity of derivations; production stamp gate; certification writer; whether a learned ranker beats the exact fold."
  - `a10-writers-retract-xlog-pr`: "Exact-degree pairing in the fork optimizer: BLOCKED."
  - `coresearch-d-staunen-arc`: "OPEN: root = agent or report; which probes to run" (root part ruled by #1916).

### Arc 6. DR-select, Quack boundary, selection vector, derived planes (D-DR-SELECT-K, D-QUACK-BOUNDARY-LAB, D-SELVEC-COMPAT)
- PRs: #2032-#2046, #2057-#2059, #2140 (about 22). Fork: DuckDB C-API compare #2042.
- What it established: one DR selection, several physical views (`dr-select-k-0`, geometry roles, #2036); the lab ran and more than one view wins (`dr-select-k-1-lab`, #2035), then a byte-accounting and tie-rule correction (`dr-select-k-1-correction`, #2037). R1's falsifier ran (ELIMINATE to ADAPT, `selvec-compat-r1-falsifier`, #2038). A position selection is admitted for COUNT and SUM (`dr-select-k-sv-1`, #2039), witnessed with its lanes (`sv-2`, #2040), compared with DuckDB's SelectionVector C-API (`sv-3`, #2042). A Quack program is admitted where its lanes are (`quack-boundary-lab`, #2045). Derived planes inherit what P1-P4 measured (`quack-derived-plane-p1`, `p2-p4`, #2058). In-place arm (#2140).
- Closed ends: SV-1/-2/-3; the question "do Quack and mask-risc admit an explicit selection operand" is answered by #2039 (inferred); R7 (range operators, grouped count in Draft) by #2130 (`exo-292-draft-ranges-count-by`).
- Loose ends:
  - `dr-select-k-1-lab`: "where the density threshold between positions and mask lies (two density points only); any storage-backed source; gates 1 and 2 of plan §5." Repeated in `selvec-compat-r1-falsifier` and `dr-select-k-sv-1`.
  - `dr-select-k-sv-2`: "Open (BLOCKED): no authority issues a reader Scope, so authorization is not established; no non-lab order witness."
  - `dr-select-k-sv-3`: "R4 and R7 stay UNDER REVIEW; x86-64 Linux only; allocation failure not compared; authorization untouched."
  - `quack-derived-plane-p2-p4`: "no producer witness for lane contents; no carrier for a foreign population's frame in a derived plane; plane identity has no socket."
  - `dr-select-k-1-correction`: "whether a position-list consumer should exist at all is a decision."
  - `quack-boundary-lab`: "the eleven contracts in the plan, section 8. None is closed by this lab."

### Arc 7. Bag fold, DR fold, Abakus, DRISA, zero-copy driver (D-BAG-FOLD, D-DR-FOLD, D-DRAM, D-DRISA, D-DRIVER-ZC, D-GLOBAL-WHERE)
- PRs: #1937-#1946, #1949-#1957 (about 20). ndarray #402.
- What it established: multi-hop existence from resident presence, no stored relation (`d-bag-fold-witnessed-existential`); v3 Arm L against production witness contracts; v4 attested pointer lens and candidate-gated execution (21/21 checks). Storage, addressing and algebra kept apart (`d-dr-fold-map`). Direct-routed virtual SoA views plus a DuckDB parity gate (D-GLOBAL-WHERE P0 #1943; P2 criteria-aware WHERE and Lance selective read, `BLOCKED_D4`, #1950). Carry-word census (`d-dram-1-census`); k-of-n on the carry words: kill fired (`d-dram-p3-5-kofn`, #1954). A sealed DR frame works as a local register file (`d-drisa-1-register-proof`); carry words express threshold, Kleene and Bag exactly but lose to native words in every shape (`d-drisa-3-alu-semantics`).
- Closed ends: k-of-n KILLED (verdict row 1); D-DRAM-1 census; D-GLOBAL-WHERE P0/P2 measured; DRISA-1/-3 measured.
- Loose ends:
  - `d-bag-fold-witnessed-existential`: "OPEN. The P-vs-L operator ruling; a plane-naming witness; a mint guard that edges refine their source key" (= ISS-MULTIHOP-PRESENCE-ARM-UNRULED, verdict row 20).
  - `d-dr-fold-map`: "OPEN, owner decisions D1-D4": range-expand op, scatter-then-AND classification, binding lifetime vs presence writer, owner of DR to Lance read planning.
  - `d-global-where-p2`: "Boundary: BLOCKED_D4. No witness binds a DR Frame to a Lance version."
  - `d-drisa-1-register-proof`: three BLOCKED rows (source-location receipt/lease host EXO-46/89/151; cross-call result register lifetime; DR to Lance _rowid authority D4/S-INGRESS).
  - `d-driver-zc-0-virtual-bus-contract`: ZC-3 foreign amortization BLOCKED EXO-46/89/151/140; ZC-5 BLOCKED D4/S-INGRESS, EXO-114.
  - `d-bag-fold-v3-production-witnesses` / `v4`: owner decisions on the EXO-140/EXO-46 rulings, contract extensions (presence, uniqueness, pointer origin, lens), register-gated Gather lowering in Quack.

### Arc 8. Codegraph, Charon, Kythe, SAG, Thought-Git (D-CODEGRAPH, D-SAG-6, D-CGMM, D-CG-BEH, D-TGL)
- PRs: #1876, #1878, #1883, #1894, #1918-#1934, #1957, #1958, #1999-#2007, #2144 (about 35). ruff #129-#138. OGAR none.
- What it established: SAG-6-1 capability audit (rust-analyzer, Charon, Kythe, ruff) and a bounded lab projection with span join and move-surviving identity (`d-sag-6-1-capability-audit`, `d-sag-6-2-lab-projection`, `d-sag-6-2-span-join-identity`). Kythe entries produced from rust-analyzer facts (`kythe-bridge-ra-facts`, #1894). `ruff_charon` reads Charon LLBC, with an offline store, per-crate incremental refresh and a binary rows format with per-row FunDeclId (ruff #129-#133, #135). `lance-graph-codegraph` (#1921, #1923, #1929). Semantic Grammar Recovery gates S1-S6 (`sgr-recovery-gates`, #1932). Codegraph over real Charon stores: false "unaffected" for public symbols found, then coverage-aware impact (`codegraph-recon-1`, `-2`, #2000, #2002). CG-BEH-1/-2 trace a changed dispatch arm to registered methods and read ON, WHEN, DO as separate facts (#2003, #2005). The Thought-Git lifecycle lab (F-LAND/F-DISCARD) (#1924-#1926).
- Closed ends: D-CODEGRAPH-0 slices A-C; BIN reader unchanged refresh 2.58 us (`codegraph-bin-reader-timing-gate`); P1-P3 review decisions (`codegraph-review-operator-decisions`); D-TGL-PUB steps 1-3 (#1925, #1926); `impact_report` duplicate (#2001 vs #2002) resolved by "Wave 0: one impact_report" (#2004); effects rows (CG-8, #2006); ruff_codegraph projection (ruff #138) opens in the reader (`d-codegraph-n1-ruff-source-rows-open`, #2144).
- Loose ends:
  - `codegraph-bin-reader-timing-gate`: "changed-crate intake 2.5 ms fails the 1 ms gate" and "OPEN: B3 (interner) and the re-run of this gate; Quack counting (F7) behind the codec9-column gap." This is a failed gate, not a result.
  - `d-codegraph-0-gap-map`: "OPEN: the method body anchor (G7), whether R2IL can carry Rust places, drops and panics (G4), merge order of ruff's claude/r2il-pr2-* sink branches (G8); a cross-crate callee is named by Charon as cg_lib::shapes::Impl::area."
  - `cg5-defuse-probe`: "OPEN: def-use rows as a store row kind (CG-3 format decision, EXO-235); interprocedural provenance."
  - `codegraph-recon-2`: "OPEN: per-configuration stores (features, test targets); recording targets and opaque modules in the store instead of declaring them."
  - `cg-beh-2`: "OPEN. Executed parity per word. Source attribution for ON and WHEN. Whether the guard belongs in the binding digest and whether admit should see the basis."
  - `sgr-recovery-gates`: "A10 BLOCKED: raw5 = 0; S6 fork rule conflict and the decline fix"; `a10-probes-s6-magic-nnue-baselines`: query-relative Magic Sets hazard measured.
  - `d-sag-6-2-lab-projection`: four admission gates before leaving the lab; F05 and F07 NOT RUN (`d-sag-6-2-span-join-identity`).
  - `p4-operator-decisions-lifecycle-lab`: production gates G9-G12 OPEN; `p4-thought-git-codegraph-dt64-review`: preconditions EXO-148, D-TGA-1/-3/-4, D-TGB-2/-3/-5/-7.

### Arc 9. Intake as a compiler, register contract, PostgreSQL polyfill (D-INTAKE, D-REGC, D-QSP, dir-sim)
- PRs: #2017, #2018, #2022, #2098-#2108, #2145-#2157 (about 28).
- What it established: a field mapping over DT64 lane socket plus `MapTable::frames` (D-MAPPING-POLYFILL-1, #2098); directory re-read stored as verified changes, delta read, linear restore (`d-dirsim-reobserve`, `d-dirsim-delta-read-1`, #2099-#2102); licence planes through the DT64 lane and selection sockets (#2105-#2108). `borrow_all_selected(_present)` and the zammad-rs gate (#2147, #2148). D-REGC reconciliation: seven contradictions measured in one consumer (`d-regc-register-contract-reconciliation`), the PostgreSQL row polyfill (`d-regc-12-pg-row-polyfill`, #2152), data and class as config (`d-regc-14-intake`), read by reference and the cube over three tiles (`d-regc-bound-read-tiled-cube`, #2153). Intake as a compiler measured against DuckDB: two failed gates D-INTAKE-2G and 2H (`d-intake-2-compiler`, #2155); #2156 is titled "D-INTAKE-2G/2H: streaming cell sink and per-version validity planes" (so likely the repair; inferred).
- Closed ends: reobserve and delta read; REGC-1/-2/-8; D-QSP-VROW gate (#2148); literals bind through `quack::bind`, row egress in `intake::pg` (#2157).
- Loose ends:
  - `d-regc-12-pg-row-polyfill`: "Open: the presence 0, value != 0 arm has no fixture row; ORDER BY; the issuer of join positions (EXO-329); the outer side's plane (D-REGC-4); zammad-rs adoption."
  - `d-regc-14-intake`: "Open: the caveats table ...: 64-bit lanes; a delete that does not advance the data version (EXO-337); the fallback join form (D-REGC-9); the owner's Vecs instead of SoA rows (D-REGC-11)."
  - `d-regc-bound-read-tiled-cube`: "Open: basis share of the census; D-REGC-7 (needs D-REGC-5 / EXO-329); D-REGC-3, -4, -13 need decisions; D-REGC-6 needs the predicate count."
  - `d-intake-2-compiler`: failed gates 2G and 2H recorded.

### Arc 10. Virtual address algebra, bag as a thinking object, weather planes (D-VAS, D-VIRTUAL-ADDRESS-ALGEBRA, D-WX, logic-neutrino)
- PRs: #2048-#2071, #2076 (about 20). Entries: `coresearch-bag-object`, `weather-bit-planes`, `weather-dr256-ternlog`, `wx-calibrated-plane-admission`. PR titles for D-VAS R9-R14 only (#2066, #2067, #2069-#2071); `D-VIRTUAL-ADDRESS-ALGEBRA-1` owner-backed scalar read is "LAB VERIFIED" (#2060, PR title).
- What it established: the bag is "a bound description over existing carriers, not a stored object" (`coresearch-bag-object`); a guarded read-only bag COUNT is admitted through mask-risc, R2IL and Quack (#2069, PR title); act drops two allocations per call with no measurable time gain (#2071, PR title). A one-byte weather code as resident bit-planes and weather x DR256 slabs x one ternlog x an integer fold (labs #2055, #2057). Calibrated plane admission: "representation stays, interpretation is refused" (#2059). The logic-neutrino P0/P1 lab was audited and its six findings closed (#2051-#2054).
- Closed ends: D-VAS R9-R14 (PR titles); logic-neutrino audit findings (#2054); the bag closeout "evidence points to a bound description" is a map, adopts nothing.
- Loose ends:
  - `coresearch-bag-object`: "OPEN: D1-D4" plus 4 probes, 2 adopt-now tests.
  - `weather-bit-planes`: "min and max over bit-planes (a bit-descent of Any tests, not written); the slab-stack form; whether a 13-op chain can be shortened."
  - `weather-dr256-ternlog`: "whether resident predicate planes are affordable when the bounds change per query; per-station statistics on a slab stack; throughput at 10^9 rows."
  - `wx-calibrated-plane-admission`: "a carrier for the current floor that the admission path can read; the bake recording its floor stamp; DerivedPlane over a source of arbitrary size."

### Arc 11. MeTTa, MORK, grammar, Mistaber, global pattern words (D-COGNITIVE-POLYFILL, D-GRAMMAR, D-MIST-NATIVE, D-GPW)
- PRs: #1873-#1880, #1944, #1947, #1948, #2142 (about 12). OGAR none direct.
- What it established: four frontends (MORK / MeTTa / NAL / Atomese) over one native algebra (`d-cognitive-polyfill-p0`, #1944); German dative roles as typed constraints and a grammatical WHEN through DT64 (`d-grammar-constraints-0`, `d-grammar-when-1`); Mistaber's Kripke holdings as a least fixpoint over DR64 compose, with HHTL16 Register128 rows and makor addressed the KJV way with no classid (`d-mist-native-0`, `-1a`, `-1b`). `LIKE` resolved at Quack bind (#2142, `d-gpw-quack-like-at-bind`); spaCy Matcher format as a front end (`d-gpw-lance-graph-spacy-front-end`); global pattern words proposed (`d-gpw-global-pattern-words`).
- Closed ends: Mistaber makor addressing ("OPEN: none for addressing; the intake belongs in deepnsm-v2", `d-mist-native-1b-makor-hydrate`); LIKE at bind (MEASURED).
- Loose ends:
  - `d-cognitive-polyfill-p0`: "W-1: one cold symbol -> DR id step for MeTTa/MORK. W-2: Guards::check needs a FrameStore."
  - `d-grammar-when-1`: EXO-140 classify EqU32Via; EXO-46 / D-RMAP-1; three D-DT64-6 questions; promotion needs hydration/Rubicon gate. `d-grammar-constraints-0`: "the DT64 WHEN (SetExpr) has no hop leaf" (now addressed in part by `d-dt64-reach-1`, #2139).
  - `d-mist-native-0-kripke-fixpoint-over-dr64`: "a production home for compose (blocks the lance-graph-mistaber facade); axes past 64; lineage into SupportLedger; intake from compiled HLL." `coresearch-mistaber-hydration`: undeclared-world policy, hilchot data, SupportLedger derived-from owner, D-ARG-1..3 order.
  - `d-gpw-global-pattern-words`: "the four decisions in the plan's §5; whether the absent-case policy can be a compile-time marker; D-GPW-1's census has not been run." `d-gpw-lance-graph-spacy-front-end`: "TEST-PINNED against a table read from spaCy's source. NOT run against spaCy."

### Arc 12. ONNX intake, TFLP, Tract, Codec8, NSL (D-TFLP, D-ONNX-INTAKE, D-TRACT-HARVEST, D-NAM)
- PRs: #1800-#1803, #1806-#1810, #1846, #1849-#1858, #1865, #1867, #1870, #1872, #1876 (about 30). ndarray #397-#401, #398.
- What it established: softmax evidence and Codec8 semantic admission (`d-tflp-65-landing`, #1800, #1801); the TANH proof receipts (`d-tflp-tanh-proof-learning-tract-harvest`); the cold frontend bake-off (`onnx-frontend-intake-bakeoff`, S1-S5 NOT RUN there); Tract as a cold ONNX frontend admitted onto native methods (`d-tract-harvest-0-native-algebra-bridge`, #1850), the coverage map where all 346 unregistered Jina nodes became bound (`d-tract-onnx-coverage-map`, outcome section), and cold shape folding (`d-tract-shape-1-cold-shape-folding`, #1853). ONNX is one classid (#1870). 
- Closed ends: Tract harvest lab implemented; 346 nodes bound, Jina 7,620 registered, 0 unregistered (`d-tract-onnx-coverage-map`); ndarray #399 vserf, #401 re-exports and checked i64.
- Loose ends:
  - `d-tflp-65-landing`: "OPEN: generation identity owner (EXO-152); binding store (D-NFR S5); contract placement of Receipt."
  - `d-tract-harvest-0-native-algebra-bridge`: "OPEN: float reduction tolerance contract (D-ONNX-NATIVE-1); admitted f32 MatMul (first whole-model blocker); Gather reclassification; ORT_DISABLE_ALL as the standing oracle setting."
  - `d-tract-shape-1-cold-shape-folding`: "OPEN: int64 shape-arithmetic semantics; axis as DO parameter vs classid per axis; vserf ~4x slower than Tract's erf; ConstantOfShape/Expand not yet registered."
  - `d-tract-codegraph-address-map`: four open questions for DT64: one table or a separate concept (0xC402), OGAR mint of 0xC400-0xC402, the r2ssa import owner, where the ONNX lens moves.
  - `2026-10-29-palette512-turbovec9-nsl-worksteal`: work packages W1-W8 with admission gates; "research / falsifier plan, not an implementation".

### Arc 13. Foundry parity, ISA stream lab, GEMM shape (D-FOUNDRY-PARITY, D-DT64-GEMM-SHAPE, D-DT64-G2L)
- PRs: #1983, #1987-#1991, #1992, #1995, #1996, #1997 (about 10). ndarray #414, #415, #418-#422, #429.
- What it established: one lane binding feeds a cube fold and an AMX BF16 GEMM (`dr-isa-stream-lab`, #1983); one witnessed translation reused across GEMMs (`dt64-g2l-lab`); 64x64 and 256x256 BF16 objects (`dt64-gemm-shape`); INT8 u8xi8 arm and facet-stride audit (`dt64-int8-facet-stride`); INT8 on a prepacked B (`int8-direct-packed-b`, ndarray #418); `LaneBinding::borrow_all` (`dt64-lane-borrow-all`, #1988; lab adopts it, `lab-resolve-borrow-all`, #1989). The 64-store operating network (`foundry-parity-1-lab`) and per-store cost as one INT8 GEMM (`foundry-parity-1-arm-c`, #1995); CanCook as a registered DT64 method (G6, #1996).
- Closed ends: G2L row (measured in lab); borrow_all (SHIPPED); strided-A and prepacked INT8 B asks answered by ndarray #414/#415/#418; G6.
- Loose ends:
  - `foundry-parity-1-lab`: "OPEN: G1 element-wise arithmetic decision; G2 IAM in the read path; G3 overlay frame; G4 action gate; G5 bound relations; G6 method lowering." (G6 done by #1996, PR title.)
  - `foundry-parity-1-arm-c`: "OPEN: G1 remainder (P1b); D-DRISA-M1 layout witness; generator from Resolved to a typed vocabulary."
  - `dr-isa-stream-lab`: MKL comparator not executed.
  - `lab-resolve-borrow-all`: "Open: consumers that call addr per field elsewhere should move to borrow_all."

### Arc 14. ndarray simd, abakus, coordinate folds (ndarray #378-#437)
- PRs (ndarray): #378-#383, #385-#403, #406-#411, #414-#437 (58). Plus lance-graph #2079 argmin/argmax terminals, #2080 jc extrema.
- What it established: NaN/presence bijection and SQL NULL=NaN predicates (#381); `simd_stride.rs` (lut_u8, lut9) and the FACET STRIDE rule (#382, plus the 2026-10-27 facet-stride audit in ndarray CLAUDE.md); U9 head-to-head and U9 abacus probes (#388-#393); half/full adder mask words and vectorized exp (#402); position-form folds (#423), count_set_at (#424), masked i32 coordinate folds and argmin/argmax (#425-#427, `argminmax-cube-coordinate`); u64x64 whole-tile probe and route F/G (#419-#421, #429); `hpc::abakus::pm` add_pm/neg_pm/sub_pm as an experiment (#430-#433); additive float/mask/integer API on every backend with an IEEE bit-exact CI gate (#435), then an AVX2 `F64x8::reduce_min/max` regression fix (#437).
- Closed ends: k-of-n register model killed on the carry words (verdict row 1; the pm library continued as #433 and #2090); strided BF16 and prepacked INT8 B (#414, #415, #418).
- Loose ends:
  - `dynbag-1-n-compose`: bit-packed i1 x i1 -> i32 popcount word, AMX tier host.
  - `d-dram-p3-5-kofn`: "## Open: a carry fold over U64x8 would need a wide adder; repeat with target-cpu=native and on AVX2-only and NEON hosts; the model may still win on counts above 255."
  - `dt64-int8-facet-stride`: (now answered by #418; inferred).
  - `cypher-df-vs-quack-bench`: "Open: the SIMD gather in ndarray ... two-hop needs a foreign-value sum" (gather is quarantined, verdict row 38).
  - `pm-ledger-abakus-and-2086-review-exception`: wasm tiers of `tests/add_pm.rs` not in ndarray CI.
  - verdict row 29 BF16-POLYFILL-PER-ELEMENT-ALLOC: #414/#415 touched the entry, not the loop.

### Arc 15. OGAR: HIRO action handler, OGIT tables, shared DO, ERP/directory mints (OGAR #363-#415)
- PRs (OGAR): #363-#415 (50). lance-graph mirrors: #1904, #1959, #2068, #2072, #2103.
- What it established: OGIT allowed-edge whitelist and per-type row ranges through Quack (#363, #389, #390); `ogar-vocab` reserves the 0xFFXX hardcoded-behaviour block (#365; mirrored in lance-graph #1904); directory concepts minted (#370, mirrored #1959); `erp_polyfill (0x020C)` and the ERP capability table (#378, #379); EffectClass in `ogar-vocab` and on DT64 `Registered` (#386); the HIRO action handler (stateful submit, wire codec, session adapter, completeness and the invoker state machine, #392-#397, #405-#407, #410, #411, #413); every ActionDef registers as a DT64 method (#394, #404, #409); KI-1 runs the Kleene guard through the class binding (#391, follows lance-graph #2132); `ogar-from-ki` reads HIRO Knowledge Items (#412, #414); dir-core keeps the Entra to EXO ordinal (#371).
- Closed ends: KI-1 red-pin turns green with #391 (inferred: `exo-326-kleene-gate` said it "turns red when OGAR moves to this revision"); all 50 defs registered (#404); frozen DISCOVERY-MAP/EPIPHANIES (#372).
- Loose ends:
  - `exo-326-kleene-gate`: "Native gated plans (the literal ask of EXO-326). Needs a gate operand in the frozen programs; not started."
  - `classid-on-do-call-shape`: "Open: whether MethodBinding.on is redundant; the effect class on the binding and the call-parameter channel (EXO-291); EqU32Via / Semijoin admission (EXO-140); cube sides filtering on a property (EXO-327)." (OGAR #386, #404 touch the first two; inferred.)
  - `d-omr-bindstore-is-the-registry` (D-OMR-AMEND-1): decision recorded, documentation only.
  - Verdict row 22 ISS-RECIPE-LOCO-VOCABULARY-EXISTS-TWICE (two `impl Vocabulary`).

### Arc 16. ruff: Ruby/Rails frontend, ruff_charon, Codegraph projection (ruff #127-#150)
- PRs (ruff): #127-#150 (24).
- What it established: predicate roles and a role census (#127); `ruff_charon` Rust intake from Charon LLBC, offline store, rows v2, binary rows format, no `?` dispatch targets (#129-#135); `ruff_cpp_spo` keeps callee resolution status (#136); Rails enums kept in declared form (#137); `ruff_codegraph` projects the Ruby harvest into Codegraph store files (#138); the Ruby frontend reads seed calls, class-body macros, routes, validations, nested routes, Module.nesting (#139-#149); a CoffeeScript frontend (#143); docs "from a Rails harvest to the register contract" (#150).
- Closed ends: ruff_charon slices A-C intake half; ruff_codegraph store opens in the lance-graph reader (#2144).
- Loose ends: from PR titles only (no entry text): the Ruby frontend counts every unknown route DSL call (#146), so unknown DSL shapes are measured, not lifted. `d-codegraph-0-gap-map`: merge order of ruff `claude/r2il-pr2-*` sink branches (G8).

### Arc 17. report-rs as a consumer of the lance-graph surfaces (report-rs #96-#182, 83 PRs; lance-graph touch list, 92 PRs)
- Source limits: PR titles plus a direct read of report-rs `origin/main` at `f2e55c3` (#182). Nothing here says how report-rs is built or deployed, and nothing proposes a change inside report-rs (it is in production).
- Verified imports (git grep on report-rs main): `lance_graph_contract` 15 files, `lance_graph_dir_sim` 5 (`directory_view.rs`, `iam_preview_view.rs`, `identity_ws.rs`, `licreport.rs`, `tests/p68_iam_policy_preview.rs`), `lance_graph_report` 3, `lance_graph_rbac` 3, `lance_graph_dt64` 2 (`src/xverweis.rs`, `tests/p66_xverweis_pins.rs`).
- The DT64 consumer fact that matters: `src/xverweis.rs:35` imports `lance_graph_dt64::dr::{Frame, FrameStore, LocalMap, MapRefusal, MapTable, Scope}`; `:219` declares `const READER: Scope = Scope { id: 1 };` ("the one reader of the relation: the cube that owns it") and `:537` calls `store.bind_lanes(...)` with it. The binding class is a process-local `BINDINGS_VIEW: ClassId = 0x7F1D`, documented as not an OGAR concept ("a mint is OGAR's call, not this crate's"). So report-rs is a production consumer of exactly the API shape ISS-DT64-SCOPE-ID-IS-NOT-A-GRANT is about: a public `Scope` id constructed by the caller. Any council ruling on that row (private id, or `bind_lanes` taking a sealed scope) is a breaking change for this call site.
- What the titles show on the report-rs side: XVERWEIS answers pinned "before R(x, y) moves to DT64 lanes" (#175), then "XVERWEIS R(x, y) as two aligned DT64 lanes" (#176); the licence report "over Graph and dir-sim" (#166) with run receipt and Nachweise beside it (#120) and an XLSX export (#173); directory group memberships and group-granted roles (#171), role raised by test-tenant groups at login (#172), inherited-role policy preview and JML (#177, #178); level, customer and service roles in the policy (#160); the project plan "as ClassView x WideFieldMask classes" (#101); field mapping studio (#124) where "an edited field mapping survives Fokus and preference" (#133); the EXO-263 unified ADUC / licence / Address Console plan and AC-1i identity workspace "over one admitted observation" (#113-#118); the spatial-nav graph walked through the router (#174).
- What it shows on the lance-graph side (titles): lance-graph #2098 D-MAPPING-POLYFILL-1 names report-rs as its consumer (`d-mapping-polyfill-1`: "Consumer: report-rs (read-only)"); #1969 and #1970 are "report: AbiBatch lanes through the DT64 lane socket" and "one field basis ... test-only scope admission"; #2040 is "dt64, report: a position selection witnessed with its lanes"; #2002 recovers "report-rs #33 from compiler facts"; dir-sim #2099-#2108 (reobserve, delta read, licence planes, lanes bound by predicate, "licence coverage follows the nextLink chain").
- Closed ends (lance-graph side): lane socket 1-3 (#1967, #1969, #1970); mapping polyfill with `MapTable::frames` WIRE (#2098); dir-sim reobserve, delta read, restore, licence planes (#2099-#2108); position selection witnessed (#2040).
- Loose ends: none read from entries for report-rs itself. The lance-graph-side open lines these PRs touch: `abibatch-lane-socket` ("no authoritative reader-scope witness"), `d-regc-12-pg-row-polyfill` (EXO-329, zammad-rs adoption), `coresearch-workbook-hypercube-soa-projection` (G1 GATHER-DECISION still OPEN on the #1590 ledger; PR #33).

---

## B. EXISTING FRUITS, REVISED

Actions are copied from verdicts.md. "Arc evidence" names the arc in section A that adds new evidence.

| # | row | old tier | new action | reason | arc evidence |
|---|---|---|---|---|---|
| 1 | k-of-n carry words (IDEAS) | High-signal | CLOSE (KILLED, keep as record) | Status already "Measured 2026-11-03: kill fired" (#1954). pm library lives on as `hpc::abakus::pm` (#433, #2090). | Arc 7 (`d-dram-p3-5-kofn`), Arc 14 |
| 2 | support masks vs `DerivationLedger::pooled` | High-signal | MERGE-INTO-8 | Same fixture: diamond + cycle + bypass; mask vs ledger vs path oracle in one run. | Arc 4 (`d-xlog-w7-bodies` gives a support mask per variable for cyclic bodies), Arc 5 (ledger `sensitivity`) |
| 3 | certificate replaces deltas | High-signal | MERGE-INTO-5 | Both are the Thought-Git replay witness. #2150 kept MapTable mappings as replayable patterns is adjacent, not closing. | Arc 9 (dir-sim stores verified changes, #2099), Arc 1 |
| 4 | Lumpability as closure check | High-signal | KEEP | Open; nothing closes it. | none found |
| 5 | one replay checks proof, receipt, witness | High-signal | MERGE-INTO-3 | Same as 3. | Arc 8 (Thought-Git lab, `p4-operator-decisions-lifecycle-lab`) |
| 6 | TLAC-P01 first kill test | High-signal | KEEP | No TLAC in `crates/` or the OGAR log. | none found |
| 7 | five unrun D-DRMC probes | High-signal | KEEP (X8 half-run) | X6, X7, X16, X18 open. | none found |
| 8 | independent support is a mask question | Low-hanging | KEEP (absorbs 2) | See 2. Residual follow-up from row 21: pin ONE reading (origin) in P-AGENT-VS-REPORT. | Arc 5 (#1916 ruling), Arc 4 |
| 9 | recursion never emits a count | Low-hanging | KEEP | Bag refusal exists for multiplicity (`Refusal::BagNeedsMultiplicity`), not the recursive-count refusal. | Arc 4 (`Cube::reach` #2112/#2115 is existence only; counts over cyclic bodies are factorized, #2124) |
| 10 | MeTTa four-binding vs D-MPS-9 | Low-hanging | KEEP (Low stays; cheaper) | #1944 adds a MeTTa frontend over the native algebra. | Arc 11 (W-1, W-2 in `d-cognitive-polyfill-p0`) |
| 11 | Kythe `ref/call` | Heavy | KEEP, partly UNBLOCKED | `ruff_charon` (#129-#135) is a call-fact source; #2000 runs over real Charon stores. "Charon is not installed here" is stale. | Arc 8, Arc 16 |
| 12 | certified-compiler boundary | Heavy | KEEP | Open. | Arc 5 (D-CFA-5 docs only, #1818) |
| 13 | TD-FROZEN-POLYFILL-NAMED-BY-R2IL | High-signal | MERGE-INTO-24 | `lance-graph-r2il/Cargo.toml:25-26` still non-optional git main. #1899/#1902 moved the lens to position and alias views, not the compile-time table or optional OGAR. Same closing edit: the `fold` feature makes OGAR optional. | Arc 1 (`polyfill-positional-lens`: "DT64 does not yet act as the authority for 0xFFFF"; #1927 census ratchet) |
| 14 | TD-NARS-REVISION-UNGUARDED-CAUSAL-EDGE-ISA-1 | High-signal | KEEP | No PR. | Arc 5 (#1936 writers, `pearl::retract`; does not guard `Opcode::Revision`, inferred) |
| 15 | TD-CLASSID-CANON-COMPAT | Low-hanging | KEEP (scope narrowed to legacy 0x0000_XXXX), Low | No PR. | Arc 1 (L1 full-classid reading, #1903) |
| 16 | TD-XLOG-NATIVE-COUNT-OVERFLOW-UNTESTED | Low-hanging | KEEP | xlog grew (#1907/#1911/#1913) but no Count refusal test. | Arc 3 (`d-xlog-native-6`: "the overflow refusal has no reachable fixture"), Arc 4 (#2083 i128 checked products, #2092 checked i64 escape change the shape of the test; inferred) |
| 17 | TD-BGZ-TENSOR-INTEGRATION-TESTS-UNGATED-1 | Low-hanging | CLOSE (RESOLVED 2026-10-26 on branch) | CI runs `--tests`, 241 pass. Closes when branch merges. | none |
| 18 | TD-CE64-W-SLOT-DOC-SAYS-NO-ANCHOR | Low-hanging | CLOSE (same) | `edge.rs:944`, `layout.rs:53`. | none |
| 19 | TD-STATUS-BOARD-ROWS-MISSING-FOR-OCT-PLANS-1 | Low-hanging | KEEP, RE-SCOPE to Oct+Nov | Worse: November plans D-INTAKE, D-REGC, D-DRDT-SPLIT, D-LANESOCK all lack rows. | Arcs 2, 9, 13 name the plans |
| 20 | ISS-MULTIHOP-PRESENCE-ARM-UNRULED | High-signal | KEEP (operator decision) | Still ARM-L-DECISION-MEMO. | Arc 7 (`d-bag-fold-witnessed-existential`, v3, v4) |
| 21 | ISS-SOURCE-ROOT-AGENT-OR-REPORT | High-signal | CLOSE (RULED) | DECISION (operator, 2026-11-01), d-staunen-5 plan line 157 (#1916): a source root identifies the independent evidential origin; observer, report and lineage kept separately. Residual: P-AGENT-VS-REPORT pins ONE reading; re-check probe `root()`. | Arc 5 |
| 22 | ISS-RECIPE-LOCO-VOCABULARY-EXISTS-TWICE | High-signal | KEEP | OGAR `nars.rs:135` NarsVocabulary and lance-graph-ogar `recipe_vocab.rs:138` both stand. | Arc 15 |
| 23 | ISS-SIX-DECISIONS-...-ACTION-GATE | High-signal | KEEP | `action.rs:266 pub fn commit` still present. | Arc 13 (Foundry G4 action gate is open), Arc 15 (OGAR #384 Validator) |
| 24 | ISS-SIX-DECISIONS-...-FLOAT-REFS | High-signal | KEEP (absorbs 13) | `crates/lance-graph/Cargo.toml:58` r2il unconditional. | Arc 1 |
| 25 | WIRE50-ISA-LEGACY | High-signal | KEEP | No code. | Arc 5 (`d-epi-cot-c3b-scent`: bits 50..52 unwritten) |
| 26 | SUPERVISOR-DISPATCH-CAPACITY-DRAIN | High-signal | KEEP | `drain.rs:97 unbounded_channel` still. | none |
| 27 | D-CGW-4 PROMOTION GATE | High-signal | KEEP | `counterfactual_verdict` has no caller outside `pearl.rs`. | Arc 5 (D-CFA docs, #1813-#1818) |
| 28 | DURABLEREF-THOUGHTCOMMITID | High-signal | KEEP | No struct in `crates/`; 2026-11-02 P4 review uses BaseRef as placeholder. | Arc 8 (`p4-thought-git-codegraph-dt64-review` lists ISS-DURABLEREF as precondition) |
| 29 | BF16-POLYFILL-PER-ELEMENT-ALLOC | High-signal | KEEP | `simd_ops.rs:640 vec![0.0f32; k]` in loop; #414/#415 did not touch it. | Arc 14 |
| 30 | RA-STATICINDEX macro refs | Low-hanging | KEEP | Open. | Arc 8 (`kythe-bridge-ra-facts`: "OPEN: the macro-argument gap") |
| 31 | FISHERZ saturates | Low-hanging | KEEP (Low) | Open. | none |
| 32 | ISS-CAUSAL-EDGE-...-CLIPPY | Low-hanging | CLOSE (RESOLVED 2026-10-26 on branch) | Clippy gated in `rust-test.yml`. | none |
| 33 | DT64-SCOPE-ID-IS-NOT-A-GRANT | Heavy | KEEP (waits on council plan v3 W11) | Open. | Arc 2 (#2024/#2025 pin it), Arc 6 (`dr-select-k-sv-2` BLOCKED on reader Scope authority) |
| 34 | ORDER-AND-CODEBOOK-WITNESS | Heavy | KEEP | Partial movement: #1977 exact predicate-basis witness; #2026 refusal means "not witnessed for this version". Per-site witnesses, no single contract type. | Arcs 1, 2, 6 (see synergy S1) |
| 35 | ADMISSION-B call-site classification | Heavy | KEEP (ownership decision) | Open. | Arcs 7, 11 name EXO-140 (see N-items) |
| 36 | RECURSION-DRIVER | Heavy | KEEP | #1896 is candidate evidence only. | Arc 4 (`Cube::reach` over pair planes, #2112, #2131, may be a co-addressed driver; inferred) |
| 37 | RETRACTION-PROVENANCE | Heavy | KEEP | #1896 labels and `DerivationLedger` exist; the per-cell plane does not. | Arc 5 (#1936 `pearl::retract`), Arc 9 (stored deltas) |
| 38 | GATHER-HOP-1616 | Heavy | KEEP (quarantined) | Open. | Arc 14 (`cypher-df-vs-quack-bench` asks for a gather) |
| 39 | D-NAM-2 TFLite int8 leaf (IDEAS 2026-10-26, High-signal) | High-signal | no verdict supplied; suggest KEEP, re-scope | Its stated need ("arithmetic below it") is now met: ndarray #414 strided-A BF16, #418 prepacked INT8 B, `dt64-int8-facet-stride`, `int8-direct-packed-b`; first gate is open. (inferred) | Arcs 12, 13, 14 (synergy S6) |

Already closed (2): T-2026-10-23-1 (KILLED, #1731); ISS-ACTIVATION-IGNORES-APP-HALF (RESOLVED 2026-11-12).

---

## C. NEW LOOSE ENDS NOT ON THE BOARD

Checked against FRUITS.md and verdicts.md. Ledger = where to file. Sources cited as entry file or PR. "Close:" is the kill/close condition.

**Decisions and owner-blocked (mostly ISSUES)**

N1. Method identity: MethodSignature vs MethodRef vs MethodKey (EXO-50). Ledger ISSUES. Tier High-signal (unlocks N-items below). Step: owner picks one identity; then switch the probe owner to `dt64::dr`, finish `bind_dr64`, give the DR #4 router a `MethodRef`. Close: one type named, three entries' OPEN lines closed. Sources: `d-dt64-6-dr-addressing-module`, `d-xlog-native-5-bindstore-epi5`, `d-drca-3-dr4-method-routing` ("the method key is test-minted"), `d-dt64-6-bs-bind-store-crate`. Related: #1979 stamps MethodRef with issuer and digest (PR title only).

N2. D4 / S-INGRESS: DR to Lance `_rowid` authority, version-checked. Ledger ISSUES. Tier Heavy (owner ruling). Step: operator names the owner; then one witness binds a DR Frame to a Lance version. Close: `BLOCKED_D4` lines removed from the four entries. Sources: `d-global-where-p2` ("BLOCKED_D4. No witness binds a DR Frame to a Lance version"), `d-drisa-1-register-proof`, `d-driver-zc-0-virtual-bus-contract` (ZC-5), `d-dram-p3-consolidated` (D-DRAM-4), `abibatch-lane-socket`.

N3. Source-location receipt / lease host (EXO-46, EXO-89, EXO-151). Ledger ISSUES. Tier Heavy (owner). Step: choose a host outside the excluded lab crates. Close: D-DRAM-2, D-DRISA-5, D-DRIVER-ZC-3 gates open. Sources: `d-dram-1-census` ("D-DRAM-2 stays GATED"), `d-drisa-0-remote-register-contract`, `d-drisa-1-register-proof`, `d-driver-zc-0-virtual-bus-contract`.

N4. Reader Scope authority: no authority issues a reader `Scope`. Ledger ISSUES. Tier Heavy; closes with verdict row 33's council ruling. Step: attach SV-2, derived-plane and lanesock-4 blockers to row 33 as co-blockers. Sources: `dr-select-k-sv-2` ("Open (BLOCKED)"), `reader-scope-seam-inventory`, `abibatch-lane-socket`, #2030 sealed reader scope sketch (PR title).

N5. Generation identity owner (EXO-152), binding store (D-NFR S5), `Receipt` contract placement. Ledger ISSUES. Tier Heavy. Step: operator assigns the owner. Close: three names resolved. Sources: `d-tflp-65-landing`, `d-tflp-v2-synergy-revision`, `d-epi-cot-c3c-learn` ("production generation binding stays with EXO-152 PZL-4/G1"), `d-drca-2-dr256-where-dr3`.

N6. Detached `Keep` accepted under a new seal: the seal attests the caller's claim, not mask provenance. Ledger ISSUES. Tier High-signal. Step: pin a can-fire test that refuses it, or document the claim scope. Close: refusal test or ruling. Source: `d-drca-2-dr256-where-dr3` (## Open); `dt64-hotplug`/D-NFR S5 independent gate (as named there). Feeds row 34.

N7. Who issues `ExecPermit` / `DoGrant`. Ledger ISSUES. Tier Heavy. Step: name an issuer; wire to OGAR `CapabilityAuthority` (D6). Close: permit has an issuer in code. Sources: `r1-r2il-binding-reading` ("who issues ExecPermit; 0xC401 is provisional"), `x2-dt64-admission-policy` ("DoGrant has no issuer"), `d-dt64-6-bs-bind-store-crate` (CapabilityAuthority wiring D6).

N8. Does `admit` see the basis? (`admit` accepts a MethodRef without basis identity; guard in binding digest?). Ledger ISSUES. Tier High-signal. Step: decide, then flip the pinned finding `finding_admit_does_not_carry_the_basis_identity`. Close: finding becomes a refusal test. Sources: `dt64-l5a-exact-basis-witness` (L5c), `cg-beh-2` (EXO-155 owner question). #1979 may cover part (inferred).

N9. `MethodBinding.on` redundancy; effect class on the binding; call-parameter channel (EXO-291). Ledger ISSUES. Tier High-signal (small decision). Step: read OGAR #386 and #404 against the plan, then rule. Close: decision recorded. Source: `classid-on-do-call-shape` ("Open: whether MethodBinding.on is redundant ...").

N10. Production stamp gate and certification writer for epistemic state; rule identity of derivations. Ledger ISSUES. Tier Heavy. Step: scope from `ISS-NO-EVIDENCE-WRITER-FOR-EPISTEMIC-STATE` against #1936 writers. Close: gate named in code. Sources: `a10-epistemic-admission-and-nnue-compiler`, `a10-writers-retract-xlog-pr`.

N11. Thought-Git production gates G9-G12 and EXO-148. Ledger ISSUES. Tier Heavy. Step: list preconditions D-TGA-1/-3/-4 and D-TGB-2/-3/-5/-7 as one plan row. Close: gates have owners. Sources: `p4-operator-decisions-lifecycle-lab`, `p4-thought-git-codegraph-dt64-review`; lab steps shipped #1924-#1926.

N12. Schema wardrobe: `on_behalf` enforcement seam; descriptors in `collect_casts`; owner of a resident codegraph population (BLOCKED). Ledger ISSUES. Tier Heavy. Step: one seam PR after owner is named. Close: seam enforced and tested. Sources: `schema-wardrobe-p0`; ownership audit #2044 (PR title).

N13. EXO-337: a delete that does not advance the data version. Ledger ISSUES. Tier High-signal (one fix, correctness). Step: test that pins the stale-version read, then fix. Close: delete bumps version. Source: `d-regc-14-intake` (Open list).

N14. EXO-329: issuer of join positions (and D-REGC-5 / -7 dependency). Ledger ISSUES. Tier Heavy. Source: `d-regc-12-pg-row-polyfill`, `d-regc-bound-read-tiled-cube`. Unlocks D-REGC-7.

N15. OGAR mint of provisional concepts 0xC400-0xC402 (and 0xC401 provisional). Ledger ISSUES. Tier Heavy (OGAR owner). Step: mint together or split. Close: OGAR codebook entry + lance-graph mirror. Sources: `d-tract-codegraph-address-map` (## Open for DT64, Q1-Q2), `r1-r2il-binding-reading`. I found no OGAR PR title minting 0xC4xx (inferred).

**Probes and fixtures (IDEAS / TECH_DEBT)**

N16. Magic Sets: query-relative hazard measured; S6 fork rule conflict and decline fix; A10 encoding decision BLOCKED (raw5 = 0). Ledger ISSUES. Tier Heavy. Step: operator decides the fork-rule conflict, then the decline fix. Close: S6 runs. Sources: `sgr-recovery-gates`, `a10-probes-s6-magic-nnue-baselines`, `a10-writers-retract-xlog-pr` (xlog fork PR #7).

N17. Codegraph gap-map items G4, G7, G8 and cross-crate callee naming (`cg_lib::shapes::Impl::area`). Ledger IDEAS. Tier Heavy. Step: one reading of "where function realises word W" from a lens-declared table or Charon attribute, never a name match. Sources: `d-codegraph-0-gap-map`, `codegraph-review-operator-decisions` ("OPEN: where 'function realises word W' comes from"), `coresearch-mandatory-codegraph` (method body anchor).

N18. Codegraph intake gate: B3 interner and re-run of the failed 1 ms gate; Quack counting F7 behind codec9-column gap. Ledger TECH_DEBT. Tier Low-hanging (inferred scope: one PR). Step: interner PR, re-run the measurement in units under the rule. Close: gate passes in us, or the gate is rewritten by decision. Source: `codegraph-bin-reader-timing-gate`.

N19. Codegraph store extensions: def-use rows as a row kind (CG-3, EXO-235); per-configuration stores; targets and opaque modules recorded not declared. Ledger IDEAS. Tier Heavy. Sources: `cg5-defuse-probe`, `codegraph-recon-2`, `d-codegraph-n1-ruff-source-rows-open` ("events.tsv and receipt.json have no reader here").

N20. SAG-6-2 admission: F05 and F07 NOT RUN; four admission gates (symbol identity surviving a move, span join, F05/F07, owner approval for DT64/DR). Ledger IDEAS. Tier High-signal (one probe gates leaving the lab). Sources: `d-sag-6-2-lab-projection`, `d-sag-6-2-span-join-identity`, `d-sag-6-1-checklist`.

N21. D-EPI-COT C3c label: change to "the ablation discriminated" using ledger `sensitivity`. Ledger IDEAS. Tier Low-hanging (evidence exists). Close: test shows learned ordering trains on the new label. Sources: `d-epi-cot-c3c-learn`, `d-nsl-w0-w1` ("C3c's training label ... is still open").

N22. `CertificationModel` reports no effect size, sample size, uncertainty or multiple-comparison control; evidence horizon k undeclared. Ledger TECH_DEBT. Tier Low-hanging (field additions, evidence in `d-epi-cot-c3-warrant`). Sources: `d-epi-cot-c3-warrant`, `d-epi-cot-c0-c2`.

N23. NSL: plain HashMap + SoA control from H1 and B01 at Cora scale. Ledger IDEAS. Tier Low-hanging. Source: `d-nsl-w0-w1` (## Open).

N24. Palette512 x TurboVec9 x NSL work packages W1-W8 unrun. Ledger IDEAS. Tier Heavy. Source: `2026-10-29-palette512-turbovec9-nsl-worksteal` (ndarray #383/#384, turbovec #3/#4/#7).

N25. Density threshold between positions and mask (two density points only); storage-backed source; DuckDB adapter; whether a position-list consumer should exist. Ledger IDEAS. Tier Low-hanging for the sweep (one lab run), decision part Heavy. Sources: `dr-select-k-1-lab`, `selvec-compat-r1-falsifier`, `dr-select-k-sv-1`, `dr-select-k-1-correction`, `d-dr-select-k-1-inplace`.

N26. mask-risc parallel tiles: no second-host crossover measurement. Ledger IDEAS. Tier Low-hanging. Source: `exo-290-par-tiles-crossover` ("OPEN: no measurement on a second host").

N27. Derived-plane inheritance: no producer witness for lane contents; no frame carrier for a foreign population in a derived plane; plane identity has no socket. Ledger ISSUES. Tier High-signal (feeds row 34). Sources: `quack-derived-plane-p2-p4`, `wx-calibrated-plane-admission` (carrier for the floor stamp; DerivedPlane over source of arbitrary size).

N28. Weather probes: min/max over bit-planes; slab-stack; per-query bounds cost; 10^9-row throughput. Ledger IDEAS. Tier Low-hanging for min/max bit-descent (one probe), rest Heavy. Sources: `weather-bit-planes`, `weather-dr256-ternlog`.

N29. Bag as thinking object: operator questions D1-D4 and probes. Ledger IDEAS. Tier Heavy (needs ruling). Source: `coresearch-bag-object`.

N30. Foundry parity G1 remainder (P1b), G2 IAM in the read path, G3 overlay frame, G4 action gate, G5 bound relations; D-DRISA-M1 layout witness; Resolved-to-typed-vocabulary generator. Ledger IDEAS. Tier Heavy. Sources: `foundry-parity-1-lab`, `foundry-parity-1-arm-c`.

N31. Move per-field `addr` consumers to `LaneBinding::borrow_all`; ISA lab MKL comparator not executed. Ledger TECH_DEBT. Tier Low-hanging (a grep and mechanical edits; the grep must be followed by reading each call site, per FIRST-HAND-SOURCE-LAW). Sources: `lab-resolve-borrow-all`, `dt64-gemm-shape`, `dr-isa-stream-lab`.

N32. ndarray asks: bit-packed i1 x i1 -> i32 popcount word (D-DYNBAG-3); wide adder for U64x8 carry fold; repeat k-of-n on AVX2-only and NEON hosts. Ledger IDEAS. Tier Heavy. Sources: `dynbag-1-n-compose`, `d-dram-p3-5-kofn`.

N33. ndarray CI: wasm tiers of `tests/add_pm.rs` not run; P4 economics. Ledger TECH_DEBT. Tier Low-hanging. Source: `pm-ledger-abakus-and-2086-review-exception`.

N34. onebrc lane K: cold page cache, 10,000-station key set, more than 4 workers. Ledger IDEAS. Tier Low-hanging. Source: `onebrc-lane-k-codebook-mmap`.

N35. XLOG consumer half: fork routes `0xF100` aliases to `Refusal::Behaviour` (xlog#2); many-to-many join home (D-XLOG-NATIVE-3); MISSING-1..4 in owner crates; arrow major alignment or a facts-free xlog-core; CudaBuffer::View. Ledger IDEAS. Tier Heavy. Sources: `xlog-behaviour-lens-f100`, `d-xlog-dr64-0-xlog-rir-over-dr64-relcomp`, `lance-graph-xlog-crate`, `xlog-resident-hypercube-context`, `dynbag-0-xlog-through-quack-mask-risc`.

N36. XLOG W7-2 steps 4-5: bodies, reach, closure do not read through the socket; presence-writer lifetime (#1941 D3); guard constants tied to one frame's map. Ledger IDEAS. Tier Heavy. Sources: `d-xlog-w7-2-cube-lanes`, `d-xlog-w7-2-cube-planes`, `d-xlog-w7-2-method-over-cube-lanes`, `d-xlog-w7-2-method-is-a-called-do`, `d-xlog-w7-2-admission`.

N37. XLOG census: tiled path not exercised by any census row (fork builds single-tile cubes); remaining W7 refusals; 44 unavailable refused rules from G1b. Ledger IDEAS. Tier Low-hanging for the fork-route PR (xlog fork). Sources: `d-xlog-w5d-tiled-reach-w7-park`, `d-xlog-cov-2-census-007bb19`, `d-xlog-knowledge-g1b-reference`.

N38. D-REGC: presence 0 / value != 0 arm has no fixture row; ORDER BY; D-REGC-3, -4, -9, -11, -13 decisions; basis share of the census. Ledger TECH_DEBT (fixture) + ISSUES (decisions). Tier Low-hanging for the fixture. Sources: `d-regc-12-pg-row-polyfill`, `d-regc-14-intake`, `d-regc-bound-read-tiled-cube`.

N39. MeTTa/MORK W-1 (one cold symbol to DR id step) and W-2 (`Guards::check` needs a FrameStore). Ledger IDEAS. Tier Low-hanging (probe against existing oracle; joins verdict row 10). Source: `d-cognitive-polyfill-p0` (## Open (owner)).

N40. Grammar and GPW: D-GPW-1 census not run; four decisions in plan §5; spaCy front end not run against spaCy (EXO-345 scope). Ledger IDEAS. Tier Low-hanging for the census. Sources: `d-gpw-global-pattern-words`, `d-gpw-lance-graph-spacy-front-end`, `d-grammar-when-1`.

N41. Mistaber: production home for compose (blocks the facade); axes past 64; SupportLedger lineage; undeclared-world policy; D-ARG-1..3 order. Ledger IDEAS. Tier Heavy. Sources: `d-mist-native-0-kripke-fixpoint-over-dr64`, `coresearch-mistaber-hydration`.

N42. ONNX native: float reduction tolerance contract (D-ONNX-NATIVE-1); admitted f32 MatMul (first whole-model blocker); Gather reclassification; int64 shape-arithmetic semantics; ConstantOfShape/Expand registration. Ledger IDEAS. Tier Heavy. Sources: `d-tract-harvest-0-native-algebra-bridge`, `d-tract-shape-1-cold-shape-folding`, `d-tract-onnx-coverage-map`.

N43. D-LITHO Q1-Q4 (does the rod belong to DR's address or DT's field law?) against the 2026-11-12 ruling "DR routes, DT64 gives meaning". Ledger IDEAS. Tier Low-hanging (read-and-reconcile, docs; inferred that #2031 answers Q1). Sources: `d-litho-photolithography-contract`, #2031 (PR title).

N44. TD-SIMD-SWEEP-W2, TD-NDARRAY-PATCH-0_16, EXO-112 dependency shape (open on 2026-10-27). Ledger TECH_DEBT. Tier Low-hanging: verify whether #1985/#1986 float pins and #2028 (lance 14) closed them, then close or re-date. Source: `2026-10-27-harvest-integration-composite-key-and-simd-scope` (## OPEN). Not on FRUITS; the ledger row may exist without a Fruit line (unverified).

N45. DT64 persistence and restart: P5 binding across a restart; L5d reorder gap; MapTable frames not restored; stored selections (D7' / O4). Ledger IDEAS. Tier Low-hanging for P5 (probe pre-registered). Sources: `coresearch-drdt-split`, `2026-11-10-dt64-map-table-lance` ("Frames are not restored"), `maptable-versioned-local-map`.

N46. OGAR/native gated plans (EXO-326 literal ask: gate operand in the frozen programs); `hop`/`any` in Quack `Draft` (EXO-331 option 1). Ledger IDEAS. Tier Heavy. Sources: `exo-326-kleene-gate`, `d-dt64-reach-1`, `d-grammar-constraints-0`.

**Likely CLOSE candidates the arcs suggest (verify first; all inferred)**
- `ISS-REPORT-NO-COMPOSITE-KEY-GROUP-FOLD` (`2026-10-27-harvest-integration-...`): Cube column counts (#2082), join size (#2083), grouped count in Draft (#2130, R7) and factorized count (#2124) address it.
- `cg-beh-1` "PR #2001 and #2002 both add impact_report": closed by #2004.
- `d-xlog-cube-safe-0` P1: closed by #2087.
- D-INTAKE-2G/2H: probably repaired by #2156.
- Entry `dt64-l5b-historical-basis` "until L5a lands": stale, L5a is #1977.

---

## D. CONNECTING THE DOTS

S1. Witness family to ISS-ORDER-AND-CODEBOOK-WITNESS (row 34). Ends: #1977 exact predicate-basis witness; #1982 versioned source-local codes; `dr-select-k-sv-2` witness binding a selection to its lanes; `quack-derived-plane-p2-p4` "no producer witness for lane contents"; `d-drca-2` detached Keep under a new seal (N6); `dt64-l5a` L5c (N8); #1979 stamp. Why they connect: row 34 says seven open items wait on one witness nobody owns; the arcs built at least five per-site witnesses and each names the missing one. Joint step: a one-page census of witness kinds (who issues, what it binds, what version) before any contract type. Tier High-signal.

S2. Charon call facts to Kythe `ref/call` (row 11) and codegraph. Ends: `ruff_charon` per-row FunDeclId (ruff #133-#135); #2000 Codegraph over real Charon stores; `kythe-bridge-ra-facts` ("OPEN: ... call edges"); `d-codegraph-0-gap-map` cross-crate callee naming. Why: the Kythe row's blocker "no call-fact source" no longer holds (verdict row 11); the gap map names the one naming problem left. Joint step: emit Kythe `ref/call` from `rows.bin` for one crate, assert the cross-crate callee spelling. Tier Heavy (parts Low).

S3. Foundry G4 action gate to ISS-SIX-DECISIONS-ACTION-GATE (row 23). Ends: `foundry-parity-1-lab` G4 open; `action.rs:266 pub fn commit` fails open (row 23); OGAR #384 Validator and #397 applicability. Why: both are the same absent gate seen from a consumer lab and from the contract. Joint step: use the Foundry lab as the falsifier that closing commit fails closed. Tier High-signal.

S4. One reader-Scope ruling unblocks four ends: row 33 (copied scope id), `dr-select-k-sv-2` (BLOCKED), `abibatch-lane-socket` (no authoritative reader-scope witness), Foundry G2 IAM in the read path (`foundry-parity-1-lab`). Evidence: #2024/#2025 pin the defect; `reader-scope-seam-inventory` lists the seams. Joint step: bring all four to the council as one question. Tier Heavy.

S5. D4/S-INGRESS (N2) blocks DRISA-6, DRIVER-ZC-5, D-GLOBAL-WHERE P2, D-DRAM-4 and the storage-backed source in `dr-select-k-1-lab`. The durable cycle writer for the map table (`2026-11-10-dt64-map-table-lance.md`) shows one durable path exists for mappings, not for DR frames ("Frames are not restored"). Joint step: one ruling, then one witness (frame to Lance version). Tier Heavy.

S6. NAM-2 TFLite int8 leaf (row 39) gets its arithmetic from ndarray #414 (strided-A), #418 (prepacked INT8 B) and the lab arms `dt64-int8-facet-stride` and `int8-direct-packed-b`, with Codec8 admission (#1801) and Tract coverage (`d-tract-onnx-coverage-map`). Open on the other side: tolerance contract for float reductions (N42). Joint step: run "first gate if authorized" from the IDEAS row on the new INT8 path. Tier High-signal.

S7. Support masks (rows 2 and 8) and recursion count (row 9) get fixtures from the cube. `d-xlog-w7-bodies` answers cyclic bodies (`tri`, `cyc`, `clique5`) as count, a support mask per variable and edge-only witness rows; `DerivationLedger` (#1840, `support.rs:308-323`) pools root-disjoint support; `Cube::reach` (#2112) is existence only. Joint step: one fixture (diamond + cycle + bypass) compared three ways: mask, ledger, path oracle; then P-REC-NO-COUNT over `Cube::reach`. Tier Low-hanging (fixture) feeding High-signal rows.

S8. Retraction provenance (row 37): #1936 `pearl::retract`, #1914 resident recursive plasticity, and dir-sim stored deltas (`d-dirsim-reobserve`, `d-dirsim-delta-read-1`: "reading what changed between observations, from the stored deltas") give three partial retraction signals without a per-cell plane. Joint step: a fixture where a source is retracted and the dir-sim delta chain names the affected cells (inferred feasibility). Tier Heavy.

S9. XLOG cube lanes lent through the lane socket (#2136, #2137) are the missing input for D-LITHO Q1 and for the DR/DT split: the cube is a relation lent to a method executed as a called DO (`d-xlog-w7-2-method-is-a-called-do`). Same step fixes verdict row 16: checked i128 products (#2083) and checked i64 escape (#2092) now make the overflow refusal reachable by a test (inferred). Joint step: one test pinning `NativeRefusal::Count` with can-fire and stay-silent cases; reconcile D-LITHO Q1 with #2031. Tier Low-hanging.

S10. Method identity (N1) is the missing input of five open lines: `bind_dr64`, `d-xlog-native-5`, DR #4 router, X2 admission, OGAR #404 arm ledger. One ruling retires a column of OPEN lines across Arcs 1, 3 and 15. Tier High-signal.

S11. Frozen polyfill (rows 13/24): the D-DT64-AUTH census with two-way ratchet (#1927) counts hardcoded `compose_classid` lens mints; it is the guard a compile-time table at `0xFFFF :: 0xF000` needs ("D-DT64-7-3: a parity test against R2ILVocabulary", TD row 13). Joint step: land the table, run the ratchet, make `ogar-loco`/`ogar-r2il` optional behind `fold`. Tier High-signal.

S12. MeTTa (row 10) and global pattern words: `d-cognitive-polyfill-p0` W-1 (cold symbol to DR id) and `d-gpw-global-pattern-words` (absent-case policy, census) both need one identity step from a cold token to a register. (inferred) Joint step: run W-1 and D-GPW-1 against one shared token table. Tier Low-hanging.

Consumer synergies (report-rs). Rule for this block: report-rs is in production; each item is an observation plus a lance-graph-side next step. No report-rs change is proposed. These come from PR titles and the entries named; I did not read report-rs code.

S13. XVERWEIS on DT64 lanes (report-rs #175, #176) and the relation socket. Ends: report-rs pins XVERWEIS answers first (#175), then expresses R(x, y) as two aligned DT64 lanes (#176); lance-graph has `dr::PairPlane` (#2126), the relation socket (#2127), cube lanes lent through the lane socket (#2136, `d-xlog-w7-2-cube-lanes`), and `LaneBinding::borrow_all` (#1988). Why they connect: a relation held as two aligned lanes is the shape those four lance-graph PRs lend (inferred from titles). The open ends are the join-position issuer (N14, EXO-329 in `d-regc-12-pg-row-polyfill`) and the presence-writer lifetime (`d-xlog-w7-2-cube-planes`). Joint step (lance-graph side): record report-rs #176 as the first external consumer of the relation socket in `d-xlog-w7-2-relation-socket`'s follow-up, and ask whether EXO-329's issuer question is already exercised by it. Observation only. Tier High-signal.

S14. dir-sim and the licence report (report-rs #166, #120, #173). Ends: report-rs builds the licence report "over Graph and dir-sim" (#166) with a run receipt and Nachweise beside it (#120); lance-graph shipped licences as DR64 planes (#2105), licence-state planes through the DT64 lane and selection sockets (#2107), and the nextLink coverage fix (#2108). Why they connect: the run receipt is a consumer-side instance of verdict rows 3 and 5 (certificate / one replay checks proof, receipt, witness); dir-sim stored deltas (#2099, #2100) are the replayable record those rows want. Joint step: take the receipt fields named in #120 and the dir-sim delta chain and write down which of proof, receipt, witness each already carries (docs, lance-graph side). Tier High-signal (feeds rows 3+5).

S15. Reader scope and the report-rs role policy (report-rs #160, #171, #172, #177, #178). Ends: report-rs carries its own level / customer / service roles, group-granted roles and login-time role raising; lance-graph has no authority issuing a reader `Scope` (N4, `dr-select-k-sv-2`), and ISS-DT64-SCOPE-ID-IS-NOT-A-GRANT (row 33) says a copied scope id binds another reader's fields. #1970 is the shared test-only scope admission. Why they connect: a production consumer already models who may read which service, and the council deciding a sealed `Scope` has real role shapes to check against (inferred). Joint step (lance-graph side): list the role dimensions named in these report-rs PR titles as input cases for the council's scope design and for Foundry G2 (S4). Observation only. Tier Heavy.

S16. Mapping polyfill and persistence (report-rs #124, #133; lance-graph #2098, `MapTable` log #2150). Ends: report-rs is the stated consumer of D-MAPPING-POLYFILL-1; it has a mapping studio where an edited mapping must survive focus and preference changes (#133). On the lance-graph side the mapping is versioned (#2016 "version the mapping, not the content"), kept mappings live in the MapTable log (#2150), but "Frames are not restored" (`2026-11-10-dt64-map-table-lance.md`) and P5 "binding across a restart" is unrun (`coresearch-drdt-split`, N45). Why they connect: a consumer feature that needs an edited mapping to persist is the use case behind N45 (inferred). Joint step (lance-graph side): run P5 and state whether a restored map frame is needed for that case. Tier Low-hanging.

S17. Compiler-facts recovery of report-rs #33 (lance-graph #2002, `codegraph-recon-2`). Ends: lance-graph recovered report-rs #33 from Charon facts as its real-world case; open lines are per-configuration stores and recording targets and opaque modules in the store (N19). The same PR #33 appears in `coresearch-workbook-hypercube-soa-projection` where G1 GATHER-DECISION stays open (and row 38). Why they connect: one real PR is both the codegraph impact fixture and the gather-question corpus. Joint step: reuse the #33 fixture for the per-configuration store probe (N19) without touching report-rs (read-only recovery). Tier Low-hanging.

S18. Directory concepts and IAM lifecycle (report-rs #171, #177, #178; OGAR #370, #371; lance-graph #1959, #2108). Ends: report-rs previews inherited roles and JML over the directory; OGAR minted directory_user, directory_group, group_membership, license_a... (#370, title cut off) and dir-core keeps the Entra-to-EXO ordinal (#371); lance-graph mirrors directory concepts 0x0B06-0x0B09 (#1959). Why they connect: the consumer's group-granted roles are the first user of those minted concepts (inferred from titles). Joint step: confirm that `ogar_codebook` mirror and dir-sim bind-by-predicate (#2108) cover every directory property the report-rs titles use; list any gap as a mint request (OGAR side). Tier Low-hanging.

S19. EXO-263 counterfactual handoff (report-rs #115, #116, #117) and verdict row 27 (D-CGW-4 promotion gate). Ends: report-rs docs "land counterfactual revision plan" and "AC-1i DR512/Cube synergy and counterfactual handoff"; lance-graph has D-CFA docs (#1813-#1818) and `counterfactual_verdict` with no caller outside `pearl.rs`. Why they connect: the consumer names a counterfactual handoff the contract does not yet call. Joint step (lance-graph side): read #116 and record which lance-graph symbol it expects; if it is `counterfactual_verdict`, that is the first caller row 27 lacks (inferred). Tier High-signal.

---

## E. PROPOSED PRIORITY

Low-hanging (5):
1. Pin ONE reading for P-AGENT-VS-REPORT and re-check `root()` (verdict row 21 residual): the ruling exists (#1916), so this is a test edit.
2. Switch the D-EPI-COT C3c training label to "the ablation discriminated" (N21): the ledger `sensitivity` it needs already exists.
3. Pin `NativeRefusal::Count` (verdict row 16, S9): #2083 and #2092 made checked arithmetic reachable.
4. Fixture for the presence 0 / value != 0 arm in the PG row polyfill (N38): `d-regc-12` names it as the one missing row.
5. Expand TD-STATUS-BOARD rows to Oct+Nov plans (verdict row 19): November added D-INTAKE, D-REGC, D-DRDT-SPLIT, D-LANESOCK, all unrowed.

High-signal (5):
1. Method identity ruling (N1/S10): one decision unblocks about five open lines.
2. Witness census (S1, row 34): five per-site witnesses exist; they need to be seen together.
3. Merge rows 13+24 into one closing edit: compile-time `0xF000` table plus optional OGAR via `fold` (S11).
4. Support-mask fixture against `DerivationLedger::pooled` (S7, rows 2+8): the cube now gives the missing mask side.
5. D4/S-INGRESS ruling (N2/S5): blocks four plans.

Heavy (3):
1. Kythe `ref/call` from Charon rows (S2, row 11): source of call facts now exists.
2. Reader-Scope council ruling (S4, row 33): one question closes four blockers.
3. XLOG W7-2 steps 4-5 plus consumer half in the fork (N35/N36): the cube already lends lanes; terminals still do not read through the socket.

---

Cross-check of counts: arcs 17 (16 plus the report-rs consumer arc); revised rows 39 (38 from verdicts.md plus the row with no verdict); new loose ends 46 (N1-N46), with 5 close-candidates listed separately; synergies 19 (S1-S12 general, S13-S19 report-rs consumer observations). The report-rs import fact map (map-report-rs.md) was not available; Arc 17 and S13-S19 rest on PR titles only.
