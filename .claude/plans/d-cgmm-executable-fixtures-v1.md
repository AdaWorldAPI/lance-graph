# D-CGMM-FIXTURES-0: concrete independent source and semantic oracles

> **2026-11-04 · UNEXECUTED TEST SPECIFICATION.** These fixtures define independent test inputs and expected results for future, separate CGMM PRs. They do not claim shipped AST harvest, real SAP/Rails semantics or native runtime authorization. [Blueprint](d-cgmm-implementation-blueprint-v1.md) · [EXO ledger](d-cgmm-exo-gates-v1.md) · [46 higher-level falsifiers](d-cgmm-falsifier-matrix-v1.md) · [EXO-155](https://linear.app/exo-red/issue/EXO-155).

## 0. Oracle independence

Reference data is authored BEFORE implementations. A correct oracle enumerates static facts/values by stable producer/global ID and may not call `Codegraph::partners`, candidate demand planner, FieldView scratch or native prepared fold. An optional pinned Crepe/datafrog/Soufflé/MORK comparator is a **separate oracle** with semantic/licensing gates. Every deliberate disable must turn a nonvacuous result or refusal check red.

## F-A: Ruff/Charon/Kythe-shaped source fact and grade corpus

~~~text
repo = sandbox-codeapp; revision = R1; producer = fixture-v1
source node keys (stable strings scoped to repo/producer/rev):
  f=Controller::run  g=Service::read  h=Store::save
  i=DynImpl::process t=Trait::process k=Audit::record
facts, each with exact source span, revision and producer receipt:
  CALLS_RESOLVED(f,g)
  CALLS_RESOLVED(f,h)
  MAY_DISPATCH(f,t)
  IMPLEMENTS(i,t)
  UNRESOLVED_CALL(f,?dynamic)
  READS(g,amount)       # SYNTHETIC typed producer only; not shipped intake
  WRITES(h,status)      # SYNTHETIC typed producer only; not shipped intake
  effects(i)=UNKNOWN; extraction=PARTIAL
reference CALLEES(f):
  confirmed={g,h}, sites=2
  possible={i}
  unknown={?dynamic}
reference method purity:
  h is not pure (known write); i is ABSTAIN(UNKNOWN_EFFECTS);
  g pure only with COMPLETE certified read/write coverage
~~~

**Demand D1** only confirmed callees and source locations; **D2** additionally possible; **D3** additionally unknown. These MUST preserve grades, not promote MAY to confirmed. Baseline Codegraph still constructs six Cube relations at intake: measure, do not claim they vanished.

**Kills:** promote MAY to CALLS; omit WRITES while claiming COMPLETE; equate two same-named nodes from source revisions R1/R2; accept placeholder `?` as durable identity; refresh malformed source and partially mutate committed cube. Any missing writer effect column yields **NOT_RUN / ABSTAIN**, not an invented result.

## F-B: ClassView / FieldViewDef / RelAlg value corpus

One synthetic source-owned SoA: source global IDs T1..T6, a class basis S1 with `status` U32 at position 0 and `amount` I32 at position 1, and a separate 109-field class with a test field at **position 100**.

| Global ID | Tenant | status | amount | presence | status==2 AND amount>=100 |
| --- | --- | --- | --- | --- | --- |
| T1 | A | 2 | 120 | both present | TRUE |
| T2 | A | 2 | 90 | both present | FALSE |
| T3 | A | 1 | 150 | both present | FALSE |
| T4 | A | 2 | 101 | both present | TRUE |
| T5 | B | 0 | 0 | both present | FALSE |
| T6 | A | absent | 101 | amount only | UNKNOWN |

**Expected:** For admitted *two-valued* domain T1..T5, output mask `{T1,T4}` and COUNT=**2**. With T6 under explicit K3 semantics, today's R2IL native eligibility must **refuse** three-valued execution. Any SQL WHERE UNKNOWN filtering requires independent declared semantics and separate admission, not hidden coalescing.

**Kills:** swap field positions 0/1; same label from different ClassId; truncate position 100 to a 64-bit field mask or wrap position; try `lens_fields` under only the concept half; treat missing status as present zero; authorize using a stale schema basis. Scratch candidate masks are not field-presence bits. An alias/pointer check and materialization trap demonstrate no owned canonical status/amount vectors on admitted borrowed hot path.

## F-C: OGIT M:N, tenant scope, Set and Bag

Stable IDs:
- A: W1=1001, W2=1002; P1=1101, P2=1102.
- B: W3=2001; P3=2101.

~~~text
edge-instance rows:
e1=(W1,P1,tenant=A)
e2=(W1,P2,tenant=A)
e3=(W1,P2,tenant=A)  # distinct edge identity, duplicate pair
e4=(W2,P2,tenant=A)
e5=(W3,P3,tenant=B)
e6=(W2,P3,cross-tenant)  # physically present, unauthorized
OGIT proposed link: Worker --assigned_to--> Project
Cardinality::ManyToMany
Tenant policy: same-tenant endpoints allowed; cross-tenant denied
~~~

**Expected:** Tenant A Boolean Set of legal pairs has cardinality **3**, Bag count **4**. A+B legal Set **4**, Bag **5**. e6 never appears in admitted results. Forward direction does **not** automatically permit Project→Worker inverse; LinkSpec schema is not edge-instance identity or an execution permit.

**Kills:** disable tenant filter, deduplicate Bag count, join on unshared local ordinal, reverse edge direction, treat schema cardinality as actual pair population or enumerate cross-product. Each must be visibly wrong or explicitly refused.

## F-D: OGAR ActionDef, Odoo/SAP/Rails equivalence

~~~text
SYNTHETIC app profiles only, no asserted real ERP behavior:
Odoo ApproveEntry: condition=pending, ordered effects:
  (1) status=approved; (2) log approval timestamp
SAP ApproveEntry: condition=pending, ordered effects:
  (1) status=approved; (2) post accounting document;
  (3) log approval timestamp
Rails ApproveEntry: Ruby harvester evidence missing
~~~

**Expected:** Same concept, **different method effects**, so the Odoo/SAP actions must **not** share an admitted executable binding. Rails yields `ABSTAIN(MISSING_PRODUCER)`, never inferred purity. OGAR ActionDef is a declaration, not ActionInvocation; DT64 can store a guarded method but current R2IL must refuse its execution.

**Kills:** omit SAP posting, reorder effects, approve Rails without source, cache by common method name, collapse tenant identity. All must change the refusal/equivalence verdict.

## F-E: independently versioned mutations

~~~text
R1 → R2: source adds WRITE fact; human method labels unchanged
S1 → S2: T1.amount changes 120→80; DR identity mapping unchanged
M1 → M2: MethodBinding SigRev/ordered effects change; source static
P1 → P2: tenant-A execution permit revoked; class basis unchanged
L1 → L2: physical source SoA incarnation replaced; numeric version reused
scratch-cache eviction with no source/schema change
~~~

**Expected:** R2 invalidates fact/constraint scratch; S2 invalidates value-derived result mask, not necessarily DR map; M2 invalidates admitted method recipe; P2 invalidates authorization; L2 invalidates layout lease even with matching weak digest. Scratch eviction alone must not change truth. Reusing a method name or FNV digest alone is prohibited.

## 1. Fair benchmark arms

| Arm | Work included | Claim and counters |
| --- | --- | --- |
| B0 scalar oracle | Independent enumeration on static facts/value rows | Correctness only, separate time |
| B1 current Codegraph | Parse `.rows.bin`, intern positions, build 6 relations, run `partners/or_partners` | Cold ingestion + relation build, p50/p95/p99, bytes and binds |
| B2 demand-scoped lab | Same source/result/grades, prepare demand and query | Cold compile, demanded fact count, Q=1/2/8/16/64/256, cache and memory |
| B3 FieldView scratch | Candidate roles and logical constraints | OFFER/REFUSE/ABSTAIN, cold bytes/allocations |
| B4 owner-approved RelAlg/R2IL | Pure same-row query on borrowed SoA | Cold native prepare and hot repeated fold, no source-value copy |
| B5 witness-keyed memo | B4 with prepared recipe and frame/scope/source validation | Hit/miss/evict/invalidate, total query and permission checks |

Corpus sizes: 20 for correctness; 256 and 4,096 symbols for cost; 65,536 DR rows is a **separate** source-layout experiment, not comparable with 65,536 harvested AST symbols. Vary dynamic-call rate, sparse/dense fact edges, 1 vs 16 methods and Q independently. If B2 loses after including cold cost, report **NO BENEFIT**.

## 2. Stage acceptance gates

| Proposed PR | Positive | Deliberate disable |
| --- | --- | --- |
| CGMM-1 | Stable source identity + preserved grades/effect gaps | Delete/write wrong effect -> purity admission rejected |
| CGMM-2 | Confirmed/possible/unknown parity + explicit cold/hot economy | MAY promoted; stale demand cache |
| CGMM-3 | S1/wide ClassView exact, scratch eviction stable | Wrong full_classid/position -> refusal |
| CGMM-4 | Tenant A Set=3, Bag=4; legal tenant scope; distinct method effects | Tenant filter or action posting removed |
| CGMM-5 | COUNT=2 on T1..T5 and native read parity | Missing T6 coerced zero; foreign M:N "join" silently accepted |
| CGMM-6 | Owner-issued guard-free method exact native parity | Guarded/K3 or stale permit refuses |
| CGMM-7 | Valid recipe caching, no copied SoA value columns | Value/source/scope revised but cache reused |
| CGMM-8 | Independent oracle + scoped Codegraph caller coverage | Insert copied-value bypass; CI fails inside claimed roots |

**All gates NOT_RUN** in this documentation PR. Future PRs must report actual raw results, commit SHAs, source/owner decisions and EXO-50 handoff; no automatic production admission.
