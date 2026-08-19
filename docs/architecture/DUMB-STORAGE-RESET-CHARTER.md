# ARCHITECTURE RESET CHARTER — DUMB STORAGE × JAVA MECHANICAL API × HHTL EPISTEMIC SPINE (operator, 2026-08-19, verbatim)

> **READING NOTE — the name of the concept (operator, 2026-08-19, follow-up
> ruling "not literally dumb, not literally replace" + handle "A7A"):**
> "dumb"/"intentionally stupid" in the verbatim text below is shorthand for
> a posture, not the concept's name. Read it as **A7A** — the
> **agnostically-encoded hierarchical pattern** for separation of concerns:
> dependency-free mask-based addressing + parent-node HHTL for the
> CLAM/CHAODA trie. A reusable, reliable STORAGE CONCERN — it answers
> "where"/"which references", never "what" — built to be the floor for
> arbitrarily sophisticated patterns above it (Java ABI / Panama /
> Valhalla, Ontology, OpenStreetMap alike), with structural referencing as
> a globally available pattern. This note is interpretive only: no
> historical text is edited, no file is renamed, the verbatim block below
> stays byte-identical.
>
> Operator ruling, received 2026-08-19 ~14:05Z, immediately after merging
> PR #968 (the seal STORNO + finalization map + register-grid correction +
> 5+3-ratified spec). **This charter STOPS implementation of the
> freeze/seal-centered architecture** (task #25's W1–W4 do NOT launch on
> the #968 merge) while preserving that arc's research history and
> falsifiers by exact reference: merge commit `66fec27` (PR #968, head
> `88210f7`), `docs/lotus/CASCADE-ACCUMULATED-SEAL-SPEC.md`,
> `docs/lotus/SEAL-FINALIZATION-MAP.md`,
> `.claude/plans/cascade-seal-register-grid-v1.md` (RATIFIED v3),
> `crates/rp-seal-t0-probe/` (X-C2-1 harness + held scaffold),
> `docs/lotus/RP-SEAL-CONSOLIDATION-PASS1.md`.
>
> Everything below the rule is the operator's text, verbatim.

---

ARCHITECTURE RESET / PREPARATION ARC
DUMB STORAGE × JAVA MECHANICAL API × HHTL EPISTEMIC SPINE
THEN META-AWARENESS / EPISTEMIC CAUSALITY

STOP implementation of the old freeze/seal-centered architecture.

Preserve its research history and falsifiers by exact commit/path references.
Do not erase useful experiments.
But do not let obsolete freeze/batch-barrier assumptions become the new
production architecture.

This arc is SOURCE-AUDIT + ARCHITECTURE CONTRACT + INTEGRATION PLAN FIRST.

============================================================
0. THE NEW CENTRAL MODEL
============================================================

The substrate is intentionally stupid.

It must know only:

    stable references
    hierarchical reference relations
    ClassView / field projection vocabulary
    WideFieldMask ergonomics
    DatasetVersion
    temporal coordinates
    zero-copy / borrowed payload access
    scalable versioned parent/child traversal

It MUST NOT know:

    ontology semantics
    causal meaning
    Pearl rung meaning
    known-unknown semantics
    NARS
    AriGraph semantics
    orchestration policy
    "awareness"
    Java domain semantics
    OSM semantics

Those are interpretations ABOVE the substrate.

Canonical decomposition:

    REFERENCE
        exact identity

    HHTL
        hierarchical locality / addressing

    EXPLICIT HHTL NODE
        reference-set connective tissue

    CLASSVIEW
        interpretation of positions

    WIDEFIELDMASK
        foveated selection

    DATASETVERSION / TEMPORAL
        historical truth

Then ABOVE storage:

    ontology rails
    episodic witness
    epistemic causality
    rung decomposition
    meta-awareness
    orchestration nudges

============================================================
1. HHTL HAS TWO DOORS — KEEP BOTH
============================================================

Do not collapse these:

A. HHTL TRIE / NiblePath

    implicit routing structure
    16^n nibble address
    prefix / parent / common-prefix arithmetic
    cheap addressing
    identity/locality

B. EXPLICIT HHTL HIERARCHY NODES

    materialized REFERENCE NODES, not payload containers

    they carry:
        parent ref
        child/ref set
        projection/presence mask
        version coordinates
        optional small structural metadata

    they do NOT carry copied child payload.

The trie answers:

    WHERE DOES THIS BELONG?

The explicit node answers:

    WHAT LOCALITY / REFERENCE SET DOES THIS NODE REPRESENT?

Do not require the explicit hierarchy to use the same fanout as NiblePath.

Keep open:

    16^n routing
    4:1 Lotus-style locality pyramids
    two-nibble 16^2 = 256 attention tiles
    4096 → 1024 → 256 → 64 → 16 abstractions
    future physical/locality layouts

Hierarchy geometry is an accelerator, never semantic authority.

Falsifier:

    rebuilding / changing hierarchy geometry must not alter the exact result
    set obtained from canonical references.

============================================================
2. WIDEFIELDMASK IS THE FOVEA, NOT THE ADDRESS
============================================================

Do NOT redefine WideFieldMask as identity.

Keep:

    address/ref = where
    ClassView   = what positions mean
    WideFieldMask = which positions matter now

The current WideFieldMask field-position vocabulary naturally permits
a 256-position fovea.

Exploit that carefully.

An explicit HHTL node may expose child/reference positions THROUGH its
ClassView, allowing WideFieldMask to select up to a local 256-ish reference
surface without global pairwise search.

This is the desired attention economy:

    parent locality node
        ↓
    <= local bounded ref surface
        ↓
    WideFieldMask
        ↓
    selected child localities
        ↓
    descend only where necessary

NOT:

    256:256 × 256:256 Cartesian explosion.

============================================================
3. KEEP 256:256 POLYMORPHIC
============================================================

Do not globally assign one meaning to an 8:8 / 256:256 pair.

Keep all existing/future ClassView-elected readings open, including:

    field : field
    context : role
    part_of : is_a
    basin : relation
    area : location
    palette-x : palette-y
    palette : ranked-result
    episodic-witness basin : role
    other registered readings

One coordinate is an address atom.

Its semantics are elected by ClassView.

No byte-sniffing.
No storage-level ontology semantics.

============================================================
4. THE EPISTEMIC SPINE
============================================================

This is the conceptual reason for making hierarchy nodes explicit.

Ontology rails:

    part_of
    is_a

act as a sparse spine through the HHTL hierarchy.

Think "sparse RNA strand":

    parent node
        contains/inherits the connective tissue
        ↓
    child node
        references parent knowledge
        + adds local delta

If epistemic knowledge is proven about a parent locality, descendants do NOT
need to copy or rediscover it.

They may inherit the reference to that knowledge subject to:

    ClassView
    WideFieldMask projection
    TemporalPov / knowledge horizon
    epistemic rung
    local exceptions/deltas

This creates the desired bridge between:

    abstract ontology knowledge

and

    contextual occurrence knowledge

analogous to:

    lexical/global meaning
        +
    corpus/context occurrence

without conflating them.

A child's context may refine or contradict inherited knowledge.

Inheritance is NEVER:

    copy parent payload to every child.

It is:

    follow sparse parent epistemic reference
    + apply local delta.

============================================================
5. EXPLICIT HIERARCHY NODES AMORTIZE ATTENTION
============================================================

A parent is not merely an index node.

Above the storage layer, it may be INTERPRETED as an epistemic object.

Example interpretation:

    expected child/reference positions
    grounded positions
    unresolved expected positions
    contradictions
    supporting episodic witnesses

Then:

    known unknown
        = an expected epistemic position
          with no grounded filler at this horizon.

Do NOT store "unknown meaning" in storage.

Storage only stores the reference topology/masks/version.

The epistemic layer interprets an unresolved expected sibling as:

    I KNOW THAT THIS SLOT EXISTS
    I DO NOT KNOW ITS FILLER.

That unknown can itself become a reasoning target.

Higher-order cognition can therefore reason about:

    parent abstraction
    sibling relations
    unresolved sibling
    expected relation
    contradiction

without descending into every leaf.

============================================================
6. STORAGE PREPARATION ARC — BUILD THIS FIRST
============================================================

Before any epistemic-causality implementation, source-audit lance-graph and
produce the minimal storage contract needed for:

    versioned hierarchical reference nodes
    parent/child reference traversal
    ClassView projection
    WideFieldMask selection
    stable exact refs
    zero-copy payload ownership
    DatasetVersion publication
    temporal projection

Audit and reuse before minting anything:

    hhtl::NiblePath
    ClassView
    FieldMask
    WideFieldMask
    selection / NamedView / ViewRegistry
    standing_mask
    canonical_node / NodeRowPacket
    BatchWriter
    planner temporal.rs
    temporal_pov.rs
    existing Lance version APIs
    any current hierarchy/refset type
    any existing graph mask/native frontier representation

Do not create another graph model if these pieces compose.

Return a matrix:

    REQUIRED CAPABILITY
    EXISTING TYPE/API
    EXACT SOURCE
    ALREADY SUFFICIENT?
    MINIMAL GAP
    PROPOSED CHANGE
    FALSIFIER

============================================================
7. NO FREEZE / NO BATCH WALL
============================================================

This remains absolute.

BatchWriter initially stays AS-IS unless a source-proven defect requires a
change.

No architecture may require:

    freeze
    close-the-world
    wait-for-version
    writer-authorizes-next-thought
    fixed batch barrier
    reverse barrier on failure
    "16 outstanding means stop"

The live field continues evolving.

DatasetVersion is:

    a durable frontier through evolving references.

It is NOT:

    permission to compute the next state.

Async hot history may carry many generations while persistence catches up.

============================================================
8. TEMPORAL.RS IS A TOOL, NOT THE ORCHESTRATOR
============================================================

Current temporal.rs already correctly declares epistemology a query/planner
annotation rather than storage semantics.

Preserve that boundary.

Use temporal mechanisms for:

    historical coordinate
    knowable_from
    contemporary vs hindsight
    strict/aware/retro projection
    owner-local trajectory reconstruction
    later target × horizon distinction

Do NOT use temporal.rs as:

    scheduler
    backpressure mechanism
    global clock authority
    "may next think" gate

New APIs must avoid baking the current u64::MAX Strict-default sentinel into
new architecture.

Source-audit the registered target/horizon work before changing the canonical
temporal type.

============================================================
9. JAVA PROJECT — MECHANICAL MIRROR ONLY
============================================================

Audit AdaWorldAPI/lance-graph-java CURRENT MAIN first.

Do not re-propose already-shipped mask-native work.

The Java project already aims for:

    ordinary Java
        ↓
    typed semantic descriptors
        ↓
    View / Mask
        ↓
    Panama FFM
        ↓
    Rust mask-native data plane

Preserve this.

Java must see the SAME mechanical vocabulary:

    stable Ref / Node handle
    ClassView/View identity
    WideFieldMask
    hierarchy/root reference
    parent/children traversal
    DatasetVersion
    temporal point-of-view where required

Public Java API must remain boring.

No FFM types in public semantic signatures.
No arrays of 64K row ids as graph currency.
No Java object graph mirroring the Rust hierarchy.

Target ergonomics should feel approximately like:

    root/ref
        .view(...)
        .children(mask)
        .hop(...)
        .at(version)

but DO NOT freeze those method names before auditing current RowStore / Graph /
Mask APIs.

Prefer extending the shipped mask-native idiom.

============================================================
10. JAVA GENERICITY PROOFS
============================================================

Use at least TWO mechanically different consumers to prove the storage/Java
surface is not cognition-specific:

A. ontology read cache

    hierarchy = ontology / part_of / is_a
    masks = field/reference projection

B. OpenStreetMap / geographic graph

    hierarchy = geographic/locality reference structure
    masks = projection / area / route/ref interests

Both use the SAME substrate operations.

Neither requires the storage layer to know ontology or geography.

This is an important genericity falsifier.

============================================================
11. LATER ARC — EPISODIC VS EPISTEMIC CAUSALITY
============================================================

Prepare the seams now.
Do NOT implement the full cognitive layer in the storage arc.

Later distinguish explicitly:

EPISODIC WITNESS

    grounded historical evidence
    "this observation/event witnessed/supports this"

EPISTEMIC CAUSALITY

    what a reader can currently infer about:
        support
        contradiction
        missing support
        expected-but-unknown relation
        candidate explanation

Do not collapse them.

The existing CausalWitnessFacet remains pointer-like/contextual evidence.

Known unknowns are epistemic topology, not null payload.

============================================================
12. RUNG IS A SINGLE SHARED AXIS
============================================================

Do not create:

    one rung ladder for ontology
    another for causality
    another for orchestration
    another vertical copy in meta-awareness

One canonical rung/decomposition vocabulary.

The HHTL hierarchy answers locality/abstraction.

Rung answers:

    WHAT KIND / DEPTH OF EPISTEMIC OPERATION IS BEING PERFORMED?

Temporal answers:

    WHAT KNOWLEDGE MAY THIS READER ADMIT?

WideFieldMask answers:

    WHERE IS ATTENTION PROJECTED?

These axes compose.
They do not duplicate one another.

============================================================
13. META-AWARENESS = TOP-DOWN SELF-ORGANIZATION
============================================================

This is the future layer the storage interfaces must permit.

Meta-awareness observes abstract HHTL nodes first.

It should NOT scan 64K leaves to discover organization.

It sees:

    hierarchy locality
    active masks
    epistemic holes
    contradictions
    rung distribution
    local awareness horizons
    hot frontier
    durable frontier
    ontology activation
    incoming/outgoing intakes

Then it may descend selectively.

Think:

    HHTL top
        ↓
    abstract locality
        ↓
    foveated child mask
        ↓
    interesting region
        ↓
    leaf only if necessary

Top-down self-organization instead of chaotic flat alignment.

============================================================
14. META-AWARENESS FEEDBACK / NUDGES
============================================================

Meta-awareness must be able to feed small explicit NUDGES back into the
system.

Examples conceptually:

    catch-up knowledge horizon
    revisit known-unknown
    widen attention
    narrow attention
    descend hierarchy
    ascend to abstraction
    hydrate ontology context
    increase/decrease epistemic rung
    seek supporting witness
    resolve contradiction

Do NOT mint these exact enums until source audit proves where they belong.

The important contract is:

    nudge is versioned/replayable intent
    nudge may alter future attention/reasoning priority

BUT:

    nudge never becomes storage backpressure
    nudge never becomes "permission to think"

============================================================
15. CRITICAL DISTINCTION:
    EPISTEMIC PARTICIPATION != EXECUTION PERMISSION
============================================================

We DO want the following behavior:

    "your awareness timeline is behind;
     catch up before I count you in this current-horizon reasoning."

We DO NOT want:

    "your timeline is behind;
     stop computing/writing until storage catches up."

A stale node/SoA/intake may continue:

    computing
    casting
    writing
    evolving

But a higher-rung/current-horizon reasoning projection may mark its evidence:

    NOT CURRENTLY ADMISSIBLE TO THIS QUORUM

until its knowledge horizon catches up.

This is epistemic admission, not execution blocking.

This distinction is load-bearing.

Falsifier:

    node A horizon = V100
    orchestration current epistemic horizon = V116

    A may continue producing V101... independently.

    A's V100-conditioned observation must NOT silently vote as if it knew V116.

    After explicit catch-up / new AwarenessRef at V116,
    its contemporary evidence may participate.

No wall.
No time travel.
No fake consensus.

============================================================
16. META-AWARENESS SHOULD ITSELF BE REFERENCEABLE
============================================================

Prepare for second-order observation.

An awareness/meta-awareness record should be able to reference:

    object/locality
    ontology projection
    orchestration projection
    another awareness reference

without requiring new payload storage.

Second order:

    awareness-of-awareness

must use the same:

    reference
    hierarchy
    mask
    temporal
    version

substrate.

Do not invent a homunculus MetaAgent.

============================================================
17. STORAGE FEEDBACK REMAINS STUPID
============================================================

When meta-awareness emits a nudge or awareness update, storage sees only:

    reference record
    hierarchy target
    projection/mask ref
    temporal/version coordinates
    small opaque typed state if contractually required

Storage does NOT interpret:

    "catch up"
    "known unknown"
    "causal"
    "rung"

The cognitive/orchestration layer interprets those records.

This keeps Java/OSM/ontology consumers on the same substrate.

============================================================
18. REQUIRED FALSIFIERS
============================================================

Pre-register at least:

F-HIERARCHY-NOT-AUTHORITY
    changing hierarchy geometry does not change exact reference query result.

F-TRIE-VS-NODE
    implicit NiblePath routing and explicit hierarchy nodes remain separable.

F-WFM-FOVEA
    a 256-ish local projection does not trigger global pairwise enumeration.

F-SPARSE-INHERITANCE
    child epistemic interpretation can consume parent proof/reference
    without copying the parent's payload into every child.

F-CONTEXT-DELTA
    local/contextual evidence may refine/contradict inherited parent knowledge.

F-ONTOLOGY-READONLY
    attention/epistemic activation never mutates the ontology source/cache.

F-NO-FREEZE
    cognition continues while multiple DatasetVersions are outstanding.

F-NO-BACKPRESSURE-AUTHORITY
    persistence lag cannot become permission-to-think.

F-EPISTEMIC-PARTICIPATION
    stale-horizon evidence cannot masquerade as current-horizon evidence,
    while its producer continues running.

F-STRICT-HINDSIGHT
    later knowledge cannot appear in an earlier strict projection.

F-KNOWN-UNKNOWN
    an expected unresolved position exists without a future filler value.

F-META-SECOND-ORDER
    meta-awareness can reference another awareness object without payload copy.

F-JAVA-PARITY
    Java and Rust resolve identical refs/masks/hierarchy/version projections.

F-DOMAIN-GENERICITY
    ontology and OSM both consume the same hierarchical-reference substrate
    without domain semantics entering storage.

F-NO-64K-JAVA-OBJECTS
    Java hierarchy/attention remains mask/native-frontier based.

============================================================
19. DELIVERABLE ORDER
============================================================

DO NOT jump directly into the cognitive/meta layer.

Deliver in this order:

ARC A — SOURCE ARCHAEOLOGY

    lance-graph
    lance-graph-java
    exact capability map
    no implementation

ARC B — DUMB STORAGE CONTRACT

    minimum missing primitives only
    hierarchy refs
    masks
    versions
    temporal compatibility
    zero-copy ownership

ARC C — JAVA MECHANICAL INTEGRATION

    reuse current mask-native facade
    hierarchy/ref/version navigation
    ontology + OSM genericity proof

ARC D — EPISODIC / EPISTEMIC MODEL

    rails
    inherited epistemic spine
    known unknown siblings
    causality decomposition
    rung composition

ARC E — ORCHESTRATION META-AWARENESS

    top-down HHTL observation
    second-order awareness stacking
    versioned nudges
    epistemic participation/catch-up
    self-organizing attention

Do not mix ARC D/E semantics into ARC B storage types just because we know
they are coming.

============================================================
20. FIRST RESPONSE BEFORE ANY CODE
============================================================

Return:

1. CURRENT SOURCE MAP
   with exact file:line anchors in both repos.

2. ALREADY-SHIPPED / GAP MATRIX.

3. MINIMAL STORAGE PR SEQUENCE.

4. JAVA INTEGRATION PR SEQUENCE.

5. TYPES/APIS WE SHOULD REUSE UNCHANGED.

6. TYPES/APIS THAT NEED EXTENSION.

7. ANY PROPOSED NEW TYPE
   with a proof that an existing type cannot express it.

8. HOW THE CURRENT #968 / seal work changes under:
       no freeze
       hierarchy refs
       versioned frontier
   while preserving its research history by commit reference.

9. ONE architecture diagram showing:

       ontology/OSM/mechanical inputs
               ↓
       refs + ClassView + WideFieldMask
               ↓
       HHTL trie + explicit hierarchy nodes
               ↓
       versioned dumb substrate
               ↓
       Java mechanical facade

   and, clearly separated as FUTURE:

       episodic witness
               ↓
       epistemic causality / known unknown
               ↓
       rung
               ↓
       orchestration meta-awareness
               ↓
       top-down versioned nudges

10. RATIFICATION QUESTIONS ONLY where source leaves a genuine fork.

NO CODE until this map is returned.

============================================================
CANON
============================================================

THE STORAGE LAYER KNOWS REFERENCES, NOT MEANING.

HHTL MAKES LOCALITY EXPLICIT.

THE TRIE ROUTES.
THE HIERARCHY NODE REMEMBERS THE NEIGHBORHOOD.

WIDEFIELDMASK IS THE FOVEA.

CLASSVIEW GIVES THE FOVEA MEANING.

ONTOLOGY RAILS ARE A SPARSE EPISTEMIC SPINE.

PARENT KNOWLEDGE IS REFERENCED, NOT COPIED.

KNOWN UNKNOWNS ARE HOLES WITH ADDRESSES.

TEMPORAL.RS MAKES KNOWLEDGE HONEST.
IT DOES NOT SCHEDULE THOUGHT.

DATASETVERSION IS DURABLE HISTORY.
IT IS NEVER PERMISSION TO THINK.

A STALE OBSERVER MAY KEEP THINKING.
IT MAY NOT PRETEND IT HAS CAUGHT UP.

META-AWARENESS ORGANIZES TOP-DOWN.
IT NUDGES ATTENTION.
IT DOES NOT BUILD A WALL.
