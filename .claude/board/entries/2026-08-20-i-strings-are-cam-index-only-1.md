## 2026-08-20 — I-STRINGS-ARE-CAM-INDEX-ONLY-1 — strings in the hot path resolve through a codebook; the ONLY string home is the CAM index codebook; NEVER in any SoA

**Status:** OPERATOR RULING (verbatim: *"All strings in hot path are mandatory to
use codebooks, the only occurrence of strings is in content addressable memory
index codebook / Never in any SoA"*; sharpened same message: *"In hot path, only
patient data or KV Side Car table"*). **Confidence:** High — iron rule.

**The rule.** In the hot path a semantic identity is an ORDINAL. A `String` may
exist in exactly two places: (a) the **CAM index codebook** — the
content-addressed cold store / family codebook that maps bytes or a label to its
ordinal; (b) a **KV side-car table** or patient data. A string may NEVER live in
an SoA row, a value tenant, an overlay, or any reasoning-path struct.

**The sanctioned homes already exist — this rule is not a request for new
machinery, it is a demand to stop bypassing what is built:**
- `content_store.rs` — content-addressed cold text/blob store, `ContentAddress`
  = fnv1a-64 of the stored bytes, write side membrane-only. This IS the CAM index.
- `codebook.rs` — `family -> Codebook`, <=255 entries, 1-byte in-family index,
  index 0 reserved as the `EdgeBlock` empty-slot sentinel; a family that
  outgrows 255 SPLITS rather than widening the byte.

**Measured compliance in this repo (2026-08-20 audit).**

COMPLIANT, do not touch: `nars/belief.rs` carries ZERO strings — `CStmt { s: u16,
cop: Copula, p: u16 }`, `Copula::Rel(u16)`; `stance.rs:51 Interner
{ map: HashMap<String,u16>, names: Vec<String> }` is the correct string->ordinal
membrane feeding it; `recipes.rs` `code`/`name`/`substrate` are `&'static str`
catalogue metadata reached by `id: u8`, never used for dispatch or equality.

VIOLATIONS:
- `literal_graph.rs:72` `label_codebook: Vec<String>` — an ad-hoc codebook
  duplicating `codebook::Codebook`'s exact shape. Worse: `ensure_label` COMPUTES
  the ordinal on every `add_node`/`add_edge` and then DISCARDS it, keeping only
  the `String` on the node/edge.
- `literal_graph.rs:21,25,39,41,43` — `id`/`label`/`source`/`target` as `String`
  IDENTITY, plus three `HashMap<String,_>` adjacency indices.
- `exploration.rs:140-142` `FrontierEdge.source/target/label: String` — the LIVE
  ranking substrate (`curiosity()` + MUL-weighted next-edge choice).
- `exploration.rs:309-311` `ExplorationResult.confirmed/denied:
  Vec<(String,String,String,NarsTruth)>` — an SPO triple driving NARS revision
  as three heap Strings, in a repo whose NARS statement type is three `u16`s.
- Additional interners beside the sanctioned two: `deepnsm/vocabulary.rs:82`
  `forms: HashMap<String,u16>`; `lance-graph-cognitive/fabric/gel.rs:97`
  `labels: HashMap<String,u16>`.

**Predicate ordinals are TWO bytes, not one.** `ogar-loco/src/lib.rs:347` sets
`DOMAIN_FLOOR = 0x90`: `0x00..=0x8F` is universal ABI forever, `0x90..=0xFF` is
DOMAIN-LOCAL. `ogar-dismech` mints `0x90..=0xA2` (19, test-pinned) and `ogar-ro`
mints `0x90..=0xA5` (22) — the SAME range. A bare `FnIndex` is therefore
ambiguous; predicate identity is `(vocabulary, FnIndex)`, or the vocabulary is
implied by the lane's classid exactly as `ogar-obo` already does it.

**Correction recorded with the rule:** an audit pass this session reported
`ClassRowSchema`/`RowField`/`ValueSchema` as ABSENT. They exist —
`ogar-obo/src/layout.rs:36,61` and `lance_graph_contract::canonical_node::ValueSchema`.
What IS absent is narrower: a slot->ROLE mapping (subject/predicate/mediator).
`ClassRowSchema` carves FIELDS (`entity_type`, `edge_lanes`), not roles.

**Consequence for the reasoning overlay.** An overlay stores an ordinal, a slot
position and state bits — never a label, a relation name, or a path. Slot
position and schema ARE information; re-encoding them as prose is the redundancy
this rule exists to kill.

---

