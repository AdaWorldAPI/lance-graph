//! D-SAG-1: a software-awareness graph — the methods of a concept, their
//! compiled programs and the evidence about them, as typed graph rows queried
//! through Quack.
//!
//! A [`SoftwareGraph`] is a PROJECTION. It is rebuilt from a [`MethodTable`]
//! and the specs it was prepared from; it is never an authority the compiler
//! reads. Every edge says how it is known ([`Tier`]) and where it came from
//! ([`Provenance`]):
//!
//! | tier | written by | example |
//! |---|---|---|
//! | [`Tier::Declared`] | the spec as written | `billable_hours READS_FIELD duration` |
//! | [`Tier::Derived`] | source read at projection (claims, compute edges) | `duration DEPENDS_ON_CLAIM hypothesized` |
//! | [`Tier::Compiled`] | `MethodTable::prepare` and `lower` | `program USES_PRIMITIVE MaskedSumI32` |
//! | [`Tier::Verified`] | an executed oracle, under a [`Certificate`] | `billable_hours VERIFIED_BY …` |
//! | [`Tier::Hypothesis`] | a person or an analysis, unproven | `iam_reconcile MAY_REQUIRE FirstOfRun` |
//!
//! Validation is the compiler's: a refusal is the [`PrepareError`] the table
//! holds, recorded as a [`Rel::RefusedFor`] edge. The graph adds no rule that
//! could disagree with it.
//!
//! The edge table is a Quack table: [`SoftwareGraph`] implements [`Binder`],
//! so a dependency question is a `Draft` that binds, lowers and runs on
//! mask-risc like any other query (see [`SoftwareGraph::methods_using`]).
//! Nothing here runs on a method's warm path.
//!
//! Plan: `.claude/plans/software-awareness-graph-v1.md`.
use std::fmt::Write as _;

use lance_graph_contract::class_view::{compute_dag_is_acyclic, ComputeEdge};
use lance_graph_mask_risc::{
    execute_into, materialize_rows, words_for, ExecError, Foreign, LaneRef, Lowering, MaskOp,
    Operand, Planes, Pred, Program, Scratch, Terminal,
};
use lance_graph_quack::bind::{BindError, Binder, BoundField, Draft, FieldKind, TableId};
use lance_graph_quack::{lower, Agg, Col, LowerError, Mask, Query};

use crate::basis::{FieldMap, Grade};
use crate::method::{
    Cond, Fold, Law, MethodBody, MethodKey, MethodRef, MethodSpec, MethodTable, Order,
    PrepareError, ReadingId, SchemaSeal,
};

/// What a node is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    Class,
    Method,
    /// A canonical field of one concept under one schema seal.
    Field,
    /// One condition of a body, under its reading.
    Cond,
    /// One fold of a body, under its reading.
    Fold,
    Law,
    /// One lowered program of a prepared method.
    Program,
    /// A mask-risc op or terminal kind.
    Primitive,
    /// How mask-risc executes a program ([`Program::lowering`]).
    Lowering,
    /// A compiler refusal, by variant.
    Refusal,
    /// A claim grade on a field.
    Claim,
    /// A reusable operation, shipped or not.
    Capability,
    /// An executed check.
    Test,
}

/// What an edge says. Directions are fixed: `src REL dst`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rel {
    /// class → method.
    HasMethod,
    /// method → field.
    ReadsField,
    /// method → cond.
    Filters,
    /// method → fold.
    Folds,
    /// method → law.
    FinishedBy,
    /// method → method: a `Cond::Method` reference (G1).
    Calls,
    /// method → program.
    LowersTo,
    /// program → primitive (compiled), or method → primitive (a claim).
    UsesPrimitive,
    /// program → lowering.
    LowersAs,
    /// method or class → refusal.
    RefusedFor,
    /// field → claim.
    DependsOnClaim,
    /// field → field (`target` recomputed from `input`).
    ComputesFrom,
    /// method or refusal → capability: the capability would admit it.
    BlockedBy,
    /// method → capability, unproven.
    MayRequire,
    /// method → test, under a certificate.
    VerifiedBy,
}

/// How an edge is known.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Tier {
    Declared,
    Derived,
    Compiled,
    Verified,
    Hypothesis,
}

/// Where a projection's facts came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Provenance {
    /// The file the specs are declared in.
    pub source: String,
    /// The source revision (a commit, or `worktree`).
    pub rev: String,
    pub seal: SchemaSeal,
    pub reading: ReadingId,
}

/// One node: a kind and an identity string. Labels are display text and
/// never part of the identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    pub kind: Kind,
    pub ident: String,
    pub label: Option<String>,
}

/// One edge row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Edge {
    pub src: u32,
    pub rel: Rel,
    pub dst: u32,
    pub tier: Tier,
    /// The method (or class) whose projection wrote the edge: the hop a
    /// dependency query lands on.
    pub ctx: u32,
    pub prov: u32,
}

/// Proof that a method's current body ran against an oracle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Certificate {
    pub method: MethodKey,
    /// [`SoftwareGraph::digest`] of the body when the oracle ran.
    pub digest: u64,
    pub seal: SchemaSeal,
    pub test: String,
}

/// Why a certificate is not accepted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CertError {
    UnknownMethod(MethodKey),
    /// The body changed since the oracle ran.
    StaleBody {
        method: MethodKey,
    },
    /// The schema changed since the oracle ran.
    StaleSeal {
        method: MethodKey,
    },
}

/// A dependency query that did not run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryError {
    Bind(BindError),
    Lower(LowerError),
    Exec(ExecError),
}

/// Whether a reusable operation exists, from the graph's evidence alone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapStatus {
    /// A compiled program contains it.
    Shipped,
    /// Not shipped; a compiler refusal names it as the missing piece.
    Demonstrated {
        refusals: Vec<String>,
    },
    /// Not shipped; only declared or hypothesized needs.
    Claimed {
        by: Vec<String>,
    },
    Unknown,
}

/// A program that names a primitive its claim edges do not match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimMismatch {
    pub method: String,
    pub primitive: String,
}

/// A structural optimization candidate. Never a rewrite.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    pub kind: CandidateKind,
    /// The method nodes that would change.
    pub methods: Vec<String>,
    /// What must hold for the rewrite to be legal, each open or discharged.
    pub obligations: Vec<Obligation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CandidateKind {
    /// The same conditions in several methods: factor one predicate.
    SharedPredicate { conds: Vec<String> },
    /// One method's programs evaluate the same selection more than once.
    SharedSelection { programs: Vec<String> },
    /// A tiled program whose selection feeds a masked sum: H1, fuse the
    /// predicate into the fold.
    FusePredicateIntoSum { program: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Obligation {
    pub what: &'static str,
    /// `Some(reason)` when the graph's evidence discharges it.
    pub discharged: Option<String>,
}

/// The CPU a measurement ran on: the compiled target and the host feature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuTier {
    pub target: String,
    pub avx512f: bool,
}

/// One timing of one program on one dataset and tier.
#[derive(Debug, Clone, PartialEq)]
pub struct Measurement {
    pub program: String,
    pub rows: usize,
    pub tier: CpuTier,
    pub median_ns: f64,
}

/// A measurement refused for the tier it was asked about.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CostError {
    Unmeasured,
    TierMismatch { measured: CpuTier },
}

/// One workload's before/after timing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Workload {
    pub before_ns: f64,
    pub after_ns: f64,
}

/// What a set of workloads says about a transformation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Unmeasured,
    Improves,
    Regresses,
    /// Faster on some, slower on others: not an improvement.
    Mixed,
}

/// Faster everywhere, slower everywhere, or mixed. A tie is not faster.
#[must_use]
pub fn verdict(w: &[Workload]) -> Verdict {
    if w.is_empty() {
        return Verdict::Unmeasured;
    }
    let up = w.iter().filter(|x| x.after_ns < x.before_ns).count();
    let down = w.iter().filter(|x| x.after_ns > x.before_ns).count();
    match (up, down) {
        (_, 0) if up == w.len() => Verdict::Improves,
        (0, _) if down == w.len() => Verdict::Regresses,
        _ => Verdict::Mixed,
    }
}

/// The software-awareness graph.
#[derive(Debug, Default, Clone)]
pub struct SoftwareGraph {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    provs: Vec<Provenance>,
    /// `(method, body digest, seal)` of every projected spec.
    digests: Vec<(MethodKey, u64, SchemaSeal)>,
    measurements: Vec<Measurement>,
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

/// A node's kind, from its identity's prefix: one identity, one kind.
#[must_use]
pub fn kind_of(ident: &str) -> Option<Kind> {
    const PREFIXES: [(&str, Kind); 13] = [
        ("class:", Kind::Class),
        ("method:", Kind::Method),
        ("field:", Kind::Field),
        ("cond:", Kind::Cond),
        ("fold:", Kind::Fold),
        ("law:", Kind::Law),
        ("program:", Kind::Program),
        ("prim:", Kind::Primitive),
        ("lowering:", Kind::Lowering),
        ("refusal:", Kind::Refusal),
        ("claim:", Kind::Claim),
        ("cap:", Kind::Capability),
        ("test:", Kind::Test),
    ];
    PREFIXES
        .iter()
        .find(|(p, _)| ident.starts_with(p))
        .map(|(_, k)| *k)
}

/// The identity of a method node.
#[must_use]
pub fn method_ident(key: MethodKey) -> String {
    format!("method:{}", hex(&key.0))
}

/// The identity of a field node: concept, seal and canonical name. Two
/// concepts that share a field name have two field nodes.
#[must_use]
pub fn field_ident(concept: u16, seal: SchemaSeal, name: &str) -> String {
    format!("field:{concept:#06x}/{:016x}/{name}", seal.0)
}

/// The primitive kind of a mask op, by variant.
#[must_use]
pub fn op_name(op: &MaskOp) -> String {
    match op {
        MaskOp::Pred { pred, .. } => format!("Pred.{}", pred_name(pred)),
        MaskOp::And { .. } => "And".into(),
        MaskOp::Or { .. } => "Or".into(),
        MaskOp::Xor { .. } => "Xor".into(),
        MaskOp::AndNot { .. } => "AndNot".into(),
        MaskOp::Not { .. } => "Not".into(),
        MaskOp::Ternlog { .. } => "Ternlog".into(),
        MaskOp::Gather { .. } => "Gather".into(),
        MaskOp::GatherChain { .. } => "GatherChain".into(),
    }
}

fn pred_name(p: &Pred) -> &'static str {
    match p {
        Pred::GtI32 { .. } => "GtI32",
        Pred::LtI32 { .. } => "LtI32",
        Pred::GeI32 { .. } => "GeI32",
        Pred::LeI32 { .. } => "LeI32",
        Pred::EqI32 { .. } => "EqI32",
        Pred::NeI32 { .. } => "NeI32",
        Pred::EqU32 { .. } => "EqU32",
        Pred::NeU32 { .. } => "NeU32",
        Pred::MatchU32 { .. } => "MatchU32",
        Pred::MatchU64 { .. } => "MatchU64",
        Pred::EqU32Via { .. } => "EqU32Via",
        Pred::Range { .. } => "Range",
        Pred::EqU32Strided { .. } => "EqU32Strided",
        Pred::NeU32Strided { .. } => "NeU32Strided",
        Pred::MatchFacetStrided { .. } => "MatchFacetStrided",
        Pred::MatchFacet16Strided { .. } => "MatchFacet16Strided",
    }
}

/// The terminal kind, by variant.
#[must_use]
pub fn terminal_name(t: &Terminal) -> &'static str {
    match t {
        Terminal::Count { .. } => "Count",
        Terminal::Any { .. } => "Any",
        Terminal::All { .. } => "All",
        Terminal::MaskedSumI32 { .. } => "MaskedSumI32",
        Terminal::MaskedMinI32 { .. } => "MaskedMinI32",
        Terminal::MaskedMaxI32 { .. } => "MaskedMaxI32",
        Terminal::MaskedStridedGroupSum { .. } => "MaskedStridedGroupSum",
        Terminal::BlendI32 { .. } => "BlendI32",
        Terminal::Keep { .. } => "Keep",
        Terminal::ScatterOrU32 { .. } => "ScatterOrU32",
        Terminal::ScatterCountU32 { .. } => "ScatterCountU32",
        Terminal::CountKeyRunsU32 { .. } => "CountKeyRunsU32",
        Terminal::GroupSumI32 { .. } => "GroupSumI32",
        Terminal::GroupSumViaI32 { .. } => "GroupSumViaI32",
        Terminal::GroupReduce { .. } => "GroupReduce",
        Terminal::GroupPowerSumsI32 { .. } => "GroupPowerSumsI32",
        Terminal::GroupCrossPowerSumsI32 { .. } => "GroupCrossPowerSumsI32",
    }
}

fn lowering_name(l: Lowering) -> &'static str {
    match l {
        Lowering::Range(_) => "Range",
        Lowering::Ternlog(_) => "Ternlog",
        Lowering::TernlogKeep(_) => "TernlogKeep",
        Lowering::Tern2(_) => "Tern2",
        Lowering::Tern3(_) => "Tern3",
        Lowering::Tiled => "Tiled",
    }
}

fn refusal_name(e: &PrepareError) -> String {
    let d = format!("{e:?}");
    d.split(|c: char| !c.is_alphanumeric())
        .next()
        .unwrap_or_default()
        .to_owned()
}

/// FNV-1a over a spec's rendering. Stable within one build: a toolchain
/// change makes every certificate stale (never falsely fresh).
fn digest(spec: &MethodSpec<'_>) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in format!("{spec:?}").bytes() {
        h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

const EDGE_TABLE: &str = "edges";
const SRC: u16 = 0;
const REL: u16 = 1;
const DST: u16 = 2;
const TIER: u16 = 3;
const CTX: u16 = 4;

const RELS: [(Rel, &str); 15] = [
    (Rel::Calls, "calls"),
    (Rel::HasMethod, "has_method"),
    (Rel::ReadsField, "reads_field"),
    (Rel::Filters, "filters"),
    (Rel::Folds, "folds"),
    (Rel::FinishedBy, "finished_by"),
    (Rel::LowersTo, "lowers_to"),
    (Rel::UsesPrimitive, "uses_primitive"),
    (Rel::LowersAs, "lowers_as"),
    (Rel::RefusedFor, "refused_for"),
    (Rel::DependsOnClaim, "depends_on_claim"),
    (Rel::ComputesFrom, "computes_from"),
    (Rel::BlockedBy, "blocked_by"),
    (Rel::MayRequire, "may_require"),
    (Rel::VerifiedBy, "verified_by"),
];

const TIERS: [(Tier, &str); 5] = [
    (Tier::Declared, "declared"),
    (Tier::Derived, "derived"),
    (Tier::Compiled, "compiled"),
    (Tier::Verified, "verified"),
    (Tier::Hypothesis, "hypothesis"),
];

fn rel_code(r: Rel) -> u32 {
    RELS.iter()
        .position(|(x, _)| *x == r)
        .expect("every rel is listed") as u32
}

fn tier_code(t: Tier) -> u32 {
    TIERS
        .iter()
        .position(|(x, _)| *x == t)
        .expect("every tier is listed") as u32
}

impl SoftwareGraph {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    #[must_use]
    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    #[must_use]
    pub fn provenance(&self, edge: &Edge) -> &Provenance {
        &self.provs[edge.prov as usize]
    }

    /// The node with this identity.
    #[must_use]
    pub fn node(&self, ident: &str) -> Option<u32> {
        self.nodes
            .iter()
            .position(|n| n.ident == ident)
            .map(|i| i as u32)
    }

    fn intern(&mut self, kind: Kind, ident: String) -> u32 {
        debug_assert_eq!(kind_of(&ident), Some(kind), "{ident}");
        if let Some(i) = self.node(&ident) {
            return i;
        }
        self.nodes.push(Node {
            kind,
            ident,
            label: None,
        });
        (self.nodes.len() - 1) as u32
    }

    /// Add an edge unless the same `(src, rel, dst, tier)` is already there.
    pub fn link(&mut self, src: u32, rel: Rel, dst: u32, tier: Tier, ctx: u32, prov: u32) {
        let dup = self
            .edges
            .iter()
            .any(|e| (e.src, e.rel, e.dst, e.tier) == (src, rel, dst, tier));
        if !dup {
            self.edges.push(Edge {
                src,
                rel,
                dst,
                tier,
                ctx,
                prov,
            });
        }
    }

    /// The edges `src REL ?`, any tier.
    #[must_use]
    pub fn out(&self, src: &str, rel: Rel) -> Vec<(&str, Tier)> {
        let Some(s) = self.node(src) else {
            return Vec::new();
        };
        self.edges
            .iter()
            .filter(|e| e.src == s && e.rel == rel)
            .map(|e| (self.nodes[e.dst as usize].ident.as_str(), e.tier))
            .collect()
    }

    /// The current body digest of a projected method.
    #[must_use]
    pub fn digest(&self, key: MethodKey) -> Option<u64> {
        self.digests
            .iter()
            .rev()
            .find(|(k, ..)| *k == key)
            .map(|(_, d, _)| *d)
    }

    /// Project a prepared table and the specs it was prepared from.
    ///
    /// `claims` are the glove's claims, read so that a field's grade is a
    /// node (a method whose field is `Hypothesized` is refused, and the
    /// graph says why). A spec's identity facts are `Declared`, the claims
    /// `Derived`, the table's verdicts and programs `Compiled`.
    pub fn project(
        &mut self,
        table: &MethodTable,
        specs: &[(&str, MethodSpec<'_>)],
        claims: &[FieldMap],
        source: &str,
        rev: &str,
    ) {
        for (label, spec) in specs {
            let prov = self.provs.len() as u32;
            self.provs.push(Provenance {
                source: source.into(),
                rev: rev.into(),
                seal: spec.schema,
                reading: spec.reading,
            });
            self.digests.push((spec.key, digest(spec), spec.schema));
            let class = self.intern(Kind::Class, format!("class:{:#06x}", spec.concept));
            let m = self.intern(Kind::Method, method_ident(spec.key));
            // A method's projection replaces its earlier one: every edge
            // written within it (structure, programs, verdicts,
            // certificates, notes) belongs to the body it described.
            self.edges.retain(|e| e.ctx != m);
            self.nodes[m as usize].label = Some((*label).to_owned());
            self.link(class, Rel::HasMethod, m, Tier::Declared, m, prov);
            let r = spec.reading.0;
            let field = |g: &mut Self, name: &str| {
                let f = g.intern(Kind::Field, field_ident(spec.concept, spec.schema, name));
                g.link(m, Rel::ReadsField, f, Tier::Declared, m, prov);
                if let Some(c) = claims.iter().find(|c| c.canonical == name) {
                    let grade = match c.grade {
                        Grade::Exact => "exact",
                        Grade::Converted => "converted",
                        Grade::Hypothesized => "hypothesized",
                    };
                    let claim = g.intern(Kind::Claim, format!("claim:{}:{grade}", c.native));
                    g.link(f, Rel::DependsOnClaim, claim, Tier::Derived, m, prov);
                }
                g.nodes[f as usize].ident.clone()
            };
            let (filter, folds, law): (&[Cond<'_>], &[Fold<'_>], Option<Law<'_>>) = match spec.body
            {
                MethodBody::Query { filter, folds, law } => (filter, folds, Some(law)),
                MethodBody::Select { filter } => (filter, &[], None),
                MethodBody::Action { .. } => (&[], &[], None),
            };
            for c in filter {
                let (name, op, lit) = match *c {
                    Cond::Eq { field, literal } => (field, "=", literal),
                    Cond::Ne { field, literal } => (field, "!=", literal),
                    // A reference is a CALLS edge to the method it names, as
                    // the table resolves it; one that does not resolve has no
                    // edge, and the table's refusal says why.
                    Cond::Method(r) => {
                        let target = match r {
                            MethodRef::Key(k) => Some(k),
                            MethodRef::Label(l) => table.resolve(l),
                        };
                        if let Some(k) = target {
                            let t = self.intern(Kind::Method, method_ident(k));
                            self.link(m, Rel::Calls, t, Tier::Declared, m, prov);
                        }
                        continue;
                    }
                    // A hop's key and target field belong to the glove, not
                    // the spec: the condition is recorded as written.
                    Cond::Order {
                        field: name,
                        op,
                        value,
                    } => {
                        let op = match op {
                            Order::Lt => "<",
                            Order::Le => "<=",
                            Order::Gt => ">",
                            Order::Ge => ">=",
                        };
                        let f = field(self, name);
                        let n = self.intern(Kind::Cond, format!("cond:{f}{op}{value}@r{r}"));
                        self.link(m, Rel::Filters, n, Tier::Declared, m, prov);
                        continue;
                    }
                    Cond::Reaches { rel } => {
                        let n = self.intern(Kind::Cond, format!("cond:out({rel})@r{r}"));
                        self.link(m, Rel::Filters, n, Tier::Declared, m, prov);
                        continue;
                    }
                    Cond::Via {
                        rel,
                        field,
                        literal,
                        negate,
                    } => {
                        let op = if negate { "!=" } else { "=" };
                        let n = self.intern(
                            Kind::Cond,
                            format!("cond:out({rel}).{field}{op}{literal}@r{r}"),
                        );
                        self.link(m, Rel::Filters, n, Tier::Declared, m, prov);
                        continue;
                    }
                };
                let f = field(self, name);
                let n = self.intern(Kind::Cond, format!("cond:{f}{op}{lit}@r{r}"));
                self.link(m, Rel::Filters, n, Tier::Declared, m, prov);
            }
            for fold in folds {
                let ident = match *fold {
                    Fold::Count => "fold:count(*)".to_owned(),
                    Fold::Sum(f) => format!("fold:sum({})", field(self, f)),
                    Fold::Min(f) => format!("fold:min({})", field(self, f)),
                    Fold::Max(f) => format!("fold:max({})", field(self, f)),
                    Fold::GroupSum { key, val, groups } => format!(
                        "fold:group_sum({},{},{groups})",
                        field(self, key),
                        field(self, val)
                    ),
                    // A relation's key and target belong to the glove; the
                    // fold is recorded as written.
                    Fold::Distinct { rel } => format!("fold:distinct({rel})"),
                };
                let n = self.intern(Kind::Fold, format!("{ident}@r{r}"));
                self.link(m, Rel::Folds, n, Tier::Declared, m, prov);
            }
            if let Some(law) = law {
                let law = match law {
                    Law::Values => "law:values".to_owned(),
                    Law::NarsRevision {
                        independent,
                        scale,
                        horizon,
                    } => format!("law:nars_revision({independent},{scale},{horizon})"),
                };
                let n = self.intern(Kind::Law, law);
                self.link(m, Rel::FinishedBy, n, Tier::Declared, m, prov);
            }
            match table.get(spec.key) {
                Some(Ok(p)) => {
                    for (i, program) in p.programs().iter().enumerate() {
                        let pid = format!("program:{}#{i}", hex(&spec.key.0));
                        self.add_program(m, &pid, program, prov);
                    }
                }
                Some(Err(e)) => {
                    let n = self.intern(Kind::Refusal, format!("refusal:{}", refusal_name(e)));
                    self.link(m, Rel::RefusedFor, n, Tier::Compiled, m, prov);
                }
                None => {}
            }
        }
    }

    /// Record one lowered program of `owner`: its primitives and lowering,
    /// read from the program itself.
    pub fn add_program(&mut self, owner: u32, ident: &str, program: &Program, prov: u32) {
        let p = self.intern(Kind::Program, ident.to_owned());
        self.link(owner, Rel::LowersTo, p, Tier::Compiled, owner, prov);
        let names = program
            .ops
            .iter()
            .map(op_name)
            .chain([terminal_name(&program.terminal).to_owned()]);
        for name in names {
            let n = self.intern(Kind::Primitive, format!("prim:{name}"));
            self.link(p, Rel::UsesPrimitive, n, Tier::Compiled, owner, prov);
        }
        let l = self.intern(
            Kind::Lowering,
            format!("lowering:{}", lowering_name(program.lowering())),
        );
        self.link(p, Rel::LowersAs, l, Tier::Compiled, owner, prov);
    }

    /// A provenance row for facts not projected from a table.
    pub fn note(&mut self, source: &str, rev: &str) -> u32 {
        self.provs.push(Provenance {
            source: source.into(),
            rev: rev.into(),
            seal: SchemaSeal(0),
            reading: ReadingId(0),
        });
        (self.provs.len() - 1) as u32
    }

    /// Record a need, or a claim, by identity: `src REL target` at `tier`.
    /// Each node's kind is read from its identity's prefix, so a noted
    /// `refusal:` is the same refusal node the compiler writes, never a
    /// second node of another kind. Recording never ships anything (see
    /// [`Self::capability`]).
    ///
    /// # Errors
    ///
    /// A compiled or verified tier, or an identity whose prefix names no
    /// kind.
    pub fn claim(
        &mut self,
        src: &str,
        rel: Rel,
        target: &str,
        tier: Tier,
        prov: u32,
    ) -> Result<(), String> {
        // Compiled and verified edges are written by `project`, `add_program`,
        // `project_compute`, `blocked_by` and `certify`, never by a note.
        if matches!(tier, Tier::Compiled | Tier::Verified) {
            return Err(format!(
                "{tier:?} is written by the compiler or a certificate"
            ));
        }
        let m = self.intern_ident(src)?;
        let c = self.intern_ident(target)?;
        self.link(m, rel, c, tier, m, prov);
        Ok(())
    }

    fn intern_ident(&mut self, ident: &str) -> Result<u32, String> {
        let kind = kind_of(ident).ok_or_else(|| ident.to_owned())?;
        Ok(self.intern(kind, ident.to_owned()))
    }

    /// A method node by identity, for programs not prepared by a table.
    pub fn add_method(&mut self, ident: &str) -> u32 {
        self.intern(Kind::Method, ident.to_owned())
    }

    /// Record that `capability` would admit what a compiler refusal
    /// rejects. Only a refusal the compiler actually recorded (a compiled
    /// `RefusedFor` edge) can be named: a gap is demonstrated by a refusal,
    /// never by a note.
    ///
    /// # Errors
    ///
    /// The refusal identity, when no compiled refusal carries it.
    pub fn blocked_by(&mut self, refusal: &str, capability: &str, prov: u32) -> Result<(), String> {
        let r = self
            .node(refusal)
            .filter(|&r| {
                self.edges
                    .iter()
                    .any(|e| e.dst == r && e.rel == Rel::RefusedFor && e.tier == Tier::Compiled)
            })
            .ok_or_else(|| refusal.to_owned())?;
        let c = self.intern(Kind::Capability, capability.to_owned());
        self.link(r, Rel::BlockedBy, c, Tier::Compiled, r, prov);
        Ok(())
    }

    /// Project a class's compute DAG and record the contract's verdict.
    ///
    /// The cycle check is `class_view::compute_dag_is_acyclic`, run on the
    /// declaration; the graph only records what it said.
    pub fn project_compute(
        &mut self,
        concept: u16,
        seal: SchemaSeal,
        basis: &[String],
        edges: &[ComputeEdge],
        prov: u32,
    ) {
        let class = self.intern(Kind::Class, format!("class:{concept:#06x}"));
        let name = |p: u8| {
            basis
                .get(usize::from(p))
                .cloned()
                .unwrap_or(format!("#{p}"))
        };
        for e in edges {
            let t = self.intern(Kind::Field, field_ident(concept, seal, &name(e.target)));
            for &i in e.inputs {
                let f = self.intern(Kind::Field, field_ident(concept, seal, &name(i)));
                self.link(t, Rel::ComputesFrom, f, Tier::Derived, class, prov);
            }
        }
        if !compute_dag_is_acyclic(edges) {
            let r = self.intern(Kind::Refusal, "refusal:ComputeCycle".into());
            self.link(class, Rel::RefusedFor, r, Tier::Compiled, class, prov);
        }
    }

    /// Accept a certificate when the body and seal it names are current.
    ///
    /// # Errors
    ///
    /// [`CertError`] when the method is unknown or changed since.
    pub fn certify(&mut self, cert: &Certificate, prov: u32) -> Result<(), CertError> {
        let Some(&(_, d, s)) = self.digests.iter().rev().find(|(k, ..)| *k == cert.method) else {
            return Err(CertError::UnknownMethod(cert.method));
        };
        if d != cert.digest {
            return Err(CertError::StaleBody {
                method: cert.method,
            });
        }
        if s != cert.seal {
            return Err(CertError::StaleSeal {
                method: cert.method,
            });
        }
        let m = self.intern(Kind::Method, method_ident(cert.method));
        let t = self.intern(Kind::Test, format!("test:{}", cert.test));
        self.link(m, Rel::VerifiedBy, t, Tier::Verified, m, prov);
        Ok(())
    }

    /// Record a measurement.
    pub fn measure(&mut self, m: Measurement) {
        self.measurements.push(m);
    }

    /// The measured cost of `program` on `tier`.
    ///
    /// # Errors
    ///
    /// [`CostError::TierMismatch`] when it was measured only on another
    /// tier; a number from one CPU is never reported for another.
    pub fn cost(&self, program: &str, rows: usize, tier: &CpuTier) -> Result<f64, CostError> {
        let mut other = None;
        for m in self
            .measurements
            .iter()
            .filter(|m| m.program == program && m.rows == rows)
        {
            if &m.tier == tier {
                return Ok(m.median_ns);
            }
            other = Some(m.tier.clone());
        }
        Err(
            other.map_or(CostError::Unmeasured, |measured| CostError::TierMismatch {
                measured,
            }),
        )
    }

    // ---- queries over the edge table, through Quack -------------------

    fn columns(&self) -> ([Vec<u32>; 5], Vec<u64>) {
        let n = self.edges.len();
        let mut cols: [Vec<u32>; 5] = Default::default();
        for e in &self.edges {
            cols[SRC as usize].push(e.src);
            cols[REL as usize].push(rel_code(e.rel));
            cols[DST as usize].push(e.dst);
            cols[TIER as usize].push(tier_code(e.tier));
            cols[CTX as usize].push(e.ctx);
        }
        let mut live = vec![u64::MAX; words_for(n)];
        if !n.is_multiple_of(64) {
            if let Some(last) = live.last_mut() {
                *last = (1u64 << (n % 64)) - 1;
            }
        }
        (cols, live)
    }

    /// Run `draft` over the edge table and hop each surviving edge to its
    /// `ctx` node: one Quack program, `ScatterOrU32` as the demanded result.
    ///
    /// # Errors
    ///
    /// [`QueryError`] from binding, lowering or execution.
    pub fn hop_to_ctx(&self, draft: &Draft) -> Result<Vec<&str>, QueryError> {
        let bound = draft.bind(self).map_err(QueryError::Bind)?;
        let q = Query {
            filter: bound.filter,
            agg: Agg::ScatterOrU32 {
                fk: Col(CTX),
                out_rows: self.nodes.len() as u32,
            },
        };
        let program = lower(&q).map_err(QueryError::Lower)?;
        let (cols, live) = self.columns();
        let lanes: Vec<LaneRef<'_>> = cols.iter().map(|c| LaneRef::U32(c)).collect();
        let planes = Planes {
            n_rows: self.edges.len(),
            masks: &[live.as_slice()],
            lanes: &lanes,
        };
        let mut scratch =
            Scratch::for_program(&program, planes.n_rows).map_err(QueryError::Exec)?;
        let mut out = vec![0u64; words_for(self.nodes.len())];
        execute_into(
            &program,
            &planes,
            &Foreign::NONE,
            &mut scratch,
            lance_graph_mask_risc::Out::Mask(&mut out),
        )
        .map_err(QueryError::Exec)?;
        Ok(materialize_rows(&out, self.nodes.len())
            .into_iter()
            .map(|i| self.nodes[i].ident.as_str())
            .collect())
    }

    /// Query A: the methods whose compiled programs use `primitive`.
    ///
    /// # Errors
    ///
    /// [`QueryError`]; an unknown primitive is `BindError::UnknownValue`.
    pub fn methods_using(&self, primitive: &str) -> Result<Vec<&str>, QueryError> {
        self.hop_to_ctx(
            &lance_graph_quack::bind::table(EDGE_TABLE)
                .where_eq("rel", "uses_primitive")
                .where_eq("tier", "compiled")
                .where_eq("dst", format!("prim:{primitive}").as_str()),
        )
    }

    /// Query B: the methods (or classes) the compiler refused, for `reason`.
    ///
    /// # Errors
    ///
    /// As [`Self::hop_to_ctx`].
    pub fn refused_for(&self, reason: &str) -> Result<Vec<&str>, QueryError> {
        self.hop_to_ctx(
            &lance_graph_quack::bind::table(EDGE_TABLE)
                .where_eq("rel", "refused_for")
                .where_eq("dst", format!("refusal:{reason}").as_str()),
        )
    }

    /// Falsifier 8: every primitive a method is CLAIMED to use (any tier but
    /// compiled) must appear in one of its compiled programs.
    #[must_use]
    pub fn primitive_claim_mismatches(&self) -> Vec<ClaimMismatch> {
        self.edges
            .iter()
            .filter(|e| e.rel == Rel::UsesPrimitive && e.tier != Tier::Compiled)
            .filter(|claim| {
                !self.edges.iter().any(|c| {
                    c.rel == Rel::UsesPrimitive
                        && c.tier == Tier::Compiled
                        && c.ctx == claim.src
                        && c.dst == claim.dst
                })
            })
            .map(|e| ClaimMismatch {
                method: self.nodes[e.src as usize].ident.clone(),
                primitive: self.nodes[e.dst as usize].ident.clone(),
            })
            .collect()
    }

    /// Falsifier 9: a capability is shipped only when a COMPILED program
    /// uses it; a declared need never ships it.
    #[must_use]
    pub fn capability(&self, ident: &str) -> CapStatus {
        let Some(c) = self.node(ident) else {
            return CapStatus::Unknown;
        };
        let into = |p: &dyn Fn(&Edge) -> bool| -> Vec<String> {
            self.edges
                .iter()
                .filter(|e| e.dst == c && p(e))
                .map(|e| self.nodes[e.src as usize].ident.clone())
                .collect()
        };
        if !into(&|e| e.rel == Rel::UsesPrimitive && e.tier == Tier::Compiled).is_empty() {
            return CapStatus::Shipped;
        }
        let refusals =
            into(&|e| e.rel == Rel::BlockedBy && self.nodes[e.src as usize].kind == Kind::Refusal);
        if !refusals.is_empty() {
            return CapStatus::Demonstrated { refusals };
        }
        let by = into(&|_| true);
        if by.is_empty() {
            CapStatus::Unknown
        } else {
            CapStatus::Claimed { by }
        }
    }

    fn methods(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.nodes.len() as u32).filter(|&i| self.nodes[i as usize].kind == Kind::Method)
    }

    fn targets(&self, src: u32, rel: Rel) -> Vec<u32> {
        let mut v: Vec<u32> = self
            .edges
            .iter()
            .filter(|e| e.src == src && e.rel == rel)
            .map(|e| e.dst)
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Query D (a): condition sets shared by two or more prepared methods.
    /// Conditions are compared by identity (field under concept and seal,
    /// operator, literal, reading), never by label.
    #[must_use]
    pub fn shared_predicates(&self) -> Vec<Candidate> {
        let prepared: Vec<u32> = self
            .methods()
            .filter(|&m| !self.targets(m, Rel::LowersTo).is_empty())
            .collect();
        let mut out: Vec<Candidate> = Vec::new();
        for (i, &a) in prepared.iter().enumerate() {
            let ca = self.targets(a, Rel::Filters);
            if ca.is_empty() {
                continue;
            }
            let mut group = vec![a];
            for &b in &prepared[i + 1..] {
                if self.targets(b, Rel::Filters) == ca {
                    group.push(b);
                }
            }
            let conds: Vec<String> = ca
                .iter()
                .map(|&c| self.nodes[c as usize].ident.clone())
                .collect();
            let seen = out.iter().any(
                |c| matches!(&c.kind, CandidateKind::SharedPredicate { conds: x } if *x == conds),
            );
            if group.len() > 1 && !seen {
                out.push(Candidate {
                    kind: CandidateKind::SharedPredicate { conds },
                    methods: group
                        .iter()
                        .map(|&m| self.nodes[m as usize].ident.clone())
                        .collect(),
                    obligations: vec![
                        Obligation {
                            what: "same concept, seal, reading and effect",
                            discharged: Some(
                                "condition identities include concept, seal and reading".into(),
                            ),
                        },
                        Obligation {
                            what: "the predicate method lowers to the same program",
                            discharged: None,
                        },
                    ],
                });
            }
        }
        out
    }

    /// Query D (b): a method whose programs carry identical selection ops,
    /// i.e. the same population scanned once per fold.
    #[must_use]
    pub fn shared_selections(&self, table: &MethodTable) -> Vec<Candidate> {
        let mut out = Vec::new();
        for m in self.methods() {
            let ident = &self.nodes[m as usize].ident;
            let Some(key) = parse_key(ident) else {
                continue;
            };
            let Some(Ok(p)) = table.get(key) else {
                continue;
            };
            let progs = p.programs();
            if progs.len() > 1 && progs.iter().all(|x| x.ops == progs[0].ops) {
                out.push(Candidate {
                    kind: CandidateKind::SharedSelection {
                        programs: (0..progs.len())
                            .map(|i| format!("program:{}#{i}", hex(&key.0)))
                            .collect(),
                    },
                    methods: vec![ident.clone()],
                    obligations: vec![Obligation {
                        what: "one mask feeding several terminals is a population \
                               intermediate the executor forbids; needs a multi-terminal \
                               program, not a shared mask",
                        discharged: None,
                    }],
                });
            }
        }
        out
    }

    /// H1: tiled programs whose selection feeds a `MaskedSumI32`. Each
    /// candidate lists what fusing the predicate into the sum must keep.
    #[must_use]
    pub fn fusion_candidates(&self, programs: &[(String, &Program)]) -> Vec<Candidate> {
        programs
            .iter()
            .filter(|(_, p)| {
                matches!(p.terminal, Terminal::MaskedSumI32 { .. })
                    && p.lowering() == Lowering::Tiled
            })
            .map(|(id, p)| {
                let gates = gate_planes(p);
                let owner = self
                    .node(id)
                    .and_then(|pid| {
                        self.edges
                            .iter()
                            .find(|e| e.dst == pid && e.rel == Rel::LowersTo)
                    })
                    .map(|e| self.nodes[e.src as usize].ident.clone());
                Candidate {
                    kind: CandidateKind::FusePredicateIntoSum {
                        program: id.clone(),
                    },
                    methods: owner.into_iter().collect(),
                    obligations: vec![
                        Obligation {
                            what: "NULL/UNKNOWN: every validity plane the program ANDs must \
                                   still gate the fused sum",
                            discharged: (gates.len() <= 1).then(|| {
                                "the program reads one plane (live); no validity gate".into()
                            }),
                        },
                        Obligation {
                            what: "row bound: at most MASKED_SUM_I32_MAX_ROWS rows",
                            discharged: None,
                        },
                        Obligation {
                            what: "the fused terminal must exist: FusedFold has only Count/Any",
                            discharged: None,
                        },
                    ],
                }
            })
            .collect()
    }
}

/// The resident planes a program reads as gates.
fn gate_planes(p: &Program) -> Vec<u16> {
    let mut v = Vec::new();
    let mut see = |o: Operand| {
        if let Operand::Plane(k) = o {
            if !v.contains(&k) {
                v.push(k);
            }
        }
    };
    for op in &p.ops {
        match *op {
            MaskOp::Pred { under, .. }
            | MaskOp::Gather { under, .. }
            | MaskOp::GatherChain { under, .. } => {
                if let Some(u) = under {
                    see(u);
                }
            }
            MaskOp::And { a, b, .. }
            | MaskOp::Or { a, b, .. }
            | MaskOp::Xor { a, b, .. }
            | MaskOp::AndNot { a, b, .. } => {
                see(a);
                see(b);
            }
            MaskOp::Not { a, .. } => see(a),
            MaskOp::Ternlog { a, b, c, .. } => {
                see(a);
                see(b);
                see(c);
            }
        }
    }
    v
}

fn parse_key(ident: &str) -> Option<MethodKey> {
    let h = ident.strip_prefix("method:")?;
    if h.len() != 32 {
        return None;
    }
    let mut k = [0u8; 16];
    for (i, b) in k.iter_mut().enumerate() {
        *b = u8::from_str_radix(&h[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(MethodKey(k))
}

/// The edge table as a Quack table: `edges(src, rel, dst, tier, ctx)`, every
/// field a code. Node identities, relation and tier names are the codebook;
/// nothing is minted.
impl Binder for SoftwareGraph {
    fn table(&self, name: &str) -> Option<TableId> {
        (name == EDGE_TABLE).then_some(TableId(0))
    }
    fn live(&self, _: TableId) -> Mask {
        Mask(0)
    }
    fn field(&self, _: TableId, name: &str) -> Option<BoundField> {
        let col = match name {
            "src" => SRC,
            "rel" => REL,
            "dst" => DST,
            "tier" => TIER,
            "ctx" => CTX,
            _ => return None,
        };
        Some(BoundField {
            col: Col(col),
            kind: FieldKind::Code,
            validity: None,
        })
    }
    fn code(&self, _: TableId, field: Col, literal: &str) -> Option<u32> {
        match field.0 {
            REL => RELS
                .iter()
                .position(|(_, n)| *n == literal)
                .map(|i| i as u32),
            TIER => TIERS
                .iter()
                .position(|(_, n)| *n == literal)
                .map(|i| i as u32),
            _ => self.node(literal),
        }
    }
}

/// Description length of the methods' declared bodies (D-SAG-1 §5).
///
/// Transparent, deliberately crude: a body is the sequence of its condition,
/// fold and law tokens. `L(P)` = each atom × `log2(atoms)`, plus each used
/// composite × one `log2(|V|)` reference per atom it is defined over;
/// `L(M|P)` = each method's tokens × `log2(|V|)`, where `V` is the atoms and
/// the used composites. A composite (a factored predicate) replaces its
/// conditions as one token in every method that carries all of them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dl {
    pub primitives: f64,
    pub methods: f64,
}

impl Dl {
    #[must_use]
    pub fn total(&self) -> f64 {
        self.primitives + self.methods
    }
}

/// `bodies`: each method's token identities. `composites`: factored
/// condition sets. Identities are compared as given; the caller decides
/// whether they are typed identities or bare labels.
#[must_use]
pub fn description_length(bodies: &[Vec<String>], composites: &[Vec<String>]) -> Dl {
    let mut atoms: Vec<&String> = bodies.iter().flatten().collect();
    atoms.sort();
    atoms.dedup();
    let atom_bits = (atoms.len().max(2) as f64).log2();
    let rewritten: Vec<Vec<String>> = bodies
        .iter()
        .map(|b| {
            let mut b = b.clone();
            for (i, c) in composites.iter().enumerate() {
                if c.iter().all(|x| b.contains(x)) {
                    b.retain(|x| !c.contains(x));
                    b.push(format!("composite#{i}"));
                }
            }
            b
        })
        .collect();
    // The vocabulary is every token a rewritten body uses, plus every atom a
    // used composite is defined over: a composite is spelled in atoms, so
    // its atoms stay in the alphabet even when no body names them alone.
    let used: Vec<usize> = (0..composites.len())
        .filter(|i| {
            rewritten
                .iter()
                .flatten()
                .any(|t| *t == format!("composite#{i}"))
        })
        .collect();
    let mut vocab: Vec<&String> = rewritten
        .iter()
        .flatten()
        .filter(|t| !t.starts_with("composite#"))
        .chain(used.iter().flat_map(|&i| composites[i].iter()))
        .collect();
    vocab.sort();
    vocab.dedup();
    let v_bits = ((vocab.len() + used.len()).max(2) as f64).log2();
    // Atoms cost their spelling; a composite costs one reference per atom.
    let primitives = vocab.len() as f64 * atom_bits
        + used
            .iter()
            .map(|&i| composites[i].len() as f64 * v_bits)
            .sum::<f64>();
    let methods = rewritten.iter().map(|b| b.len() as f64 * v_bits).sum();
    Dl {
        primitives,
        methods,
    }
}

impl SoftwareGraph {
    /// Each prepared-or-not method's declared tokens, by identity.
    #[must_use]
    pub fn bodies(&self) -> Vec<(String, Vec<String>)> {
        self.methods()
            .map(|m| {
                let toks = [Rel::Filters, Rel::Folds, Rel::FinishedBy]
                    .into_iter()
                    .flat_map(|r| self.targets(m, r))
                    .map(|t| self.nodes[t as usize].ident.clone())
                    .collect();
                (self.nodes[m as usize].ident.clone(), toks)
            })
            .filter(|(_, t): &(String, Vec<String>)| !t.is_empty())
            .collect()
    }
}
