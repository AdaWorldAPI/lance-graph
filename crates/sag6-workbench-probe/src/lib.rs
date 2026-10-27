//! D-SAG-6-W1 — a read-only, source-attested project graph, evaluated through
//! the existing Quack lowering and the mask-risc executor.
//!
//! **Lab only.** Nothing here is a production binding store, a DT64 adapter,
//! a DR64³ axis, a method registry or an MCP server. It answers one question
//! shape — "which identities does this identity reach through this predicate,
//! and on what evidence" — and refuses every shape that has no admitted
//! execution path.
//!
//! Three things are kept apart and never inferred from one another:
//!
//! - **Identity** ([`GlobalId`]): the upstream system's own key, plus the
//!   revision for keys that only mean something at a revision (a source
//!   symbol is `path@commit`, never `path`).
//! - **Coordinate** ([`LocalRef`]): a frame-local ordinal. It is only valid
//!   together with the [`Frame`] it was issued under, and only while that
//!   frame is the store's current seal. Frame staleness is
//!   `dr_math_binding_probe::FrameStore`, reused, not restated.
//! - **Evidence** ([`Witness`]): which snapshot an edge was read from, where
//!   to find it upstream, and how well it is established ([`Grade`]).
//!
//! An answer is never a bare `bool`. An empty result is `NotHeld` only when
//! the edge set for that subject, predicate and direction was read in full;
//! otherwise it is `Undetermined` and names what was not read.

pub mod fixture;

use std::collections::{BTreeMap, BTreeSet};

use dr_math_binding_probe::{Frame, FrameRevisions, FrameStore, Refusal as FrameRefusal};
use lance_graph_contract::hash::fnv1a;
use lance_graph_mask_risc::{Foreign, LaneRef, Out, Planes, Scratch, execute_into, words_for};
use lance_graph_quack::{Agg, Cmp, Col, Filter, Query, lower};

/// The upstream system that owns an identity. The workbench reads from these
/// and never writes to them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SourceSystem {
    /// GitHub: pull requests, commits, files and symbols at a commit, and the
    /// board rows committed to a repository.
    GitHub,
    /// Linear: issues and their typed relations.
    Linear,
    /// A Claude Code session id, as recorded by the artifact that names it.
    ClaudeSession,
}

/// A globally stable, source-attested identity. Equality is the upstream key
/// and, for revision-scoped keys, the revision. Two symbols with the same
/// path at different commits are different identities.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GlobalId {
    pub system: SourceSystem,
    pub key: String,
    /// `Some` for keys that are only meaningful at a revision (source
    /// symbols, board rows); `None` for keys the upstream keeps stable
    /// (a Linear issue id, a pull request number).
    pub revision: Option<String>,
}

impl GlobalId {
    pub fn linear(issue: &str) -> Self {
        Self {
            system: SourceSystem::Linear,
            key: issue.to_owned(),
            revision: None,
        }
    }

    pub fn pull_request(repo: &str, number: u32) -> Self {
        Self {
            system: SourceSystem::GitHub,
            key: format!("{repo}#{number}"),
            revision: None,
        }
    }

    /// A source location at a commit: `path#Lline`, scoped to `commit`.
    pub fn symbol(repo: &str, commit: &str, path: &str, line: u32) -> Self {
        Self {
            system: SourceSystem::GitHub,
            key: format!("{repo}:{path}#L{line}"),
            revision: Some(commit.to_owned()),
        }
    }

    /// A technical-debt row (`.claude/board/ISSUES.md` anchor) at a commit.
    pub fn debt_row(repo: &str, commit: &str, row: &str) -> Self {
        Self {
            system: SourceSystem::GitHub,
            key: format!("{repo}:.claude/board/ISSUES.md#{row}"),
            revision: Some(commit.to_owned()),
        }
    }

    pub fn session(id: &str) -> Self {
        Self {
            system: SourceSystem::ClaudeSession,
            key: id.to_owned(),
            revision: None,
        }
    }
}

/// One read of one upstream object: which system, which revision of it, and
/// when it was read. An edge is only ever as current as its snapshot.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Snapshot {
    pub system: SourceSystem,
    /// The upstream revision: a commit SHA, or a Linear `updatedAt`.
    pub revision: String,
    /// The object the snapshot was read from (an issue id, a repository).
    pub object: String,
    pub fetched_at: String,
}

/// The predicate catalogue of this fixture. Ordinals are fixture-local; they
/// are not DT64 slots and not OGAR classids.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Predicate {
    /// Linear `blocks`. Its inverse, `blocked_by`, is a projection of the
    /// same rows and is never stored a second time.
    Blocks,
    /// Linear `relatedTo`, symmetric.
    RelatedTo,
    /// An issue names a source location.
    MentionsSymbol,
    /// An issue is tracked by a technical-debt row on the GitHub board.
    TrackedByDebtRow,
    /// An issue cites a pull request.
    CitesPr,
    /// An artifact records the Claude Code session that wrote it.
    AuthoredInSession,
}

impl Predicate {
    pub const ALL: [Predicate; 6] = [
        Predicate::Blocks,
        Predicate::RelatedTo,
        Predicate::MentionsSymbol,
        Predicate::TrackedByDebtRow,
        Predicate::CitesPr,
        Predicate::AuthoredInSession,
    ];

    fn ordinal(self) -> u32 {
        self as u32
    }

    pub fn spec(self) -> PredicateSpec {
        use Predicate::*;
        let (key, symmetric, inverse, convergence) = match self {
            Blocks => (
                "blocks",
                false,
                Some("blocked_by"),
                Some("ogar_vocab::class_ids::PROJECT_RELATION"),
            ),
            RelatedTo => (
                "related_to",
                true,
                None,
                Some("ogar_vocab::class_ids::PROJECT_RELATION"),
            ),
            MentionsSymbol => ("mentions_symbol", false, Some("mentioned_by"), None),
            TrackedByDebtRow => ("tracked_by_debt_row", false, Some("tracks"), None),
            CitesPr => ("cites_pr", false, Some("cited_by"), None),
            AuthoredInSession => ("authored_in_session", false, Some("authored"), None),
        };
        PredicateSpec {
            predicate: self,
            key,
            symmetric,
            inverse,
            convergence,
        }
    }
}

/// What a predicate is, as declared by this fixture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PredicateSpec {
    pub predicate: Predicate,
    pub key: &'static str,
    pub symmetric: bool,
    /// The name of the inverse projection, if the predicate is directed.
    pub inverse: Option<&'static str>,
    /// The existing OGAR concept this predicate would converge onto, if one
    /// exists. A candidate, not a binding: no GitHub or Linear port maps it.
    pub convergence: Option<&'static str>,
}

/// How well an edge is established. The first five are observations or
/// derivations with a source; `Hypothesis` is an inference nobody has
/// checked against its source, and is never reported as a proven relation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Grade {
    ObservedSource,
    CompilerDerived,
    LogicallyDerived,
    Tested,
    IndependentlyVerified,
    Hypothesis,
}

/// Which way a directed predicate is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Direction {
    /// subject → object (`blocks`).
    Forward,
    /// object → subject (`blocked_by`), read from the same rows.
    Inverse,
}

/// One source-attested edge as imported. The canonical record; everything
/// else in a [`Workbench`] is derived from these and rebuildable.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Record {
    pub subject: GlobalId,
    pub predicate: Predicate,
    pub object: GlobalId,
    pub snapshot: Snapshot,
    /// Where to see the edge upstream.
    pub locator: String,
    pub grade: Grade,
}

/// The edge sets that were read in full. Only these can answer `NotHeld`.
pub type Covered = (GlobalId, Predicate, Direction);

/// The canonical, cold-side record set. Import is idempotent: the same
/// record imported twice is one record.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Corpus {
    records: BTreeSet<Record>,
    coverage: BTreeSet<Covered>,
}

impl Corpus {
    pub fn import(&mut self, records: impl IntoIterator<Item = Record>) {
        self.records.extend(records);
    }

    /// Declare that every `predicate` edge of `subject` in `direction` was
    /// read. For a symmetric predicate the direction is ignored.
    pub fn mark_complete(&mut self, subject: GlobalId, predicate: Predicate, direction: Direction) {
        let direction = if predicate.spec().symmetric {
            Direction::Forward
        } else {
            direction
        };
        self.coverage.insert((subject, predicate, direction));
    }

    pub fn records(&self) -> impl Iterator<Item = &Record> {
        self.records.iter()
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

/// A witnessed answer item: the identity reached, and the evidence for it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Witness {
    pub id: GlobalId,
    pub snapshot: Snapshot,
    pub locator: String,
    pub grade: Grade,
}

/// Why a question was not answered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reason {
    /// The coordinate was issued under a frame that is not the current seal.
    Frame(FrameRefusal),
    /// Multi-hop has no admitted execution path: `Filter::Semijoin` is
    /// unclassified under ADMISSION B (ISSUES.md
    /// ISS-ADMISSION-B-CALL-SITE-CLASSIFICATION-NOT-STARTED, Linear EXO-140),
    /// and mask-risc's `OutgoingHop` is quarantined behind the off-by-default
    /// `quarantined-gather-hop` feature (ISS-GATHER-HOP-1616).
    MultiHopNotAdmitted,
    /// The predicate has no OGAR classid binding, so no prepared method can
    /// be bound to it, whatever classid the caller names.
    PredicateUnbound,
    /// The lowering or the executor refused the program.
    Executor(String),
}

/// An answer. Never a bare `bool`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Answer {
    /// At least one edge was found. `complete` is true only when the edge set
    /// was read in full; otherwise more edges may exist upstream.
    Witnessed {
        witnesses: Vec<Witness>,
        complete: bool,
    },
    /// No edge, and the edge set was read in full.
    NotHeld,
    /// No edge found, but the edge set was not read in full. This is UNKNOWN,
    /// not FALSE.
    Undetermined {
        unread: Vec<Covered>,
    },
    Refused(Reason),
    /// The question shape has no implementation here.
    Unsupported(&'static str),
}

/// An identity lookup.
#[derive(Clone, Debug)]
pub enum Resolution {
    Resolved(LocalRef),
    /// No read snapshot mentions this identity. Not evidence that it does
    /// not exist upstream.
    NotInSnapshot,
}

/// A frame-local coordinate. Only [`Workbench::resolve_identity`] issues one,
/// and it is only valid under the frame it carries.
#[derive(Clone, Debug)]
pub struct LocalRef {
    frame: Frame,
    ordinal: u32,
}

impl LocalRef {
    pub fn ordinal(&self) -> u32 {
        self.ordinal
    }
}

/// What [`Workbench::inspect_predicate`] reports.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PredicateInspection {
    pub spec: PredicateSpec,
    /// Always `None` here: no GitHub or Linear port exists in OGAR, so no
    /// predicate of this fixture resolves to a classid.
    pub class_id: Option<u16>,
    pub method: Answer,
}

/// [`Workbench::explain_blocker`]'s report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockerReport {
    pub blocked_by: Answer,
    pub debt_rows: Answer,
    /// What blocks the blockers. Multi-hop, so refused.
    pub transitive: Answer,
}

/// The difference between two corpora, as source records.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SnapshotDiff {
    pub added: Vec<Record>,
    pub removed: Vec<Record>,
}

// Lane positions in `Planes::lanes`.
const SUBJECT: Col = Col(0);
const PREDICATE: Col = Col(1);
const OBJECT: Col = Col(2);
const GRADE: Col = Col(3);

/// The sealed, queryable view of a [`Corpus`]: interned identities, four
/// `u32` lanes over the records in their stored order, and a frame.
#[derive(Debug)]
pub struct Workbench {
    corpus: Corpus,
    rows: Vec<Record>,
    ids: Vec<GlobalId>,
    index: BTreeMap<GlobalId, u32>,
    subject: Vec<u32>,
    predicate: Vec<u32>,
    object: Vec<u32>,
    grade: Vec<u32>,
    store: FrameStore,
}

impl Workbench {
    pub fn new(corpus: Corpus) -> Self {
        let mut wb = Self {
            corpus: Corpus::default(),
            rows: Vec::new(),
            ids: Vec::new(),
            index: BTreeMap::new(),
            subject: Vec::new(),
            predicate: Vec::new(),
            object: Vec::new(),
            grade: Vec::new(),
            store: FrameStore::new(FrameRevisions {
                source: 0,
                relation_map: 0,
                coordinates: 0,
            }),
        };
        wb.load(corpus);
        wb
    }

    /// Replace the corpus. Every coordinate issued before is refused from now
    /// on, even when the new corpus is identical: re-sealing is a new frame.
    pub fn reload(&mut self, corpus: Corpus) {
        self.load(corpus);
    }

    fn load(&mut self, corpus: Corpus) {
        let rows: Vec<Record> = corpus.records().cloned().collect();
        let mut ids = BTreeSet::new();
        for r in &rows {
            ids.insert(r.subject.clone());
            ids.insert(r.object.clone());
        }
        // An identity read in full with no edge at all still resolves, so its
        // NotHeld is reachable.
        for (subject, _, _) in &corpus.coverage {
            ids.insert(subject.clone());
        }
        let ids: Vec<GlobalId> = ids.into_iter().collect();
        let index: BTreeMap<GlobalId, u32> = ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id.clone(), i as u32))
            .collect();
        self.subject = rows.iter().map(|r| index[&r.subject]).collect();
        self.predicate = rows.iter().map(|r| r.predicate.ordinal()).collect();
        self.object = rows.iter().map(|r| index[&r.object]).collect();
        self.grade = rows.iter().map(|r| r.grade as u32).collect();
        let revisions = FrameRevisions {
            source: digest(rows.iter().map(|r| {
                format!(
                    "{:?}|{}|{}|{}",
                    r.snapshot.system,
                    r.snapshot.object,
                    r.snapshot.revision,
                    r.snapshot.fetched_at
                )
            })),
            relation_map: digest(Predicate::ALL.iter().map(|p| p.spec().key.to_owned())),
            coordinates: digest(ids.iter().map(|id| format!("{id:?}"))),
        };
        self.rows = rows;
        self.ids = ids;
        self.index = index;
        self.corpus = corpus;
        self.store.advance(revisions);
    }

    /// The identity's frame-local coordinate under the current seal.
    pub fn resolve_identity(&self, id: &GlobalId) -> Resolution {
        match self.index.get(id) {
            Some(&ordinal) => Resolution::Resolved(LocalRef {
                frame: self.store.seal(),
                ordinal,
            }),
            None => Resolution::NotInSnapshot,
        }
    }

    /// One hop from `at` through `predicate`, read in `direction`.
    pub fn query_relationships(
        &self,
        at: &LocalRef,
        predicate: Predicate,
        direction: Direction,
    ) -> Answer {
        if let Err(r) = self.store.check(&at.frame) {
            return Answer::Refused(Reason::Frame(r));
        }
        let spec = predicate.spec();
        let p = Filter::cmp(PREDICATE, Cmp::EqU32(predicate.ordinal()));
        let as_subject = Filter::and([Filter::cmp(SUBJECT, Cmp::EqU32(at.ordinal)), p.clone()]);
        let as_object = Filter::and([Filter::cmp(OBJECT, Cmp::EqU32(at.ordinal)), p]);
        let filter = match (spec.symmetric, direction) {
            (true, _) => Filter::or([as_subject, as_object]),
            (false, Direction::Forward) => as_subject,
            (false, Direction::Inverse) => as_object,
        };
        let kept = match self.keep(filter) {
            Ok(kept) => kept,
            Err(reason) => return Answer::Refused(reason),
        };
        let mut witnesses = Vec::new();
        for row in set_bits(&kept) {
            let other = if self.subject[row] == at.ordinal && self.object[row] != at.ordinal {
                self.object[row]
            } else if self.object[row] == at.ordinal {
                self.subject[row]
            } else {
                self.object[row]
            };
            witnesses.push(self.witness(row, other));
        }
        let direction = if spec.symmetric {
            Direction::Forward
        } else {
            direction
        };
        let covered = (self.ids[at.ordinal as usize].clone(), predicate, direction);
        let complete = self.corpus.coverage.contains(&covered);
        match (witnesses.is_empty(), complete) {
            (false, _) => Answer::Witnessed {
                witnesses,
                complete,
            },
            (true, true) => Answer::NotHeld,
            (true, false) => Answer::Undetermined {
                unread: vec![covered],
            },
        }
    }

    /// More than one hop. There is no admitted execution path; see
    /// [`Reason::MultiHopNotAdmitted`].
    pub fn query_chain(&self, _at: &LocalRef, _path: &[(Predicate, Direction)]) -> Answer {
        Answer::Refused(Reason::MultiHopNotAdmitted)
    }

    /// Every edge that touches `at`, with its snapshot, locator and grade.
    pub fn trace_provenance(&self, at: &LocalRef) -> Answer {
        if let Err(r) = self.store.check(&at.frame) {
            return Answer::Refused(Reason::Frame(r));
        }
        let filter = Filter::or([
            Filter::cmp(SUBJECT, Cmp::EqU32(at.ordinal)),
            Filter::cmp(OBJECT, Cmp::EqU32(at.ordinal)),
        ]);
        match self.keep(filter) {
            Ok(kept) => {
                let witnesses: Vec<Witness> = set_bits(&kept)
                    .map(|row| {
                        let other = if self.subject[row] == at.ordinal {
                            self.object[row]
                        } else {
                            self.subject[row]
                        };
                        self.witness(row, other)
                    })
                    .collect();
                if witnesses.is_empty() {
                    Answer::Undetermined { unread: Vec::new() }
                } else {
                    Answer::Witnessed {
                        witnesses,
                        complete: false,
                    }
                }
            }
            Err(reason) => Answer::Refused(reason),
        }
    }

    pub fn inspect_predicate(&self, predicate: Predicate) -> PredicateInspection {
        PredicateInspection {
            spec: predicate.spec(),
            class_id: None,
            method: Answer::Unsupported(
                "no prepared method: the predicate has no OGAR classid binding (no GitHub or Linear port)",
            ),
        }
    }

    /// Bind a prepared method to `predicate` under `class_id`. Refused for
    /// every predicate and every classid here, because none is bound; a
    /// caller naming the convergence candidate's classid does not create a
    /// binding.
    pub fn prepare_method(&self, predicate: Predicate, class_id: u16) -> Result<(), Reason> {
        let _ = (predicate, class_id);
        Err(Reason::PredicateUnbound)
    }

    /// What blocks `at`, and which debt rows track it. The blockers'
    /// blockers would be a second hop, so that part is refused.
    pub fn explain_blocker(&self, at: &LocalRef) -> BlockerReport {
        BlockerReport {
            blocked_by: self.query_relationships(at, Predicate::Blocks, Direction::Inverse),
            debt_rows: self.query_relationships(
                at,
                Predicate::TrackedByDebtRow,
                Direction::Forward,
            ),
            transitive: self.query_chain(
                at,
                &[
                    (Predicate::Blocks, Direction::Inverse),
                    (Predicate::Blocks, Direction::Inverse),
                ],
            ),
        }
    }

    /// Edges graded `Hypothesis`: claims nobody has checked against their
    /// source.
    pub fn list_unverified_claims(&self) -> Result<Vec<&Record>, Reason> {
        let kept = self.keep(Filter::cmp(GRADE, Cmp::EqU32(Grade::Hypothesis as u32)))?;
        Ok(set_bits(&kept).map(|row| &self.rows[row]).collect())
    }

    /// The source records in `after` that are not in `before`, and the
    /// reverse. A cold comparison of the canonical records, not a query.
    pub fn compare_snapshots(before: &Corpus, after: &Corpus) -> SnapshotDiff {
        SnapshotDiff {
            added: after.records.difference(&before.records).cloned().collect(),
            removed: before.records.difference(&after.records).cloned().collect(),
        }
    }

    pub fn corpus(&self) -> &Corpus {
        &self.corpus
    }

    fn witness(&self, row: usize, other: u32) -> Witness {
        let r = &self.rows[row];
        Witness {
            id: self.ids[other as usize].clone(),
            snapshot: r.snapshot.clone(),
            locator: r.locator.clone(),
            grade: r.grade,
        }
    }

    /// Lower `filter` with Quack and run it on mask-risc over the four lanes.
    /// The kept rows come back as a mask; nothing is copied out of the lanes.
    fn keep(&self, filter: Filter) -> Result<Vec<u64>, Reason> {
        let n = self.rows.len();
        let mut kept = vec![0u64; words_for(n)];
        if n == 0 {
            return Ok(kept);
        }
        let program = lower(&Query {
            filter,
            agg: Agg::Rows,
        })
        .map_err(|e| Reason::Executor(format!("{e:?}")))?;
        let lanes = [
            LaneRef::U32(&self.subject),
            LaneRef::U32(&self.predicate),
            LaneRef::U32(&self.object),
            LaneRef::U32(&self.grade),
        ];
        let planes = Planes {
            n_rows: n,
            masks: &[],
            lanes: &lanes,
        };
        let mut scratch =
            Scratch::for_program(&program, n).map_err(|e| Reason::Executor(format!("{e:?}")))?;
        execute_into(
            &program,
            &planes,
            &Foreign::NONE,
            &mut scratch,
            Out::Mask(&mut kept),
        )
        .map_err(|e| Reason::Executor(format!("{e:?}")))?;
        Ok(kept)
    }
}

fn digest(parts: impl Iterator<Item = String>) -> u64 {
    let mut joined = String::new();
    for p in parts {
        joined.push_str(&p);
        joined.push('\n');
    }
    fnv1a(joined.as_bytes())
}

/// Positions of the set bits, in row order.
fn set_bits(words: &[u64]) -> impl Iterator<Item = usize> + '_ {
    words.iter().enumerate().flat_map(|(w, &word)| {
        let mut bits = word;
        std::iter::from_fn(move || {
            if bits == 0 {
                return None;
            }
            let t = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            Some(w * 64 + t)
        })
    })
}
