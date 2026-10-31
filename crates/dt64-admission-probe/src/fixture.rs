//! The D-SAG-6 fixture (`crates/sag-ra-probe/fixtures/sag_fixture`) as
//! committed evidence, plus a DT64 store binding its functions.
//!
//! # Provenance of the rows
//!
//! No row-level output of `project_rows.py` is committed anywhere in the
//! repository: the board entries record counts and per-function findings, not
//! row lists. The rows below are therefore RECONSTRUCTED in that tool's
//! format from facts the entries state, one citation per row group:
//!
//! - E1 = `.claude/board/entries/2026-10-31-d-sag-6-2-lab-projection.md`
//!   §3 (relation table) and §4 (binding table);
//! - E2 = `.claude/board/entries/2026-10-31-d-sag-6-2-span-join-identity.md`
//!   §1 (span join: subjects are rust-analyzer monikers, closures are
//!   `<fn>::{closure@L<line>}::<method>`, and the path join leaves 17 rows
//!   `UNJOINED` on this fixture).
//!
//! What is NOT reproduced: the full 38 `CALLS`, 8 `MAY_PANIC` and 23
//! `DEFINES` rows of E1 §3. Only the rows bearing on the functions below are
//! here. The `CALLS` objects (`core::slice::...`, `core::iter::...`) are
//! illustrative spellings of the standard-library callees each body makes;
//! the policy reads them only by prefix against its trusted list. Closure
//! line numbers are the closures' lines in the fixture source. Profile-
//! dependent overflow asserts (E1 F03) are not listed; adding them would
//! not change any outcome below, since every function they would touch is
//! already refused or is pure arithmetic outside this set.
//!
//! Re-running Charon and the two tools on the fixture would replace these
//! rows with measured ones; that is not done here (no `/tmp` input, no
//! Charon run at test time).

use lance_graph_contract::class_view::WideFieldMask;
use lance_graph_contract::set_algebra::{Cmp, Logic, SetExpr};
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::{
    AlgebraRef, Arg, BindStore, ClassSchema, Effect, Extent, Guard, MethodBinding, MethodId,
    MethodRef, SchemaRev, SigRev, ValueDemand,
};

use crate::evidence::{parse_rows, Row};
use crate::{Attachment, EvidenceSet};

/// The span-joined rows (E2 §1: `sag_fixture` joins 29 of 29 bodies).
pub const SPAN_JOINED_ROWS: &str = "\
relation\tsubject\tobject\twitness\tevidence
DEFINES\tsag_fixture::count_below\tFunction\trust-analyzer\tWITNESSED
CALLS\tsag_fixture::count_below\tcore::slice::<impl [T]>::iter\tcharon\tWITNESSED
CALLS\tsag_fixture::count_below\tcore::iter::traits::iterator::Iterator::filter\tcharon\tWITNESSED
CALLS\tsag_fixture::count_below\tcore::iter::traits::iterator::Iterator::count\tcharon\tWITNESSED
DEFINES\tsag_fixture::count_with_tally\tFunction\trust-analyzer\tWITNESSED
CALLS\tsag_fixture::count_with_tally\tcore::slice::<impl [T]>::iter\tcharon\tWITNESSED
CALLS\tsag_fixture::count_with_tally\tcore::iter::traits::iterator::Iterator::filter\tcharon\tWITNESSED
CALLS\tsag_fixture::count_with_tally\tcore::iter::traits::iterator::Iterator::count\tcharon\tWITNESSED
WRITES_CALLER_MEMORY\tsag_fixture::count_with_tally::{closure@L42}::call_mut\t1\tcharon\tWITNESSED
DEFINES\tsag_fixture::count_div\tFunction\trust-analyzer\tWITNESSED
CALLS\tsag_fixture::count_div\tcore::slice::<impl [T]>::iter\tcharon\tWITNESSED
CALLS\tsag_fixture::count_div\tcore::iter::traits::iterator::Iterator::filter\tcharon\tWITNESSED
CALLS\tsag_fixture::count_div\tcore::iter::traits::iterator::Iterator::count\tcharon\tWITNESSED
MAY_PANIC\tsag_fixture::count_div::{closure@L51}::call_mut\t1 assert\tcharon\tWITNESSED
DEFINES\tsag_fixture::count_generic\tFunction\trust-analyzer\tWITNESSED
CALLS\tsag_fixture::count_generic\tcore::slice::<impl [T]>::iter\tcharon\tWITNESSED
CALLS\tsag_fixture::count_generic\tcore::iter::traits::iterator::Iterator::filter\tcharon\tWITNESSED
CALLS\tsag_fixture::count_generic\tcore::iter::traits::iterator::Iterator::count\tcharon\tWITNESSED
MAY_DISPATCH\tsag_fixture::count_generic::{closure@L31}::call_mut\ttrait:Clause\tcharon\tPOSSIBLE
DEFINES\tsag_fixture::count_dyn\tFunction\trust-analyzer\tWITNESSED
CALLS\tsag_fixture::count_dyn\tcore::slice::<impl [T]>::iter\tcharon\tWITNESSED
CALLS\tsag_fixture::count_dyn\tcore::iter::traits::iterator::Iterator::filter\tcharon\tWITNESSED
CALLS\tsag_fixture::count_dyn\tcore::iter::traits::iterator::Iterator::count\tcharon\tWITNESSED
MAY_DISPATCH\tsag_fixture::count_dyn::{closure@L36}::call_mut\tdynamic\tcharon\tPOSSIBLE
DEFINES\tsag_fixture::mask_below\tFunction\trust-analyzer\tWITNESSED
CALLS\tsag_fixture::mask_below\tcore::slice::<impl [T]>::iter\tcharon\tWITNESSED
CALLS\tsag_fixture::mask_below\tcore::iter::traits::iterator::Iterator::enumerate\tcharon\tWITNESSED
WRITES_THROUGH_LOCAL_REF\tsag_fixture::mask_below\t1\tcharon\tPOSSIBLE
DEFINES\tsag_fixture::impl::Below::Pred::test\tFunction\trust-analyzer\tWITNESSED
";

/// The path-joined rows for `Below::test` (E1 §3: impls and closures do not
/// join by path, so Charon's `sag_fixture::Impl::test` is `UNJOINED`; E2 §1:
/// the two `Impl::test` methods share that one Charon path).
pub const PATH_JOINED_ROWS: &str = "\
relation\tsubject\tobject\twitness\tevidence
DEFINES\tsag_fixture::impl::Below::Pred::test\tFunction\trust-analyzer\tWITNESSED
UNJOINED\tsag_fixture::Impl::test\tno rust-analyzer moniker with this path\tjoin\tUNKNOWN
";

/// The bound class. This is the lab classid `lance-graph-dt64`'s own tests
/// bind (`tests/bind_store.rs`, `ORDER`); reused, not minted.
pub const CLASS: u32 = 0x0A1E_0001;
/// The polyfill lens; `lance-graph-dt64`'s test lens (`POLYFILL`), reused.
pub const LENS: u32 = 0x7F0A_0001;
/// Word 0 of the lens: a one-operand popcount over a mask field (E1 §4: the
/// count is written "through mask-risc's popcount word").
pub const POPCOUNT: AlgebraRef = AlgebraRef {
    lens: LENS,
    word: 0,
};

/// Field 0: the element `x`; field 1: the count.
pub const FIELD_X: u8 = 0;
pub const FIELD_COUNT: u8 = 1;

pub const COUNT_BELOW: MethodId = MethodId(1);
pub const COUNT_WITH_TALLY: MethodId = MethodId(2);
pub const COUNT_DIV: MethodId = MethodId(3);
pub const COUNT_GENERIC: MethodId = MethodId(4);
pub const COUNT_DYN: MethodId = MethodId(5);
pub const MASK_BELOW: MethodId = MethodId(6);
pub const BELOW_TEST: MethodId = MethodId(7);
/// Bound, but no evidence attachment.
pub const UNATTACHED: MethodId = MethodId(8);

/// The functions and the subjects their evidence is attached to.
pub const SUBJECTS: [(MethodId, &str); 7] = [
    (COUNT_BELOW, "sag_fixture::count_below"),
    (COUNT_WITH_TALLY, "sag_fixture::count_with_tally"),
    (COUNT_DIV, "sag_fixture::count_div"),
    (COUNT_GENERIC, "sag_fixture::count_generic"),
    (COUNT_DYN, "sag_fixture::count_dyn"),
    (MASK_BELOW, "sag_fixture::mask_below"),
    (BELOW_TEST, "sag_fixture::impl::Below::Pred::test"),
];

/// One binding: ON `x`, reads `x`, one effect writing the count through the
/// popcount word. `count_below` alone carries its WHEN (`x < 6`, two-valued);
/// the others are unguarded.
pub fn binding(id: MethodId, name: &str) -> MethodBinding {
    let when = (id == COUNT_BELOW).then(|| {
        Guard::new(
            SetExpr::i32(u16::from(FIELD_X), Cmp::Lt, 6),
            Logic::TwoValued,
        )
    });
    MethodBinding {
        id,
        sig_rev: SigRev(1),
        name: name.to_string(),
        on: WideFieldMask::from_positions(&[FIELD_X]),
        when,
        reads: WideFieldMask::from_positions(&[FIELD_X]),
        effects: vec![Effect {
            target: FIELD_COUNT,
            algebra: POPCOUNT,
            args: vec![Arg::Field(FIELD_X)],
        }],
        population: Population::Distinct,
        truth: Truth::TwoValued,
        demand: ValueDemand::Values,
    }
}

/// Every function bound in one class. All of them admit: binding is
/// authentication, not authorization.
pub fn store() -> BindStore {
    let mut s = BindStore::new();
    s.declare_algebra(LENS, SchemaRev(1), Extent::DR64_CUBE, vec![1])
        .expect("lens");
    let mut methods: Vec<MethodBinding> = SUBJECTS
        .iter()
        .map(|(id, subj)| binding(*id, subj.rsplit("::").next().unwrap_or(subj)))
        .collect();
    methods.push(binding(UNATTACHED, "count_below"));
    s.register(
        CLASS,
        ClassSchema {
            schema_rev: SchemaRev(1),
            field_count: 2,
            extent: Extent::DR64_CUBE,
        },
        methods,
    )
    .expect("register");
    s
}

/// The faithful reference to a bound method.
pub fn method_ref(store: &BindStore, id: MethodId) -> MethodRef {
    let b = store
        .admit(&MethodRef {
            class: CLASS,
            method: id,
            sig_rev: SigRev(1),
            guard: (id == COUNT_BELOW).then(|| binding(id, "").when.unwrap().witness()),
        })
        .expect("bound");
    MethodRef::of(CLASS, b)
}

fn attachments() -> Vec<Attachment> {
    SUBJECTS
        .iter()
        .map(|(id, subj)| Attachment {
            class: CLASS,
            method: *id,
            sig_rev: SigRev(1),
            subject: subj.to_string(),
        })
        .collect()
}

pub fn rows(tsv: &str) -> Vec<Row> {
    parse_rows(tsv).expect("committed rows parse")
}

/// The span-joined evidence set.
pub fn span_joined() -> EvidenceSet {
    EvidenceSet {
        rows: rows(SPAN_JOINED_ROWS),
        attachments: attachments(),
    }
}

/// The path-joined evidence set: `Below::test` attaches to Charon's path.
pub fn path_joined() -> EvidenceSet {
    EvidenceSet {
        rows: rows(PATH_JOINED_ROWS),
        attachments: vec![Attachment {
            class: CLASS,
            method: BELOW_TEST,
            sig_rev: SigRev(1),
            subject: "sag_fixture::Impl::test".to_string(),
        }],
    }
}
