//! XLOG's own behaviour lens, `0xFFFF :: 0xF100` (plan
//! `d-dt64-7-behaviour-addressing-registration-v1.md`, decision O7: the
//! `F1__` range is XLOG-exclusive semantics, minted here).
//!
//! XLOG's relational core (scans, joins, projections, `col == const`,
//! unions) has an equivalent in the shared algebra at `0xFFFF :: 0xF0__` and
//! gets no word here. What XLOG adds beyond it is term inspection: the
//! built-ins `functor/3`, `arg/3` and `=../2`. XLOG's compiler resolves each
//! into an atom over a helper relation (`__xlog_meta_functor`, `_arg`,
//! `_univ`) whose rows it emits as facts about the program's ground terms.
//! The rows are data; the built-in is behaviour, and it is addressed here.
//!
//! **Position binds; labels are aliases**, as for the `F000` polyfill
//! (`lance-graph-r2il::polyfill`). Word `p` is [`XLOG_WORDS`]`[p]`; a name
//! selects a word only if exactly one position lists it.
//!
//! XLOG's `__xlog_meta_fail` helper is not a word. It is a relation with no
//! rows, so a rule reading it never fires; as an empty data relation it
//! already means exactly that.
//!
//! # What `arity` means for this lens
//!
//! [`declare`] passes each word's COLUMN COUNT (the helper relation's arity:
//! `functor` has term, name, arity) as `BindStore::declare_algebra`'s
//! `arity`. For the `F000` polyfill the same field is a STACK OPERAND COUNT,
//! because those words are scalar folds. The store does not interpret the
//! number; [`declare`]'s docs list every reader and what each does with it.

use lance_graph_contract::ogar_codebook::{compose_classid, HARDCODED_BEHAVIOUR};
use lance_graph_dt64::{AlgebraRef, BindStore, Extent, Refusal, SchemaRev};

/// XLOG's behaviour lens, `0xFFFF :: 0xF100`.
pub const XLOG_LENS: u32 = compose_classid(HARDCODED_BEHAVIOUR, 0xF100);

/// One word of [`XLOG_LENS`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XlogWord {
    /// The columns of the relation the word reads.
    pub columns: u8,
    /// Every name a view may give this position: the Prolog name first,
    /// then XLOG's helper-relation name.
    pub aliases: &'static [&'static str],
}

/// The words, by position. Positions never move: a changed meaning takes a
/// new position or a new lens.
pub const XLOG_WORDS: [XlogWord; 3] = [
    // functor(Term, Name, Arity)
    XlogWord {
        columns: 3,
        aliases: &["functor", "__xlog_meta_functor"],
    },
    // arg(Term, Index, Arg)
    XlogWord {
        columns: 3,
        aliases: &["arg", "__xlog_meta_arg"],
    },
    // Term =.. [Name | Args]: the term and its parts as a list.
    XlogWord {
        columns: 2,
        aliases: &["=..", "__xlog_meta_univ"],
    },
];

/// The position whose aliases include `label`, if exactly one does.
#[must_use]
pub fn position_of(label: &str) -> Option<u8> {
    let mut hits = XLOG_WORDS
        .iter()
        .enumerate()
        .filter(|(_, w)| w.aliases.contains(&label));
    match (hits.next(), hits.next()) {
        (Some((p, _)), None) => u8::try_from(p).ok(),
        _ => None,
    }
}

/// The address of the word `label` names, if it names one.
#[must_use]
pub fn word_of(label: &str) -> Option<AlgebraRef> {
    position_of(label).map(|word| AlgebraRef {
        lens: XLOG_LENS,
        word,
    })
}

/// Bind [`XLOG_LENS`] in `store`, one field per word, each declared with its
/// column count as `arity` (see the module docs). Idempotent; a different
/// earlier binding of the lens is refused (`AlgebraImmutable`).
///
/// Readers of the declared `arity`, and what each does with a column count:
///
/// - `BindStore::insert` (`lance-graph-dt64`, via `register`/`derive`):
///   a `MethodBinding` effect naming an XLOG word must carry exactly
///   `columns` arguments, else `ArityMismatch`. **Side effect:** an effect
///   on `functor` takes 3 arguments and on `=..` takes 2, one per column of
///   the relation it reads, not a stack operand count.
/// - `lance-graph-r2il::binding` (the executor): never reaches an XLOG
///   word. It refuses any lens that is neither `POLYFILL_LENS` nor an
///   R2IL-fold concept (`NotR2ilReading`) before reading `arity`, and its
///   frozen-word check reads the `F000` table, not the store.
///
/// No other crate reads the field.
pub fn declare(store: &mut BindStore) -> Result<(), Refusal> {
    store.declare_algebra(
        XLOG_LENS,
        SchemaRev(1),
        Extent::DR64_CUBE,
        XLOG_WORDS.iter().map(|w| w.columns).collect(),
    )
}
