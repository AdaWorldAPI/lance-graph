//! The hardcoded relational polyfill as a positional lens (D-DT64-7 §1a, §1b).
//!
//! Hardcoded behaviour is addressed `0xFFFF :: 0xF###`: left side `0xFFFF`,
//! right side a behaviour lens in a concern range. [`POLYFILL_LENS`] is
//! `0xFFFF :: 0xF000`, the relational / mask-risc polyfill.
//!
//! **Position binds; labels are aliases.** Word `p` of this lens is
//! [`POLYFILL`]`[p]`, whatever any view calls it. A view over the lens
//! supplies its language's names for the positions (an R2IL opcode name, a
//! SQL keyword, a Quack variant), and each name is checked against the
//! aliases fixed at that position. A name that belongs to another position,
//! or to none, is refused; it never selects a different word.
//!
//! An alias states that the word computes the same thing as the named
//! operation over a non-empty selection. Over an empty selection SQL answers
//! `NULL` for `SUM`, `MIN` and `MAX`, while this reading answers the fold's
//! identity (`0`, `i64::MAX`, `i64::MIN`); a three-valued binding is refused
//! by [`crate::binding::prepare`], so `NULL` never reaches it.
//!
//! The positions are the [`crate::binding::EXECUTABLE`] opcodes in that
//! order. The opcode, arity and R2IL name of each entry are checked against
//! OGAR's `R2ILVocabulary` by test, not at run time.

use lance_graph_contract::class_view::{ClassId, ClassView};
use lance_graph_contract::ogar_codebook::{classid_concept, compose_classid};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use ogar_loco::FnIndex;
use ogar_r2il::{ALL, ANY, KEY_RUNS, MAX, MIN, SUM};

use crate::{
    INT_ADD, INT_AND, INT_EQUAL, INT_MULT, INT_NOT, INT_OR, INT_SUB, INT_S_DIV, INT_S_LESS,
    INT_XOR, POP_COUNT,
};

/// The left side of every hardcoded behaviour classid.
pub const HARDCODED: u16 = 0xFFFF;

/// The relational / mask-risc polyfill lens, `0xFFFF :: 0xF000`.
pub const POLYFILL_LENS: u32 = compose_classid(HARDCODED, 0xF000);

/// One word of the polyfill lens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrozenWord {
    /// The fold-dialect opcode the word runs as.
    pub opcode: FnIndex,
    /// The number of stack operands it takes; it always pushes one value.
    pub arity: u8,
    /// Every name a view may give this position. The first is the R2IL
    /// opcode name.
    pub aliases: &'static [&'static str],
    /// Which alias is this word's SQL name, if SQL has one.
    pub sql_at: Option<u8>,
    /// Which alias is this word's Quack name, if Quack has one.
    pub quack_at: Option<u8>,
}

/// A language whose names a view gives the polyfill positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AliasLanguage {
    /// OGAR's R2IL opcode names.
    R2il,
    /// SQL keywords and operators.
    Sql,
    /// `lance-graph-quack` variant names (`Agg::*`, `Cmp::*`, `Filter::*`).
    Quack,
}

impl FrozenWord {
    /// This word's name in `lang`, if that language has one.
    #[must_use]
    pub fn name_of(&self, lang: AliasLanguage) -> Option<&'static str> {
        let at = match lang {
            AliasLanguage::R2il => Some(0),
            AliasLanguage::Sql => self.sql_at,
            AliasLanguage::Quack => self.quack_at,
        };
        at.and_then(|i| self.aliases.get(usize::from(i)).copied())
    }

    /// This word's name in `lang`, or its R2IL name where `lang` has none,
    /// so a view always holds one label per position.
    #[must_use]
    pub fn name_in(&self, lang: AliasLanguage) -> &'static str {
        self.name_of(lang).unwrap_or(self.aliases[0])
    }

    /// Whether `label` is one of this word's names.
    #[must_use]
    pub fn is_alias(&self, label: &str) -> bool {
        self.aliases.contains(&label)
    }
}

const fn w(
    opcode: FnIndex,
    arity: u8,
    aliases: &'static [&'static str],
    sql_at: Option<u8>,
    quack_at: Option<u8>,
) -> FrozenWord {
    FrozenWord {
        opcode,
        arity,
        aliases,
        sql_at,
        quack_at,
    }
}

/// The polyfill lens, by position.
pub const POLYFILL: [FrozenWord; 17] = [
    w(INT_ADD, 2, &["IntAdd", "+"], Some(1), None),
    w(INT_SUB, 2, &["IntSub", "-"], Some(1), None),
    w(INT_MULT, 2, &["IntMult", "*"], Some(1), None),
    w(INT_S_DIV, 2, &["IntSDiv", "/"], Some(1), None),
    w(
        INT_EQUAL,
        2,
        &["IntEqual", "=", "Cmp::EqI32", "Cmp::EqU32"],
        Some(1),
        Some(2),
    ),
    w(
        INT_S_LESS,
        2,
        &["IntSLess", "<", "Cmp::LtI32"],
        Some(1),
        Some(2),
    ),
    w(
        INT_AND,
        2,
        &["IntAnd", "AND", "Filter::and"],
        Some(1),
        Some(2),
    ),
    w(INT_OR, 2, &["IntOr", "OR", "Filter::or"], Some(1), Some(2)),
    w(INT_XOR, 2, &["IntXor", "XOR"], Some(1), None),
    w(
        INT_NOT,
        1,
        &["IntNot", "NOT", "Filter::negate"],
        Some(1),
        Some(2),
    ),
    w(
        POP_COUNT,
        1,
        &["PopCount", "COUNT", "Agg::Count"],
        Some(1),
        Some(2),
    ),
    w(SUM, 2, &["SUM", "Agg::SumI32"], None, Some(1)),
    w(MIN, 2, &["MIN", "Agg::MinI32"], None, Some(1)),
    w(MAX, 2, &["MAX", "Agg::MaxI32"], None, Some(1)),
    w(ANY, 1, &["ANY", "EXISTS", "Agg::Any"], Some(1), Some(2)),
    w(ALL, 1, &["ALL", "Agg::All"], None, Some(1)),
    w(
        KEY_RUNS,
        2,
        &["KEY_RUNS", "COUNT DISTINCT", "Agg::CountDistinctOrderedU32"],
        Some(1),
        Some(2),
    ),
];

/// The lens's declared arities, in position order, for
/// `BindStore::declare_algebra`.
#[must_use]
pub fn polyfill_arities() -> Vec<u8> {
    POLYFILL.iter().map(|w| w.arity).collect()
}

/// The position whose aliases include `label`, if exactly one does.
#[must_use]
pub fn position_of(label: &str) -> Option<u8> {
    let mut hits = POLYFILL
        .iter()
        .enumerate()
        .filter(|(_, w)| w.aliases.contains(&label));
    match (hits.next(), hits.next()) {
        (Some((p, _)), None) => u8::try_from(p).ok(),
        _ => None,
    }
}

/// The polyfill lens's labels in one language: a [`ClassView`] whose
/// [`lens_fields`](ClassView::lens_fields) answers [`POLYFILL_LENS`] with one
/// label per [`POLYFILL`] position. It holds no concept basis: a behaviour
/// lens is read by its full classid (D-DT64-7 O9), never from [`HARDCODED`].
///
/// Every label is an alias of its own position, so
/// [`crate::binding::prepare`] binds the same plan from every language. A
/// field's `predicate_iri` names the word, not the language
/// (`lg:polyfill:f000/<R2IL name>`); only the label changes. A position with
/// no name in the language is labelled with its R2IL name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolyfillView {
    lang: AliasLanguage,
    labels: Vec<FieldRef>,
}

impl PolyfillView {
    /// The view of the polyfill lens in `lang`.
    #[must_use]
    pub fn new(lang: AliasLanguage) -> Self {
        let labels = POLYFILL
            .iter()
            .map(|w| {
                FieldRef::new(
                    format!("lg:polyfill:f000/{}", w.aliases[0]),
                    w.name_in(lang),
                )
            })
            .collect();
        Self { lang, labels }
    }

    /// The language this view labels the positions in.
    #[must_use]
    pub fn language(&self) -> AliasLanguage {
        self.lang
    }
}

impl ClassView for PolyfillView {
    fn fields(&self, _class: ClassId) -> &[FieldRef] {
        &[]
    }
    fn lens_fields(&self, lens: u32) -> Option<&[FieldRef]> {
        (lens == POLYFILL_LENS).then_some(self.labels.as_slice())
    }
    fn template(&self, _class: ClassId) -> DisplayTemplate {
        DisplayTemplate::Detail
    }
    fn dolce_category_id(&self, _class: ClassId) -> u8 {
        0
    }
}

/// A consumer's [`ClassView`] with [`HARDCODED`] answered by a
/// [`PolyfillView`]: the concept `0xFFFF` and every lens under it come from
/// the polyfill view; every other concept and lens, template and DOLCE
/// category, from the consumer's view unchanged. The consumer's own labels
/// for `0xFFFF`, if any, are not consulted.
///
/// This routes one concept and checks nothing else; whether the caller's
/// classes are activated is not this adapter's question (D-DT64-7 L3).
#[derive(Debug, Clone, Copy)]
pub struct WithPolyfill<'a, V: ?Sized> {
    inner: &'a V,
    polyfill: &'a PolyfillView,
}

impl<'a, V: ClassView + ?Sized> WithPolyfill<'a, V> {
    /// `inner` with [`HARDCODED`] answered by `polyfill`.
    #[must_use]
    pub fn new(inner: &'a V, polyfill: &'a PolyfillView) -> Self {
        Self { inner, polyfill }
    }
}

impl<V: ClassView + ?Sized> ClassView for WithPolyfill<'_, V> {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        if class == HARDCODED {
            self.polyfill.fields(class)
        } else {
            self.inner.fields(class)
        }
    }
    fn lens_fields(&self, lens: u32) -> Option<&[FieldRef]> {
        if classid_concept(lens) == HARDCODED {
            self.polyfill.lens_fields(lens)
        } else {
            self.inner.lens_fields(lens)
        }
    }
    fn template(&self, class: ClassId) -> DisplayTemplate {
        if class == HARDCODED {
            self.polyfill.template(class)
        } else {
            self.inner.template(class)
        }
    }
    fn dolce_category_id(&self, class: ClassId) -> u8 {
        if class == HARDCODED {
            self.polyfill.dolce_category_id(class)
        } else {
            self.inner.dolce_category_id(class)
        }
    }
}
