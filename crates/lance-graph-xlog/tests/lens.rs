//! XLOG's behaviour lens `0xFFFF :: 0xF100`: positions, aliases, the
//! binding in the bind store, and the range it occupies.

use lance_graph_contract::ogar_codebook::{classid_canon, classid_custom, HARDCODED_BEHAVIOUR};
use lance_graph_dt64::{AlgebraRef, BindStore, Extent, Refusal, SchemaRev};
use lance_graph_xlog::lens::{declare, position_of, word_of, XLOG_LENS, XLOG_WORDS};

#[test]
fn the_lens_sits_in_the_xlog_concern_range() {
    assert_eq!(classid_canon(XLOG_LENS), HARDCODED_BEHAVIOUR);
    assert_eq!(classid_custom(XLOG_LENS) & 0xFF00, 0xF100);
}

#[test]
fn every_alias_selects_its_own_position_and_nothing_else_does() {
    for (p, w) in XLOG_WORDS.iter().enumerate() {
        for alias in w.aliases {
            assert_eq!(position_of(alias), Some(p as u8), "{alias}");
            assert_eq!(
                word_of(alias),
                Some(AlgebraRef {
                    lens: XLOG_LENS,
                    word: p as u8
                })
            );
        }
    }
    // Relational names and XLOG's empty `fail` helper are not words here.
    for name in ["join", "edge", "__xlog_meta_fail", "Functor", ""] {
        assert_eq!(position_of(name), None, "{name}");
    }
    assert_eq!(position_of("functor"), Some(0));
    assert_eq!(position_of("=.."), Some(2));
}

#[test]
fn declared_column_counts_are_what_the_store_reads_back() {
    let mut store = BindStore::new();
    declare(&mut store).unwrap();
    declare(&mut store).unwrap();
    for (p, w) in XLOG_WORDS.iter().enumerate() {
        let word = AlgebraRef {
            lens: XLOG_LENS,
            word: p as u8,
        };
        assert_eq!(store.arity(word), Some(w.columns));
    }
    assert_eq!(
        store.arity(AlgebraRef {
            lens: XLOG_LENS,
            word: 3
        }),
        None
    );

    // A different binding of the same lens is refused.
    let mut other = BindStore::new();
    other
        .declare_algebra(XLOG_LENS, SchemaRev(1), Extent::DR64_CUBE, vec![1, 1, 1])
        .unwrap();
    assert!(matches!(
        declare(&mut other),
        Err(Refusal::AlgebraImmutable(_))
    ));
}
