//! Logic-neutrino P0/P1, lab-only: a byte selects candidate rules;
//! the existing DT64 WHEN + Quack program decides which candidates hold.
//!
//! No production selector, DSL, registered recipe, ClassId or CE64 bit change.
//! P0 is not a claim of generic Rails/MeTTa/game semantics: it uses the
//! independently annotated German dative fixture already in this lab.
//!
//! Run (requires the sibling AdaWorldAPI/ndarray checkout, per repo policy):
//! cargo test --manifest-path crates/grammar-constraints-probe/Cargo.toml
//!   --test logic_neutrino_p0
//!
//! These tests must FAIL if the head-frame guard is erased, a description
//! from another rule is substituted, or a lexicon revision is reused.

use causal_edge::{CausalEdge64, Orientation};
use grammar_constraints_probe::fixtures::{self, Gold, GOLD};
use grammar_constraints_probe::when::{
    self, ComposeRefusal, FramedPopulation, Lexicon, Physical, RuleDesc, RunRefusal, CLASS_COADDR,
    EXPERIENCER, FRAME_NONE, RECIPIENT,
};
use grammar_constraints_probe::{has, tile, Tok, Vocab, TILE};
use lance_graph_dt64::dr::{Frame, MapTable};
use lance_graph_dt64::{BindStore, MethodId};
use lance_graph_mask_risc::{execute_into, Foreign, LaneRef, Out, Planes, Scratch, Value};
use lance_graph_quack::{lower, Agg, Cmp, Col, Filter, Query};

/// LAB-LOCAL selector positions, not OGAR RecipeConceptId, F### word, or
/// CausalEdge64 opcode. Their only invariant is that both surviving rule
/// candidates are retained when the observed syntax is ambiguous.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum Candidate {
    Recipient = 1,
    Experiencer = 2,
}

const DATIVE_CANDIDATES: [Candidate; 2] = [Candidate::Recipient, Candidate::Experiencer];

impl Candidate {
    /// The DT64 method a candidate names. The selector's output reaches the
    /// store only through this.
    fn method(self) -> MethodId {
        match self {
            Candidate::Recipient => RECIPIENT,
            Candidate::Experiencer => EXPERIENCER,
        }
    }

    fn desc(self) -> RuleDesc {
        match self {
            Candidate::Recipient => RuleDesc::recipient(),
            Candidate::Experiencer => RuleDesc::experiencer(),
        }
    }
}

/// A selector that drops one of the two candidates.
fn left_corner_recipient_only(t: &Tok) -> &'static [Candidate] {
    if t.deprel == "obl:arg" && t.case == "Dat" {
        &DATIVE_CANDIDATES[..1]
    } else {
        &[]
    }
}

/// A selector that proposes both candidates for every dative token, whatever
/// its relation (determiners and the instrument included).
fn left_corner_any_dative(t: &Tok) -> &'static [Candidate] {
    if t.case == "Dat" {
        &DATIVE_CANDIDATES
    } else {
        &[]
    }
}

/// The roles each token is given when ONLY the methods `selector` proposes
/// for that token are consulted. A method is composed the first time some
/// token proposes it and never otherwise. Returns the tokens that got a role,
/// in corpus order, and how many methods were composed.
#[allow(clippy::too_many_arguments)]
fn roles_under(
    selector: fn(&Tok) -> &'static [Candidate],
    toks: &[Tok],
    store: &BindStore,
    vocab: &Vocab,
    lex: &Lexicon,
    pop: &FramedPopulation,
    table: &MapTable,
    frame: &Frame,
) -> (Vec<(u64, Vec<Candidate>)>, usize) {
    let map = table.map(frame, TILE).expect("same sealed frame");
    let where_mask = where_all(pop.base.n);
    let mut masks: [Option<[u64; 4]>; 2] = [None, None];
    let mut composed = 0;
    let mut out = Vec::new();
    for tok in toks {
        let local = map.local(tok.gid).expect("known local position");
        let mut roles = Vec::new();
        for &cand in selector(tok) {
            let slot = cand as usize - 1;
            let mask = *masks[slot].get_or_insert_with(|| {
                composed += 1;
                let mref = store
                    .method_ref(CLASS_COADDR, cand.method())
                    .expect("registered candidate");
                when::compose(
                    store,
                    &mref,
                    &cand.desc(),
                    Physical::CoAddressed,
                    vocab,
                    lex,
                )
                .expect("candidate plan")
                .run(pop, table, frame, &where_mask)
                .expect("candidate mask")
            });
            if has(&mask, local) {
                roles.push(cand);
            }
        }
        if !roles.is_empty() {
            out.push((tok.gid, roles));
        }
    }
    (out, composed)
}

/// A single left-corner discriminator on *already harvested* UD facts.
/// This deliberately cannot inspect the head, execute a rule, or certify
/// the result. It does not pretend to be a general left-corner parser.
fn left_corner(t: &Tok) -> &'static [Candidate] {
    if t.deprel == "obl:arg" && t.case == "Dat" {
        &DATIVE_CANDIDATES
    } else {
        &[]
    }
}

fn where_all(n: usize) -> Vec<u64> {
    let mut mask = vec![u64::MAX; n.div_ceil(64)];
    if !n.is_multiple_of(64) {
        *mask.last_mut().expect("nonempty when n % 64 != 0") = (1u64 << (n % 64)) - 1;
    }
    mask
}

#[test]
fn p0_byte_candidates_require_the_existing_typed_when() {
    let toks = fixtures::corpus();
    let (table, frame) = tile(11, &fixtures::gids(&toks));
    let mut vocab = Vocab::new();
    let lex = Lexicon::v0();
    let pop = FramedPopulation::produce(&toks, &mut vocab, &table, &frame, &lex)
        .expect("real lab population");
    let mut store = BindStore::new();
    let recipient = RuleDesc::recipient();
    let experiencer = RuleDesc::experiencer();
    when::register(
        &mut store,
        CLASS_COADDR,
        Physical::CoAddressed,
        &[recipient.clone(), experiencer.clone()],
        &vocab,
        &lex,
    )
    .expect("both rules carry their distinct guarded DT64 identities");

    let recipient_ref = store
        .method_ref(CLASS_COADDR, RECIPIENT)
        .expect("recipient ref");
    let experiencer_ref = store
        .method_ref(CLASS_COADDR, EXPERIENCER)
        .expect("experiencer ref");
    let mut recipient_plan = when::compose(
        &store,
        &recipient_ref,
        &recipient,
        Physical::CoAddressed,
        &vocab,
        &lex,
    )
    .expect("recipient plan");
    let mut experiencer_plan = when::compose(
        &store,
        &experiencer_ref,
        &experiencer,
        Physical::CoAddressed,
        &vocab,
        &lex,
    )
    .expect("experiencer plan");
    let where_mask = where_all(pop.base.n);
    let recipient_mask = recipient_plan
        .run(&pop, &table, &frame, &where_mask)
        .expect("admitted lab recipient mask");
    let experiencer_mask = experiencer_plan
        .run(&pop, &table, &frame, &where_mask)
        .expect("admitted lab experiencer mask");

    let map = table.map(&frame, TILE).expect("same sealed frame");
    // Expectations come from the fixture's gold annotation, which the rules
    // never read. The left corner is identical for the four `obl:arg` datives;
    // the head frame is the indispensable missing argument of WHEN.
    let mut same_corner = 0;
    for &(gid, gold) in &GOLD {
        let tok = toks.iter().find(|t| t.gid == gid).expect("gold token");
        let local = map.local(gid).expect("known local position");
        assert_eq!(
            has(&recipient_mask, local),
            gold == Gold::Recipient,
            "{gid}: recipient vs gold {gold:?}"
        );
        assert_eq!(
            has(&experiencer_mask, local),
            gold == Gold::Experiencer,
            "{gid}: experiencer vs gold {gold:?}"
        );
        if left_corner(tok) == DATIVE_CANDIDATES {
            same_corner += 1;
        }
    }
    assert_eq!(
        same_corner, 4,
        "105, 302, 405 and 505 share one left corner"
    );
    assert_eq!(
        GOLD.iter()
            .filter(|g| matches!(g.1, Gold::Recipient | Gold::Experiencer))
            .count(),
        3,
        "the gold must contain both roles or the comparison above is vacuous"
    );

    // A nonmatching syntactic form must not be smuggled into the candidate
    // pool merely because it also carries German dative Case.
    let instrument = toks.iter().find(|t| t.gid == 408).unwrap();
    assert!(left_corner(instrument).is_empty());

    // The selector feeds the composition: only the methods it proposes for a
    // token are composed and consulted for that token.
    let want: Vec<(u64, Vec<Candidate>)> = GOLD
        .iter()
        .filter_map(|&(gid, gold)| match gold {
            Gold::Recipient => Some((gid, vec![Candidate::Recipient])),
            Gold::Experiencer => Some((gid, vec![Candidate::Experiencer])),
            _ => None,
        })
        .collect();
    let run = |selector: fn(&Tok) -> &'static [Candidate]| {
        roles_under(selector, &toks, &store, &vocab, &lex, &pop, &table, &frame)
    };
    assert_eq!(run(left_corner), (want.clone(), 2));
    // Dropping a candidate from the selector loses the answer only that
    // candidate's method can give, and its method is never composed.
    let (narrow, composed) = run(left_corner_recipient_only);
    assert_eq!(composed, 1);
    assert_eq!(
        narrow,
        vec![
            (105, vec![Candidate::Recipient]),
            (405, vec![Candidate::Recipient])
        ],
        "a selector without Experiencer must lose 302"
    );
    assert_ne!(narrow, want);
    // Proposing too much changes nothing: the guards reject the extra
    // tokens (dative determiners, the instrument 408).
    let extra = toks
        .iter()
        .filter(|t| !left_corner_any_dative(t).is_empty() && left_corner(t).is_empty())
        .count();
    assert!(
        extra >= 4,
        "the wide selector must propose tokens the narrow one does not"
    );
    assert_eq!(run(left_corner_any_dative), (want, 2));
    // Proposing nothing composes nothing and answers nothing.
    assert_eq!(run(|_| &[]), (Vec::new(), 0));

    // The would-be Rosetta byte does NOT own the semantic rule. The same
    // candidate selector is compatible with two different DT64 methods,
    // but swapping the full rule description must be rejected.
    assert!(matches!(
        when::compose(
            &store,
            &recipient_ref,
            &experiencer,
            Physical::CoAddressed,
            &vocab,
            &lex,
        ),
        Err(ComposeRefusal::DescriptionMismatch)
    ));
    let mut dropped_frame = recipient.clone();
    dropped_frame.frame = None;
    assert!(matches!(
        when::compose(
            &store,
            &recipient_ref,
            &dropped_frame,
            Physical::CoAddressed,
            &vocab,
            &lex,
        ),
        Err(ComposeRefusal::DescriptionMismatch)
    ));

    // Removing the head-frame condition is the RED control, on the same
    // DT64-composed path as the answers above: two frame-less methods are
    // registered in their own store, admitted, composed and run. Both now
    // match every `obl:arg` dative, the unframed 505 included. This is an
    // intentionally wrong pair of methods, not an admitted DO.
    let frameless = |mut d: RuleDesc| {
        d.frame = None;
        d
    };
    let wrong_rules = [frameless(recipient.clone()), frameless(experiencer.clone())];
    let mut wrong_store = BindStore::new();
    when::register(
        &mut wrong_store,
        CLASS_COADDR,
        Physical::CoAddressed,
        &wrong_rules,
        &vocab,
        &lex,
    )
    .expect("frame-less methods register");
    let wrong_mask = |d: &RuleDesc| {
        let mref = wrong_store
            .method_ref(CLASS_COADDR, d.method)
            .expect("frame-less ref");
        when::compose(&wrong_store, &mref, d, Physical::CoAddressed, &vocab, &lex)
            .expect("frame-less plan")
            .run(&pop, &table, &frame, &where_mask)
            .expect("frame-less mask")
    };
    let wrong_recipient = wrong_mask(&wrong_rules[0]);
    let wrong_experiencer = wrong_mask(&wrong_rules[1]);
    let at = |gid: u64| map.local(gid).expect("known local position");
    assert!(
        has(&wrong_recipient, at(302)) && has(&wrong_experiencer, at(105)),
        "the frame ablation must create false positives; otherwise this negative control is vacuous"
    );
    assert!(has(&wrong_recipient, at(505)) && has(&wrong_experiencer, at(505)));
    assert!(!has(&wrong_recipient, at(408)) && !has(&wrong_experiencer, at(408)));
    assert!(!has(&recipient_mask, at(302)));
    assert!(!has(&experiencer_mask, at(105)));

    // Abstention is an answer of its own, not the absence of one. A third
    // method, `obl:arg AND Dat AND head_frame = NONE`, goes through the same
    // registration and composition and selects exactly 505: the dative whose
    // head has no frame entry. 408 (not a candidate) and 105 (a recipient)
    // are both outside it, so "no role" has two distinguishable causes.
    let abstain = RuleDesc {
        method: MethodId(3),
        name: "AbstainNoFrame",
        deprel: Some("obl:arg"),
        case: Some("Dat"),
        frame: Some(FRAME_NONE),
    };
    let mut abstain_store = BindStore::new();
    when::register(
        &mut abstain_store,
        CLASS_COADDR,
        Physical::CoAddressed,
        &[recipient.clone(), experiencer.clone(), abstain.clone()],
        &vocab,
        &lex,
    )
    .expect("three methods register");
    let abstain_ref = abstain_store
        .method_ref(CLASS_COADDR, abstain.method)
        .expect("abstain ref");
    let abstain_mask = when::compose(
        &abstain_store,
        &abstain_ref,
        &abstain,
        Physical::CoAddressed,
        &vocab,
        &lex,
    )
    .expect("abstain plan")
    .run(&pop, &table, &frame, &where_mask)
    .expect("abstain mask");
    let abstained: Vec<u64> = toks
        .iter()
        .map(|t| t.gid)
        .filter(|&g| has(&abstain_mask, at(g)))
        .collect();
    assert_eq!(abstained, vec![505]);
    // The three answers are disjoint and cover every left-corner candidate.
    for tok in toks.iter().filter(|t| !left_corner(t).is_empty()) {
        let l = at(tok.gid);
        let answers = [&recipient_mask, &experiencer_mask, &abstain_mask]
            .iter()
            .filter(|m| has(m, l))
            .count();
        assert_eq!(
            answers, 1,
            "{}: exactly one of role / role / abstain",
            tok.gid
        );
    }
    // Independent cross-check: the older hand-built abstention program
    // (`EqU32Via` through the head pointer, no producer lane) names the same
    // set.
    let mut older = grammar_constraints_probe::Rules::prepare(
        &grammar_constraints_probe::Ids::bind(&mut vocab),
        grammar_constraints_probe::Operands::ALL,
    );
    let base = grammar_constraints_probe::Population::bind(&toks, &mut vocab, &table, &frame)
        .expect("same fixture");
    let readout = older.run(&base, &table, &frame).expect("older readout");
    assert_eq!(readout.abstain_frame, abstain_mask);

    // Versioned producer facts are not transferable to another lexicon,
    // even though the byte selector and the source token shapes are equal.
    let swapped = Lexicon::swapped();
    // The lexicon is part of the method's identity: a method bound under one
    // lexicon does not compose under another, so no plan exists that could
    // read the swapped lane as Recipient.
    assert!(matches!(
        when::compose(
            &store,
            &recipient_ref,
            &recipient,
            Physical::CoAddressed,
            &vocab,
            &swapped,
        ),
        Err(ComposeRefusal::DescriptionMismatch)
    ));
    assert_ne!(
        recipient
            .guard(Physical::CoAddressed, &vocab, &lex)
            .witness(),
        recipient
            .guard(Physical::CoAddressed, &vocab, &swapped)
            .witness()
    );
    let swapped_pop = FramedPopulation::produce(&toks, &mut vocab, &table, &frame, &swapped)
        .expect("alternative lab reading");
    assert!(matches!(
        recipient_plan.run(&swapped_pop, &table, &frame, &where_mask),
        Err(RunRefusal::LexiconRevision { .. })
    ));
}

/// F. Das gibt mir zu denken. ("That makes me think.")
///
/// Hand-annotated in the same UD convention as the fixture corpus; not
/// checked against a treebank. `mir` is an `obl:arg` dative under *geben*,
/// so the lexicon's GIVE frame reads it as a recipient. In this idiom nothing
/// is transferred: the single-annotator gold role is Experiencer.
const F: [Tok; 6] = [
    Tok {
        gid: 601,
        form: "Das",
        lemma: "der",
        deprel: "nsubj",
        case: "Nom",
        head: 602,
    },
    Tok {
        gid: 602,
        form: "gibt",
        lemma: "geben",
        deprel: "root",
        case: "",
        head: 0,
    },
    Tok {
        gid: 603,
        form: "mir",
        lemma: "ich",
        deprel: "obl:arg",
        case: "Dat",
        head: 602,
    },
    Tok {
        gid: 604,
        form: "zu",
        lemma: "zu",
        deprel: "mark",
        case: "",
        head: 605,
    },
    Tok {
        gid: 605,
        form: "denken",
        lemma: "denken",
        deprel: "xcomp",
        case: "",
        head: 602,
    },
    Tok {
        gid: 606,
        form: ".",
        lemma: ".",
        deprel: "punct",
        case: "",
        head: 602,
    },
];
const F_GOLD: (u64, Gold) = (603, Gold::Experiencer);

/// The gold annotation is able to disagree with the lexicon. On the fixture
/// corpus every gold role is a function of the head lemma, so rules and gold
/// cannot differ there. Sentence F is a case where they do, and the test pins
/// the disagreement: a frame keyed by the head lemma alone gets it wrong.
#[test]
fn p0_gold_can_disagree_with_a_lemma_keyed_frame() {
    let mut toks = fixtures::corpus();
    toks.extend_from_slice(&F);
    let (table, frame) = tile(12, &fixtures::gids(&toks));
    let mut vocab = Vocab::new();
    let lex = Lexicon::v0();
    let pop =
        FramedPopulation::produce(&toks, &mut vocab, &table, &frame, &lex).expect("corpus plus F");
    let mut store = BindStore::new();
    when::register(
        &mut store,
        CLASS_COADDR,
        Physical::CoAddressed,
        &[RuleDesc::recipient(), RuleDesc::experiencer()],
        &vocab,
        &lex,
    )
    .expect("two methods");
    let (got, _) = roles_under(
        left_corner,
        &toks,
        &store,
        &vocab,
        &lex,
        &pop,
        &table,
        &frame,
    );
    let role_of = |gid: u64| {
        got.iter()
            .find(|g| g.0 == gid)
            .map(|g| g.1.clone())
            .unwrap_or_default()
    };
    let agrees = |gid: u64, gold: Gold| match gold {
        Gold::Recipient => role_of(gid) == [Candidate::Recipient],
        Gold::Experiencer => role_of(gid) == [Candidate::Experiencer],
        _ => role_of(gid).is_empty(),
    };
    // The fixture gold still agrees everywhere with F present.
    for &(gid, gold) in &GOLD {
        assert!(agrees(gid, gold), "{gid}: {gold:?}");
    }
    // F: same left corner as the other datives, the rules say Recipient, the
    // gold says Experiencer.
    let mir = toks.iter().find(|t| t.gid == F_GOLD.0).expect("mir");
    assert_eq!(left_corner(mir), &DATIVE_CANDIDATES);
    assert_eq!(role_of(F_GOLD.0), [Candidate::Recipient]);
    assert!(!agrees(F_GOLD.0, F_GOLD.1));
    let scored = GOLD.len() + 1;
    let correct =
        GOLD.iter().filter(|g| agrees(g.0, g.1)).count() + usize::from(agrees(F_GOLD.0, F_GOLD.1));
    assert_eq!((correct, scored), (6, 7), "rules against gold");
}

fn native_role_count(raw: &[u64], role: Orientation) -> usize {
    native_count(raw, u64::from(role.code()) << 43, 0b111 << 43)
}

fn native_count(raw: &[u64], pattern: u64, care: u64) -> usize {
    // This is the EXISTING Quack MatchU64 lowering and mask-risc executor,
    // not a second role-mask evaluator.
    let program = lower(&Query {
        filter: Filter::cmp(Col(0), Cmp::MatchU64 { pattern, care }),
        agg: Agg::Count,
    })
    .expect("existing 64-bit mask predicate");
    let lanes = [LaneRef::U64(raw)];
    let planes = Planes {
        n_rows: raw.len(),
        masks: &[],
        lanes: &lanes,
    };
    let mut scratch = Scratch::for_program(&program, raw.len()).expect("scratch");
    match execute_into(&program, &planes, &Foreign::NONE, &mut scratch, Out::None)
        .expect("native role count")
    {
        Value::Count(n) => n,
        other => panic!("unexpected terminal: {other:?}"),
    }
}

/// The independent oracle uses the canonical CE64 decoder, not the bit
/// pattern or Quack's predicate. This proves equality only for role
/// selection, not for inference, Epi5 certification or semantic equivalence.
fn accessor_role_count(raw: &[u64], role: Orientation) -> usize {
    raw.iter()
        .filter(|&&word| CausalEdge64(word).orientation() == role)
        .count()
}

/// Deterministic dense words: every CE64 field takes varied values.
fn dense_words(n: usize, seed: u64) -> Vec<u64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        })
        .collect()
}

#[test]
fn p1_role_selection_preserves_epistemic_independence() {
    for n in [0usize, 1, 63, 64, 65, 256, 257, 4099] {
        let mut edges = dense_words(n, 0x9E37_79B9_7F4A_7C15 ^ n as u64);

        // Every role, against the canonical decoder; the eight selections
        // partition the lane.
        let mut total = 0;
        for role in Orientation::ALL {
            let expected = accessor_role_count(&edges, role);
            assert_eq!(
                native_role_count(&edges, role),
                expected,
                "Quack role mask and CE64 decoder differ at n={n} for {role:?}"
            );
            total += expected;
        }
        assert_eq!(total, n);

        let expected = accessor_role_count(&edges, Orientation::Recipient);
        if n >= 256 {
            // The background is dense enough to expose a wrong mask: a care
            // mask one bit too wide (bit 46, the inference mantissa's low
            // bit) gives another count. On a zero background it would not.
            assert!(expected > 0 && expected < n);
            assert_ne!(
                native_count(&edges, 2u64 << 43, 0b1111 << 43),
                expected,
                "a too-wide care mask must be visible at n={n}"
            );
        }

        // Flip every bit of the Epi5 field (59..63), one at a time, in every
        // word. The role field must remain untouched, and selection identical.
        for bit in 59..64 {
            for word in &mut edges {
                *word ^= 1u64 << bit;
            }
            assert_eq!(
                accessor_role_count(&edges, Orientation::Recipient),
                expected
            );
            assert_eq!(
                native_role_count(&edges, Orientation::Recipient),
                expected,
                "Epi5 bit {bit} must not alter Haspelmath role selection at n={n}"
            );
        }

        if let Some(i) = edges
            .iter()
            .position(|&w| CausalEdge64(w).orientation() == Orientation::Recipient)
        {
            // A genuine role mutation MUST change the answer, unlike Epi5.
            let mut first = CausalEdge64(edges[i]);
            first.set_orientation(Orientation::Core);
            edges[i] = first.0;
            assert_eq!(
                native_role_count(&edges, Orientation::Recipient),
                expected - 1,
                "changing CE64 orientation must be observable at n={n}"
            );
        }
    }
}
