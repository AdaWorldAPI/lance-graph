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
use grammar_constraints_probe::fixtures;
use grammar_constraints_probe::when::{
    self, ComposeRefusal, FramedPopulation, Lexicon, Physical, RuleDesc, RunRefusal, CLASS_COADDR,
    EXPERIENCER, RECIPIENT,
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
    // The left corner is identical for all four tokens; the head frame
    // is the indispensable missing argument of WHEN.
    for (gid, want_recipient, want_experiencer) in [
        (105u64, true, false), // geben: recipient
        (302, false, true),    // gefallen: experiencer
        (405, true, false),    // geben: recipient
        (505, false, false),   // helfen: no declared frame, ABSTAIN
    ] {
        let tok = toks.iter().find(|t| t.gid == gid).expect("fixture token");
        assert_eq!(
            left_corner(tok),
            &DATIVE_CANDIDATES,
            "same left corner: {gid}"
        );
        let local = map.local(gid).expect("known local position");
        assert_eq!(
            has(&recipient_mask, local),
            want_recipient,
            "{gid}: recipient"
        );
        assert_eq!(
            has(&experiencer_mask, local),
            want_experiencer,
            "{gid}: experiencer"
        );
    }

    // A nonmatching syntactic form must not be smuggled into the candidate
    // pool merely because it also carries German dative Case.
    let instrument = toks.iter().find(|t| t.gid == 408).unwrap();
    assert!(left_corner(instrument).is_empty());

    // The selector feeds the composition: only the methods it proposes for a
    // token are composed and consulted for that token.
    let want = vec![
        (105u64, vec![Candidate::Recipient]),
        (302, vec![Candidate::Experiencer]),
        (405, vec![Candidate::Recipient]),
    ];
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

    // Disabling the head-frame operand is a genuine RED control: both
    // previously distinguished arguments now match both rules. This is an
    // intentionally wrong alternate program, not an admitted DO.
    let mut ablated = grammar_constraints_probe::Rules::prepare(
        &grammar_constraints_probe::Ids::bind(&mut vocab),
        grammar_constraints_probe::Operands {
            deprel: true,
            case: true,
            frame: false,
        },
    );
    let base = grammar_constraints_probe::Population::bind(&toks, &mut vocab, &table, &frame)
        .expect("same fixture");
    let wrong = ablated.run(&base, &table, &frame).expect("lab ablation");
    let local_105 = map.local(105).unwrap();
    let local_302 = map.local(302).unwrap();
    assert!(
        has(&wrong.recipient, local_302) && has(&wrong.experiencer, local_105),
        "the frame ablation must create false positives; otherwise this negative control is vacuous"
    );
    assert!(!has(&recipient_mask, local_302));
    assert!(!has(&experiencer_mask, local_105));

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

fn native_role_count(raw: &[u64], role: Orientation) -> usize {
    // This is the EXISTING Quack MatchU64 lowering and mask-risc executor,
    // not a second role-mask evaluator.
    let care = 0b111u64 << 43;
    let pattern = u64::from(role.code()) << 43;
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

#[test]
fn p1_role_selection_preserves_epistemic_independence() {
    for n in [0usize, 1, 63, 64, 65, 256, 257] {
        let mut edges: Vec<u64> = (0..n)
            .map(|i| {
                let mut edge = CausalEdge64(i as u64);
                edge.set_orientation(match i % 3 {
                    0 => Orientation::Recipient,
                    1 => Orientation::Core,
                    _ => Orientation::Beneficiary,
                });
                edge.0
            })
            .collect();

        let expected = accessor_role_count(&edges, Orientation::Recipient);
        assert_eq!(
            native_role_count(&edges, Orientation::Recipient),
            expected,
            "Quack role mask and CE64 decoder differ at n={n}"
        );

        // Flip just Topology2 bit 59 within the valid Open certification.
        // The role field must remain untouched, and selection identical.
        for word in &mut edges {
            *word ^= 1u64 << 59;
        }
        assert_eq!(
            accessor_role_count(&edges, Orientation::Recipient),
            expected
        );
        assert_eq!(
            native_role_count(&edges, Orientation::Recipient),
            expected,
            "Epi5 topology must not silently alter Haspelmath role selection at n={n}"
        );

        if n > 0 {
            // A genuine role mutation MUST change the answer, unlike Epi5.
            let mut first = CausalEdge64(edges[0]);
            assert_eq!(first.orientation(), Orientation::Recipient);
            first.set_orientation(Orientation::Core);
            edges[0] = first.0;
            assert_eq!(
                native_role_count(&edges, Orientation::Recipient),
                expected - 1,
                "changing CE64 orientation must be observable at n={n}"
            );
        }
    }
}
