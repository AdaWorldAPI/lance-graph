//! P0 research fixture: an 8-bit first-construct selector proposes rules;
//! typed WHEN must independently admit the correct meaning or refuse.
//! Test-only and synthetic: no ClassId mint, parser, real game engine, F###
//! execution, new runtime, external effect, or production ABI.
//!
//! The only concrete production read exercised here is CausalEdge64 orientation.
//! Run with:
//! cargo test --manifest-path crates/causal-edge/Cargo.toml --test logic_neutrino_p0

use causal_edge::{CausalEdge64, Orientation};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Family { Source, Role, River }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Action { CountIncomplete, ExternalNotify, SelectRecipient, Navigate }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Rule { family: Family, first: u8, action: Action }

// Local test ordinals only, NOT OGAR RecipeConceptIds or F### positions.
// Source intentionally has two rules with the identical left corner.
const RULES: [Rule; 4] = [
    Rule { family: Family::Source, first: 0x21, action: Action::CountIncomplete },
    Rule { family: Family::Source, first: 0x21, action: Action::ExternalNotify },
    Rule { family: Family::Role, first: 0x42, action: Action::SelectRecipient },
    Rule { family: Family::River, first: 0x63, action: Action::Navigate },
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Witness { source_rev: u64, schema_rev: u64, scope: u16 }
const CURRENT: Witness = Witness { source_rev: 7, schema_rev: 11, scope: 3 };

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Receiver { Checklist, Notifier }

// The witness models a declared reading, not the actual DT64 ClassView trait.
// The river fields model one resource-cost subcase, not full turn legality.
#[derive(Clone, Copy)]
enum Input {
    Source { first: u8, receiver: Receiver, witness: Option<Witness> },
    Role { first: u8, edge: CausalEdge64, witness: Option<Witness> },
    River {
        first: u8, coal: u8, turns_60: u8, speed_delta: u8,
        witness: Option<Witness>,
    },
}

impl Input {
    fn family(self) -> Family {
        match self {
            Self::Source { .. } => Family::Source,
            Self::Role { .. } => Family::Role,
            Self::River { .. } => Family::River,
        }
    }
    fn first(self) -> u8 {
        match self {
            Self::Source { first, .. } |
            Self::Role { first, .. } |
            Self::River { first, .. } => first,
        }
    }
    fn witness(self) -> Option<Witness> {
        match self {
            Self::Source { witness, .. } |
            Self::Role { witness, .. } |
            Self::River { witness, .. } => witness,
        }
    }
}

// ON: candidate discovery must not inspect the receiver, role, coal or revision.
fn candidates(input: Input) -> impl Iterator<Item = Rule> {
    RULES.iter().copied().filter(move |r| {
        r.family == input.family() && r.first == input.first()
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Refusal {
    WrongCandidate, MissingWitness, StaleSource, StaleSchema, WrongScope,
    WrongReceiver, WrongRole, InsufficientCoal, ExternalEffectNotAdmitted,
}

// WHEN: checks are separate from the u8 selector. Ok(Action) is only a
// prepared candidate. In particular, no host effect is executed in this lab.
fn when(rule: Rule, input: Input, expected: Witness) -> Result<Action, Refusal> {
    if rule.family != input.family() || rule.first != input.first() {
        return Err(Refusal::WrongCandidate);
    }
    let w = input.witness().ok_or(Refusal::MissingWitness)?;
    if w.source_rev != expected.source_rev { return Err(Refusal::StaleSource); }
    if w.schema_rev != expected.schema_rev { return Err(Refusal::StaleSchema); }
    if w.scope != expected.scope { return Err(Refusal::WrongScope); }

    match (rule.action, input) {
        (Action::CountIncomplete, Input::Source { receiver: Receiver::Checklist, .. }) =>
            Ok(Action::CountIncomplete),
        (Action::CountIncomplete, Input::Source { .. }) =>
            Err(Refusal::WrongReceiver),
        (Action::ExternalNotify, Input::Source { receiver: Receiver::Notifier, .. }) =>
            Err(Refusal::ExternalEffectNotAdmitted),
        (Action::ExternalNotify, Input::Source { .. }) =>
            Err(Refusal::WrongReceiver),
        (Action::SelectRecipient, Input::Role { edge, .. }) => {
            // MatchU64-shaped predicate on the original packed edge, not a
            // claimed Quack execution. Independent oracle uses orientation().
            let care = 0b111u64 << 43;
            let pattern = (Orientation::Recipient.code() as u64) << 43;
            if ((edge.0 ^ pattern) & care) == 0 {
                Ok(Action::SelectRecipient)
            } else { Err(Refusal::WrongRole) }
        }
        (Action::Navigate, Input::River { coal, turns_60, speed_delta, .. }) => {
            // Extra 60-degree turns/speed changes after the first free step.
            // Geometry, collision, heading and full ruleset remain out of scope.
            let cost = turns_60.saturating_sub(1) as u16
                     + speed_delta.saturating_sub(1) as u16;
            if cost <= coal as u16 {
                Ok(Action::Navigate)
            } else { Err(Refusal::InsufficientCoal) }
        }
        _ => Err(Refusal::WrongCandidate),
    }
}

fn admitted(input: Input, expected: Witness) -> Vec<Action> {
    candidates(input).filter_map(|r| when(r, input, expected).ok()).collect()
}

// Independent scalar oracle for the SPECIFIED SMALL FIXTURE, not for Rails or
// full Mississippi Queen. It does not call candidates(), when(), or bit masks.
fn reference(input: Input, expected: Witness) -> Vec<Action> {
    if input.witness() != Some(expected) { return Vec::new(); }
    match input {
        Input::Source { first: 0x21, receiver: Receiver::Checklist, .. } =>
            vec![Action::CountIncomplete],
        Input::Role { first: 0x42, edge, .. }
            if edge.orientation() == Orientation::Recipient =>
            vec![Action::SelectRecipient],
        Input::River { first: 0x63, coal, turns_60, speed_delta, .. } => {
            let mut spent = 0u16;
            for _ in 1..turns_60 { spent += 1; }
            for _ in 1..speed_delta { spent += 1; }
            if spent <= coal as u16 { vec![Action::Navigate] }
            else { Vec::new() }
        }
        _ => Vec::new(),
    }
}

fn edge(role: Orientation) -> CausalEdge64 {
    let mut word = CausalEdge64(0);
    word.set_orientation(role);
    word
}

#[test]
fn ambiguous_first_byte_requires_receiver_typing() {
    let checklist = Input::Source {
        first: 0x21, receiver: Receiver::Checklist, witness: Some(CURRENT),
    };
    let notifier = Input::Source {
        first: 0x21, receiver: Receiver::Notifier, witness: Some(CURRENT),
    };
    assert_eq!(candidates(checklist).count(), 2);
    assert_eq!(candidates(notifier).count(), 2);
    assert_eq!(admitted(checklist, CURRENT), vec![Action::CountIncomplete]);
    assert_eq!(admitted(notifier, CURRENT), Vec::<Action>::new());
    assert_eq!(admitted(checklist, CURRENT), reference(checklist, CURRENT));
    assert_eq!(admitted(notifier, CURRENT), reference(notifier, CURRENT));
    let notify = candidates(notifier).find(|r| r.action == Action::ExternalNotify).unwrap();
    assert_eq!(when(notify, notifier, CURRENT), Err(Refusal::ExternalEffectNotAdmitted));
}

#[test]
fn ce64_orientation_selection_is_independent_of_epistemic_bits() {
    for role in Orientation::ALL {
        let word = edge(role);
        let original = Input::Role { first: 0x42, edge: word, witness: Some(CURRENT) };
        assert_eq!(admitted(original, CURRENT), reference(original, CURRENT), "{role:?}");
        for raw in 0u64..32 {
            // The altered raw5 values may include reserved codes. This tests
            // only disjoint bit positions, never their epistemic validity.
            let changed = CausalEdge64((word.0 & !(0x1fu64 << 59)) | (raw << 59));
            let input = Input::Role { first: 0x42, edge: changed, witness: Some(CURRENT) };
            assert_eq!(admitted(input, CURRENT), admitted(original, CURRENT), "{role:?}/{raw}");
        }
    }
    let core = Input::Role {
        first: 0x42, edge: edge(Orientation::Core), witness: Some(CURRENT),
    };
    let recipient = Input::Role {
        first: 0x42, edge: edge(Orientation::Recipient), witness: Some(CURRENT),
    };
    assert!(admitted(core, CURRENT).is_empty());
    assert_eq!(admitted(recipient, CURRENT), vec![Action::SelectRecipient]);
}

#[test]
fn same_river_geometry_with_different_coal_changes_candidate_admission() {
    for coal in 0..=6 {
        for turns in 0..=4 {
            for delta in 0..=3 {
                let input = Input::River {
                    first: 0x63, coal, turns_60: turns, speed_delta: delta,
                    witness: Some(CURRENT),
                };
                assert_eq!(admitted(input, CURRENT), reference(input, CURRENT),
                           "coal={coal}, turns={turns}, delta={delta}");
            }
        }
    }
    let empty = Input::River {
        first: 0x63, coal: 0, turns_60: 3, speed_delta: 0, witness: Some(CURRENT),
    };
    let full = Input::River {
        first: 0x63, coal: 6, turns_60: 3, speed_delta: 0, witness: Some(CURRENT),
    };
    assert!(admitted(empty, CURRENT).is_empty());
    assert_eq!(admitted(full, CURRENT), vec![Action::Navigate]);
    assert_eq!(when(candidates(empty).next().unwrap(), empty, CURRENT),
               Err(Refusal::InsufficientCoal));
}

#[test]
fn missing_stale_or_wrong_scope_witness_refuses_by_name() {
    let src = Witness { source_rev: CURRENT.source_rev + 1, ..CURRENT };
    let schema = Witness { schema_rev: CURRENT.schema_rev + 1, ..CURRENT };
    let scope = Witness { scope: CURRENT.scope + 1, ..CURRENT };
    for (witness, reason) in [
        (None, Refusal::MissingWitness),
        (Some(src), Refusal::StaleSource),
        (Some(schema), Refusal::StaleSchema),
        (Some(scope), Refusal::WrongScope),
    ] {
        let input = Input::Source {
            first: 0x21, receiver: Receiver::Checklist, witness,
        };
        let candidate = candidates(input).next().unwrap();
        assert_eq!(when(candidate, input, CURRENT), Err(reason));
        assert!(admitted(input, CURRENT).is_empty());
        assert_eq!(admitted(input, CURRENT), reference(input, CURRENT));
    }
}

#[test]
fn disabling_when_would_create_four_false_positives() {
    let wrong_cases = [
        Input::Source {
            first: 0x21, receiver: Receiver::Notifier, witness: Some(CURRENT),
        },
        Input::Role {
            first: 0x42, edge: edge(Orientation::Core), witness: Some(CURRENT),
        },
        Input::River {
            first: 0x63, coal: 0, turns_60: 3, speed_delta: 0,
            witness: Some(CURRENT),
        },
        Input::Source {
            first: 0x21, receiver: Receiver::Checklist,
            witness: Some(Witness { source_rev: CURRENT.source_rev + 1, ..CURRENT }),
        },
    ];
    for input in wrong_cases {
        // Deliberately broken ablation: the ON stage is treated as authority.
        let unsafe_prefix_only: Vec<Action> = candidates(input).map(|r| r.action).collect();
        assert!(!unsafe_prefix_only.is_empty());
        assert_ne!(unsafe_prefix_only, reference(input, CURRENT));
        assert_eq!(admitted(input, CURRENT), reference(input, CURRENT));
        assert!(admitted(input, CURRENT).is_empty());
    }
}
