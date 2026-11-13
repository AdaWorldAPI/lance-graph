//! Game arm: chess hidden state, judged by shakmaty.
//!
//! The rules read lanes copied field-by-field from `shakmaty::Setup` (piece
//! on a square, a castling-rights bit). They contain no chess logic. The
//! authority is `Position::legal_moves`, which shares no code with them.

use hidden_state_twins_probe::{digest, Family, Rule, TwinRefusal};
use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::{CastlingMode, Chess, Color, EnPassantMode, Move, Position, Role, Square};

const CLASS_CASTLE: u32 = 0x7F31_0001;
const CLASS_DRAW: u32 = 0x7F31_0002;

// Lanes of a position row (castling family).
const KING_E1: u16 = 0;
const ROOK_H1: u16 = 1;
const F1_EMPTY: u16 = 2;
const G1_EMPTY: u16 = 3;
const RIGHT_H1: u16 = 4;

const CASTLE: u8 = 1 << 0;
const STEP_F1: u8 = 1 << 1;

fn position(fen: &str) -> Chess {
    fen.parse::<Fen>()
        .expect("fen")
        .into_position(CastlingMode::Standard)
        .expect("legal position")
}

/// The producer: one raw read of `Setup` per lane. No rule of chess.
fn castle_lanes(positions: &[Chess]) -> Vec<Vec<u32>> {
    let mut lanes = vec![Vec::new(); 5];
    for p in positions {
        let s = p.to_setup(EnPassantMode::Legal);
        let white = |sq, role| {
            s.board
                .piece_at(sq)
                .is_some_and(|pc| pc.color == Color::White && pc.role == role)
        };
        lanes[usize::from(KING_E1)].push(u32::from(white(Square::E1, Role::King)));
        lanes[usize::from(ROOK_H1)].push(u32::from(white(Square::H1, Role::Rook)));
        lanes[usize::from(F1_EMPTY)].push(u32::from(s.board.piece_at(Square::F1).is_none()));
        lanes[usize::from(G1_EMPTY)].push(u32::from(s.board.piece_at(Square::G1).is_none()));
        lanes[usize::from(RIGHT_H1)].push(u32::from(s.castling_rights.contains(Square::H1)));
    }
    lanes
}

/// The selector: what the BOARD looks like, nothing else. It cannot see the
/// castling right, so twins with one board get one byte.
fn left_corner(lanes: &[Vec<u32>], r: usize) -> u8 {
    let on = |l: u16| lanes[usize::from(l)][r] == 1;
    let mut c = 0;
    if on(KING_E1) && on(ROOK_H1) && on(F1_EMPTY) && on(G1_EMPTY) {
        c |= CASTLE;
    }
    if on(KING_E1) && on(F1_EMPTY) {
        c |= STEP_F1;
    }
    c
}

fn castle_family(with_right: bool) -> Family {
    let mut castle = vec![
        (KING_E1, vec![1]),
        (ROOK_H1, vec![1]),
        (F1_EMPTY, vec![1]),
        (G1_EMPTY, vec![1]),
    ];
    if with_right {
        castle.push((RIGHT_H1, vec![1]));
    }
    Family {
        class: CLASS_CASTLE,
        lanes: 5,
        rules: vec![
            Rule {
                name: "CastleKingside",
                when: castle,
            },
            Rule {
                name: "KingStepF1",
                when: vec![(KING_E1, vec![1]), (F1_EMPTY, vec![1])],
            },
        ],
        producer: digest(&["shakmaty::Setup", "raw-fields-v1"]),
    }
}

/// The authority: which of the two moves shakmaty generates as legal.
fn authority(p: &Chess) -> u8 {
    let mut a = 0;
    for m in p.legal_moves() {
        match m {
            Move::Castle {
                king: Square::E1,
                rook: Square::H1,
            } => a |= CASTLE,
            Move::Normal {
                role: Role::King,
                from: Square::E1,
                to: Square::F1,
                ..
            } => a |= STEP_F1,
            _ => {}
        }
    }
    a
}

// A: right present.  B: the SAME board, right absent.  C: right present, a
// black rook on f8 attacks f1.  D: a bishop stands on f1.
const A: &str = "4k3/8/8/8/8/8/8/4K2R w K - 0 1";
const B: &str = "4k3/8/8/8/8/8/8/4K2R w - - 0 1";
const C: &str = "k4r2/8/8/8/8/8/8/4K2R w K - 0 1";
const D: &str = "4k3/8/8/8/8/8/8/4KB1R w K - 0 1";

#[test]
fn castling_twins_need_the_right_as_a_lane() {
    let positions: Vec<Chess> = [A, B, C, D].iter().map(|f| position(f)).collect();
    let lanes = castle_lanes(&positions);
    let candidates: Vec<u8> = (0..positions.len())
        .map(|r| left_corner(&lanes, r))
        .collect();
    let want: Vec<u8> = positions.iter().map(authority).collect();

    // A and B are twins: one board, one candidate byte, two authorities.
    let board = |p: &Chess| p.to_setup(EnPassantMode::Legal).board;
    assert_eq!(board(&positions[0]), board(&positions[1]));
    assert_eq!(candidates[0], CASTLE | STEP_F1);
    assert_eq!(candidates[0], candidates[1]);
    assert_eq!((want[0], want[1]), (CASTLE | STEP_F1, STEP_F1));

    let family = castle_family(true);
    let got = family
        .bind()
        .unwrap()
        .decide(&family, &lanes, &candidates)
        .unwrap();
    assert_eq!(got.composed, 2);
    // With the right as a lane the guards agree with shakmaty on A, B and D.
    for r in [0, 1, 3] {
        assert_eq!(got.holds[r], want[r], "row {r}");
    }
    assert_eq!(got.holds[3], 0, "D proposes nothing: f1 is occupied");

    // RED control: drop the right from the guard. The twins collapse and B
    // is given a castle shakmaty does not generate.
    let ablated = castle_family(false);
    let wrong = ablated
        .bind()
        .unwrap()
        .decide(&ablated, &lanes, &candidates)
        .unwrap();
    assert_eq!(wrong.holds[0], wrong.holds[1]);
    assert_eq!(
        wrong.holds[1] & !want[1],
        CASTLE,
        "the ablation must admit an illegal castle"
    );

    // A description that drops the right does not pass for the bound rule.
    assert_eq!(
        family.bind().unwrap().decide(&ablated, &lanes, &candidates),
        Err(TwinRefusal::DescriptionMismatch)
    );
}

/// The limit, pinned. In C the right is present and the squares are empty,
/// so both guards hold; shakmaty generates neither move, because f1 is
/// attacked. Whether a square is attacked is a fact about the position a
/// move would produce and about other pieces' reach. No raw field of the
/// row states it. A lane for it would have to be written by something that
/// already knows the rules of chess, which is the authority itself.
#[test]
fn attacked_squares_are_not_a_raw_field() {
    let positions = vec![position(C)];
    let lanes = castle_lanes(&positions);
    let candidates = vec![left_corner(&lanes, 0)];
    let family = castle_family(true);
    let got = family
        .bind()
        .unwrap()
        .decide(&family, &lanes, &candidates)
        .unwrap();
    assert_eq!(got.holds[0], CASTLE | STEP_F1, "every raw condition holds");
    assert_eq!(
        authority(&positions[0]),
        0,
        "shakmaty generates neither move"
    );
}

/// Play UCI moves from the initial position; return every position reached,
/// the initial one first.
fn play(moves: &[&str]) -> Vec<Chess> {
    let mut history = vec![Chess::default()];
    for m in moves {
        let pos = history.last().unwrap().clone();
        let mv = UciMove::from_ascii(m.as_bytes())
            .expect("uci")
            .to_move(&pos)
            .expect("legal");
        history.push(pos.play(mv).expect("playable"));
    }
    history
}

// Draw family lanes.
const WHITE_TO_MOVE: u16 = 0;
const HALFMOVES: u16 = 1;
const OCCURRENCES: u16 = 2;
const CLAIM: u8 = 1 << 0;

fn draw_family(producer: &str) -> Family {
    Family {
        class: CLASS_DRAW,
        lanes: 3,
        rules: vec![Rule {
            name: "ClaimThreefold",
            when: vec![(OCCURRENCES, vec![3, 4, 5])],
        }],
        producer: digest(&["repetition-count", producer]),
    }
}

/// Lanes for the final position of each game. `same` decides when two
/// positions count as one: that choice is the producer.
fn draw_lanes(games: &[Vec<Chess>], same: fn(&Chess, &Chess) -> bool) -> Vec<Vec<u32>> {
    let mut lanes = vec![Vec::new(); 3];
    for g in games {
        let last = g.last().unwrap();
        let s = last.to_setup(EnPassantMode::Legal);
        lanes[usize::from(WHITE_TO_MOVE)].push(u32::from(s.turn == Color::White));
        lanes[usize::from(HALFMOVES)].push(s.halfmoves);
        lanes[usize::from(OCCURRENCES)].push(g.iter().filter(|p| same(p, last)).count() as u32);
    }
    lanes
}

/// A draw claim depends on history that no field of the position holds.
///
/// Both games end on the initial arrangement after eight knight moves. In
/// THRICE it stood at plies 0, 4 and 8; in TWICE only at plies 0 and 8. The
/// two final positions are equal as `Chess` values and as `Setup` structs,
/// clocks included. The expected answers are read off the move lists.
#[test]
fn a_draw_claim_needs_a_lane_folded_from_history() {
    const THRICE: [&str; 8] = [
        "g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8",
    ];
    const TWICE: [&str; 8] = [
        "g1f3", "g8f6", "f3h4", "f6h5", "h4f3", "h5f6", "f3g1", "f6g8",
    ];
    let games = vec![play(&THRICE), play(&TWICE)];
    let (a, b) = (games[0].last().unwrap(), games[1].last().unwrap());
    assert_eq!(a, b);
    assert_eq!(
        a.to_setup(EnPassantMode::Legal),
        b.to_setup(EnPassantMode::Legal)
    );
    let want = [CLAIM, 0];

    // FIDE equality (shakmaty's `PartialEq`) as the producer.
    let fide = draw_lanes(&games, |x, y| x == y);
    assert_eq!(fide[usize::from(OCCURRENCES)], vec![3, 2]);
    // Every lane that is a raw field of the position is equal for the twins,
    // so no guard over raw fields can tell them apart.
    for lane in [WHITE_TO_MOVE, HALFMOVES] {
        assert_eq!(fide[usize::from(lane)][0], fide[usize::from(lane)][1]);
    }
    let family = draw_family("fide-equality");
    let bound = family.bind().unwrap();
    let candidates = [CLAIM, CLAIM];
    assert_eq!(
        bound.decide(&family, &fide, &candidates).unwrap().holds,
        want
    );

    // Another producer: positions are the same only if their clocks are too.
    // It counts 1 for both games, and the same guard then refuses both.
    let strict = draw_lanes(&games, |x, y| {
        x.to_setup(EnPassantMode::Legal) == y.to_setup(EnPassantMode::Legal)
    });
    assert_eq!(strict[usize::from(OCCURRENCES)], vec![1, 1]);
    let other = draw_family("setup-equality");
    assert_eq!(
        other
            .bind()
            .unwrap()
            .decide(&other, &strict, &candidates)
            .unwrap()
            .holds,
        [0, 0]
    );
    // The producer is part of the method's identity: lanes described as
    // coming from the other producer are refused by the bound methods.
    assert_eq!(
        bound.decide(&other, &strict, &candidates),
        Err(TwinRefusal::DescriptionMismatch)
    );
}
