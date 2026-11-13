//! `Thought::in_space`: the same rules as `Thought::new`, with the bounds
//! given by the caller.

use thought_git_probe::fixtures::{LANES, Outcome, ROWS, Thought, cell};

/// FAILS IF: `in_space` keeps the fixture bounds, or `new` stops using them.
#[test]
fn the_bounds_come_from_the_caller() {
    // Outside the 12 x 4 fixture space, inside 64 x 8.
    let wide = Thought::in_space(1, 64, 8, vec![(cell(40, 6), 9)], Outcome::Commit);
    assert_eq!(wide.writes, vec![(cell(40, 6), 9)]);

    // `new` is `in_space` with the fixture bounds: same value for the same writes.
    let writes = vec![(cell(ROWS - 1, LANES - 1), 3), (cell(0, 0), 1)];
    assert_eq!(
        Thought::new(2, writes.clone(), Outcome::Plan),
        Thought::in_space(2, ROWS, LANES, writes, Outcome::Plan)
    );
}

/// FAILS IF: a row at the caller's bound is accepted.
#[test]
#[should_panic(expected = "outside the 64 x 8 space")]
fn a_row_at_the_bound_is_refused() {
    let _ = Thought::in_space(1, 64, 8, vec![(cell(64, 0), 1)], Outcome::Commit);
}

/// FAILS IF: a lane at the caller's bound is accepted.
#[test]
#[should_panic(expected = "outside the 64 x 8 space")]
fn a_lane_at_the_bound_is_refused() {
    let _ = Thought::in_space(1, 64, 8, vec![(cell(0, 8), 1)], Outcome::Commit);
}

/// FAILS IF: `new` accepts a cell the fixture space does not hold.
#[test]
#[should_panic(expected = "outside the 12 x 4 space")]
fn new_still_refuses_outside_the_fixture_space() {
    let _ = Thought::new(1, vec![(cell(40, 6), 9)], Outcome::Commit);
}

/// FAILS IF: one cell may be written twice in a caller-sized space.
#[test]
#[should_panic(expected = "twice")]
fn a_cell_written_twice_is_refused() {
    let _ = Thought::in_space(
        1,
        64,
        8,
        vec![(cell(5, 5), 1), (cell(5, 5), 2)],
        Outcome::Commit,
    );
}
