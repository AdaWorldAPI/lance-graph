//! D-DRCA-1: HHTL16 Cartesian direct routing, the first falsifiable proof.
//!
//! Fixture: two tables with explicit, independently written facts.
//!
//! - Table A (`EMP`, class 21): 6 employees. Properties of A: 5 skills.
//!   Same-table relation: `reports_to` (employee → employee).
//!   Foreign key: `works_in` (employee → department).
//! - Table B (`DEPT`, class 22): 3 departments. Properties of B: 4 sites.
//!
//! Every global id is unique across the fixture, and the two tables
//! deliberately share LOCAL ordinals (employee local 0 and department local 0
//! both exist), so check 9 has something to refuse.
//!
//! The oracle answers each question by scanning the fact arrays below. It
//! shares no code with `hhtl16`, `relcomp` or the binding layer.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use dr_math_binding_probe::hhtl16::{
    self, LocalMap, MapRefusal, address, cell, cell_rotated, column, column_validity, cooccurrence,
    cooccurrence_count, coordinates, rotate_yz,
};
use dr_math_binding_probe::relcomp::{Chain, ComposeRefusal, RelExpr, Rows};
use dr_math_binding_probe::{
    Axis, AxisKind, BoundDr64, Focus, FrameRevisions, FrameStore, Guards, MethodSignature, Operand,
    Population, Refusal, Scope, Truth, bind_dr64,
};

// ── allocation counter (check 10) ─────────────────────────────────────────

// Per thread, so other tests running in parallel are not counted.
struct Counting;
thread_local! {
    static ALLOCS: Cell<u64> = const { Cell::new(0) };
}
fn allocs() -> u64 {
    ALLOCS.with(Cell::get)
}
// SAFETY: forwards to `System` unchanged; only counts calls.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.with(|c| c.set(c.get() + 1));
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: same contract as the caller's.
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static GLOBAL: Counting = Counting;

// ── the fixture: facts written as global ids, nothing else ────────────────

const EMP: u16 = 21;
const DEPT: u16 = 22;

const EMPLOYEES: [u64; 6] = [1001, 1002, 1003, 1004, 1005, 1006];
const SKILLS: [u64; 5] = [2001, 2002, 2003, 2004, 2005];
const DEPARTMENTS: [u64; 3] = [3001, 3002, 3003];
const SITES: [u64; 4] = [4001, 4002, 4003, 4004];

/// employee has skill.
const HAS_SKILL: [[u64; 2]; 9] = [
    [1001, 2001],
    [1001, 2003],
    [1002, 2001],
    [1002, 2002],
    [1003, 2003],
    [1004, 2004],
    [1004, 2001],
    [1005, 2005],
    [1006, 2003],
];
/// employee reports to employee (not symmetric).
const REPORTS_TO: [[u64; 2]; 4] = [[1002, 1001], [1003, 1001], [1004, 1002], [1005, 1002]];
/// employee works in department (the witnessed foreign key).
const WORKS_IN: [[u64; 2]; 6] = [
    [1001, 3001],
    [1002, 3001],
    [1003, 3002],
    [1004, 3002],
    [1005, 3003],
    [1006, 3003],
];
/// department is located at site.
const LOCATED_AT: [[u64; 2]; 5] = [
    [3001, 4001],
    [3001, 4002],
    [3002, 4002],
    [3003, 4003],
    [3003, 4004],
];
/// Three-valued: is the skill certified for the employee? `Some(true)` known
/// yes, `Some(false)` known no, missing = unknown.
const CERTIFIED: [(u64, u64, bool); 4] = [
    (1001, 2001, true),
    (1001, 2003, false),
    (1002, 2001, true),
    (1004, 2004, false),
];

// ── independent oracle ────────────────────────────────────────────────────

fn holds(facts: &[[u64; 2]], s: u64, o: u64) -> bool {
    facts.iter().any(|f| f[0] == s && f[1] == o)
}
fn certified(e: u64, s: u64) -> Option<bool> {
    CERTIFIED.iter().find(|c| c.0 == e && c.1 == s).map(|c| c.2)
}
/// employee e is at site t, through the department it works in.
fn at_site(e: u64, t: u64) -> bool {
    DEPARTMENTS
        .iter()
        .any(|&d| holds(&WORKS_IN, e, d) && holds(&LOCATED_AT, d, t))
}
/// some employee holds both skills.
fn co_skill(p: u64, q: u64) -> usize {
    EMPLOYEES
        .iter()
        .filter(|&&e| holds(&HAS_SKILL, e, p) && holds(&HAS_SKILL, e, q))
        .count()
}

// ── plate side: prepare once, then bind ───────────────────────────────────

struct Maps {
    emp: LocalMap,
    skill: LocalMap,
    dept: LocalMap,
    site: LocalMap,
}
fn maps() -> Maps {
    // A source may hand ids in any order; the local order is the source's.
    Maps {
        emp: LocalMap::prepare(&[1004, 1001, 1006, 1002, 1005, 1003]).unwrap(),
        skill: LocalMap::prepare(&[2005, 2003, 2001, 2004, 2002]).unwrap(),
        dept: LocalMap::prepare(&[3002, 3003, 3001]).unwrap(),
        site: LocalMap::prepare(&[4004, 4001, 4003, 4002]).unwrap(),
    }
}
/// Build a register from facts through two maps (the preparation step).
fn plane(facts: &[[u64; 2]], rows: &LocalMap, cols: &LocalMap) -> Rows {
    let mut r = [0u64; 64];
    for f in facts {
        let (x, y) = (rows.local(f[0]).unwrap(), cols.local(f[1]).unwrap());
        r[x as usize] |= 1 << y;
    }
    r
}
fn certified_planes(m: &Maps) -> (Rows, Rows) {
    let (mut val, mut known) = ([0u64; 64], [0u64; 64]);
    for &(e, s, v) in &CERTIFIED {
        let (x, y) = (m.emp.local(e).unwrap(), m.skill.local(s).unwrap());
        known[x as usize] |= 1 << y;
        if v {
            val[x as usize] |= 1 << y;
        }
    }
    (val, known)
}

const FOCUS: Focus = Focus { revision: 1 };
const SCOPE: Scope = Scope { id: 7 };

fn axis(class: u16, kind: AxisKind) -> Axis {
    Axis { class, kind }
}
fn sig(ord: u16, row: Axis, col: Axis) -> MethodSignature {
    MethodSignature::presence(row.class, ord, row, col)
}
fn store() -> FrameStore {
    FrameStore::new(FrameRevisions {
        source: 1,
        relation_map: 1,
        coordinates: 1,
    })
}
fn bind<'a>(st: &FrameStore, m: MethodSignature, rows: &'a Rows) -> BoundDr64<'a> {
    let at = Guards {
        frame: st.seal(),
        focus: FOCUS,
        scope: SCOPE,
    };
    bind_dr64(
        st,
        m,
        m,
        &at,
        (FOCUS, SCOPE),
        Operand {
            rows,
            validity: None,
        },
    )
    .unwrap()
}

fn has_skill_sig() -> MethodSignature {
    sig(
        1,
        axis(EMP, AxisKind::Identity),
        axis(EMP, AxisKind::Property),
    )
}

// ── 1. same-table Identity × Property ─────────────────────────────────────

#[test]
fn identity_by_property_matches_the_oracle_cell_for_cell() {
    let m = maps();
    let r = plane(&HAS_SKILL, &m.emp, &m.skill);
    let b = bind(&store(), has_skill_sig(), &r);
    let mut hits = 0;
    for &e in &EMPLOYEES {
        let x = m.emp.local(e).unwrap();
        for &s in &SKILLS {
            let y = m.skill.local(s).unwrap();
            let got = cell(b.rows, address(x, y).unwrap()).unwrap();
            assert_eq!(got, holds(&HAS_SKILL, e, s), "({e}, {s})");
            hits += usize::from(got);
        }
    }
    // Anti-vacuity: the fixture has true AND false cells.
    assert_eq!(hits, HAS_SKILL.len());
    assert!(hits < EMPLOYEES.len() * SKILLS.len());
    // Nothing outside the fixture is set.
    assert_eq!(hhtl16::count(b.rows) as usize, HAS_SKILL.len());
}

// ── 2. both mirror orientations and the YZ rotation ───────────────────────

#[test]
fn yz_rotation_reads_the_same_relationship_from_the_other_axis() {
    let m = maps();
    let r = plane(&HAS_SKILL, &m.emp, &m.skill);
    // The fixture is not symmetric, so an identity "rotation" cannot pass.
    assert!((0..64u8).any(|x| (0..64u8).any(|y| {
        cell(&r, address(x, y).unwrap()).unwrap() != cell(&r, address(y, x).unwrap()).unwrap()
    })));
    for x in 0..64u8 {
        for y in 0..64u8 {
            let a = address(x, y).unwrap();
            let rot = rotate_yz(a);
            assert_eq!(coordinates(rot).unwrap(), (y, x));
            assert_eq!(rotate_yz(rot), a, "rotation is its own inverse");
            assert_eq!(cell_rotated(&r, rot).unwrap(), cell(&r, a).unwrap());
        }
    }
}

#[test]
fn property_orientation_is_the_column_and_matches_the_oracle() {
    let m = maps();
    let r = plane(&HAS_SKILL, &m.emp, &m.skill);
    for &s in &SKILLS {
        let y = m.skill.local(s).unwrap();
        let col = column(&r, y).unwrap();
        for &e in &EMPLOYEES {
            let x = m.emp.local(e).unwrap();
            assert_eq!(col >> x & 1 == 1, holds(&HAS_SKILL, e, s), "({e}, {s})");
        }
        // Only mapped employees can appear.
        assert_eq!(col >> m.emp.len(), 0);
    }
    assert_eq!(column(&r, 64), None);
}

// ── 3. optional Identity × Identity ───────────────────────────────────────

#[test]
fn identity_by_identity_reads_forward_and_through_the_rotation_as_inverse() {
    let m = maps();
    let r = plane(&REPORTS_TO, &m.emp, &m.emp);
    let st = store();
    let b = bind(
        &st,
        sig(
            2,
            axis(EMP, AxisKind::Identity),
            axis(EMP, AxisKind::Identity),
        ),
        &r,
    );
    for &a in &EMPLOYEES {
        for &c in &EMPLOYEES {
            let (x, y) = (m.emp.local(a).unwrap(), m.emp.local(c).unwrap());
            // forward: a reports to c
            assert_eq!(
                cell(b.rows, address(x, y).unwrap()).unwrap(),
                holds(&REPORTS_TO, a, c)
            );
            // rotated address (c, a) reads the same fact: c is reported to by a.
            assert_eq!(
                cell_rotated(b.rows, address(y, x).unwrap()).unwrap(),
                holds(&REPORTS_TO, a, c)
            );
        }
    }
    // The rotation is not an asserted symmetric relation: (1001 reports to 1002) is false.
    let (x, y) = (m.emp.local(1001).unwrap(), m.emp.local(1002).unwrap());
    assert!(!cell(&r, address(x, y).unwrap()).unwrap());
    assert!(cell(&r, address(y, x).unwrap()).unwrap());
}

// ── 4. derived Property × Property ────────────────────────────────────────

#[test]
fn property_by_property_cooccurrence_matches_the_oracle() {
    let m = maps();
    let r = plane(&HAS_SKILL, &m.emp, &m.skill);
    let b = bind(&store(), has_skill_sig(), &r);
    let p = cooccurrence(&b).unwrap();
    let mut pairs = 0;
    for &s in &SKILLS {
        for &t in &SKILLS {
            let (ys, yt) = (m.skill.local(s).unwrap(), m.skill.local(t).unwrap());
            let want = co_skill(s, t);
            assert_eq!(p[ys as usize] >> yt & 1 == 1, want > 0, "({s}, {t})");
            assert_eq!(cooccurrence_count(&r, ys, yt).unwrap() as usize, want);
            pairs += usize::from(s != t && want > 0);
        }
    }
    // Anti-vacuity: at least one off-diagonal co-occurrence, and not all.
    assert!(pairs > 0 && pairs < SKILLS.len() * (SKILLS.len() - 1));
    // A binding that is not Identity × Property is refused.
    let rr = plane(&REPORTS_TO, &m.emp, &m.emp);
    let ii = bind(
        &store(),
        sig(
            2,
            axis(EMP, AxisKind::Identity),
            axis(EMP, AxisKind::Identity),
        ),
        &rr,
    );
    assert!(cooccurrence(&ii).is_none());
}

// ── 5 + 9. cross-table Identity × Property through a witnessed FK ────────

#[test]
fn cross_table_projection_through_the_foreign_key_matches_the_oracle() {
    let m = maps();
    let fk = plane(&WORKS_IN, &m.emp, &m.dept);
    let loc = plane(&LOCATED_AT, &m.dept, &m.site);
    let st = store();
    let fk_b = bind(
        &st,
        sig(
            3,
            axis(EMP, AxisKind::Identity),
            axis(DEPT, AxisKind::Identity),
        ),
        &fk,
    );
    let loc_b = bind(
        &st,
        sig(
            1,
            axis(DEPT, AxisKind::Identity),
            axis(DEPT, AxisKind::Property),
        ),
        &loc,
    );
    let chain = Chain::new(&st, vec![fk_b, loc_b]).unwrap();
    let order = RelExpr::comp(RelExpr::Leaf(0), RelExpr::Leaf(1));
    let (out, _) = chain.eval(&order).unwrap();
    let mut hits = 0;
    for &e in &EMPLOYEES {
        for &t in &SITES {
            let (x, y) = (m.emp.local(e).unwrap(), m.site.local(t).unwrap());
            let got = cell(&out, address(x, y).unwrap()).unwrap();
            assert_eq!(got, at_site(e, t), "({e}, {t})");
            hits += usize::from(got);
        }
    }
    assert!(hits > 0 && hits < EMPLOYEES.len() * SITES.len());
}

#[test]
fn equal_local_ordinals_from_different_tables_do_not_alias() {
    let m = maps();
    // Employee local 0 and department local 0 both exist and are different ids.
    assert_ne!(m.emp.global(0), m.dept.global(0));
    // A department id is not an employee: no silent local slot.
    assert_eq!(m.emp.local(3001), Err(MapRefusal::Unmapped));

    let skills = plane(&HAS_SKILL, &m.emp, &m.skill);
    let loc = plane(&LOCATED_AT, &m.dept, &m.site);
    let st = store();
    // Composing employee skills with department sites joins EMP.Property to
    // DEPT.Identity on equal numbers only; the typing refuses it.
    let a = bind(&st, has_skill_sig(), &skills);
    let b = bind(
        &st,
        sig(
            1,
            axis(DEPT, AxisKind::Identity),
            axis(DEPT, AxisKind::Property),
        ),
        &loc,
    );
    assert_eq!(
        Chain::new(&st, vec![a, b]).unwrap_err(),
        ComposeRefusal::DomainMismatch { at: 0 }
    );
    // An FK sealed by another frame owner, at equal revision numbers, is refused.
    let other = store();
    let fk = plane(&WORKS_IN, &m.emp, &m.dept);
    let fk_foreign = bind(
        &other,
        sig(
            3,
            axis(EMP, AxisKind::Identity),
            axis(DEPT, AxisKind::Identity),
        ),
        &fk,
    );
    let b2 = bind(
        &st,
        sig(
            1,
            axis(DEPT, AxisKind::Identity),
            axis(DEPT, AxisKind::Property),
        ),
        &loc,
    );
    assert_eq!(
        Chain::new(&st, vec![fk_foreign, b2]).unwrap_err(),
        ComposeRefusal::DifferentFrame { at: 1 }
    );
}

// ── 6. source-qualified identity and presence survive the round trip ──────

#[test]
fn presence_decodes_back_to_the_source_ids_exactly() {
    let m = maps();
    let r = plane(&HAS_SKILL, &m.emp, &m.skill);
    let mut decoded = 0;
    for x in 0..64u8 {
        for y in 0..64u8 {
            if cell(&r, address(x, y).unwrap()).unwrap() {
                let e = m.emp.global(x).unwrap();
                let s = m.skill.global(y).unwrap();
                assert!(holds(&HAS_SKILL, e, s), "invented fact ({e}, {s})");
                decoded += 1;
            }
        }
    }
    assert_eq!(decoded, HAS_SKILL.len(), "a fact was lost");
}

// ── 7. inverse coordinate mapping ─────────────────────────────────────────

#[test]
fn coordinate_and_local_mappings_invert_exactly() {
    for x in 0..64u8 {
        for y in 0..64u8 {
            assert_eq!(coordinates(address(x, y).unwrap()).unwrap(), (x, y));
        }
    }
    assert_eq!(address(64, 0), Err(MapRefusal::OutOfRegister));
    assert_eq!(address(0, 64), Err(MapRefusal::OutOfRegister));
    let m = maps();
    for &e in &EMPLOYEES {
        assert_eq!(m.emp.global(m.emp.local(e).unwrap()), Some(e));
    }
    assert_eq!(
        LocalMap::prepare(&[1, 2, 1]).unwrap_err(),
        MapRefusal::Duplicate
    );
    let many: Vec<u64> = (0..65).collect();
    assert_eq!(LocalMap::prepare(&many).unwrap_err(), MapRefusal::TooMany);
}

// ── K3 / NULL, permission, invalidation ───────────────────────────────────

#[test]
fn unknown_is_not_false_in_either_orientation() {
    let m = maps();
    let (val, known) = certified_planes(&m);
    let st = store();
    let at = Guards {
        frame: st.seal(),
        focus: FOCUS,
        scope: SCOPE,
    };
    let s3 = MethodSignature {
        truth: Truth::ThreeValued,
        population: Population::Distinct,
        ..sig(
            4,
            axis(EMP, AxisKind::Identity),
            axis(EMP, AxisKind::Property),
        )
    };
    let b = bind_dr64(
        &st,
        s3,
        s3,
        &at,
        (FOCUS, SCOPE),
        Operand {
            rows: &val,
            validity: Some(&known),
        },
    )
    .unwrap();
    let (mut t, mut f, mut u) = (0, 0, 0);
    for &e in &EMPLOYEES {
        for &s in &SKILLS {
            let (x, y) = (m.emp.local(e).unwrap(), m.skill.local(s).unwrap());
            let vcol = column(b.rows, y).unwrap();
            let kcol = column_validity(&b, y).unwrap();
            let read = (kcol >> x & 1 == 1).then_some(vcol >> x & 1 == 1);
            assert_eq!(read, certified(e, s), "({e}, {s})");
            match read {
                Some(true) => t += 1,
                Some(false) => f += 1,
                None => u += 1,
            }
        }
    }
    // All three values occur, so a FALSE/UNKNOWN collapse would be caught.
    assert!(t > 0 && f > 0 && u > 0);
    // Boolean co-occurrence reads payload bits only, so it refuses a
    // three-valued binding. Payload under an UNKNOWN cell is set here, so a
    // fold that accepted the binding would report a pair no fact supports.
    let mut dirty = val;
    let (x, y) = (m.emp.local(1001).unwrap(), m.skill.local(2005).unwrap());
    assert_eq!(certified(1001, 2005), None);
    dirty[x as usize] |= 1 << y;
    let bd = bind_dr64(
        &st,
        s3,
        s3,
        &at,
        (FOCUS, SCOPE),
        Operand {
            rows: &dirty,
            validity: Some(&known),
        },
    )
    .unwrap();
    assert!(cooccurrence(&bd).is_none());
    assert!(cooccurrence(&b).is_none());
}

#[test]
fn a_changed_scope_or_an_advanced_frame_refuses() {
    let m = maps();
    let r = plane(&HAS_SKILL, &m.emp, &m.skill);
    let mut st = store();
    let at = Guards {
        frame: st.seal(),
        focus: FOCUS,
        scope: SCOPE,
    };
    let s = has_skill_sig();
    let op = Operand {
        rows: &r,
        validity: None,
    };
    assert_eq!(
        bind_dr64(&st, s, s, &at, (FOCUS, Scope { id: 8 }), op).unwrap_err(),
        Refusal::ScopeChanged
    );
    st.advance(FrameRevisions {
        source: 1,
        relation_map: 1,
        coordinates: 2,
    });
    assert_eq!(
        bind_dr64(&st, s, s, &at, (FOCUS, SCOPE), op).unwrap_err(),
        Refusal::CoordinatesChanged
    );
}

// ── 10. no hot-path rebind, no materialized product ───────────────────────

#[test]
fn projections_never_call_the_global_map_and_never_allocate() {
    let m = maps();
    let r = plane(&HAS_SKILL, &m.emp, &m.skill);
    let st = store();
    let b = bind(&st, has_skill_sig(), &r);
    let before = (m.emp.lookups(), m.skill.lookups());
    let a0 = allocs();
    let mut acc = 0u64;
    for x in 0..64u8 {
        acc ^= hhtl16::row(&b, x).unwrap();
        acc ^= column(b.rows, x).unwrap();
        for y in 0..64u8 {
            acc ^= u64::from(cell_rotated(b.rows, address(y, x).unwrap()).unwrap());
        }
    }
    let p = cooccurrence(&b).unwrap();
    acc ^= p[0] ^ u64::from(cooccurrence_count(b.rows, 0, 1).unwrap());
    let a1 = allocs();
    std::hint::black_box(acc);
    assert_eq!(
        (m.emp.lookups(), m.skill.lookups()),
        before,
        "a projection rebound"
    );
    assert_eq!(a1 - a0, 0, "a projection allocated");
}
