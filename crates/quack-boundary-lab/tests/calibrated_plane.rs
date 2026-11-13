//! D-WX-CALIBRATED-PLANE-ADMISSION-1: a resident bit-plane is a representation;
//! a temperature predicate derived from it carries a calibration
//! interpretation that admission can refuse.
//!
//! Pass conditions A-E, fixed before running:
//! `.claude/board/entries/2026-11-13-wx-calibrated-plane-admission.md`.
//!
//! No issuer is added. The interpretation is a registered DT64 method whose
//! guard carries `Sym(digest(floor version, theta, code threshold))`; the
//! caller's description is compared with the bound guard through
//! `Guard::witness`. The evaluation below is composed only of existing calls.

use dr256_ternlog_lab::bit_planes::{gt, plane_op};
use hidden_state_twins_probe::digest;
use lance_graph_contract::class_view::WideFieldMask;
use lance_graph_contract::set_algebra::{Logic, SetExpr};
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::{
    Guard, LaneBinding, LaneRefusal, MethodBinding, MethodId, SigRev, ValueDemand,
};
use lance_graph_mask_risc::{fuse_program, Out, Scratch, Terminal, Value};
use quack_boundary_lab::derived::{
    admitted_planes, check_planes, DerivedPlane, PlaneRefusal, Recipe,
};
use quack_boundary_lab::{Prepared, Refused, RowOrder, Source, World, AMOUNT, CLASS, READER, ROWS};
use weather_poc::floor::{calibrate, CalibratedFloor};

const THETA: f64 = 283.15;
const M_A: MethodId = MethodId(1);
const M_B: MethodId = MethodId(2);

/// `n` values spread evenly over `lo..=hi`; no RNG.
fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| lo + (hi - lo) * i as f64 / (n as f64 - 1.0))
        .collect()
}

fn q(floor: &CalibratedFloor) -> u8 {
    floor.quantize(THETA)
}

/// Identity of one interpretation: calibration epoch, physical threshold and
/// the code threshold they produce.
fn token(floor: &CalibratedFloor) -> u32 {
    digest(&[
        "wx-calibrated-plane",
        &floor.floor_version().to_string(),
        &THETA.to_string(),
        &q(floor).to_string(),
    ])
}

fn describe(floor: &CalibratedFloor) -> Guard {
    Guard::new(SetExpr::Sym(token(floor)), Logic::TwoValued)
}

fn method(id: u32, floor: &CalibratedFloor) -> MethodBinding {
    let fields = WideFieldMask::from_positions(&[AMOUNT]);
    MethodBinding {
        id: MethodId(id),
        sig_rev: SigRev(1),
        name: format!("temperature_above_theta_floor_{id}"),
        on: fields.clone(),
        when: Some(describe(floor)),
        reads: fields,
        effects: Vec::new(),
        population: Population::Distinct,
        truth: Truth::TwoValued,
        demand: ValueDemand::Values,
    }
}

/// The code of a row: the low byte of its amount.
fn code(amount: i32) -> u8 {
    (amount as u32 & 0xFF) as u8
}

/// Rows whose code is above `q`, from the source's own values.
fn oracle(source: &Source, q: u8) -> usize {
    (0..ROWS)
        .filter(|&r| code(source.row(r).amount) > q)
        .count()
}

fn codes(source: &Source) -> Vec<u8> {
    (0..ROWS).map(|r| code(source.row(r).amount)).collect()
}

struct Fx {
    w: World,
    source: Source,
    floor_a: CalibratedFloor,
    floor_b: CalibratedFloor,
}

impl Fx {
    fn new() -> Fx {
        let floor_a = calibrate(&linspace(240.0, 300.0, 4096)).expect("sample a calibrates");
        let floor_b = calibrate(&linspace(250.0, 320.0, 4096)).expect("sample b calibrates");
        let w = World::with(
            quack_boundary_lab::View::standard(),
            vec![method(1, &floor_a), method(2, &floor_b)],
        );
        Fx {
            w,
            source: Source::new(RowOrder::ById),
            floor_a,
            floor_b,
        }
    }

    fn binding(&self) -> LaneBinding {
        self.w
            .bind_fields(&self.source, &[AMOUNT], READER, self.source.space())
            .expect("the amount lane binds")
    }

    fn planes(&self, binding: &LaneBinding) -> Vec<DerivedPlane> {
        (0..8u8)
            .map(|k| {
                DerivedPlane::bit(&self.w, &self.w.view, binding, &self.source, AMOUNT, k)
                    .expect("the bit-plane derives")
            })
            .collect()
    }
}

fn expects() -> Vec<Recipe> {
    (0..8u8).map(|k| Recipe::Bit { field: AMOUNT, k }).collect()
}

/// Calibrated predicate `code > q(floor_now)` over resident planes, admitted
/// through the method registered under `method`.
fn evaluate(
    w: &World,
    binding: &LaneBinding,
    source: &Source,
    planes: &[&DerivedPlane],
    method: MethodId,
    floor_now: &CalibratedFloor,
) -> Result<usize, Refused> {
    let plane_err = |e| Refused::Plane(e);
    let mref = w
        .store
        .method_ref(CLASS, method)
        .map_err(|e| plane_err(PlaneRefusal::Method(e)))?;
    let bound = w
        .store
        .admit(&mref)
        .map_err(|e| plane_err(PlaneRefusal::Method(e)))?;
    let stored = bound
        .when
        .as_ref()
        .ok_or(plane_err(PlaneRefusal::Description))?;
    if describe(floor_now).witness() != stored.witness() {
        return Err(plane_err(PlaneRefusal::Description));
    }
    let expr = gt(q(floor_now), plane_op).expect("threshold below 255");
    let program = fuse_program(&expr, 0, |m| Terminal::Count { mask: m }).expect("fuses");
    let mut scratch = Scratch::for_program(&program, ROWS).map_err(Refused::Exec)?;
    let p = Prepared::over(program, source, binding);
    let v = admitted_planes(
        &p,
        &expects(),
        planes,
        w,
        &w.view,
        binding,
        source,
        &mut scratch,
        Out::None,
    )?;
    match v {
        Value::Count(n) => Ok(n),
        other => panic!("a count terminal returned {other:?}"),
    }
}

fn refs(planes: &[DerivedPlane]) -> Vec<&DerivedPlane> {
    planes.iter().collect()
}

/// Anti-vacuity for the two floors: the physical threshold lands on two
/// different, interior code thresholds.
fn assert_distinct_thresholds(fx: &Fx) -> (u8, u8) {
    let (qa, qb) = (q(&fx.floor_a), q(&fx.floor_b));
    assert_ne!(qa, qb, "the two calibrations must disagree");
    assert_ne!(fx.floor_a.floor_version(), fx.floor_b.floor_version());
    assert!(
        (2..=253).contains(&qa) && (2..=253).contains(&qb),
        "{qa} {qb}"
    );
    assert_ne!(token(&fx.floor_a), token(&fx.floor_b));
    (qa, qb)
}

/// A. FAILS IF: the admitted predicate over the eight resident planes differs
/// from a scalar loop over the code bytes, or is refused, or the answer is
/// trivial (0 or every row).
#[test]
fn a_same_floor_predicate_equals_the_code_space_oracle() {
    let fx = Fx::new();
    let (qa, _) = assert_distinct_thresholds(&fx);
    let b = fx.binding();
    let planes = fx.planes(&b);
    let want = oracle(&fx.source, qa);
    assert!(0 < want && want < ROWS, "trivial answer {want}");
    assert_eq!(
        evaluate(&fx.w, &b, &fx.source, &refs(&planes), M_A, &fx.floor_a),
        Ok(want)
    );
}

/// B1-B3. FAILS IF: the planes stop passing the frame checks after only the
/// calibration changed; the method bound under floor A accepts a description
/// under floor B; the method bound under floor B is refused or answers other
/// than the scalar loop at floor B's threshold over the same plane objects; or
/// the planes were re-derived (their words changed).
#[test]
fn b_the_representation_survives_a_calibration_change_and_the_stale_predicate_is_refused() {
    let fx = Fx::new();
    let (qa, qb) = assert_distinct_thresholds(&fx);
    let b = fx.binding();
    let planes = fx.planes(&b);
    let before: Vec<Vec<u64>> = planes.iter().map(|p| p.words().to_vec()).collect();
    let r = refs(&planes);

    // B1: the representation is still valid; only the calibration moved.
    assert_eq!(check_planes(&expects(), &r, &b, &fx.source), Ok(()));

    // B2: the predicate interpreted under floor A is refused under floor B.
    assert_eq!(
        evaluate(&fx.w, &b, &fx.source, &r, M_A, &fx.floor_b),
        Err(Refused::Plane(PlaneRefusal::Description))
    );

    // B3: the predicate registered under floor B is admitted over the SAME planes.
    let (want_a, want_b) = (oracle(&fx.source, qa), oracle(&fx.source, qb));
    assert_ne!(want_a, want_b, "the two answers must differ");
    assert!(0 < want_b && want_b < ROWS);
    assert_eq!(
        evaluate(&fx.w, &b, &fx.source, &r, M_B, &fx.floor_b),
        Ok(want_b)
    );

    // Nothing was re-derived.
    let after: Vec<Vec<u64>> = planes.iter().map(|p| p.words().to_vec()).collect();
    assert_eq!(before, after);
}

/// B4, the limit. A caller that keeps describing with floor A after the
/// calibration changed is NOT refused.
///
/// This PINS a limit: nothing in the store, the binding or the source knows
/// which floor is current; the caller states it. The refusal in B2 only
/// catches a caller whose description disagrees with the method it names.
///
/// FAILS IF: some existing interface starts refusing it (then re-pin
/// deliberately and record what now holds the current floor).
#[test]
fn b4_a_caller_that_keeps_describing_with_the_old_floor_is_not_refused() {
    let fx = Fx::new();
    let (qa, qb) = assert_distinct_thresholds(&fx);
    let b = fx.binding();
    let planes = fx.planes(&b);
    let r = refs(&planes);

    // "The calibration changed": only floor_b is current. The caller does not
    // say so; it still names method A and describes with floor A.
    let current = &fx.floor_b;
    assert_ne!(current.floor_version(), fx.floor_a.floor_version());
    let stale = evaluate(&fx.w, &b, &fx.source, &r, M_A, &fx.floor_a);
    assert_eq!(stale, Ok(oracle(&fx.source, qa)));
    assert_ne!(stale, Ok(oracle(&fx.source, qb)));
}

/// C. FAILS IF: after a new timestep (changed codes, advanced frame) the old
/// binding is not refused by its own check; a fresh binding accepts the old
/// planes; fresh planes do not equal the scalar loop on the new bytes; or the
/// new answer equals the old one (nothing changed).
#[test]
fn c_a_new_timestep_refuses_old_planes_through_the_frame() {
    let mut fx = Fx::new();
    let (qa, _) = assert_distinct_thresholds(&fx);
    let b = fx.binding();
    let old_planes = fx.planes(&b);
    let old_answer = oracle(&fx.source, qa);
    assert_eq!(
        evaluate(&fx.w, &b, &fx.source, &refs(&old_planes), M_A, &fx.floor_a),
        Ok(old_answer)
    );

    fx.source.rewrite_amount(|v| -v);
    fx.source.advance_source();
    let new_answer = oracle(&fx.source, qa);
    assert_ne!(
        new_answer, old_answer,
        "the new timestep must change the answer"
    );

    // The old binding is refused by its own check.
    assert!(matches!(
        evaluate(&fx.w, &b, &fx.source, &refs(&old_planes), M_A, &fx.floor_a),
        Err(Refused::Binding(LaneRefusal::Frame(_)))
    ));

    // A fresh binding is valid; the old planes are not.
    let fresh = fx.binding();
    assert!(matches!(
        evaluate(
            &fx.w,
            &fresh,
            &fx.source,
            &refs(&old_planes),
            M_A,
            &fx.floor_a
        ),
        Err(Refused::Plane(PlaneRefusal::Frame { at: 0, .. }))
    ));

    // Freshly derived planes answer from the new bytes.
    let new_planes = fx.planes(&fresh);
    assert_eq!(
        evaluate(
            &fx.w,
            &fresh,
            &fx.source,
            &refs(&new_planes),
            M_A,
            &fx.floor_a
        ),
        Ok(new_answer)
    );
}

/// D. An undeclared rewrite (bytes change, frame does not) is not detectable:
/// the old planes are admitted and answer from the old bytes. Reproduces
/// PR 2058's finding (plan `d-quack-derived-plane-probe.md` section 13); it is
/// pinned here, not fixed.
///
/// FAILS IF: the stale planes are refused (the boundary would then be closed
/// and this pin wrong), the old and new answers coincide (the case would show
/// nothing), or freshly derived planes do not follow the new bytes.
#[test]
fn d_an_undeclared_rewrite_is_not_detectable_and_the_planes_answer_from_old_bytes() {
    let mut fx = Fx::new();
    let (qa, _) = assert_distinct_thresholds(&fx);
    let b = fx.binding();
    let old = oracle(&fx.source, qa);
    let planes = fx.planes(&b);

    fx.source.rewrite_amount(|v| -v);
    let new = oracle(&fx.source, qa);
    assert_ne!(new, old);

    // Admitted, and stale.
    assert_eq!(
        evaluate(&fx.w, &b, &fx.source, &refs(&planes), M_A, &fx.floor_a),
        Ok(old)
    );
    // Fresh planes (same binding, same frame) follow the new bytes.
    let fresh = fx.planes(&b);
    assert_eq!(
        evaluate(&fx.w, &b, &fx.source, &refs(&fresh), M_A, &fx.floor_a),
        Ok(new)
    );
}

/// E. Re-encoding the same physical values under floor B changes the code
/// bytes. Resident planes are caught only if the owner advances its frame.
///
/// Whether a real code-lane owner advances its frame on re-encoding is not
/// decided by this lab; both branches are run.
///
/// FAILS IF: the re-encoding leaves most code bytes unchanged; branch 1 (no
/// advance) refuses, or its answer equals the answer over the new codes (the
/// stale answer would be indistinguishable); or branch 2 (advance) admits the
/// old planes through a fresh binding.
#[test]
fn e_reencoding_is_caught_only_when_the_owner_advances_the_frame() {
    let reencode = |fx: &mut Fx| {
        let (a, b) = (fx.floor_a.clone(), fx.floor_b.clone());
        fx.source
            .rewrite_amount(|v| i32::from(b.quantize(a.bucket_center(code(v)))));
    };

    // Branch 1: the owner does not advance the frame.
    let mut fx = Fx::new();
    let (_, qb) = assert_distinct_thresholds(&fx);
    let b = fx.binding();
    let planes = fx.planes(&b);
    let old_codes = codes(&fx.source);
    let old_at_qb = oracle(&fx.source, qb);
    reencode(&mut fx);
    let new_codes = codes(&fx.source);
    let changed = old_codes
        .iter()
        .zip(&new_codes)
        .filter(|(o, n)| o != n)
        .count();
    assert!(changed * 2 > ROWS, "only {changed} of {ROWS} codes changed");
    let new_at_qb = oracle(&fx.source, qb);
    assert_ne!(new_at_qb, old_at_qb);
    assert_eq!(
        evaluate(&fx.w, &b, &fx.source, &refs(&planes), M_B, &fx.floor_b),
        Ok(old_at_qb)
    );

    // Branch 2: the owner advances the frame.
    let mut fx = Fx::new();
    let b = fx.binding();
    let planes = fx.planes(&b);
    reencode(&mut fx);
    fx.source.advance_source();
    let fresh = fx.binding();
    assert!(matches!(
        evaluate(&fx.w, &fresh, &fx.source, &refs(&planes), M_B, &fx.floor_b),
        Err(Refused::Plane(PlaneRefusal::Frame { at: 0, .. }))
    ));
}
