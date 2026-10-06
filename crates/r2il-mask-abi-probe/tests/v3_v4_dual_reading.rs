//! **Round 3 — V3 data and V4 IR as two readings of the same resident bytes.**
//!
//! The hypothesis (`r2il-machine-semantic-contract-v1.md` §7.8, doctrine 1):
//! *"V4 = V3 + executable content. Nothing about the bytes moves. A V4 row IS
//! a V3 row; a V3 reader just doesn't know it can run."*
//!
//! The resident object is ONE 64-aligned byte buffer of `NodeRow`s. Two
//! shipped readings are taken over it, with no translation step between them:
//!
//! - **V3**: `node_rows_from_le_bytes` → `NodeRow::value` → each 16-byte slot
//!   reinterpreted as a `FacetCascade` (`facet_classid` + 6 × `FacetTier
//!   {lo, hi}`), via `ref_from_bytes`.
//! - **V4**: the same `value` slab read as loco calls by `ogar_loco::
//!   call_in_slab` under a `LaneShape`, and selected by `ogar_r2il::r2il_mask`
//!   / `project`.
//!
//! `G6D2` (`6 × (u8:u8)`) and `LaneShape::Pairs` (`6 × (function : value)`)
//! are the same carving of a slot's 12 payload bytes, so tier `k` of slot `l`
//! IS call `6l + k`: `lo` is the function byte, `hi` the immediate.
//!
//! What this file pins:
//!
//! | claim | test |
//! |---|---|
//! | same bytes, same addresses | `both_readings_address_the_same_bytes` |
//! | other carvings read the same slab in place | `every_lane_shape_reads_the_slab_in_place` |
//! | a byte written by the owner is visible to both readings at once | `an_owner_write_is_seen_by_both_readings` |
//! | taking either reading writes nothing | `reading_never_mutates_the_bytes` |
//! | readings allocate nothing | `readings_allocate_nothing` |
//! | per-slot classids carry no V4 content | `slot_classids_do_not_reach_the_v4_reading` |
//! | the shape is not in the bytes | `the_reading_is_not_recoverable_from_the_bytes` |
//! | **executing** a stored body through loco is NOT a reading — it gathers a copy | `loco_execution_needs_a_gathered_copy` |
//!
//! No storage tier is minted, no classid is registered, and nothing is
//! serialized after the intake write (`write_into_value_slab`, the one
//! intake-arm scatter).

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

use lance_graph_contract::canonical_node::{node_rows_from_le_bytes, NodeRow};
use lance_graph_contract::facet::FacetCascade;
use ogar_loco::{
    call_in_slab, Call, FnIndex, FunctionBody, LaneShape, Program, CLASSID_BYTES, CONTENT_SLOTS,
    SLOT_STRIDE, VALUE_SLAB_LEN,
};
use ogar_r2il::{project, r2il_mask, R2ILFn, R2IL_BASE};

// ── heap meter (the `mask-risc/tests/no_alloc.rs` pattern) ───────────────

struct Counting;

static BYTES: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static MEASURING: Cell<bool> = const { Cell::new(false) };
}

// SAFETY: a pure pass-through to `System`; the counter is the only addition.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if MEASURING.try_with(Cell::get).unwrap_or(false) {
            BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        // SAFETY: same layout, same contract as the caller's.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` above with this `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static A: Counting = Counting;

fn measure<R>(f: impl FnOnce() -> R) -> (R, usize) {
    let before = BYTES.load(Ordering::Relaxed);
    MEASURING.with(|m| m.set(true));
    let r = f();
    MEASURING.with(|m| m.set(false));
    (r, BYTES.load(Ordering::Relaxed) - before)
}

// ── the resident object ──────────────────────────────────────────────────

const ROWS: usize = 2;
const ROW: usize = 512;
/// `NodeRow::value` starts after `key(16) | edges(16)`.
const VALUE_AT: usize = 32;

/// The ONE owner: 64-aligned bytes, exactly what a `FixedSizeBinary(512)`
/// column hands over.
#[repr(C, align(64))]
struct Resident([u8; ROWS * ROW]);

fn r2il_op(ordinal: u8) -> FnIndex {
    FnIndex(R2IL_BASE + ordinal)
}

/// The intake: a body mixing R2IL ops (`0x90..`) with loco core ops
/// (`< 0x90`), scattered into row 0's value slab by the one intake-arm
/// write. Per-slot classids are set to a recognisable pattern AFTER the
/// scatter, which `write_into_value_slab` leaves untouched by contract.
fn resident() -> Box<Resident> {
    let mut calls = Vec::new();
    for i in 0..LaneShape::Pairs.calls_per_function() {
        let call = if i % 3 == 0 {
            Call::new(FnIndex::ADD)
        } else {
            Call::with_value(r2il_op((i % 82) as u8), (i * 7 % 251) as u8 + 1)
        };
        calls.push(call);
    }
    let body = FunctionBody::from_calls(LaneShape::Pairs, &calls).expect("fits");
    let mut r = Box::new(Resident([0; ROWS * ROW]));
    let slab: &mut [u8; VALUE_SLAB_LEN] = (&mut r.0[VALUE_AT..VALUE_AT + VALUE_SLAB_LEN])
        .try_into()
        .expect("480");
    body.write_into_value_slab(slab);
    for l in 0..CONTENT_SLOTS {
        let c = 0xC0DE_0000u32 | l as u32;
        slab[l * SLOT_STRIDE..l * SLOT_STRIDE + CLASSID_BYTES].copy_from_slice(&c.to_le_bytes());
    }
    r
}

fn rows(r: &Resident) -> &[NodeRow] {
    node_rows_from_le_bytes(&r.0).expect("aligned, whole rows")
}

fn slot(row: &NodeRow, l: usize) -> &[u8; 16] {
    (&row.value[l * SLOT_STRIDE..(l + 1) * SLOT_STRIDE])
        .try_into()
        .expect("16")
}

// ── tests ────────────────────────────────────────────────────────────────

/// The V3 facet tier and the V4 call are the same two bytes at the same
/// address, for all 180 calls — and the row itself is the buffer, not a copy.
#[test]
fn both_readings_address_the_same_bytes() {
    let r = resident();
    let rows = rows(&r);
    assert_eq!(rows.len(), ROWS);
    assert_eq!(
        rows[0].value.as_ptr(),
        r.0[VALUE_AT..].as_ptr(),
        "row is a reinterpret"
    );
    let slab = &rows[0].value;
    let mut r2il_calls = 0;
    for l in 0..CONTENT_SLOTS {
        let facet = FacetCascade::ref_from_bytes(slot(&rows[0], l)).expect("16-aligned slot");
        assert_eq!(
            facet as *const FacetCascade as *const u8,
            slot(&rows[0], l).as_ptr(),
            "V3 facet is a reinterpret of the slot"
        );
        assert_eq!(facet.facet_classid, 0xC0DE_0000 | l as u32);
        for k in 0..6 {
            let i = l * 6 + k;
            let call = call_in_slab(slab, LaneShape::Pairs, i);
            let tier = &facet.tiers[k];
            assert_eq!(tier.lo, call.function.0, "call {i}: function byte");
            assert_eq!(tier.hi, call.values[0], "call {i}: immediate byte");
            assert_eq!(
                tier as *const _ as usize,
                slab.as_ptr() as usize + FunctionBody::call_slab_offset(LaneShape::Pairs, i),
                "call {i}: V3 tier and V4 call are not at the same address"
            );
            r2il_calls += usize::from(R2ILFn::ordinal(call.function).is_some());
        }
    }
    // Anti-vacuity: both kinds of call are present, so the mask below selects.
    let mask = r2il_mask(slab, LaneShape::Pairs);
    assert_eq!(mask.count() as usize, r2il_calls);
    assert!(r2il_calls > 0 && r2il_calls < 180);
}

/// `Triples` and `Quads` read the same slab in place too: every call's bytes
/// are the slab's bytes at `call_slab_offset`. A different carving is a
/// different reading, never a rewrite.
#[test]
fn every_lane_shape_reads_the_slab_in_place() {
    let r = resident();
    let slab = &rows(&r)[0].value;
    for shape in LaneShape::ALL {
        for i in 0..shape.calls_per_function() {
            let at = FunctionBody::call_slab_offset(shape, i);
            let call = call_in_slab(slab, shape, i);
            assert_eq!(call.function.0, slab[at]);
            for v in 0..shape.values_per_call() {
                assert_eq!(call.values[v], slab[at + 1 + v]);
            }
        }
    }
}

/// The owner writes one byte. Without rebuilding anything, the V3 facet, the
/// V4 call and the R2IL mask all see it.
#[test]
fn an_owner_write_is_seen_by_both_readings() {
    let mut r = resident();
    // Call 1 is an R2IL op (1 % 3 != 0). It sits in slot 0, tier 1.
    let i = 1;
    let at = VALUE_AT + FunctionBody::call_slab_offset(LaneShape::Pairs, i);
    let before = r2il_mask(&rows(&r)[0].value, LaneShape::Pairs);
    assert!(before.contains(i as u32));

    r.0[at] = FnIndex::ADD.0; // the owner's write: R2IL op -> core ADD

    let rows = rows(&r);
    let facet = FacetCascade::ref_from_bytes(slot(&rows[0], 0)).expect("aligned");
    assert_eq!(
        facet.tiers[1].lo,
        FnIndex::ADD.0,
        "V3 did not see the write"
    );
    assert_eq!(
        call_in_slab(&rows[0].value, LaneShape::Pairs, i).function,
        FnIndex::ADD,
        "V4 did not see the write"
    );
    let after = r2il_mask(&rows[0].value, LaneShape::Pairs);
    assert!(
        !after.contains(i as u32),
        "R2IL selection did not see the write"
    );
    assert_eq!(after.count() + 1, before.count());
}

/// Taking every reading leaves the bytes exactly as they were. The snapshot is
/// the test's oracle, not part of either reading.
#[test]
fn reading_never_mutates_the_bytes() {
    let r = resident();
    let snapshot = r.0.to_vec();
    let rows = rows(&r);
    let slab = &rows[0].value;
    let mut sink = 0u64;
    for l in 0..CONTENT_SLOTS {
        let f = FacetCascade::ref_from_bytes(slot(&rows[0], l)).expect("aligned");
        sink = sink.wrapping_add(u64::from(f.facet_classid));
    }
    for shape in LaneShape::ALL {
        for i in 0..shape.calls_per_function() {
            sink = sink.wrapping_add(u64::from(call_in_slab(slab, shape, i).function.0));
        }
        let m = r2il_mask(slab, shape);
        sink = sink.wrapping_add(project(slab, shape, &m).count() as u64);
    }
    assert_ne!(sink, 0, "anti-vacuity: the readings read something");
    assert_eq!(
        r.0.as_slice(),
        snapshot.as_slice(),
        "a reading wrote to the bytes"
    );
}

/// Both readings allocate nothing: reinterpret, in-place call reads, an
/// inline `CallMask`, and a lazy `project`.
#[test]
fn readings_allocate_nothing() {
    let r = resident();
    let ((), heap) = measure(|| {
        let rows = rows(&r);
        let slab = &rows[0].value;
        let mut n = 0usize;
        for l in 0..CONTENT_SLOTS {
            let f = FacetCascade::ref_from_bytes(slot(&rows[0], l)).expect("aligned");
            n += usize::from(f.tiers[0].lo != 0);
        }
        for i in 0..LaneShape::Pairs.calls_per_function() {
            n += usize::from(call_in_slab(slab, LaneShape::Pairs, i).function.0 != 0);
        }
        let m = r2il_mask(slab, LaneShape::Pairs);
        n += project(slab, LaneShape::Pairs, &m).count();
        assert!(n > 0);
    });
    assert_eq!(heap, 0, "a reading allocated");
    // Can-fire: the meter sees a list being built over the same reading.
    let (_, listed) = measure(|| {
        let m = r2il_mask(&rows(&r)[0].value, LaneShape::Pairs);
        m.materialize_indices()
    });
    assert!(listed > 0, "the heap meter cannot fire");
}

/// The 30 per-slot classids are V3 data the V3 reading sees, and the V4
/// reading never consults them: rewriting every one changes no call.
#[test]
fn slot_classids_do_not_reach_the_v4_reading() {
    let mut r = resident();
    let calls_before: Vec<Call> = (0..180)
        .map(|i| call_in_slab(&rows(&r)[0].value, LaneShape::Pairs, i))
        .collect();
    for l in 0..CONTENT_SLOTS {
        let at = VALUE_AT + l * SLOT_STRIDE;
        r.0[at..at + CLASSID_BYTES].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
    }
    let rows = rows(&r);
    let calls_after: Vec<Call> = (0..180)
        .map(|i| call_in_slab(&rows[0].value, LaneShape::Pairs, i))
        .collect();
    assert_eq!(
        calls_after, calls_before,
        "a slot classid leaked into a V4 call"
    );
    let f = FacetCascade::ref_from_bytes(slot(&rows[0], 3)).expect("aligned");
    assert_eq!(
        f.facet_classid, 0xFFFF_FFFF,
        "V3 must see the classid it owns"
    );
}

/// The bytes do not say which carving they are: the same slab under two
/// shapes yields two different call streams. Which reading applies has to
/// come from OUTSIDE the payload — the classid's job — and no shipped
/// function maps a classid to a `LaneShape`; every caller passes it in.
#[test]
fn the_reading_is_not_recoverable_from_the_bytes() {
    let r = resident();
    let slab = &rows(&r)[0].value;
    let pairs: Vec<u8> = (0..30)
        .map(|i| call_in_slab(slab, LaneShape::Pairs, i).function.0)
        .collect();
    let triples: Vec<u8> = (0..30)
        .map(|i| call_in_slab(slab, LaneShape::Triples, i).function.0)
        .collect();
    assert_ne!(
        pairs, triples,
        "two carvings agreed; the selector would be inert"
    );
}

/// The copy point. loco's `Interpreter` runs a `Program`, and a `Program`
/// OWNS gathered `FunctionBody` values. The only way from resident bytes to
/// one is `read_from_value_slab`, which copies the 360 payload bytes into a
/// second representation at a different address. So the stored-body V4
/// reading is a reading; executing it through loco today is not.
#[test]
fn loco_execution_needs_a_gathered_copy() {
    let r = resident();
    let slab = &rows(&r)[0].value;
    let (program, heap) = measure(|| Program {
        functions: vec![FunctionBody::read_from_value_slab(LaneShape::Pairs, slab)],
    });
    let gathered = program.entry().as_body_bytes();
    let owner = r.0.as_ptr_range();
    assert!(
        !owner.contains(&gathered.as_ptr()),
        "the gathered body lives inside the resident buffer — no copy was made"
    );
    for l in 0..CONTENT_SLOTS {
        assert_eq!(
            &gathered[l * 12..(l + 1) * 12],
            &slab[l * SLOT_STRIDE + CLASSID_BYTES..(l + 1) * SLOT_STRIDE],
            "lane {l}: the gather is a copy of the payload bytes"
        );
    }
    assert!(
        heap >= core::mem::size_of::<FunctionBody>(),
        "the Program owns a heap copy"
    );
    assert_eq!(program.entry().len(), 180);
}
