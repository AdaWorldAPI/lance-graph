//! Contract tests that need no DuckDB. The differential run against the real
//! library is `harness/run.sh`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, AtomicUsize, Ordering};

use duckdb_selvec_compat::{
    capacity, create_selection_vector, destroy_selection_vector, positions,
    selection_from_positions, selection_vector_get_data_ptr, CopyReceipt,
};
use lance_graph_mask_risc::{execute_sel, ExecError, LaneRef, Planes, SelFold, Value};

static LIVE: AtomicIsize = AtomicIsize::new(0);
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

struct Counting;

// SAFETY: forwards to `System` unchanged; it only counts.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LIVE.fetch_add(layout.size() as isize, Ordering::Relaxed);
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        LIVE.fetch_add(layout.size() as isize, Ordering::Relaxed);
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as isize, Ordering::Relaxed);
        // SAFETY: `ptr` came from this allocator with this layout.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static A: Counting = Counting;

/// One test function: the allocator counters are process-global.
#[test]
fn the_adapter_contract() {
    // --- sizes: a handle and a pointer for every size, also 0 -------------
    for size in [0u64, 1, 64, 256, 2048, 65_536] {
        let sel = create_selection_vector(size);
        assert!(!sel.is_null());
        // SAFETY: `sel` is live.
        unsafe {
            let p = selection_vector_get_data_ptr(sel);
            assert!(!p.is_null());
            assert_eq!(
                p,
                selection_vector_get_data_ptr(sel),
                "the pointer is stable"
            );
            assert_eq!(capacity(sel), Some((size as usize).max(1)));
            assert_eq!(positions(sel, size).map(<[u32]>::len), Some(size as usize));
            assert!(positions(sel, size).unwrap().iter().all(|&x| x == 0));
            // A length the handle does not own is refused, not read.
            assert!(positions(sel, (size.max(1)) + 1).is_none());
            assert!(positions(sel, u64::MAX).is_none());
            destroy_selection_vector(sel);
        }
    }

    // --- null ---------------------------------------------------------------
    // SAFETY: null is an accepted argument for all four.
    unsafe {
        destroy_selection_vector(std::ptr::null_mut());
        assert!(selection_vector_get_data_ptr(std::ptr::null_mut()).is_null());
        assert_eq!(capacity(std::ptr::null_mut()), None);
        assert!(positions(std::ptr::null_mut(), 0).is_none());
    }

    // --- native selection -> C buffer: order and repeats kept, counted ------
    let native: Vec<u32> = vec![9, 4, 4, 4, 250, 0, 9, 3];
    let (sel, receipt) = selection_from_positions(&native);
    assert_eq!(
        receipt,
        CopyReceipt {
            entries: 8,
            bytes: 32,
            allocations: 2
        }
    );
    // SAFETY: `sel` is live; nothing else writes while the slices are read.
    unsafe {
        assert_eq!(positions(sel, 8).unwrap(), native.as_slice());
        let p = selection_vector_get_data_ptr(sel);
        // The buffer is its own storage, not the native list.
        assert_ne!(p.cast_const(), native.as_ptr());

        // --- a write through the raw pointer is what the next read sees -----
        *p.add(1) = 7;
        assert_eq!(positions(sel, 8).unwrap()[1], 7);
        assert_eq!(native[1], 4, "the native selection is untouched");

        // --- consumption: untrusted integers, checked by the consumer -------
        let values: Vec<i32> = (0..256).map(|i| i * 3 - 100).collect();
        let lanes = [LaneRef::I32(&values)];
        let planes = Planes {
            n_rows: 256,
            masks: &[],
            lanes: &lanes,
        };
        let pos = positions(sel, 8).unwrap();
        let want: i64 = pos.iter().map(|&q| i64::from(values[q as usize])).sum();
        assert_eq!(
            execute_sel(SelFold::SumI32 { lane: 0 }, &planes, pos),
            Ok(Value::SumI64(want))
        );
        // An external write puts a position outside the frame: refused at the
        // next consumption, with the offender named.
        *p.add(5) = 256;
        assert_eq!(
            execute_sel(
                SelFold::SumI32 { lane: 0 },
                &planes,
                positions(sel, 8).unwrap()
            ),
            Err(ExecError::SelOutOfRange {
                at: 5,
                position: 256
            })
        );
        destroy_selection_vector(sel);
    }

    // --- every allocation is returned ---------------------------------------
    let live_before = LIVE.load(Ordering::Relaxed);
    let allocs_before = ALLOCS.load(Ordering::Relaxed);
    for i in 0..500u64 {
        let sel = create_selection_vector(i % 97);
        // SAFETY: `sel` is live.
        unsafe {
            *selection_vector_get_data_ptr(sel) = i as u32;
            destroy_selection_vector(sel);
        }
    }
    assert_eq!(LIVE.load(Ordering::Relaxed), live_before, "bytes leaked");
    assert_eq!(
        ALLOCS.load(Ordering::Relaxed) - allocs_before,
        1000,
        "two allocations per selection: the handle and its buffer"
    );
}
