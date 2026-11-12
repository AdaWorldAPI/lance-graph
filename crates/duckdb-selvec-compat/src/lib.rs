//! DuckDB C-API `SelectionVector` compatibility (D-DR-SELECT-K-SV-3). LAB.
//!
//! Reference: DuckDB tag `v1.5.6`, commit `069cc9f9b5be802405797faecc284961b07c70ef`,
//! `src/include/duckdb.h` and `src/main/capi/data_chunk-c.cpp`. The three
//! functions are marked stable since v1.5.6 in that header:
//!
//! ```c
//! duckdb_selection_vector duckdb_create_selection_vector(idx_t size);
//! void duckdb_destroy_selection_vector(duckdb_selection_vector sel);
//! sel_t *duckdb_selection_vector_get_data_ptr(duckdb_selection_vector sel);
//! ```
//!
//! with `idx_t = uint64_t`, `sel_t = uint32_t` and the handle a pointer to
//! `struct _duckdb_selection_vector { void *internal_ptr; }`.
//!
//! ## What is reproduced
//!
//! - `create(size)` returns a non-null handle owning `max(size, 1)` entries,
//!   also for `size == 0` (upstream: `MaxValue(count, 1) * sizeof(sel_t)`).
//! - `get_data_ptr` returns the same writable pointer on every call until
//!   `destroy`. Writes through it are what the next read sees; nothing is
//!   cached, sorted or deduplicated.
//! - `destroy(NULL)` does nothing (upstream: `delete nullptr`).
//!
//! ## Where this differs, and what is not covered
//!
//! - **Handles are not interchangeable with DuckDB's.** Upstream's handle is
//!   a `duckdb::SelectionVector *`; this one points to [`SelBuf`]. A handle
//!   from here must never be passed to a real `duckdb_slice_vector` or
//!   `duckdb_vector_copy_sel`, and those two functions are not provided.
//! - Fresh entries are zero here. Upstream leaves them uninitialised in a
//!   release build, so a conforming caller cannot observe the difference.
//! - Allocation failure aborts the process here. Upstream throws a C++
//!   exception across the C boundary. Not compared.
//! - `get_data_ptr(NULL)`, use after `destroy` and double `destroy` are
//!   undefined upstream. Here `get_data_ptr(NULL)` returns null; the other
//!   two stay undefined.
//! - Upstream's handle does not know its size and checks no index. This one
//!   records its capacity so that [`positions`] can refuse a length it does
//!   not own. Whether a position addresses a row is checked by the consumer
//!   (`execute_sel` refuses out-of-range positions).
//!
//! ## Authority
//!
//! A buffer made here is caller-owned, writable memory. It carries no frame,
//! source version, scope or permission. Positions read from it are untrusted
//! integers: they may be folded over lanes the caller already holds, and they
//! never become a witnessed DR selection (`LaneBinding::selected` is the only
//! way to obtain one).

use std::alloc::{alloc_zeroed, dealloc, handle_alloc_error, Layout};
use std::ffi::c_void;

/// DuckDB `idx_t`.
pub type IdxT = u64;
/// DuckDB `sel_t`.
pub type SelT = u32;

/// DuckDB `struct _duckdb_selection_vector`. Never read; the handle is an
/// opaque pointer.
#[repr(C)]
pub struct DuckdbSelectionVectorOpaque {
    internal_ptr: *mut c_void,
}

/// DuckDB `duckdb_selection_vector`.
pub type DuckdbSelectionVector = *mut DuckdbSelectionVectorOpaque;

/// What a handle points to: one owned allocation and its capacity in entries.
struct SelBuf {
    data: *mut SelT,
    cap: usize,
}

fn layout(cap: usize) -> Option<Layout> {
    Layout::array::<SelT>(cap).ok()
}

/// `duckdb_create_selection_vector`. Owns `max(size, 1)` zeroed entries.
/// Aborts if the size cannot be allocated.
#[cfg_attr(
    feature = "duckdb-symbols",
    export_name = "duckdb_create_selection_vector"
)]
pub extern "C" fn create_selection_vector(size: IdxT) -> DuckdbSelectionVector {
    let cap = usize::try_from(size).ok().map(|n| n.max(1));
    let Some((cap, layout)) = cap.and_then(|c| Some((c, layout(c)?))) else {
        std::process::abort();
    };
    // SAFETY: `layout` has non-zero size (cap >= 1, SelT is 4 bytes).
    let data = unsafe { alloc_zeroed(layout) }.cast::<SelT>();
    if data.is_null() {
        handle_alloc_error(layout);
    }
    Box::into_raw(Box::new(SelBuf { data, cap })).cast()
}

/// `duckdb_destroy_selection_vector`. Null is accepted and ignored.
///
/// # Safety
/// `sel` is null or a handle from [`create_selection_vector`] /
/// [`selection_from_positions`] that has not been destroyed. Every pointer
/// obtained from it is invalid afterwards.
#[cfg_attr(
    feature = "duckdb-symbols",
    export_name = "duckdb_destroy_selection_vector"
)]
pub unsafe extern "C" fn destroy_selection_vector(sel: DuckdbSelectionVector) {
    if sel.is_null() {
        return;
    }
    // SAFETY: per contract, `sel` came from `Box::into_raw` of a `SelBuf`.
    let buf = unsafe { Box::from_raw(sel.cast::<SelBuf>()) };
    if let Some(layout) = layout(buf.cap) {
        // SAFETY: `data` was allocated with exactly this layout.
        unsafe { dealloc(buf.data.cast(), layout) };
    }
}

/// `duckdb_selection_vector_get_data_ptr`. The same pointer on every call.
///
/// # Safety
/// `sel` is null or a live handle from this crate.
#[cfg_attr(
    feature = "duckdb-symbols",
    export_name = "duckdb_selection_vector_get_data_ptr"
)]
pub unsafe extern "C" fn selection_vector_get_data_ptr(sel: DuckdbSelectionVector) -> *mut SelT {
    if sel.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: per contract, `sel` points to a live `SelBuf`.
    unsafe { (*sel.cast::<SelBuf>()).data }
}

/// What copying a native selection into a C buffer cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CopyReceipt {
    /// Entries copied.
    pub entries: usize,
    /// Bytes copied (`entries * 4`).
    pub bytes: usize,
    /// Allocations made (the handle and its buffer).
    pub allocations: usize,
}

/// Copy a borrowed native position list into a new owned C buffer, in list
/// order, repeats included. The source is not retained; the two share
/// nothing afterwards. This is the only way a native selection reaches a
/// writable pointer.
pub fn selection_from_positions(positions: &[SelT]) -> (DuckdbSelectionVector, CopyReceipt) {
    let sel = create_selection_vector(positions.len() as IdxT);
    // SAFETY: `sel` is live and owns at least `positions.len()` entries; the
    // fresh allocation cannot overlap the borrowed source.
    unsafe {
        let data = selection_vector_get_data_ptr(sel);
        std::ptr::copy_nonoverlapping(positions.as_ptr(), data, positions.len());
    }
    let receipt = CopyReceipt {
        entries: positions.len(),
        bytes: std::mem::size_of_val(positions),
        allocations: 2,
    };
    (sel, receipt)
}

/// Entries the handle owns. `None` for null.
///
/// # Safety
/// `sel` is null or a live handle from this crate.
pub unsafe fn capacity(sel: DuckdbSelectionVector) -> Option<usize> {
    if sel.is_null() {
        return None;
    }
    // SAFETY: per contract.
    Some(unsafe { (*sel.cast::<SelBuf>()).cap })
}

/// The first `len` entries as they are now. `None` for a null handle or a
/// `len` beyond what the handle owns. The entries are untrusted: nothing
/// here says they address rows of any frame.
///
/// # Safety
/// `sel` is null or a live handle from this crate, and nothing writes
/// through its data pointer or destroys it while the slice is alive.
pub unsafe fn positions<'a>(sel: DuckdbSelectionVector, len: IdxT) -> Option<&'a [SelT]> {
    // SAFETY: forwarded contract.
    let cap = unsafe { capacity(sel) }?;
    let len = usize::try_from(len).ok().filter(|&l| l <= cap)?;
    // SAFETY: `data` is valid for `cap >= len` initialised entries; aliasing
    // is the caller's contract.
    Some(unsafe { std::slice::from_raw_parts(selection_vector_get_data_ptr(sel), len) })
}
