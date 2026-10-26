//! Amendment 6.5 P0.5: the i32 accumulator contract of `ndarray::hpc::amx_matmul::matmul_i8_to_i32`,
//! debug vs release, before and after the proposed `wrapping_*` change. Run against the production
//! ndarray and against a worktree with the proposal applied (README).
//!
//! Each case reports the tier taken, the i64 oracle, the i32 result and whether it equals the oracle.
//! A case whose oracle does not fit i32 cannot be "exact"; it shows what the function does instead.
use ndarray::Array2;

fn run(label: &str, m: usize, n: usize, k: usize, a: i8, w: i8) {
    let lhs = Array2::from_elem((m, k), a);
    let rhs = Array2::from_elem((k, n), w);
    let mut out = Array2::<i32>::zeros((m, n));
    let want = k as i64 * i64::from(a) * i64::from(w);
    let fits = want >= i64::from(i32::MIN) && want <= i64::from(i32::MAX);
    let tier = int8sem::matmul_tier(m, n, k);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ndarray::hpc::amx_matmul::matmul_i8_to_i32(lhs.view(), rhs.view(), out.view_mut()).unwrap();
        out[[0, 0]]
    }));
    let dbg = cfg!(debug_assertions);
    match r {
        Ok(v) => println!(
            "F9W {label} tier={tier} mnk={m}x{n}x{k} a={a} w={w} debug={dbg} got={v} want={want} fits_i32={fits} exact={} mod2^32={}",
            i64::from(v) == want,
            (i64::from(v) - want).rem_euclid(1 << 32) == 0
        ),
        Err(_) => println!("F9W {label} tier={tier} mnk={m}x{n}x{k} a={a} w={w} debug={dbg} PANIC want={want} fits_i32={fits}"),
    }
}

fn main() {
    std::panic::set_hook(Box::new(|_| {}));
    // final fits i32; the biased u8 x i8 accumulator does not
    run("i_biased_acc_overflow", 1, 16, 70_000, 127, 127);
    // final fits i32 (0); 128 * colsum does not
    run("ii_colsum_x128_overflow", 1, 16, 140_000, 0, 127);
    // same as i with k % 4 = 3, so the scalar K-tail adds onto an already-wrapped accumulator
    run("iii_ktail_after_overflow", 1, 16, 70_003, 127, 127);
    // AMX-aligned shape (16 x 16 x k, k % 64 = 0), final fits
    run("iv_amx_tier_biased_overflow", 16, 16, 70_016, 127, 127);
    // final does NOT fit i32: no correct i32 answer exists
    run("v_true_result_out_of_range", 1, 16, 200_000, 127, 127);
    // control: nothing overflows
    run("control_in_range", 1, 16, 1_000, 127, 127);
}
