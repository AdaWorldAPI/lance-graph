//! F9 on the public entry point: ndarray::hpc::amx_matmul::matmul_i8_to_i32 with a result that fits i32
//! while the u8 x i8 accumulator (and, in case ii, 128 * colsum) does not. Run in release and debug builds.
use ndarray::Array2;
fn run(label: &str, k: usize, a: i8, w: i8) {
    let lhs = Array2::from_elem((1, k), a);
    let rhs = Array2::from_elem((k, 16), w);
    let mut out = Array2::<i32>::zeros((1, 16));
    let want = k as i64 * i64::from(a) * i64::from(w);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ndarray::hpc::amx_matmul::matmul_i8_to_i32(lhs.view(), rhs.view(), out.view_mut()).unwrap();
        out[[0, 0]]
    }));
    match r {
        Ok(v) => println!("F9PUB {label} k={k} a={a} w={w} debug_assertions={} got={v} want={want} exact={}", cfg!(debug_assertions), i64::from(v) == want),
        Err(_) => println!("F9PUB {label} k={k} a={a} w={w} debug_assertions={} PANIC want={want}", cfg!(debug_assertions)),
    }
}
fn main() {
    run("i_unsigned_acc_overflow", 70_000, 127, 127);
    run("ii_colsum_x128_overflow", 140_000, 0, 127);
    run("control_in_range", 1_000, 127, 127);
}
