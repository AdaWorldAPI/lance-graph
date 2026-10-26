//! stdin: f32 values as hex bits, whitespace separated. stdout: ndarray `vstanh` of each,
//! as hex bits, one line. Used by `.claude/probes/tanh-sweep-v1/sweep.py` as the native arm.
use ndarray::hpc::vml::vstanh;
use ndarray::{Array1, ArrayView1};
use std::io::Read;

fn main() {
    let mut s = String::new();
    std::io::stdin().read_to_string(&mut s).expect("stdin");
    let xs: Vec<f32> = s.split_whitespace().map(|t| f32::from_bits(u32::from_str_radix(t, 16).expect("hex"))).collect();
    let mut out = Array1::<f32>::zeros(xs.len());
    vstanh(ArrayView1::from(&xs[..]), out.view_mut());
    let line: Vec<String> = out.iter().map(|v| format!("{:08x}", v.to_bits())).collect();
    println!("{}", line.join(" "));
}
