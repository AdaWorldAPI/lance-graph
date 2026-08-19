use std::time::Instant;
const ROW: usize = 512;
fn lcg(s:&mut u64)->u64{*s=s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);*s}
fn main(){
  println!("{:>10} {:>10} {:>12} {:>12} {:>8}","rows","image_MiB","seq_GB/s","rand_GB/s","ratio");
  for &n in &[16_384usize, 65_536, 262_144, 1_048_576, 4_194_304] {
    let bytes=n*ROW;
    let mut image=vec![0u8;bytes];
    let src=vec![0xABu8;ROW];
    // warm
    for i in 0..n { image[i*ROW]=1; let _=i; }
    let t=Instant::now();
    for i in 0..n { image[i*ROW..i*ROW+ROW].copy_from_slice(&src); }
    let seq=t.elapsed().as_secs_f64();
    // random permutation via a full-period LCG-ish index set (distinct indices)
    let mut perm:Vec<u32>=(0..n as u32).collect();
    let mut s=0x9E3779B97F4A7C15u64;
    for i in (1..n).rev(){ let j=(lcg(&mut s) as usize)%(i+1); perm.swap(i,j); }
    let t=Instant::now();
    for &i in &perm { let i=i as usize; image[i*ROW..i*ROW+ROW].copy_from_slice(&src); }
    let rnd=t.elapsed().as_secs_f64();
    println!("{:>10} {:>10.1} {:>12.2} {:>12.2} {:>8.2}", n, bytes as f64/1048576.0,
      bytes as f64/seq/1e9, bytes as f64/rnd/1e9, rnd/seq);
    std::hint::black_box(&image);
  }
}
