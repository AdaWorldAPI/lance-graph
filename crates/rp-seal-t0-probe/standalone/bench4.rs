use std::fs::{File, OpenOptions};
use std::io::Write;
use std::time::Instant;
fn main(){
  let dir = std::env::args().nth(1).unwrap_or("/tmp/e2fsync".into());
  std::fs::create_dir_all(&dir).unwrap();
  // 1. fsync latency on an existing file (the "one durable boundary" cost)
  for &sz in &[512usize, 8*1024, 32*1024*1024] {
    let p = format!("{dir}/f_{sz}");
    let mut f = OpenOptions::new().create(true).write(true).truncate(true).open(&p).unwrap();
    let buf = vec![0x5Au8; sz];
    let iters = if sz > 1_000_000 {5} else {50};
    let t=Instant::now();
    for _ in 0..iters { f.write_all(&buf).unwrap(); f.sync_data().unwrap(); }
    let e=t.elapsed();
    println!("append+fsync sz={sz:>9}B  {:>8.3} ms/op", e.as_secs_f64()*1e3/iters as f64);
  }
  // 2. file CREATE + write + fsync + dir fsync (the per-fragment cost)
  for &n in &[64usize, 512, 4096] {
    let buf = vec![0x5Au8; 8*1024];
    let t=Instant::now();
    for i in 0..n {
      let p = format!("{dir}/frag_{i}");
      let mut f = File::create(&p).unwrap();
      f.write_all(&buf).unwrap(); f.sync_data().unwrap();
    }
    let e=t.elapsed();
    println!("create+8KiB+fsync x{n:<5} {:>8.2} ms total  {:>7.3} ms/file", e.as_secs_f64()*1e3, e.as_secs_f64()*1e3/n as f64);
    for i in 0..n { let _=std::fs::remove_file(format!("{dir}/frag_{i}")); }
  }
}
