//! Warm latency of one owner-backed virtual-property read. Release only.
//! `cargo run --release -p virtual-address-lab --example owner_read_cost`

use std::hint::black_box;
use std::time::Instant;

use cognitive_shader_driver::mailbox_soa::MailboxSoA;
use lance_graph_contract::cognitive_shader::MetaWord;
use lance_graph_contract::soa_view::MailboxSoaView;
use virtual_address_lab::owner::{Reads, Value, VirtualProperty};
use virtual_address_lab::recipe::Recipe;

fn main() {
    const N: usize = 64;
    let mut mailbox: Box<MailboxSoA<N>> = Box::new(MailboxSoA::new(7, 3, 1.0));
    mailbox.set_populated(N);
    for r in 0..N {
        mailbox.set_meta(r, MetaWord(r as u32 * 7 + 1));
    }
    let reads = Reads::default();
    let chain = [Recipe::TRANSPOSE, Recipe::ROTATE90, Recipe::ROTATE270];
    const ROUNDS: usize = 200_000;
    let mut medians = Vec::new();
    let mut direct = Vec::new();
    for _ in 0..21 {
        let t = Instant::now();
        let mut acc = 0u64;
        for i in 0..ROUNDS {
            let p = VirtualProperty {
                row: black_box(i % N),
                property: black_box(0),
                chain: black_box(&chain),
            };
            if let Ok(Value::U32(v)) = p.read(&*mailbox, &reads) {
                acc += u64::from(v);
            }
        }
        black_box(acc);
        medians.push(t.elapsed().as_nanos() as f64 / ROUNDS as f64);
        let t = Instant::now();
        let mut acc = 0u64;
        for i in 0..ROUNDS {
            acc += u64::from(mailbox.meta_raw()[black_box(i % N)]);
        }
        black_box(acc);
        direct.push(t.elapsed().as_nanos() as f64 / ROUNDS as f64);
    }
    medians.sort_by(f64::total_cmp);
    direct.sort_by(f64::total_cmp);
    println!("owner bytes            {}", mailbox.byte_footprint());
    println!(
        "descriptor bytes       {}",
        core::mem::size_of::<VirtualProperty<'_>>()
    );
    println!(
        "virtual read, 3-recipe chain   {:.2} ns (median of 21, min {:.2}, max {:.2})",
        medians[10], medians[0], medians[20]
    );
    println!(
        "direct meta_raw()[row]         {:.2} ns (median of 21, min {:.2}, max {:.2})",
        direct[10], direct[0], direct[20]
    );
}
