//! Writes the Gate A facts and the dispatch reading as TSV.
//!
//! ```text
//! CG_BEH_SOURCE_REV=$(git rev-parse HEAD) cargo run --release \
//!   --manifest-path crates/cg-beh-probe/Cargo.toml --bin facts -- <out-dir>
//! ```
//! Without a source revision nothing is written.

use std::path::PathBuf;

use cg_beh_probe::dispatch;
use cg_beh_probe::facts::Snapshot;
use cg_beh_probe::fixture::{self, FoundryVariant, Variant};

fn main() {
    let out = PathBuf::from(std::env::args().nth(1).expect("usage: facts <out-dir>"));
    let rev = std::env::var("CG_BEH_SOURCE_REV").unwrap_or_default();

    let lab = fixture::world(Variant::Base);
    let mut test = Snapshot::new(&rev, "cg-beh-probe lab class").expect("CG_BEH_SOURCE_REV");
    test.add_class(
        &lab.store,
        &lab.view,
        fixture::CLASS,
        &fixture::METHODS,
        &fixture::READ,
    )
    .expect("the lab class reads");

    let world = fixture::foundry(FoundryVariant::Base);
    let mut foundry = Snapshot::new(&rev, "foundry-parity-lab CanCook").expect("a revision");
    foundry
        .add_class(
            &world.store,
            &world.view,
            fixture::FOUNDRY_CLASS,
            &fixture::foundry_ids(),
            &fixture::FOUNDRY_READ,
        )
        .expect("the Foundry class reads");

    let src_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../lance-graph-r2il/src/lib.rs"
    );
    let src = std::fs::read_to_string(src_path).expect("the dispatch source");
    let arms = dispatch::read(&src).expect("the dispatch arms");

    std::fs::create_dir_all(&out).expect("the output directory");
    for (name, body) in [
        ("lab.facts.tsv", test.to_tsv()),
        ("foundry.facts.tsv", foundry.to_tsv()),
        ("dispatch.tsv", arms.to_tsv()),
    ] {
        std::fs::write(out.join(name), body).expect("a facts file");
        println!("wrote {}", out.join(name).display());
    }
}
