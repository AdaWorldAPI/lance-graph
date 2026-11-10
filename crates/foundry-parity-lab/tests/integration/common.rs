use foundry_parity_lab::arm_b::{demanded_fields, Answer, ArmRefusal, ItemQuery};
use foundry_parity_lab::bind::{World, READER};
use foundry_parity_lab::oracle::Cook;
use foundry_parity_lab::source::Network;
use lance_graph_dt64::LaneSource;

/// Bind the fields item `m` demands under the source's own row order, then
/// open the query (one `borrow_all`).
pub fn open<'s>(
    world: &World,
    net: &'s Network,
    m: usize,
    cost: bool,
    delivery: bool,
) -> Result<ItemQuery<'s>, ArmRefusal> {
    let binding = world.bind(
        net,
        &demanded_fields(m, cost, delivery),
        net.order().space(),
        READER,
    )?;
    ItemQuery::open(m, &binding, &world.store, &world.view, net.frames(), net)
}

/// The oracle's answer without the parts arm B cannot produce.
pub fn coarse(c: Cook) -> Answer {
    match c {
        Cook::NotAuthorized => Answer::NotAuthorized,
        Cook::Unknown { .. } => Answer::Unknown,
        Cook::Known { feasible, max, .. } => Answer::Known { feasible, max },
    }
}
