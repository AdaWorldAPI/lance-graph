//! Thought-git lab: a bound row space and a land through the writer contract.
//!
//! Two independent parts, both lab only:
//!
//! - [`bound`]: a thought whose rows are local positions of a DT64
//!   [`LocalMap`](lance_graph_dt64::dr::LocalMap). It records the map it was
//!   written under and is admitted later only under that map or a pure
//!   append of it.
//! - [`land`]: the thoughts of one supervisor cycle are merged by the probe's
//!   from-scratch oracle, cast as one batch through
//!   [`persist_cycle`](lance_graph_planner::persist_sink::persist_cycle) into
//!   any [`WalSink`](lance_graph_planner::persist_sink::WalSink). The write
//!   is fire and forget: reads are answered by the resident version the land
//!   returned, under the probe's read contract (given, else landed, else
//!   absent). The durable image is read only on recovery, to rebuild that
//!   resident version.
//!
//! What this is not: the row encoding in [`land`] is a lab encoding, not a
//! decision about where a thought's values live in the canonical 512-byte
//! row. No Lance store is linked; the sink in the tests is in memory.

pub mod bound;
pub mod land;
