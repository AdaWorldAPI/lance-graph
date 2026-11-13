//! WHEN over one publication of a mailbox (D-VAS-C3 to C6). Stub: the test
//! is committed before the implementation.

use lance_graph_contract::class_view::ClassView;
use lance_graph_dt64::dr::Scope;
use lance_graph_dt64::{BindStore, Guard, LaneRefusal, RouteAt, SchemaRev};
use lance_graph_mask_risc::ExecError;
use lance_graph_quack::set::SetLowerError;
use lance_graph_quack::LowerError;

use crate::MailboxLanes;

/// Why a guard was not evaluated.
#[derive(Debug, Clone, PartialEq)]
pub enum WhenRefusal {
    Set(SetLowerError),
    PlaneUnissued { plane: u16 },
    ValidityUnstated,
    FieldOutOfRange { lane: u16 },
    Lane(LaneRefusal),
    Lower(LowerError),
    Exec(ExecError),
    RowOutsideLane { row: u32, rows: usize },
}

/// Where a guard holds.
#[derive(Debug)]
pub struct Fired {}

impl Fired {
    pub fn rows(&self) -> usize {
        0
    }
    pub fn count(&self) -> u64 {
        0
    }
    pub fn fields(&self) -> &[u8] {
        &[]
    }
    pub fn holds<V: ClassView + ?Sized, const N: usize>(
        &self,
        _store: &BindStore,
        _view: &V,
        _lanes: &MailboxLanes<'_, N>,
        _row: u32,
    ) -> Result<bool, WhenRefusal> {
        Err(WhenRefusal::ValidityUnstated)
    }
}

/// What a guard is evaluated for.
#[derive(Debug, Clone, Copy)]
pub struct WhenRequest {
    pub route: RouteAt,
    pub rev: SchemaRev,
    pub scope: Scope,
}

pub fn when<V: ClassView + ?Sized, const N: usize>(
    _lanes: &MailboxLanes<'_, N>,
    _store: &BindStore,
    _view: &V,
    _request: WhenRequest,
    _guard: &Guard,
) -> Result<Fired, WhenRefusal> {
    Err(WhenRefusal::ValidityUnstated)
}
