//! Consumer orchestration only. Quack lowers; mask-RISC executes. This module
//! never evaluates a predicate, visits source rows, or looks up a label.
use crate::bind::{numc, utc, BindError, CatsBatch, ACTIVITY, EMPLOYEE, HOURS, WORK_DAY};
use lance_graph_mask_risc::{
    execute_into, ExecError, Foreign, Out, Planes, Program, Scratch, Value,
};
use lance_graph_quack::{lower, Agg, Cmp, Col, Filter, Query};

/// Bound to one immutable batch, so dictionary IDs cannot cross populations.
pub struct CatsQuery<'a> {
    batch: &'a CatsBatch,
    plan: Program,
    selection: Program,
    filter_scratch: Scratch<'static>,
    group_scratch: Scratch<'static>,
}
impl<'a> CatsQuery<'a> {
    /// Cold phase: employee and inclusive UTC calendar dates bind once.
    /// ApprovalStatus is absent from the ABAP/SIMAFPort schema and is not
    /// inferred from an approver or a timestamp.
    pub fn prepare(
        batch: &'a CatsBatch,
        employee: &str,
        from: &str,
        to: &str,
    ) -> Result<Self, BindError> {
        let employee = numc(employee)?;
        let from = utc(&format!("{from}T00:00:00Z"))? / 1_000_000;
        let to = utc(&format!("{to}T00:00:00Z"))? / 1_000_000;
        if from > to {
            return Err(BindError("reversed date range".into()));
        }
        let filter = Filter::and([
            Filter::cmp(Col(EMPLOYEE as u16), Cmp::EqU32(employee)),
            Filter::cmp(Col(WORK_DAY as u16), Cmp::GeI32(from as i32)),
            Filter::cmp(Col(WORK_DAY as u16), Cmp::LeI32(to as i32)),
        ]);
        let plan = lower(&Query {
            filter: filter.clone(),
            agg: Agg::GroupSumI32 {
                key: Col(ACTIVITY as u16),
                val: Col(HOURS as u16),
            },
        })
        .map_err(|e| BindError(format!("Quack lowering: {e:?}")))?;
        let selection = lower(&Query {
            filter,
            agg: Agg::Rows,
        })
        .map_err(|e| BindError(format!("Quack lowering: {e:?}")))?;
        let filter_scratch = Scratch::for_program(&selection, batch.len())
            .map_err(|e| BindError(format!("scratch: {e:?}")))?;
        let group_scratch = Scratch::for_program(&plan, batch.len())
            .map_err(|e| BindError(format!("scratch: {e:?}")))?;
        Ok(Self {
            batch,
            plan,
            selection,
            filter_scratch,
            group_scratch,
        })
    }
    pub fn groups(&self) -> usize {
        self.batch.activity_groups() as usize
    }
    pub fn plan(&self) -> &Program {
        &self.plan
    }
    /// Scratch is bounded by the substrate tile size, independent of population.
    pub fn scratch_words(&self) -> usize {
        self.group_scratch.words() * self.group_scratch.slots()
            + self.filter_scratch.words() * self.filter_scratch.slots()
    }
    /// One Quack grouped fold, with no retained population mask or rows.
    /// Overwrites sums on every call, including an empty selection.
    pub fn execute_into(&mut self, sums: &mut [i64]) -> Result<(), ExecError> {
        if sums.len() != self.groups() {
            return Err(ExecError::LenMismatch {
                what: "group sums",
                expected: self.groups(),
                found: sums.len(),
            });
        }
        let lanes = self.batch.lanes();
        let input = Planes {
            n_rows: self.batch.len(),
            masks: &[],
            lanes: &lanes,
        };
        let value = execute_into(
            &self.plan,
            &input,
            &Foreign::NONE,
            &mut self.group_scratch,
            Out::I64(sums),
        )?;
        debug_assert_eq!(value, Value::GroupSummed);
        Ok(())
    }
    /// Optional boundary projection. Only a sink demanding original assignments
    /// supplies a population-sized mask; the aggregate does not retain one.
    pub fn select_into(&mut self, mask: &mut [u64]) -> Result<(), ExecError> {
        let lanes = self.batch.lanes();
        let input = Planes {
            n_rows: self.batch.len(),
            masks: &[],
            lanes: &lanes,
        };
        execute_into(
            &self.selection,
            &input,
            &Foreign::NONE,
            &mut self.filter_scratch,
            Out::Mask(mask),
        )?;
        Ok(())
    }
}
