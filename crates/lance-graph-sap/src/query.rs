//! Consumer orchestration only. Quack lowers; mask-RISC executes. This module
//! never evaluates a predicate, visits source rows, or looks up a label.
use crate::bind::{numc, utc, BindError, CatsBatch, ACTIVITY, EMPLOYEE, HOURS, WORK_DAY};
use lance_graph_mask_risc::{execute, ExecError, Operand, Planes, Scratch, Value};
use lance_graph_quack::{lower_group_by, Agg, Cmp, Col, Filter, GroupBy, GroupPlan};

/// Bound to one immutable batch, so dictionary IDs cannot cross populations.
pub struct CatsQuery<'a> {
    batch: &'a CatsBatch,
    plan: GroupPlan,
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
        let plan = lower_group_by(
            &GroupBy {
                filter: Filter::and([
                    Filter::cmp(Col(EMPLOYEE as u16), Cmp::EqU32(employee)),
                    Filter::cmp(Col(WORK_DAY as u16), Cmp::GeI32(from as i32)),
                    Filter::cmp(Col(WORK_DAY as u16), Cmp::LeI32(to as i32)),
                ]),
                key: Col(ACTIVITY as u16),
                groups: batch.activity_groups(),
                agg: Agg::SumI32(Col(HOURS as u16)),
            },
            0,
        )
        .map_err(|e| BindError(format!("Quack lowering: {e:?}")))?;
        let filter_scratch = Scratch::for_program(&plan.filter, batch.len())
            .map_err(|e| BindError(format!("scratch: {e:?}")))?;
        let group_scratch = Scratch::for_program(&plan.groups[0], batch.len())
            .map_err(|e| BindError(format!("scratch: {e:?}")))?;
        Ok(Self {
            batch,
            plan,
            filter_scratch,
            group_scratch,
        })
    }
    pub fn groups(&self) -> usize {
        self.plan.groups.len()
    }
    pub fn plan(&self) -> &GroupPlan {
        &self.plan
    }
    /// Hot phase: no allocation, text, reflection, DTOs or intermediate rows.
    /// Sums are in the batch's explicit decimal scale. Returns the BORROWED
    /// survivor mask for an optional terminal BAPI sink; no mask copy.
    /// Quack currently runs one kept filter plus one fold per dictionary key,
    /// not a claimed single-pass grouped aggregate.
    pub fn execute_into(&mut self, sums: &mut [i64]) -> Result<&[u64], ExecError> {
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
        let kept = execute(&self.plan.filter, &input, &mut self.filter_scratch, None)?;
        let Value::Mask(Operand::Scratch(slot)) = kept else {
            unreachable!("lowered comparison filter")
        };
        let mask = self.filter_scratch.slot(slot).expect("validated scratch");
        let masks = [mask];
        let grouped = Planes {
            n_rows: self.batch.len(),
            masks: &masks,
            lanes: &lanes,
        };
        for (program, sum) in self.plan.groups.iter().zip(sums) {
            let Value::SumI64(value) = execute(program, &grouped, &mut self.group_scratch, None)?
            else {
                unreachable!("SumI32 terminal")
            };
            *sum = value;
        }
        Ok(mask)
    }
}
