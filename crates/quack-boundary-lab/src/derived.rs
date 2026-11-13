//! LAB: mask planes derived from lanes read through a [`LaneBinding`].
//!
//! D-QUACK-DERIVED-PLANE-PROBE, P2 and P3. A [`DerivedPlane`] has no public
//! constructor: its words are written by one of the three derivations below,
//! each of which reads its input through the binding's own check. It keeps
//! what it was derived from ([`Recipe`]), the frame it was derived under and
//! the row order.
//!
//! What that does and does not give is the subject of `tests/derived_plane.rs`.
//! Nothing here is an authority: privacy inside one lab crate is not an
//! issuer, and the three recipes do not have the same dependencies.

use lance_graph_dt64::dr::{Frame, FrameRefusal};
use lance_graph_dt64::{CoordSpace, Guard, LaneBinding, LaneSource, MethodId, Refusal};
use lance_graph_mask_risc::{
    execute_into, ExecError, Foreign, LaneRef, Out, Planes, Program, Scratch, Value,
};
use lance_graph_quack::{lower, Agg, Filter, Query};

use crate::{admitted, Prepared, Refused, Source, View, World, CLASS, FIELDS, ROWS};

/// Most planes one program is handed.
pub const MAX_PLANES: usize = 8;

/// How a plane's words were computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Recipe {
    /// Bit `k` of every row's value of bound field `field`.
    Bit { field: u8, k: u8 },
    /// The rows where registered method `method`'s guard holds.
    Guard { method: MethodId },
    /// The rows whose foreign key `fk` names a row some caller-supplied
    /// visibility set contains.
    ForeignVisible { fk: u8 },
}

/// Why a derived plane was not derived or not accepted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlaneRefusal {
    /// The program expects another number of planes.
    Count { expected: usize, got: usize },
    /// Plane `at` was derived by another recipe than the program expects.
    Recipe {
        at: usize,
        expected: Recipe,
        got: Recipe,
    },
    /// Plane `at` was derived under a frame that is not the source's current
    /// one.
    Frame { at: usize, why: FrameRefusal },
    /// Plane `at` is in another row order than the binding.
    Space { at: usize },
    /// Plane `at` was derived from a field this binding does not bind.
    FieldNotBound { at: usize, field: u8 },
    /// The lane is not a 32-bit lane.
    LaneKind,
    /// The method is not admitted by the store.
    Method(Refusal),
    /// The method has no guard, or the description is not the bound guard.
    Description,
    /// A foreign key names no row of the visibility set's domain.
    ForeignKeyOutOfRange { row: usize },
}

/// A mask plane derived inside this lab from lanes read through a binding.
#[derive(Debug, Clone)]
pub struct DerivedPlane {
    recipe: Recipe,
    frame: Frame,
    space: CoordSpace,
    words: Vec<u64>,
}

impl DerivedPlane {
    pub fn recipe(&self) -> Recipe {
        self.recipe
    }

    pub fn space(&self) -> CoordSpace {
        self.space
    }

    /// The words, for comparison with an oracle.
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// Bytes held.
    pub fn bytes(&self) -> usize {
        self.words.len() * 8
    }

    fn new(recipe: Recipe, source: &Source, binding: &LaneBinding, words: Vec<u64>) -> Self {
        DerivedPlane {
            recipe,
            frame: source.frames().seal(),
            space: binding.selection(),
            words,
        }
    }

    /// Pure representation: bit `k` of bound field `field`.
    pub fn bit(
        world: &World,
        view: &View,
        binding: &LaneBinding,
        source: &Source,
        field: u8,
        k: u8,
    ) -> Result<DerivedPlane, Refused> {
        let lane = binding
            .lane(&world.store, view, source.frames(), source, field)
            .map_err(Refused::Binding)?;
        let mut words = vec![0u64; lane.len().div_ceil(64)];
        let mut set = |r: usize, v: u32| {
            if k < 32 && (v >> k) & 1 == 1 {
                words[r / 64] |= 1u64 << (r % 64);
            }
        };
        match lane {
            LaneRef::U32(v) => v.iter().enumerate().for_each(|(r, &x)| set(r, x)),
            LaneRef::I32(v) => v.iter().enumerate().for_each(|(r, &x)| set(r, x as u32)),
            _ => return Err(Refused::Plane(PlaneRefusal::LaneKind)),
        }
        Ok(Self::new(Recipe::Bit { field, k }, source, binding, words))
    }

    /// Derived interpretation: the rows where `method`'s guard holds.
    ///
    /// The method is admitted and `described` must reproduce its bound
    /// guard's witness. `filter` is the caller's lowering of that guard over
    /// the binding's operands; that `filter` and `described` say the same
    /// thing is NOT checked here.
    pub fn guard(
        world: &World,
        view: &View,
        binding: &LaneBinding,
        source: &Source,
        method: MethodId,
        described: &Guard,
        filter: Filter,
    ) -> Result<DerivedPlane, Refused> {
        let plane = |r| Refused::Plane(r);
        let mref = world
            .store
            .method_ref(CLASS, method)
            .map_err(|e| plane(PlaneRefusal::Method(e)))?;
        let bound = world
            .store
            .admit(&mref)
            .map_err(|e| plane(PlaneRefusal::Method(e)))?;
        let stored = bound
            .when
            .as_ref()
            .ok_or(plane(PlaneRefusal::Description))?;
        if described.witness() != stored.witness() {
            return Err(plane(PlaneRefusal::Description));
        }
        let program = lower(&Query {
            filter,
            agg: Agg::Rows,
        })
        .map_err(|_| plane(PlaneRefusal::Description))?;
        let mut words = vec![0u64; ROWS.div_ceil(64)];
        let mut scratch = Scratch::for_program(&program, ROWS).map_err(Refused::Exec)?;
        let p = Prepared::over(program, source, binding);
        admitted(
            &p,
            world,
            view,
            binding,
            source,
            &mut scratch,
            Out::Mask(&mut words),
        )?;
        Ok(Self::new(Recipe::Guard { method }, source, binding, words))
    }

    /// Cross-population derivation: row `r` is set when `visible` holds the
    /// row the foreign key names. `visible` is a caller-supplied set over the
    /// foreign population; it has no frame, no row order and no owner here.
    pub fn foreign_visible(
        world: &World,
        view: &View,
        binding: &LaneBinding,
        source: &Source,
        fk: u8,
        visible: &[u64],
    ) -> Result<DerivedPlane, Refused> {
        let lane = binding
            .lane(&world.store, view, source.frames(), source, fk)
            .map_err(Refused::Binding)?;
        let LaneRef::U32(keys) = lane else {
            return Err(Refused::Plane(PlaneRefusal::LaneKind));
        };
        let mut words = vec![0u64; keys.len().div_ceil(64)];
        for (r, &key) in keys.iter().enumerate() {
            let word = visible
                .get(key as usize / 64)
                .ok_or(Refused::Plane(PlaneRefusal::ForeignKeyOutOfRange { row: r }))?;
            if word >> (key % 64) & 1 == 1 {
                words[r / 64] |= 1u64 << (r % 64);
            }
        }
        Ok(Self::new(
            Recipe::ForeignVisible { fk },
            source,
            binding,
            words,
        ))
    }
}

/// The field a recipe reads through the binding, when it reads exactly one.
fn field_of(recipe: Recipe) -> Option<u8> {
    match recipe {
        Recipe::Bit { field, .. } => Some(field),
        Recipe::ForeignVisible { fk } => Some(fk),
        Recipe::Guard { .. } => None,
    }
}

/// The checks a derived plane adds on top of the binding's own: recipe,
/// frame, row order, bound field. Allocates nothing.
pub fn check_planes(
    expects: &[Recipe],
    planes: &[&DerivedPlane],
    binding: &LaneBinding,
    source: &Source,
) -> Result<(), PlaneRefusal> {
    if expects.len() != planes.len() || planes.len() > MAX_PLANES {
        return Err(PlaneRefusal::Count {
            expected: expects.len(),
            got: planes.len(),
        });
    }
    for (at, (plane, &expected)) in planes.iter().zip(expects).enumerate() {
        if plane.recipe != expected {
            return Err(PlaneRefusal::Recipe {
                at,
                expected,
                got: plane.recipe,
            });
        }
        source
            .frames()
            .check(&plane.frame)
            .map_err(|why| PlaneRefusal::Frame { at, why })?;
        if plane.space != binding.selection() {
            return Err(PlaneRefusal::Space { at });
        }
        if let Some(field) = field_of(plane.recipe) {
            if !binding.fields().has(field) {
                return Err(PlaneRefusal::FieldNotBound { at, field });
            }
        }
    }
    Ok(())
}

/// [`admitted`] for a program that also reads derived planes: plane operand
/// `i` is `planes[i]`, and must have been derived by `expects[i]`.
///
/// Order: the binding's joint check, the program's operand fields and frame,
/// then [`check_planes`], then execution. No foreign table is admitted.
#[allow(clippy::too_many_arguments)]
pub fn admitted_planes(
    p: &Prepared,
    expects: &[Recipe],
    planes: &[&DerivedPlane],
    world: &World,
    view: &View,
    binding: &LaneBinding,
    source: &Source,
    scratch: &mut Scratch<'_>,
    out: Out<'_>,
) -> Result<Value, Refused> {
    let mut lanes = [LaneRef::U32(&[]); FIELDS];
    let mut bound = 0usize;
    for item in binding
        .borrow_all(&world.store, view, source.frames(), source)
        .map_err(Refused::Binding)?
    {
        let (_field, lane) = item.map_err(Refused::Binding)?;
        lanes[bound] = lane;
        bound += 1;
    }
    let lanes = &lanes[..bound];
    if !(0..=u8::MAX)
        .filter(|&f| binding.fields().has(f))
        .eq(p.fields.iter().copied())
    {
        return Err(Refused::Operands {
            program: p.fields.clone(),
            binding: crate::bound_positions(binding),
        });
    }
    source.frames().check(&p.frame).map_err(Refused::Program)?;
    check_planes(expects, planes, binding, source).map_err(Refused::Plane)?;
    let mut masks: [&[u64]; MAX_PLANES] = [&[]; MAX_PLANES];
    for (slot, plane) in masks.iter_mut().zip(planes) {
        *slot = &plane.words;
    }
    run(&p.program, lanes, &masks[..planes.len()], scratch, out)
}

fn run(
    program: &Program,
    lanes: &[LaneRef<'_>],
    masks: &[&[u64]],
    scratch: &mut Scratch<'_>,
    out: Out<'_>,
) -> Result<Value, Refused> {
    let planes = Planes {
        n_rows: lanes.first().map_or(0, LaneRef::len),
        masks,
        lanes,
    };
    execute_into(program, &planes, &Foreign::NONE, scratch, out).map_err(|e| match e {
        ExecError::ForeignOutOfRange(_) => {
            Refused::Blocked("foreign plane: no owner-issued residency admission")
        }
        ExecError::ForeignLaneOutOfRange(_) => {
            Refused::Blocked("foreign lane: no admission of the fk's coordinate domain")
        }
        other => Refused::Exec(other),
    })
}
