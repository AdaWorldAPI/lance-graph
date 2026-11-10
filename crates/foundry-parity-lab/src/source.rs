//! The source owner: one typed plane per field, in the source's own row
//! order, with the frame store that lives with them. It implements DT64's
//! [`LaneSource`]; it never hands out a copy.

use std::cell::Cell;

use lance_graph_contract::ontology::FieldRef;
use lance_graph_dt64::dr::{Frame, FrameRevisions, FrameStore, Scope};
use lance_graph_dt64::{CoordSpace, LaneAddr, LaneSource, SchemaRev};

use crate::bind::{field, Kind, CLASS, FIELD_COUNT, READER, SPACE_A, SPACE_REVERSED};
use crate::fixture::{self as fx, STORES};

/// One borrowed plane: a value lane or a row mask (bit `r` = row `r`).
#[derive(Debug, Clone, Copy)]
pub enum Plane<'a> {
    I32(&'a [i32]),
    U32(&'a [u32]),
    Mask(&'a [u64]),
}

#[derive(Debug)]
enum Owned {
    I32(Box<[i32]>),
    U32(Box<[u32]>),
    Mask(Box<[u64]>),
}

impl Owned {
    fn bytes(&self) -> usize {
        match self {
            Owned::I32(v) => v.len() * 4,
            Owned::U32(v) => v.len() * 4,
            Owned::Mask(v) => v.len() * 8,
        }
    }
}

/// Which fixture row sits at each physical row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    /// Physical row `r` is store row `r`.
    Identity,
    /// Physical row `r` is store row `63 - r`: same stores, another order.
    Reversed,
}

impl Order {
    fn row(self, r: usize) -> usize {
        match self {
            Order::Identity => r,
            Order::Reversed => STORES - 1 - r,
        }
    }

    /// The coordinate space this order is published under.
    pub fn space(self) -> CoordSpace {
        match self {
            Order::Identity => SPACE_A,
            Order::Reversed => SPACE_REVERSED,
        }
    }
}

/// The owner of every plane of the network.
#[derive(Debug)]
pub struct Network {
    planes: Vec<Owned>,
    order: Order,
    frames: FrameStore,
    restocked: bool,
    /// The scope every lane serves.
    pub scope: Scope,
    /// A field the source declines to map (models a missing source lane).
    pub unbound: Option<u8>,
    /// Lane borrows handed out.
    pub borrows: Cell<u64>,
}

fn mask_from(pred: impl Fn(usize) -> bool) -> Box<[u64]> {
    let mut w = 0u64;
    for r in 0..STORES {
        if pred(r) {
            w |= 1 << r;
        }
    }
    vec![w].into_boxed_slice()
}

fn lane_i32(f: impl Fn(usize) -> i32) -> Box<[i32]> {
    (0..STORES).map(f).collect()
}

impl Network {
    /// Ingest the fixture into planes in `order`.
    pub fn ingest(order: Order) -> Network {
        let mut n = Network {
            planes: Vec::with_capacity(FIELD_COUNT),
            order,
            frames: FrameStore::new(FrameRevisions {
                source: 1,
                relation_map: 1,
                coordinates: 1,
            }),
            restocked: false,
            scope: READER,
            unbound: None,
            borrows: Cell::new(0),
        };
        n.rebuild();
        n
    }

    fn rebuild(&mut self) {
        let o = self.order;
        let rs = self.restocked;
        self.planes = (0..FIELD_COUNT)
            .map(|f| match field(f as u8) {
                Kind::Stock(k) => Owned::I32(lane_i32(|r| fx::stock(o.row(r), k, rs))),
                Kind::Known(k) => Owned::Mask(mask_from(|r| fx::known(o.row(r), k))),
                Kind::Cost(k) => Owned::I32(lane_i32(|r| fx::unit_cost(o.row(r), k))),
                Kind::Staff(j) => Owned::I32(lane_i32(|r| fx::staff(o.row(r), j))),
                Kind::Equipment(e) => Owned::Mask(mask_from(|r| fx::equipment(o.row(r), e))),
                Kind::Offered(m) => Owned::Mask(mask_from(|r| fx::offered(o.row(r), m))),
                Kind::DeliverySlots => Owned::I32(lane_i32(|r| fx::delivery_slots(o.row(r)))),
                Kind::DeliveryMinutes => Owned::I32(lane_i32(|r| fx::delivery_minutes(o.row(r)))),
                Kind::StoreId => Owned::U32((0..STORES).map(|r| fx::store_id(o.row(r))).collect()),
            })
            .collect();
    }

    /// The source-side change: store 06 receives cheese. The planes change in
    /// place and the frame advances, so every binding sealed before is stale.
    pub fn restock(&mut self) {
        let r = (0..STORES)
            .find(|&r| self.order.row(r) == fx::case::RESTOCKED)
            .expect("the restocked store is in the population");
        let f = crate::bind::STOCK0 + fx::CHEESE;
        if let Owned::I32(lane) = &mut self.planes[f] {
            lane[r] = fx::stock(fx::case::RESTOCKED, fx::CHEESE, true);
        }
        self.restocked = true;
        let next = self.frames.revisions().source + 1;
        self.frames.advance(FrameRevisions {
            source: next,
            relation_map: 1,
            coordinates: 1,
        });
    }

    /// A source-side price change at store row `s`, category `k`. The plane
    /// changes in place and the frame advances.
    pub fn set_unit_cost(&mut self, s: usize, k: usize, value: i32) {
        let r = (0..STORES)
            .find(|&r| self.order.row(r) == s)
            .expect("the store is in the population");
        if let Owned::I32(lane) = &mut self.planes[crate::bind::COST0 + k] {
            lane[r] = value;
        }
        let next = self.frames.revisions().source + 1;
        self.frames.advance(FrameRevisions {
            source: next,
            relation_map: 1,
            coordinates: 1,
        });
    }

    pub fn order(&self) -> Order {
        self.order
    }

    /// Bytes the source holds (the only canonical copy).
    pub fn resident_bytes(&self) -> usize {
        self.planes.iter().map(Owned::bytes).sum()
    }
}

impl LaneSource for Network {
    type Lane<'a>
        = Plane<'a>
    where
        Self: 'a;

    fn lane_of(
        &self,
        class: u32,
        rev: SchemaRev,
        field: u8,
        _basis: &FieldRef,
        _frame: &Frame,
    ) -> Option<LaneAddr> {
        if class != CLASS
            || rev != SchemaRev(1)
            || usize::from(field) >= self.planes.len()
            || self.unbound == Some(field)
        {
            return None;
        }
        Some(LaneAddr {
            space: self.order.space(),
            lane: u16::from(field),
            scope: self.scope,
        })
    }

    fn borrow(&self, addr: &LaneAddr) -> Option<Plane<'_>> {
        self.borrows.set(self.borrows.get() + 1);
        Some(match self.planes.get(usize::from(addr.lane))? {
            Owned::I32(v) => Plane::I32(v),
            Owned::U32(v) => Plane::U32(v),
            Owned::Mask(v) => Plane::Mask(v),
        })
    }

    fn frames(&self) -> &FrameStore {
        &self.frames
    }
}
