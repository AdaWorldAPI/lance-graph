//! Generic consumer hot-plug — the plug-and-play pattern EVERY consumer
//! migrates to (operator, 2026-07-07).
//!
//! Three roles, three homes:
//!
//! 1. **This module (the SOCKET, zero-dep):** the shapes a consumer uses to
//!    declare WHICH classids it hot-plugs, and the [`CapabilityAuthority`]
//!    trait the authority implements. No OGAR dep — the contract is a
//!    workspace member and MUST stay dependency-free (a path dep here breaks
//!    every CI cargo invocation at workspace-load time; learned 2026-07-07).
//! 2. **OGAR (the AUTHORITY):** resolves the hot-plugged classids to the vocab
//!    rows, the action definitions, AND the storage reading
//!    ([`Activation::read_modes`]), and verifies the registration (expected
//!    consumer, coverage both directions, ids minted exactly once).
//! 3. **The consumer:** declares one [`HotPlug`] const naming its classids +
//!    covered capabilities, calls `activate` in its own binary/tests — drift
//!    bangs once, no pins, no serialization, no per-consumer plug crate.
//!
//! The classid is the join key on BOTH sides: the consumer says "0x0805,
//! 0x0808, 0x0809 are hot", the authority hands back the concepts and every
//! action whose subject is one of those ids.
//!
//! # SPOG × slab metadata → tenant reading (the ONE resolution path)
//!
//! A row is read under three separate concerns, composed here and nowhere
//! else — there is no second registry:
//!
//! | authority | carrier | answers |
//! |---|---|---|
//! | SPOG context | the row key, via [`crate::spog_tenants::graph_of`] | which concept (graph) the row belongs to |
//! | OGAR registry | [`Activation`] (from [`CapabilityAuthority::activate`]) | does that concept exist here, and its CURRENT canonical reading |
//! | slab metadata | [`SlabDeclaration`] (beside the slab, never inside it) | the physical truth about bytes ALREADY written |
//! | this build | [`SlabReading::from_tag`], [`ENVELOPE_LAYOUT_VERSION`](crate::soa_envelope::ENVELOPE_LAYOUT_VERSION) | whether this binary implements that physical reading |
//!
//! [`Activation::resolve_for_context`] composes them into one
//! [`ResolvedReading`] or a named [`ActivationDrift`].
//!
//! - **An explicit declaration wins for reading existing data.** The OGAR
//!   reading is the current dispatch for new writes; it never makes an
//!   already persisted slab unreadable because the class has since migrated.
//! - **No declaration means inherited behaviour**, not "this is Facet96": the
//!   OGAR reading applies and [`ResolvedReading::slab`] is `None`.
//! - **No per-concept permission table.** A slab chooses its own physical
//!   reading; OGAR validates only that the semantic context exists, and this
//!   build only that it knows the reading.
//!
//! The declaration carries NO semantic identity: SPOG supplies it. Tenant
//! bytes stay content-blind: no resolution method takes a payload.

/// A consumer's hot-plug declaration: which classids it activates and which
/// capability names its executor covers. One `const` per consumer — the
/// whole registration surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HotPlug {
    /// Consumer name (crate name by convention) — the authority checks it
    /// against the expected-executor list of the tables it resolves.
    pub consumer: &'static str,
    /// The canon-high concept ids the consumer hot-plugs.
    pub classids: &'static [u16],
    /// Capability names the consumer's executor covers.
    pub covered: &'static [&'static str],
}

/// What the authority returns for a green activation: the vocab rows, the
/// capability names, and the STORAGE READING resolved for the hot-plugged
/// classids. Plain owned `std` types — zero-dep, no serialization.
///
/// **No `Default`, deliberately.** `Activation::default()` would be a
/// green-looking activation carrying no reading at all — precisely the shape
/// a consumer then turns into [`ReadMode::DEFAULT`] (V1). An activation is
/// something an authority RESOLVED; there is no meaningful empty one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Activation {
    /// `(concept, classid)` vocab rows for every hot-plugged id.
    pub concepts: Vec<(String, u16)>,
    /// Capability names whose subject is one of the hot-plugged ids.
    pub capabilities: Vec<String>,
    /// `(concept id, ReadMode)` — how a row addressed under each hot-plugged
    /// concept is READ: which tail the key carries, which value tenants
    /// materialise, how the edge block is carved.
    ///
    /// **Why this rides the activation rather than a registry entry**
    /// (D-BLOCKS-HOTPLUG-1, operator, 2026-09-07). The reading is not
    /// recorded in the key — [`crate::canonical_node::NodeGuid`] has no
    /// tail-variant accessor, and `decode()` (V1 `family:u24 ++ identity:u24`)
    /// versus `decode_v2()` (leaf + `u16`/`u16`) is chosen by classid alone.
    /// So SOMETHING must map a classid to its reading, and
    /// [`BUILTIN_READ_MODES`](crate::canonical_node::classid_read_mode)
    /// *"holds only the canon builtins"* — every entry there is a canon
    /// DOMAIN. Registering each consumer's seat there would make adding a
    /// frontend an edit to this crate plus a substrate recompile: the central
    /// lockstep the retired `COUNT_FUSE` belonged to, rebuilt one layer up.
    ///
    /// The authority already resolves the other two infos for exactly the
    /// hot-plugged ids; the reading is the third, from the same call, keyed
    /// the same way. A consumer composes its own full `u32`
    /// (`concept << 16 | its app prefix`) and reads rows with the mode the
    /// authority handed it — never by asking the canon registry about a class
    /// the canon does not own.
    ///
    /// **Owned, and per-plug — corrected 2026-09-07.** This was `&'static`,
    /// argued as "plug-and-play at COMPILE time: a reading is looked up, not
    /// computed". That argument had a consequence I did not weigh: a
    /// `&'static` table cannot be built per plug, so the authority could only
    /// hand back a table it already held ALL of, which forced an
    /// all-or-nothing guard and, behind it, one hard-coded row per consumer
    /// seat. That is the central lockstep `D-BLOCKS-HOTPLUG-1` retired,
    /// rebuilt one level up — and its failure mode is the worst kind: a
    /// consumer missing from the table keeps compiling and silently loses its
    /// V3 tail, discoverable only weeks later and nowhere near the edit.
    ///
    /// Owned lets the authority answer for exactly the ids a plug declared,
    /// derived from the plug rather than looked up in a shared list. Being
    /// plugged in IS the declaration
    /// ([`ReadMode::PLUG_AND_PLAY_V3`](crate::canonical_node::ReadMode::PLUG_AND_PLAY_V3)),
    /// and an authority may still override any single concept.
    ///
    /// **Fails closed by MECHANISM, not by prose** (operator, 2026-09-07:
    /// *"don't silently enforce V1 fallback in hotplug, that's
    /// unacceptable"*). This field is PRIVATE and the only lookup is
    /// [`Activation::read_mode_for`], which returns a `Result` — there is no
    /// `Option` to `.unwrap_or(ReadMode::DEFAULT)` and no public slice to
    /// scan. An earlier cut left it public with a doc comment saying an empty
    /// slice "is not assume-the-default"; a rule a caller can violate with a
    /// one-liner is not a rule.
    ///
    /// An empty table is still legitimate — a capability-only consumer (an
    /// executor that mints no keys) has no reading to declare. What the type
    /// now guarantees is that *asking* for an absent one bangs instead of
    /// quietly yielding a V1 tail.
    read_modes: Vec<(u16, crate::canonical_node::ReadMode)>,
}

impl Activation {
    /// Build an activation. Authority-side constructor — the fields are not
    /// public, so this is how [`CapabilityAuthority`] implementations return
    /// one.
    #[must_use]
    pub fn new(
        concepts: Vec<(String, u16)>,
        capabilities: Vec<String>,
        read_modes: Vec<(u16, crate::canonical_node::ReadMode)>,
    ) -> Self {
        Self {
            concepts,
            capabilities,
            read_modes,
        }
    }

    /// The reading for one hot-plugged concept — the ONLY lookup, and it
    /// fails closed.
    ///
    /// Returns [`ActivationDrift::NoReadingFor`] when the authority declared
    /// no reading for `concept`. It deliberately does NOT return an `Option`:
    /// an `Option` invites `.unwrap_or(ReadMode::DEFAULT)`, and
    /// [`ReadMode::DEFAULT`] is a **V1** tail. A consumer that mints keys
    /// under a hot-plugged seat must get its reading from here or bang; there
    /// is no third outcome.
    ///
    /// [`ReadMode::DEFAULT`]: crate::canonical_node::ReadMode::DEFAULT
    ///
    /// # Errors
    ///
    /// [`ActivationDrift::NoReadingFor`] if `concept` has no declared reading.
    pub fn read_mode_for(
        &self,
        concept: u16,
    ) -> Result<crate::canonical_node::ReadMode, ActivationDrift> {
        self.read_modes
            .iter()
            .find_map(|(c, m)| (*c == concept).then_some(*m))
            .ok_or(ActivationDrift::NoReadingFor(concept))
    }

    /// Every declared `(concept, reading)` pair — the AUDIT surface.
    ///
    /// For tests and authority conformance checks that assert the whole
    /// table. Consumers resolving a single seat use
    /// [`read_mode_for`](Activation::read_mode_for): this returns a plain
    /// slice, so scanning it and defaulting is once again expressible, and
    /// that is exactly what the lookup exists to avoid.
    #[must_use]
    pub fn declared_readings(&self) -> &[(u16, crate::canonical_node::ReadMode)] {
        &self.read_modes
    }
}

/// A physical reading a slab declares for itself.
///
/// The variants are exactly the readings THIS build implements. A new
/// reading (as the classid-free 128-bit register and its signed carvings
/// did) lands as a new variant; which slabs use it is decided by the slab's own declaration, never
/// by a per-concept table. The on-wire tag is a `u8` in the slab's metadata
/// envelope, decoded by [`SlabReading::from_tag`], which refuses any tag this
/// build does not implement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SlabReading {
    /// The canon self-describing `classid(4) + payload(12)` facet
    /// ([`crate::facet::FacetCascade`]). Declared only by a slab whose bytes
    /// really are that facet.
    Facet96 = 0,
    /// The 128-bit working register with NO classid in the payload
    /// (`D-LXC-29`, [`crate::register128::Register128`]), held in the
    /// [`ValueTenant::Register0`](crate::canonical_node::ValueTenant::Register0) /
    /// [`ValueTenant::Register1`](crate::canonical_node::ValueTenant::Register1)
    /// rails. The register's semantic identity is the SPOG context this
    /// declaration is resolved under, not its bytes. Declaring it does not
    /// change how Facet96 lanes are read; it is what
    /// [`ResolvedReading::bind_register128`] requires before granting the
    /// register rails.
    Register128 = 1,
    /// The same register rails, written as 32 signed `i4` values: dim `2k`
    /// is the low nibble of byte `k`, dim `2k+1` the high nibble (the
    /// [`crate::atoms::I4x32`] layout). Bound by
    /// [`ResolvedReading::bind_signed_register`], never by
    /// [`ResolvedReading::bind_register128`].
    RegisterI4x32 = 2,
    /// The same register rails, written as 16 signed `i8` values, one per
    /// byte. Bound by [`ResolvedReading::bind_signed_register`].
    RegisterI8x16 = 3,
}

impl SlabReading {
    /// Decode the envelope tag.
    ///
    /// # Errors
    ///
    /// [`ActivationDrift::UnknownSlabReading`] for any tag this build does not
    /// implement.
    pub const fn from_tag(tag: u8) -> Result<Self, ActivationDrift> {
        match tag {
            0 => Ok(SlabReading::Facet96),
            1 => Ok(SlabReading::Register128),
            2 => Ok(SlabReading::RegisterI4x32),
            3 => Ok(SlabReading::RegisterI8x16),
            other => Err(ActivationDrift::UnknownSlabReading(other)),
        }
    }
}

/// What a slab's metadata envelope DECLARES about its own, already written
/// bytes. Physical facts only — no concept, no classid, no ontology: the SPOG
/// context supplies those.
///
/// For reading existing data the declaration is the authority on the physical
/// side; the OGAR reading does not override it. A writer records the reading
/// it actually used here, so the slab stays readable after its class migrates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SlabDeclaration {
    /// The physical reading the slab was written with.
    pub reading: SlabReading,
    /// Which value tenants the slab materialised.
    pub value_schema: crate::canonical_node::ValueSchema,
    /// The envelope layout the slab was written under —
    /// [`SoaEnvelope::LAYOUT_VERSION`](crate::soa_envelope::SoaEnvelope::LAYOUT_VERSION)
    /// of its writer. A layout this build does not implement is refused,
    /// never reinterpreted.
    pub layout_version: u8,
}

/// The output of the one resolution path. `Copy + Eq + Hash`, so a caller can
/// resolve once per `(concept, declaration)` and hand the value to whatever
/// processes the population.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResolvedReading {
    /// The SPOG concept the reading was resolved under.
    pub concept: u16,
    /// The runtime reading. Tail and edge codec come from OGAR (they read the
    /// key and the edge block, which a slab does not own); the value schema
    /// comes from the slab declaration when there is one.
    pub read_mode: crate::canonical_node::ReadMode,
    /// The declared physical reading, or `None` when the slab declared
    /// nothing and is read with today's inherited behaviour.
    pub slab: Option<SlabReading>,
}

impl Activation {
    /// SPOG context × slab declaration → [`ResolvedReading`]. The single
    /// resolution path, taking the concept directly so a caller holding one
    /// population (e.g. one [`crate::spog_tenants::SpogTenants`] tenant)
    /// resolves once and never per row.
    ///
    /// 1. **OGAR** — the concept must exist in this activation
    ///    ([`read_mode_for`](Activation::read_mode_for)), else
    ///    [`ActivationDrift::NoReadingFor`].
    /// 2. **No declaration** — the OGAR reading, `slab: None`.
    /// 3. **Declaration** — its layout version must be one this build
    ///    implements, else [`ActivationDrift::SlabLayoutVersion`] (the reading
    ///    tag was already checked by [`SlabReading::from_tag`]). Then the
    ///    declared value schema and reading win: the OGAR reading is today's
    ///    dispatch for new writes and does not invalidate bytes written under
    ///    an earlier one.
    ///
    /// # Errors
    ///
    /// The named [`ActivationDrift`] arms above. No fallback on any path.
    pub fn resolve_for_context(
        &self,
        concept: u16,
        slab: Option<&SlabDeclaration>,
    ) -> Result<ResolvedReading, ActivationDrift> {
        let current = self.read_mode_for(concept)?;
        let Some(slab) = slab else {
            return Ok(ResolvedReading {
                concept,
                read_mode: current,
                slab: None,
            });
        };
        if slab.layout_version != crate::soa_envelope::ENVELOPE_LAYOUT_VERSION {
            return Err(ActivationDrift::SlabLayoutVersion {
                slab: slab.layout_version,
                expected: crate::soa_envelope::ENVELOPE_LAYOUT_VERSION,
            });
        }
        Ok(ResolvedReading {
            concept,
            read_mode: crate::canonical_node::ReadMode {
                value_schema: slab.value_schema,
                ..current
            },
            slab: Some(slab.reading),
        })
    }

    /// Convenience wrapper: the SPOG concept is
    /// [`crate::spog_tenants::graph_of`] of `key`, then
    /// [`resolve_for_context`](Activation::resolve_for_context). Takes a key,
    /// never a row or payload.
    ///
    /// # Errors
    ///
    /// As [`resolve_for_context`](Activation::resolve_for_context).
    pub fn resolve_tenant_reading(
        &self,
        key: crate::canonical_node::NodeGuid,
        slab: Option<&SlabDeclaration>,
    ) -> Result<ResolvedReading, ActivationDrift> {
        self.resolve_for_context(crate::spog_tenants::graph_of(key), slab)
    }
}

impl ResolvedReading {
    /// Bind the register rails of this population — the one place a
    /// Register128 reading is checked, done ONCE per population, never per
    /// row.
    ///
    /// Grants `rails` only when the slab declared
    /// [`SlabReading::Register128`] AND its value schema materialises every
    /// rail requested. The returned [`RegisterLanes`](crate::register128::RegisterLanes)
    /// carry the concept this reading was resolved under; that concept, not
    /// anything in the register bytes, is the registers' semantic identity.
    ///
    /// # Errors
    ///
    /// - [`ActivationDrift::NotRegister128`] when the slab declared another
    ///   reading (e.g. Facet96) or nothing: an undeclared slab is never
    ///   assumed to hold registers.
    /// - [`ActivationDrift::RegisterRailAbsent`] when the value schema does not
    ///   materialise a requested rail.
    pub fn bind_register128(
        &self,
        rails: crate::register128::RegisterRails,
    ) -> Result<crate::register128::RegisterLanes, ActivationDrift> {
        if self.slab != Some(SlabReading::Register128) {
            return Err(ActivationDrift::NotRegister128 {
                concept: self.concept,
                slab: self.slab,
            });
        }
        for &tenant in rails.tenants() {
            if !self.read_mode.value_schema.has(tenant) {
                return Err(ActivationDrift::RegisterRailAbsent {
                    concept: self.concept,
                    tenant: tenant as u8,
                });
            }
        }
        Ok(crate::register128::RegisterLanes::new(self.concept, rails))
    }

    /// Bind the register rails of this population as SIGNED values under
    /// one declared law. Checked ONCE per population, like
    /// [`bind_register128`](Self::bind_register128).
    ///
    /// The slab declares the carving (how the bytes were written:
    /// [`SlabReading::RegisterI4x32`] or [`SlabReading::RegisterI8x16`]).
    /// The caller declares the `law` (what the signed values mean for this
    /// concept: a relative offset, a position on an axis, or support). Both
    /// travel with the returned lanes, and every read and write checks the
    /// law it is asked for against the bound one, so a consumer of one law
    /// can never silently read another's values.
    ///
    /// # Errors
    ///
    /// - [`ActivationDrift::NotSignedRegister`] when the slab declared any
    ///   other reading, including the unsigned [`SlabReading::Register128`],
    ///   or nothing.
    /// - [`ActivationDrift::RegisterRailAbsent`] when the value schema does
    ///   not materialise a requested rail.
    pub fn bind_signed_register(
        &self,
        rails: crate::register128::RegisterRails,
        law: crate::register128::RegisterLaw,
    ) -> Result<crate::register128::SignedRegisterLanes, ActivationDrift> {
        let carving = match self.slab {
            Some(SlabReading::RegisterI4x32) => crate::register128::RegisterCarving::I4x32,
            Some(SlabReading::RegisterI8x16) => crate::register128::RegisterCarving::I8x16,
            other => {
                return Err(ActivationDrift::NotSignedRegister {
                    concept: self.concept,
                    slab: other,
                })
            }
        };
        for &tenant in rails.tenants() {
            if !self.read_mode.value_schema.has(tenant) {
                return Err(ActivationDrift::RegisterRailAbsent {
                    concept: self.concept,
                    tenant: tenant as u8,
                });
            }
        }
        Ok(crate::register128::SignedRegisterLanes::new(
            self.concept,
            rails,
            carving,
            law,
        ))
    }
}

/// Why an activation failed — each arm is one named bang.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActivationDrift {
    /// A hot-plugged classid is not minted in the authority's codebook.
    UnknownClassid(u16),
    /// The consumer is not an expected executor for a resolved table.
    UnexpectedConsumer(String),
    /// The authority declares a capability on a hot-plugged id that the
    /// consumer does not cover.
    Uncovered(String),
    /// The consumer claims a capability the authority does not declare on
    /// its hot-plugged ids.
    Undeclared(String),
    /// A hot-plugged classid resolves to no declared capability at all —
    /// plugging it is either premature or the table was forgotten.
    NoCapabilitiesFor(u16),
    /// [`Activation::read_mode_for`] was asked for a concept the authority
    /// declared no reading for.
    ///
    /// The named bang that replaces a silent [`ReadMode::DEFAULT`] (V1)
    /// fallback. A consumer minting keys under a hot-plugged seat reaches
    /// this instead of quietly producing legacy-tailed rows.
    ///
    /// [`ReadMode::DEFAULT`]: crate::canonical_node::ReadMode::DEFAULT
    NoReadingFor(u16),
    /// A slab's metadata envelope carries a reading tag this build does not
    /// implement. Never assumed to be [`SlabReading::Facet96`].
    UnknownSlabReading(u8),
    /// [`ResolvedReading::bind_register128`] was asked for register rails of
    /// a population whose slab did not declare [`SlabReading::Register128`].
    NotRegister128 {
        /// The concept the reading was resolved under.
        concept: u16,
        /// What the slab declared instead (`None`: nothing).
        slab: Option<SlabReading>,
    },
    /// [`ResolvedReading::bind_signed_register`] was asked for signed
    /// register rails of a population whose slab did not declare a signed
    /// carving ([`SlabReading::RegisterI4x32`] or
    /// [`SlabReading::RegisterI8x16`]).
    NotSignedRegister {
        /// The concept the reading was resolved under.
        concept: u16,
        /// What the slab declared instead (`None`: nothing).
        slab: Option<SlabReading>,
    },
    /// The resolved value schema does not materialise a requested register
    /// rail.
    RegisterRailAbsent {
        /// The concept the reading was resolved under.
        concept: u16,
        /// The missing [`ValueTenant`](crate::canonical_node::ValueTenant)
        /// discriminant.
        tenant: u8,
    },
    /// A slab was written under an envelope layout version this build does
    /// not implement.
    SlabLayoutVersion {
        /// The version the slab declared.
        slab: u8,
        /// [`crate::soa_envelope::ENVELOPE_LAYOUT_VERSION`].
        expected: u8,
    },
    /// The authority resolved a concept that this crate's zero-dep wire
    /// mirror ([`crate::ogar_codebook`]) does not carry at the same id.
    ///
    /// This is the drift the retired compile-time `COUNT_FUSE` existed to
    /// catch — now reported **per plug**, for the ids a consumer actually
    /// uses, instead of as a global equality assert that failed every build.
    /// The authority is the only place both sides are in scope: a consumer
    /// holding OGAR can see the mirror, and a mirror-only consumer cannot
    /// call [`CapabilityAuthority::activate`] at all.
    MirrorDrift {
        /// Concept name as the authority resolved it.
        concept: String,
        /// The id the authority is authoritative for.
        authority_id: u16,
        /// What the mirror said — `None` when the concept is missing entirely.
        mirror_id: Option<u16>,
    },
}

/// Cross-check an authority's resolved concepts against this crate's zero-dep
/// wire mirror, so a stale mirror is caught at the plug rather than silently
/// mis-resolving in a consumer that reads the mirror instead of the authority.
///
/// Split from the mirror lookup so the checker itself is testable against a
/// deliberately-wrong table — with the real mirror there is (by construction)
/// no concept that disagrees, so a test using it could only ever assert the
/// happy path.
#[must_use]
pub fn mirror_disagreement<F>(
    concepts: &[(String, u16)],
    mirror_lookup: F,
) -> Option<ActivationDrift>
where
    F: Fn(&str) -> Option<u16>,
{
    concepts.iter().find_map(|(concept, authority_id)| {
        let mirror_id = mirror_lookup(concept);
        (mirror_id != Some(*authority_id)).then(|| ActivationDrift::MirrorDrift {
            concept: concept.clone(),
            authority_id: *authority_id,
            mirror_id,
        })
    })
}

/// [`mirror_disagreement`] against the real [`crate::ogar_codebook`] mirror.
#[must_use]
pub fn verify_against_mirror(concepts: &[(String, u16)]) -> Option<ActivationDrift> {
    mirror_disagreement(concepts, crate::ogar_codebook::canonical_concept_id)
}

impl core::fmt::Display for ActivationDrift {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnknownClassid(id) => write!(f, "hot-plugged classid 0x{id:04X} is not minted"),
            Self::UnexpectedConsumer(c) => write!(f, "consumer `{c}` is not an expected executor"),
            Self::Uncovered(cap) => write!(f, "declared capability `{cap}` has no consumer arm"),
            Self::Undeclared(cap) => write!(f, "consumer covers `{cap}` which is not declared"),
            Self::NoCapabilitiesFor(id) => {
                write!(f, "classid 0x{id:04X} resolves to no declared capability")
            }
            Self::NoReadingFor(id) => write!(
                f,
                "no storage reading declared for concept 0x{id:04X} \
                 (a V1 default is never substituted)"
            ),
            Self::UnknownSlabReading(tag) => write!(
                f,
                "slab declares reading tag {tag}, which this build does not implement"
            ),
            Self::NotRegister128 { concept, slab } => write!(
                f,
                "concept 0x{concept:04X}: slab declares {slab:?}, not Register128; \
                 no register rails are granted"
            ),
            Self::NotSignedRegister { concept, slab } => write!(
                f,
                "concept 0x{concept:04X}: slab declares {slab:?}, not a signed register \
                 carving; no signed register rails are granted"
            ),
            Self::RegisterRailAbsent { concept, tenant } => write!(
                f,
                "concept 0x{concept:04X}: value schema does not materialise register tenant {tenant}"
            ),
            Self::SlabLayoutVersion { slab, expected } => write!(
                f,
                "slab written under envelope layout v{slab}, this build reads v{expected}"
            ),
            Self::MirrorDrift {
                concept,
                authority_id,
                mirror_id,
            } => match mirror_id {
                Some(m) => write!(
                    f,
                    "wire mirror has `{concept}`=0x{m:04X} but the authority says 0x{authority_id:04X}"
                ),
                None => write!(
                    f,
                    "wire mirror is missing `{concept}` (authority: 0x{authority_id:04X})"
                ),
            },
        }
    }
}

impl std::error::Error for ActivationDrift {}

/// Implemented by the authority (OGAR side, same binary): resolve a
/// [`HotPlug`] to its [`Activation`] or the first [`ActivationDrift`].
pub trait CapabilityAuthority {
    /// Verify the plug and hand back the vocab + capability surface for
    /// exactly the hot-plugged classids.
    fn activate(&self, plug: &HotPlug) -> Result<Activation, ActivationDrift>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TinyAuthority;
    impl CapabilityAuthority for TinyAuthority {
        fn activate(&self, plug: &HotPlug) -> Result<Activation, ActivationDrift> {
            if plug.classids.contains(&0xDEAD) {
                return Err(ActivationDrift::UnknownClassid(0xDEAD));
            }
            Ok(Activation::new(
                plug.classids
                    .iter()
                    .map(|&id| (format!("c{id:04x}"), id))
                    .collect(),
                plug.covered.iter().map(|s| (*s).to_string()).collect(),
                // This toy authority declares no reading — an authority that
                // has none says so, and asking it for one bangs.
                Vec::new(),
            ))
        }
    }

    #[test]
    fn socket_shape_round_trips_through_a_trait_object() {
        let plug = HotPlug {
            consumer: "demo",
            classids: &[0x0805],
            covered: &["recognize_line"],
        };
        let auth: &dyn CapabilityAuthority = &TinyAuthority;
        let act = auth.activate(&plug).unwrap();
        assert_eq!(act.concepts, vec![("c0805".to_string(), 0x0805)]);
        assert_eq!(act.capabilities, vec!["recognize_line".to_string()]);
        assert!(matches!(
            auth.activate(&HotPlug {
                classids: &[0xDEAD],
                ..plug
            }),
            Err(ActivationDrift::UnknownClassid(0xDEAD))
        ));
    }

    /// Asking for an absent reading BANGS — it never yields V1.
    ///
    /// Operator, 2026-09-07: *"don't silently enforce V1 fallback in hotplug,
    /// that's unacceptable."* The shape being removed is
    /// `act.read_modes.iter().find(…).map(…).unwrap_or(ReadMode::DEFAULT)` —
    /// a one-liner that compiles, reads as careful, and mints legacy-tailed
    /// keys forever. It is no longer expressible: the field is private and
    /// the lookup returns a `Result`.
    ///
    /// Two-sided on the same activation, which is what makes it a test rather
    /// than a restatement of the code: the DECLARED concept resolves, and an
    /// undeclared one bangs. A lookup that answered for everything, or for
    /// nothing, would fail one half.
    #[test]
    fn an_undeclared_reading_bangs_instead_of_defaulting_to_v1() {
        use crate::canonical_node::{EdgeCodecFlavor, ReadMode, TailVariant, ValueSchema};

        const V3_SEAT: ReadMode = ReadMode {
            tail_variant: TailVariant::V3,
            value_schema: ValueSchema::Bootstrap,
            edge_codec: EdgeCodecFlavor::CoarseOnly,
        };
        let act = Activation::new(Vec::new(), Vec::new(), vec![(0x1717, V3_SEAT)]);

        // Can-fire on the happy half: a declared seat resolves to its own
        // reading, not to some other row of the table.
        assert_eq!(act.read_mode_for(0x1717), Ok(V3_SEAT));
        assert_eq!(
            act.read_mode_for(0x1717).unwrap().tail_variant,
            TailVariant::V3
        );

        // …and an undeclared one is a NAMED bang. The anti-vacuity assertion
        // is the second one: the error must not be some value that happens to
        // compare unequal — there must be no ReadMode on this path at all.
        assert_eq!(
            act.read_mode_for(0x1718),
            Err(ActivationDrift::NoReadingFor(0x1718))
        );
        assert!(act.read_mode_for(0x1718).is_err());

        // The V1 fallback this replaces would have returned DEFAULT here.
        // Pinned so the difference is measured, not asserted in prose.
        assert_eq!(ReadMode::DEFAULT.tail_variant, TailVariant::V1);
        assert_ne!(Ok(ReadMode::DEFAULT), act.read_mode_for(0x1718));
    }

    /// An authority with NO readings still activates — and still bangs.
    ///
    /// The silence twin at the other end: a capability-only consumer (an
    /// executor that mints no keys) legitimately declares an empty table, so
    /// an empty table must not be an activation failure. What must never
    /// happen is that consumer's lookup quietly succeeding with V1.
    #[test]
    fn an_empty_table_activates_but_still_refuses_to_invent_a_reading() {
        let plug = HotPlug {
            consumer: "demo",
            classids: &[0x0805],
            covered: &["recognize_line"],
        };
        let act = TinyAuthority.activate(&plug).expect("activation is green");

        assert!(act.declared_readings().is_empty(), "no reading declared");
        assert_eq!(
            act.read_mode_for(0x0805),
            Err(ActivationDrift::NoReadingFor(0x0805)),
            "an empty table is not 'assume the default'"
        );
    }

    /// SPOG × slab resolution. Fixture: two plugged concepts with different
    /// current readings — 0x0901 (plug-and-play V3 / Full) and 0x0902 (a V1 /
    /// Cognitive / CoarseResidue class).
    mod slab_resolution {
        use super::super::*;
        use crate::canonical_node::{
            EdgeCodecFlavor, NodeGuid, NodeRow, ReadMode, TailVariant, ValueSchema,
        };
        use crate::soa_envelope::ENVELOPE_LAYOUT_VERSION;

        const A: ReadMode = ReadMode::PLUG_AND_PLAY_V3;
        const B: ReadMode = ReadMode {
            tail_variant: TailVariant::V1,
            value_schema: ValueSchema::Cognitive,
            edge_codec: EdgeCodecFlavor::CoarseResidue,
        };

        /// An activation that plugs 0x0901 (reading `A`) and 0x0902 (reading `B`).
        fn act() -> Activation {
            Activation::new(Vec::new(), Vec::new(), vec![(0x0901, A), (0x0902, B)])
        }

        /// A row whose SPOG graph (canon-high half of the classid) is `concept`.
        fn key(concept: u16, identity: u32) -> NodeGuid {
            NodeGuid::new(u32::from(concept) << 16, 1, 2, 3, 0x66, identity)
        }

        /// A Facet96 declaration at the current layout with the given value schema.
        fn decl(value_schema: ValueSchema) -> SlabDeclaration {
            SlabDeclaration {
                reading: SlabReading::Facet96,
                value_schema,
                layout_version: ENVELOPE_LAYOUT_VERSION,
            }
        }

        /// Same declaration + same SPOG context → same reading, and the key
        /// wrapper agrees with the context form.
        #[test]
        fn resolution_is_deterministic() {
            let d = decl(ValueSchema::Bootstrap);
            let first = act().resolve_for_context(0x0901, Some(&d));
            assert!(first.is_ok());
            for i in 1..9 {
                assert_eq!(act().resolve_for_context(0x0901, Some(&d)), first);
                assert_eq!(
                    act().resolve_tenant_reading(key(0x0901, i), Some(&d)),
                    first
                );
            }
        }

        /// One declaration under two registered SPOG contexts resolves
        /// differently: tail and edge codec come from each context's OGAR
        /// reading; the slab's own part is shared.
        #[test]
        fn one_slab_under_two_contexts_resolves_per_context() {
            let d = decl(ValueSchema::Bootstrap);
            let ra = act().resolve_for_context(0x0901, Some(&d)).unwrap();
            let rb = act().resolve_for_context(0x0902, Some(&d)).unwrap();
            assert_eq!(ra.read_mode.tail_variant, TailVariant::V3);
            assert_eq!(rb.read_mode.tail_variant, TailVariant::V1);
            assert_eq!(rb.read_mode.edge_codec, EdgeCodecFlavor::CoarseResidue);
            assert_ne!(ra, rb, "anti-vacuity: the context changed the answer");
            assert_eq!(ra.read_mode.value_schema, rb.read_mode.value_schema);
            assert_eq!(ra.slab, rb.slab);
            // The key wrapper derives the context from the key (graph_of),
            // so a 0x0902 row resolves as 0x0902, not as any fixed concept.
            assert_eq!(
                act().resolve_tenant_reading(key(0x0902, 3), Some(&d)),
                Ok(rb)
            );
            assert_eq!(
                act().resolve_tenant_reading(key(0x0901, 3), Some(&d)),
                Ok(ra)
            );
        }

        /// An unknown SPOG/OGAR context fails closed, with or without a
        /// declaration, including the default class 0.
        #[test]
        fn an_unknown_context_fails_closed() {
            let d = decl(ValueSchema::Bootstrap);
            for concept in [0x0903u16, 0x0000] {
                assert_eq!(
                    act().resolve_for_context(concept, None),
                    Err(ActivationDrift::NoReadingFor(concept))
                );
                assert_eq!(
                    act().resolve_for_context(concept, Some(&d)),
                    Err(ActivationDrift::NoReadingFor(concept))
                );
            }
        }

        /// A reading or layout this build does not implement fails closed;
        /// the implemented ones pass (silence twin).
        #[test]
        fn an_unsupported_physical_reading_fails_closed() {
            assert_eq!(SlabReading::from_tag(0), Ok(SlabReading::Facet96));
            assert_eq!(SlabReading::from_tag(1), Ok(SlabReading::Register128));
            assert_eq!(SlabReading::from_tag(2), Ok(SlabReading::RegisterI4x32));
            assert_eq!(SlabReading::from_tag(3), Ok(SlabReading::RegisterI8x16));
            for tag in [4u8, 5, 0x80, 0xFF] {
                assert_eq!(
                    SlabReading::from_tag(tag),
                    Err(ActivationDrift::UnknownSlabReading(tag))
                );
            }
            let stale = SlabDeclaration {
                layout_version: ENVELOPE_LAYOUT_VERSION.wrapping_sub(1),
                ..decl(ValueSchema::Full)
            };
            assert_eq!(
                act().resolve_for_context(0x0901, Some(&stale)),
                Err(ActivationDrift::SlabLayoutVersion {
                    slab: ENVELOPE_LAYOUT_VERSION.wrapping_sub(1),
                    expected: ENVELOPE_LAYOUT_VERSION,
                })
            );
            assert!(act()
                .resolve_for_context(0x0901, Some(&decl(ValueSchema::Full)))
                .is_ok());
        }

        /// No declaration: today's inherited behaviour — exactly the current
        /// OGAR reading, and NO claim about the physical reading (`slab` is
        /// `None`, not `Facet96`). No migration and no layout change.
        #[test]
        fn an_undeclared_slab_keeps_inherited_behaviour() {
            let a = act();
            for (concept, mode) in [(0x0901u16, A), (0x0902, B)] {
                let r = a.resolve_for_context(concept, None).unwrap();
                assert_eq!(Ok(r.read_mode), a.read_mode_for(concept));
                assert_eq!(r.read_mode, mode);
                assert_eq!(r.slab, None, "absence is not a Facet96 claim");
            }
            assert_eq!(core::mem::size_of::<crate::facet::FacetCascade>(), 16);
        }

        /// The class migrated; its historical slabs still read as written.
        ///
        /// Yesterday the class was registered Full; a slab was written Full and
        /// recorded that. Today OGAR registers Bootstrap. The old slab resolves
        /// to Full, a new undeclared one to Bootstrap. Both directions, so the
        /// rule is "the declaration wins", not "the wider one wins".
        #[test]
        fn a_historical_slab_survives_a_class_migration() {
            let migrated = |schema| {
                Activation::new(
                    Vec::new(),
                    Vec::new(),
                    vec![(
                        0x0901,
                        ReadMode {
                            value_schema: schema,
                            ..A
                        },
                    )],
                )
            };
            for (today, written) in [
                (ValueSchema::Bootstrap, ValueSchema::Full),
                (ValueSchema::Full, ValueSchema::Compressed),
            ] {
                let act = migrated(today);
                let old = act
                    .resolve_for_context(0x0901, Some(&decl(written)))
                    .unwrap();
                assert_eq!(
                    old.read_mode.value_schema, written,
                    "the slab's own record wins"
                );
                assert_eq!(old.slab, Some(SlabReading::Facet96));
                let new = act.resolve_for_context(0x0901, None).unwrap();
                assert_eq!(
                    new.read_mode.value_schema, today,
                    "new data reads as registered today"
                );
                assert_ne!(old, new);
            }
        }

        /// Payload does not participate: two rows with the same key and
        /// different value bytes resolve identically, and no resolution method
        /// accepts a row or payload at all.
        #[test]
        fn payload_bytes_do_not_participate() {
            let mut r1 = NodeRow {
                key: key(0x0901, 7),
                edges: crate::canonical_node::EdgeBlock::default(),
                value: [0u8; 480],
            };
            let mut r2 = r1;
            r1.value[0] = 0xAA;
            r2.value[479] = 0x55;
            let d = decl(ValueSchema::Cognitive);
            assert_ne!(r1.value, r2.value);
            assert_eq!(
                act().resolve_tenant_reading(r1.key, Some(&d)),
                act().resolve_tenant_reading(r2.key, Some(&d))
            );
        }

        /// The intended call shape: resolve ONCE for a population whose SPOG
        /// context is already known, then process the rows with no SPOG
        /// resolution, registry lookup or map lookup inside the loop.
        ///
        /// This shows the API supports that shape. It does not measure a fold,
        /// and it does not show anything is branch-free: no kernel exists yet.
        #[test]
        fn one_resolution_serves_a_whole_population() {
            /// Compiles only for types usable as a cache key or value.
            fn cacheable<T: Copy + Eq + core::hash::Hash>() {}
            cacheable::<ResolvedReading>();
            cacheable::<SlabDeclaration>();

            let population: Vec<NodeGuid> = (1..=1000).map(|i| key(0x0901, i)).collect();
            let d = decl(ValueSchema::Compressed);

            let resolved = act().resolve_for_context(0x0901, Some(&d)).unwrap();
            // Conceptual stand-in for `future_dispatch(resolved)`: a value
            // chosen once from the resolved reading, then reused.
            let tenant_bytes = resolved.read_mode.value_schema.tenant_bytes();

            let mut visited = 0usize;
            for row in &population {
                // Only the precomputed value is used here.
                let _ = (row, tenant_bytes);
                visited += 1;
            }
            assert_eq!(visited, population.len());
            assert_eq!(tenant_bytes, ValueSchema::Compressed.tenant_bytes());
            // The population really is that context (checked outside the loop).
            assert!(population
                .iter()
                .all(|k| crate::spog_tenants::graph_of(*k) == resolved.concept));
        }

        /// A Register128 declaration at the current layout.
        fn reg_decl(value_schema: ValueSchema) -> SlabDeclaration {
            SlabDeclaration {
                reading: SlabReading::Register128,
                value_schema,
                layout_version: ENVELOPE_LAYOUT_VERSION,
            }
        }

        /// FAILS IF: the register rails are granted without a Register128
        /// declaration (Facet96, or no declaration at all), or for a schema
        /// that does not materialise them. Silence twin: Register128 + Full
        /// grants both rails.
        #[test]
        fn register_rails_are_granted_only_to_a_register128_slab() {
            use crate::canonical_node::ValueTenant;
            use crate::register128::RegisterRails;
            let a = act();
            let granted = a
                .resolve_for_context(0x0901, Some(&reg_decl(ValueSchema::Full)))
                .unwrap();
            assert_eq!(granted.slab, Some(SlabReading::Register128));
            let lanes = granted.bind_register128(RegisterRails::Two).unwrap();
            assert_eq!(lanes.rails(), RegisterRails::Two);
            assert!(lanes.rail_range(1).is_some());

            let facet = a
                .resolve_for_context(0x0901, Some(&decl(ValueSchema::Full)))
                .unwrap();
            let undeclared = a.resolve_for_context(0x0901, None).unwrap();
            for (r, slab) in [(facet, Some(SlabReading::Facet96)), (undeclared, None)] {
                assert_eq!(
                    r.bind_register128(RegisterRails::One),
                    Err(ActivationDrift::NotRegister128 {
                        concept: 0x0901,
                        slab
                    })
                );
            }

            let narrow = a
                .resolve_for_context(0x0901, Some(&reg_decl(ValueSchema::Cognitive)))
                .unwrap();
            assert_eq!(
                narrow.bind_register128(RegisterRails::One),
                Err(ActivationDrift::RegisterRailAbsent {
                    concept: 0x0901,
                    tenant: ValueTenant::Register0 as u8,
                })
            );
        }

        /// FAILS IF: signed rails are granted to a slab without a signed
        /// carving (including the unsigned Register128 slab), the carving is
        /// taken from anywhere but the slab, or a rail the schema lacks is
        /// granted.
        #[test]
        fn signed_rails_are_granted_only_to_a_signed_carving() {
            use crate::canonical_node::ValueTenant;
            use crate::register128::{RegisterCarving, RegisterLaw, RegisterRails};
            let a = act();
            let carved = |reading, value_schema| SlabDeclaration {
                reading,
                value_schema,
                layout_version: ENVELOPE_LAYOUT_VERSION,
            };
            for (reading, carving) in [
                (SlabReading::RegisterI4x32, RegisterCarving::I4x32),
                (SlabReading::RegisterI8x16, RegisterCarving::I8x16),
            ] {
                let r = a
                    .resolve_for_context(0x0901, Some(&carved(reading, ValueSchema::Full)))
                    .unwrap();
                let lanes = r
                    .bind_signed_register(RegisterRails::Two, RegisterLaw::AxisPosition)
                    .unwrap();
                assert_eq!(lanes.carving(), carving);
                assert_eq!(lanes.law(), RegisterLaw::AxisPosition);
                assert_eq!(lanes.concept(), 0x0901);
                assert_eq!(
                    r.bind_register128(RegisterRails::One),
                    Err(ActivationDrift::NotRegister128 {
                        concept: 0x0901,
                        slab: Some(reading)
                    }),
                    "a signed slab is not an unsigned word register"
                );
                let narrow = a
                    .resolve_for_context(0x0901, Some(&carved(reading, ValueSchema::Cognitive)))
                    .unwrap();
                assert_eq!(
                    narrow.bind_signed_register(RegisterRails::One, RegisterLaw::Support),
                    Err(ActivationDrift::RegisterRailAbsent {
                        concept: 0x0901,
                        tenant: ValueTenant::Register0 as u8,
                    })
                );
            }
            let words = a
                .resolve_for_context(0x0901, Some(&reg_decl(ValueSchema::Full)))
                .unwrap();
            let facet = a
                .resolve_for_context(0x0901, Some(&decl(ValueSchema::Full)))
                .unwrap();
            let undeclared = a.resolve_for_context(0x0901, None).unwrap();
            for (r, slab) in [
                (words, Some(SlabReading::Register128)),
                (facet, Some(SlabReading::Facet96)),
                (undeclared, None),
            ] {
                assert_eq!(
                    r.bind_signed_register(RegisterRails::One, RegisterLaw::RelativeOffset),
                    Err(ActivationDrift::NotSignedRegister {
                        concept: 0x0901,
                        slab
                    })
                );
            }
        }

        /// FAILS IF: a register's semantic identity is read from its bytes.
        ///
        /// The same register payload — here deliberately holding the OTHER
        /// concept's id in its first word, the shape a Facet96 classid would
        /// take — is bound under two contexts. The concept follows the
        /// context both times, and the bytes come back unchanged: nothing in
        /// the binding or the resolution reads the payload.
        #[test]
        fn a_register_takes_its_concept_from_the_context_never_the_payload() {
            use crate::register128::{Register128, RegisterRails};
            let a = act();
            let d = reg_decl(ValueSchema::Full);
            let payload = Register128::from_words([0x0902_0000, 7, 8, 9]);
            // Rail 1, never rail 0: `register128::tests::register_writes_are_counted_per_tenant`
            // pins an exact Register0 count under `tenant-counters`, and tests run in parallel.
            for (concept, other) in [(0x0901u16, 0x0902u16), (0x0902, 0x0901)] {
                let mut row = NodeRow {
                    key: key(concept, 1),
                    edges: Default::default(),
                    value: [0; 480],
                };
                let lanes = a
                    .resolve_tenant_reading(row.key, Some(&d))
                    .unwrap()
                    .bind_register128(RegisterRails::Two)
                    .unwrap();
                assert!(lanes.set(&mut row, 1, payload));
                assert_eq!(lanes.concept(), concept);
                assert_ne!(lanes.concept(), other);
                assert_eq!(lanes.get(&row, 1), Some(payload));
                // Resolving again after the write is unchanged.
                let again = a
                    .resolve_tenant_reading(row.key, Some(&d))
                    .unwrap()
                    .bind_register128(RegisterRails::Two)
                    .unwrap();
                assert_eq!(again, lanes);
            }
        }
    }

    /// The drift class the retired `COUNT_FUSE` guarded: a concept the
    /// AUTHORITY knows that the MIRROR does not carry at the same id.
    ///
    /// Codex caught that hot-plug alone could not see this — `activate`
    /// consults only OGAR, so a stale mirror activated green while
    /// mirror-reading consumers resolved `None`. The checker closes that,
    /// and it is tested against a deliberately-wrong table because the real
    /// mirror is (by construction) never wrong — a test using it could only
    /// assert the happy path and would pass with the checker deleted.
    #[test]
    fn a_concept_missing_from_the_mirror_is_named_drift_not_silence() {
        let resolved = vec![("textline".to_string(), 0x0805u16)];

        // Missing entirely — the add-a-concept-without-mirroring-it case,
        // i.e. exactly what happened to osm_street_node on 2026-08-14.
        assert_eq!(
            mirror_disagreement(&resolved, |_| None),
            Some(ActivationDrift::MirrorDrift {
                concept: "textline".to_string(),
                authority_id: 0x0805,
                mirror_id: None,
            })
        );

        // Present at the WRONG id — the case a length-equality fuse could
        // never have caught at all, since the counts still match.
        assert_eq!(
            mirror_disagreement(&resolved, |_| Some(0x0806)),
            Some(ActivationDrift::MirrorDrift {
                concept: "textline".to_string(),
                authority_id: 0x0805,
                mirror_id: Some(0x0806),
            })
        );

        // Agreement is silent — without this the checker could "pass" by
        // objecting to everything, which carries no information.
        assert_eq!(mirror_disagreement(&resolved, |_| Some(0x0805)), None);
    }
}
