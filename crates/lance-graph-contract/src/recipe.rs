//! Composition layer: thinking-style recipes and persona recipes.
//!
//! This module sits above the atom layer (`contract::atoms`, D-ATOM-1) and
//! below the dispatch layer (`contract::jit`).  It implements the three-layer
//! hierarchy described in `EPIPHANIES.md` E-LADDER-SERVES-MAILBOX §2:
//!
//! ```text
//! atoms (I4-32D bipolar)  ─── composition ───►  StyleRecipe
//!                                                      │ composition
//!                                                      ▼
//!                                               PersonaRecipe  (+ β, thresholds, purpose)
//!                                                      │ compile
//!                                                      ▼
//!                                               KernelHandle  (Cranelift fused kernel)
//! ```
//!
//! **JIT placement (the key design decision):** the JIT target is the
//! *recipe*, not the per-atom dot.  A single 32-D i4 dot product is one SIMD
//! sequence — Cranelift compile overhead is on the order of microseconds and
//! only amortises when many fused operations are compiled together.  At the
//! recipe level the compiler can fuse the weighted-atom summation, the
//! threshold comparisons, the β-scaled sampling, and any style-specific
//! post-processing into a single native code path, hiding the per-recipe
//! overhead across millions of invocations.  Compiling the atom-dot alone
//! would pay the Cranelift overhead for each individual dot product — a net
//! loss.
//!
//! **Elixir-style open/closed split (hot-load protocol):**
//! - `add-atom` = *data* change (the atom catalogue, D-ATOM-1).  No
//!   recompilation needed: a new atom is a new row in the basis register.
//! - `add-style` / `add-persona` = *template* change.  A new
//!   [`RecipeTemplate`] must be registered via [`register_recipe`] so the
//!   [`jit::StyleRegistry`] can produce a new [`jit::KernelHandle`] at next
//!   activation.  This keeps the hot path closed to structural mutation while
//!   remaining open to new personas/styles at the template boundary.

// BLOCKED: `atoms::Atom` and `atoms::I4x32` — the concrete I4-32D atom
// carrier type and the `Atom` enum/catalogue are being scaffolded in
// parallel as D-ATOM-1 (`contract::atoms`).  All references below use
// the *intended names* (`atoms::Atom`, `atoms::I4x32`) as forward
// declarations; they will become real imports once D-ATOM-1 lands.

use crate::jit::{JitError, KernelHandle};

// ---------------------------------------------------------------------------
// Forward stubs for D-ATOM-1 types (BLOCKED)
// ---------------------------------------------------------------------------

/// Placeholder for the bipolar I4-32D atom vector type (`atoms::I4x32`).
///
/// BLOCKED: the real type is defined in D-ATOM-1 (`contract::atoms`).
/// Replace this stub with `use crate::atoms::I4x32;` once that module
/// lands.  All arithmetic on this type (dot product, weighted sum,
/// clamped i4 accumulation) lives in `contract::atoms`, not here.
pub type I4x32Stub = [i8; 32];

/// Placeholder for a single atom identifier from the basis catalogue
/// (`atoms::Atom`).
///
/// BLOCKED: the real type is defined in D-ATOM-1.
/// Replace with `use crate::atoms::Atom;`.
pub type AtomStub = u8;

// ---------------------------------------------------------------------------
// StyleRecipe
// ---------------------------------------------------------------------------

/// A *thinking style* as an I4-32D **composition** over atoms.
///
/// A `StyleRecipe` is **not** an atomic fingerprint.  Per
/// `EPIPHANIES.md` E-LADDER-SERVES-MAILBOX §2, styles such as
/// "Kant", "Schopenhauer", or "bookkeeping-savant" are **weighted
/// combinations** of the orthogonal bipolar atoms from the I4-32D
/// basis (`contract::atoms::Atom`).  The style itself carries no
/// intrinsic identity — its semantics are entirely determined by how
/// it weights the underlying atom poles.
///
/// # Composition semantics
///
/// Each entry in `weights` pairs an atom with a signed i4 weight:
/// positive weight activates the `+` pole, negative weight activates
/// the `−` pole, zero weight leaves the atom inert in this style.
/// The resulting composition vector is the clamped i4 accumulation
/// `Σ weight_k × atom_k` in the 32-dimensional bipolar space.
///
/// # Invariants (from I-VSA-IDENTITIES)
///
/// - `weights` must not bundle more than `√32 / 4 ≈ 1` coherent
///   super-imposed styles at a time (VSA capacity limit at 32 dims).
/// - Each `AtomStub` index must reference a valid atom in the basis
///   catalogue (D-ATOM-1); unrecognised indices are BLOCKED on
///   D-ATOM-1.
/// - This type is `Copy`-free: cloning is explicit to prevent
///   accidental duplication in the hot loop.
///
/// # Relationship to the atom layer
///
/// BLOCKED on D-ATOM-1: once `atoms::I4x32` is defined, the
/// `composition_vector` field should store the *pre-computed*
/// weighted-sum vector directly (a materialised `atoms::I4x32`)
/// rather than the sparse `weights` list, so recipe evaluation
/// reduces to a single dot product.
#[derive(Debug, Clone)]
pub struct StyleRecipe {
    /// Human-readable style name (e.g. `"Kant"`, `"bookkeeping-savant"`).
    pub name: &'static str,

    /// Sparse atom weights: `(atom_id, i4_weight)` pairs.
    ///
    /// BLOCKED on D-ATOM-1: `AtomStub` → `atoms::Atom` once available.
    /// Weights should fit within the i4 range `[−8, +7]`; values outside
    /// this range will be clamped during compilation.
    pub weights: &'static [(AtomStub, i8)],

    /// Pre-computed I4-32D composition vector.
    ///
    /// BLOCKED on D-ATOM-1: `I4x32Stub` → `atoms::I4x32`.
    /// Populated at registration time by folding `weights` into the
    /// basis; `None` until the atom layer is live.
    pub composition: Option<I4x32Stub>,
}

// ---------------------------------------------------------------------------
// PersonaRecipe
// ---------------------------------------------------------------------------

/// β (explore/exploit temperature knob) for a persona.
///
/// Controls the sampling temperature of the underlying
/// wisdom↔Staunen axis (per E-LADDER-SERVES-MAILBOX §4).  Free
/// energy self-regulates around this setpoint at runtime; `beta` is
/// the *initial* / *nominal* setpoint, not a hard ceiling.
///
/// # Calibration guidance
///
/// | Persona type | Nominal β | Rationale |
/// |---|---|---|
/// | `business` | `Beta::Cold` | Exploit: checkboxes, GoBD rules, bounded fan-out |
/// | `chat` | `Beta::Warm` | Explore: episodic modeling, witness-arc self-state |
/// | `osint` | `Beta::Annealing { start, floor }` | Hot→cold: hypothesis mailboxes, untrusted-source gates |
///
/// The `WisdomMarker` 0.1 floor (D-PERSONA-1) translates to a
/// minimum temperature that prevents annealing to absolute zero
/// (φ-1 humility invariant).
#[derive(Debug, Clone, Copy)]
pub enum Beta {
    /// Cold, exploitation-biased temperature.  Suitable for
    /// rule-governed, high-stakes business personas.
    Cold,

    /// Warm, exploration-biased temperature.  Suitable for
    /// conversational or creative personas.
    Warm,

    /// Simulated-annealing schedule: begin at `start` and cool toward
    /// `floor`.  The floor must be ≥ 0.1 (WisdomMarker minimum).
    /// Suitable for OSINT / hypothesis-generation personas.
    Annealing {
        /// Initial (hot) temperature, in `(0.0, 1.0]`.
        start: f32,
        /// Minimum temperature floor, ≥ 0.1.
        floor: f32,
    },
}

/// A *persona* as a composition of [`StyleRecipe`]s, plus thresholds,
/// purpose metadata, and the explore/exploit temperature knob (`β`).
///
/// A `PersonaRecipe` is **not** a container (per I-VSA-IDENTITIES: a
/// persona is a Layer-2 dispatch policy over one substrate, not a new
/// structural layer).  It answers: *given the current mailbox state,
/// which styles activate, at what threshold, and with what sampling
/// temperature?*
///
/// # Composition semantics
///
/// Each entry in `styles` pairs a [`StyleRecipe`] reference with a
/// normalised weight in `[0.0, 1.0]`.  At recipe-compile time the
/// weighted-style blend is fused into a single [`jit::KernelHandle`]
/// via [`RecipeTemplate::compile`].
///
/// # Threshold semantics
///
/// - `commit_threshold` — below this free-energy value F, the mailbox
///   commits without escalation (maps to the `F < 0.2` Commit gate in
///   The Click).
/// - `escalate_threshold` — above this value, a `FailureTicket` is
///   raised (maps to `F > 0.8`).
///
/// # Purpose
///
/// A short human-readable string describing the persona's intended
/// domain (e.g. `"DACH bookkeeping, GoBD-compliant"`,
/// `"open-source intelligence synthesis"`).  Stored for introspection;
/// not used at runtime.
///
/// # Relationship to the mailbox
///
/// The persona decides *what to fan out as mailboxes* and *where β
/// sits*; it does NOT own the mailbox (sea-star topology, E-BATON-1).
/// Three canonical personas are three β-policies over one substrate:
/// `business` (cold), `chat` (warm), `osint` (annealing).
#[derive(Debug, Clone)]
pub struct PersonaRecipe {
    /// Human-readable persona name (e.g. `"business"`, `"osint"`).
    pub name: &'static str,

    /// Weighted blend of [`StyleRecipe`]s.
    ///
    /// `(style_recipe, normalised_weight)` pairs.  Weights need not sum
    /// to 1.0 here; they are normalised at compile time inside
    /// [`RecipeTemplate::compile`].
    pub styles: &'static [(&'static StyleRecipe, f32)],

    /// Free-energy commit threshold (dimensionless, `[0.0, 1.0]`).
    ///
    /// Corresponds to the `F < commit_threshold` Commit gate in The Click.
    /// Default: `0.2`.
    pub commit_threshold: f32,

    /// Free-energy escalate threshold (dimensionless, `[0.0, 1.0]`).
    ///
    /// Corresponds to the `F > escalate_threshold` FailureTicket gate.
    /// Default: `0.8`.
    pub escalate_threshold: f32,

    /// Explore/exploit temperature knob (β).
    pub beta: Beta,

    /// Short prose description of this persona's intended domain.
    pub purpose: &'static str,
}

// ---------------------------------------------------------------------------
// RecipeTemplate — the Cranelift / JIT hook
// ---------------------------------------------------------------------------

/// The Cranelift/JIT hook: a recipe compiled to a fused
/// [`jit::KernelHandle`].
///
/// # Why the recipe, not the atom-dot, is the JIT target
///
/// A single 32-D i4 dot product is one short SIMD sequence — on AVX-512
/// it fits in roughly 4 instructions.  Cranelift's compilation overhead
/// is measured in microseconds per function.  At the per-atom-dot
/// granularity the compile cost would *never* amortise: millions of
/// individual one-instruction-sequence compilations would be strictly
/// worse than a hand-written scalar fallback.
///
/// At the **fused-recipe level** the compiler can emit a single native
/// function that:
/// 1. Loads the 32-D i4 atom query vector (hot in a register after
///    `I4x32::from_mailbox_state`).
/// 2. Applies the style-weight blend as a series of fused multiply-add
///    SIMD lanes (VPDPBSSD / VPDPBUSD on AVX-VNNI, or vmlal on NEON).
/// 3. Applies the β-scaled threshold comparisons.
/// 4. Emits the `CollapseHint` variant inline (no branch to a separate
///    dispatch table).
///
/// This is the same amortisation argument behind JIT-compiling SQL
/// query plans rather than individual predicates: the fixed overhead is
/// paid once per template, not once per row.
///
/// # Registration and hot-load lifecycle
///
/// A `RecipeTemplate` is created once per persona and registered with
/// the [`jit::StyleRegistry`] via [`register_recipe`].  The registry
/// holds the [`jit::KernelHandle`] and serves it to the hot path on
/// every mailbox activation.  When the persona is updated (new styles
/// added, thresholds changed), a new `RecipeTemplate` is registered
/// under the same name — the registry evicts the old handle and
/// re-compiles.  This is the "add-style/persona = template" half of
/// the Elixir-style open/closed split.
///
/// # Relationship to `jit::StyleRegistry`
///
/// BLOCKED: `StyleRegistry::get_kernel` currently expects a
/// `ThinkingStyle` argument (the existing 36-enum surface in
/// `contract::thinking`).  To accept a `RecipeTemplate` the registry
/// must gain a `register_recipe` / `get_recipe_kernel` entry-point.
/// Until `StyleRegistry`'s API is extended this scaffolding leaves
/// the registration path as `todo!()`.  Do NOT guess an extension;
/// leave the BLOCKED marker.
#[derive(Debug, Clone)]
pub struct RecipeTemplate {
    /// The persona this template compiles.
    pub persona: &'static PersonaRecipe,

    /// Parameter hash used for [`jit::KernelHandle`] cache keying.
    ///
    /// Derived from the persona's style weights, thresholds, and β at
    /// registration time.  If the persona changes, this hash changes,
    /// causing the registry to evict the old kernel.
    ///
    /// BLOCKED on D-ATOM-1: the hash must incorporate the `I4x32`
    /// composition vectors of each constituent style; until those
    /// vectors exist, hash computation is `todo!()`.
    pub param_hash: u64,
}

impl RecipeTemplate {
    /// Compile this recipe template into a fused [`jit::KernelHandle`].
    ///
    /// Calls into the [`jit::StyleRegistry`] / [`jit::JitCompiler`]
    /// chain to emit a Cranelift-compiled native function that:
    ///
    /// 1. Accepts a 32-D i4 query vector (`atoms::I4x32`, D-ATOM-1).
    /// 2. Evaluates the weighted-style blend over that vector.
    /// 3. Returns a `CollapseHint` variant appropriate to the persona's
    ///    β and threshold configuration.
    ///
    /// # Errors
    ///
    /// Returns [`jit::JitError`] if Cranelift compilation fails or if
    /// the required target feature (AVX-512 / NEON) is unavailable on
    /// the current host.
    ///
    /// # BLOCKED
    ///
    /// - BLOCKED on D-ATOM-1: `atoms::I4x32` composition vectors are
    ///   required to materialise the fused kernel body.
    /// - BLOCKED on `StyleRegistry` API extension: the current
    ///   `StyleRegistry::get_kernel(&self, style: ThinkingStyle)` takes
    ///   an enum variant, not a recipe template.  A `register_recipe` /
    ///   `get_recipe_kernel` surface must be added before this can be
    ///   wired (see type-level doc).
    pub fn compile(&self) -> Result<KernelHandle, JitError> {
        // BLOCKED: D-ATOM-1 (I4x32 composition vectors) +
        // BLOCKED: StyleRegistry API extension (register_recipe entry-point)
        todo!(
            "RecipeTemplate::compile — blocked on D-ATOM-1 (atoms::I4x32) \
             and StyleRegistry::register_recipe API extension"
        )
    }

    /// Compute the `param_hash` for this recipe from its constituent
    /// style weights, thresholds, and β setpoint.
    ///
    /// BLOCKED on D-ATOM-1: the hash must cover the materialised
    /// `atoms::I4x32` composition vectors; until those exist the hash
    /// is derived from the sparse `weights` slices only (unstable
    /// across basis changes).
    pub fn compute_param_hash(&self) -> u64 {
        // BLOCKED: D-ATOM-1 (atoms::I4x32 composition vectors required
        // for a stable hash that is invariant to atom-catalogue changes)
        todo!("RecipeTemplate::compute_param_hash — blocked on D-ATOM-1")
    }
}

// ---------------------------------------------------------------------------
// register_recipe — hot-load entry point
// ---------------------------------------------------------------------------

/// Register a recipe template with the style registry (Elixir-style hot-load).
///
/// This is the **add-style / add-persona = template** half of the
/// open/closed split described in E-LADDER-SERVES-MAILBOX §2:
///
/// - **Add-atom = data:**  a new atom in the basis catalogue (D-ATOM-1)
///   requires no call to `register_recipe`; the atom is a new row in
///   the register and becomes available immediately to all existing
///   recipes on their *next* compilation.
/// - **Add-style / add-persona = template:** a new [`StyleRecipe`] or
///   [`PersonaRecipe`] *does* require calling `register_recipe` so the
///   [`jit::StyleRegistry`] can compile a new [`jit::KernelHandle`]
///   and serve it on the next mailbox activation.
///
/// # Hot-load lifecycle
///
/// 1. Caller constructs a [`RecipeTemplate`] from the new or updated
///    [`PersonaRecipe`].
/// 2. Calls `register_recipe(registry, template)`.
/// 3. The registry compiles the template via `template.compile()` and
///    stores the resulting `KernelHandle` keyed by `template.param_hash`.
/// 4. Existing handles for the same persona name are evicted.
/// 5. Subsequent mailbox activations pick up the new handle without
///    restart.
///
/// # Idempotency
///
/// Calling `register_recipe` with a template whose `param_hash` is
/// already cached is a no-op (the registry returns the cached handle).
/// This makes recipe registration safe to call at every session start
/// without triggering unnecessary recompilation.
///
/// # BLOCKED
///
/// - BLOCKED on `StyleRegistry` API extension: a `register_recipe`
///   / `get_recipe_kernel` surface must be added to
///   `contract::jit::StyleRegistry` before this function can be fully
///   wired.  Until that extension lands, the body is `todo!()`.
/// - BLOCKED on D-ATOM-1: `RecipeTemplate::compile` is itself blocked.
///
/// # Parameters
///
/// - `registry` — a mutable reference to the [`jit::StyleRegistry`]
///   implementation.  The `&mut dyn StyleRegistry` bound requires an
///   extension method on `StyleRegistry`; see BLOCKED note above.
/// - `template` — the recipe template to register.
///
/// # Returns
///
/// The compiled [`jit::KernelHandle`] for the registered persona, or a
/// [`jit::JitError`] if compilation fails.
pub fn register_recipe(
    _registry: &mut dyn crate::jit::StyleRegistry,
    _template: &RecipeTemplate,
) -> Result<KernelHandle, JitError> {
    // BLOCKED: StyleRegistry API extension (register_recipe / get_recipe_kernel
    // entry-point does not yet exist on the StyleRegistry trait in jit.rs)
    // BLOCKED: D-ATOM-1 (RecipeTemplate::compile is blocked)
    todo!(
        "register_recipe — blocked on StyleRegistry::register_recipe API \
         extension (jit.rs) and on D-ATOM-1 (atoms::I4x32)"
    )
}
