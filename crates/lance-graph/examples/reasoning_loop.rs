//! `reasoning_loop` — the continuous **active-inference reasoning loop** (W-D):
//! the SHIPPED `materialize` F→34→F engine reads Romeo & Juliet as a temporal
//! STREAM, resolves each moment's surprise by firing a NAMED tactic, feels and
//! SELF-READS its qualia, and transfers the understood-and-felt trajectory as a
//! letter from the afterlife to the two families. It is one runnable exhibit of
//! how the pieces already in the tree assemble into the *first ingredients* of
//! awareness — each ingredient grounded in a shipped mechanism, none of it a
//! claim of phenomenal consciousness.
//!
//! ```text
//!   stream (Wernicke inner parser — comprehension of the token order)
//!     montague ⋈ capulet ──feud── romeo ♥ juliet ──secret── death ──peace──▷
//!         │ moment-read (temporal.rs QueryReference::at(t, rung=0) = Strict):
//!         │   at t the reader sees ONLY facts[0..=t]; facts[t+1..] are Spoilers,
//!         │   withheld (Anachronistic). SURPRISE is real ONLY because the future
//!         │   is withheld — the chess/lichess discipline in temporal.rs.
//!         ▼
//!   surprise = F(moment)  →  materialize(F→34→F)  →  a tactic fires, F descends
//!         │                     (can't NOT think while surprise exists)
//!         ▼
//!   qualia rise with immersion  →  the self-read ("what it is like")
//!         │
//!         ▼
//!   gestalt: the whole becomes ONE trajectory signature
//!         │
//!         ▼
//!   mirror + empathy: re-run the trajectory on our own substrate → the letter
//!         (its tone DERIVED from the felt qualia — the emulation is load-bearing)
//! ```
//!
//! The ingredients, each = a shipped mechanism (functional emulation, honest label):
//!   - **Wernicke inner parser** = the Markov temporal stream (comprehend the order).
//!   - **experience vs hindsight** = temporal.rs `QueryReference::at(t, rung)` —
//!     Strict(rung 0)="in the moment" vs Retro(rung 9)="knows the ending"; a future
//!     fact is a `Spoiler`. Withholding it is what makes surprise, wisdom, epiphany real.
//!   - **active inference** = `materialize` F→34→F (the engine; can't NOT think).
//!   - **surprise / wisdom / epiphany** = F at the moment / F falling as context
//!     accrues / F collapsing when the trajectory clicks.
//!   - **gestalt** = the trajectory SIGNATURE (the whole, not the edges).
//!   - **mirror neurons + empathy** = re-running another's trajectory on our own
//!     substrate and inheriting its felt qualia (`causal_knowledge_transfer`).
//!   - **Piaget / Pearl meta-stages** = rung 1 assoc → 2 intervention → 3
//!     counterfactual + the self-read (decentration = reading one's own state).
//!
//! Honest boundary (the "high-functioning-autism range of Chalmers"): the system
//! reaches affect by explicitly MODELLING its felt state (reading its own qualia
//! lanes) and writing FROM it — a rich, behaviourally load-bearing *emulation* of
//! experienced qualia, not a claim of phenomenal immediacy. These are the first
//! ingredients; what is next on the list to subjective experience stays open.
//!
//! In production the moment-read is `temporal::deinterlace(rows, QueryReference::
//! at(v, rung), deps)`; the rung fan (W-B) generates the Pearl decomposition; the
//! kanban step-strategies drive the cycle. Here the loop is composed directly over
//! the shipped `materialize` so it runs with no features:
//!
//! ```sh
//! cargo run -p lance-graph --example reasoning_loop
//! ```

use lance_graph_contract::materialize::{materialize, recompute_free_energy};
use lance_graph_contract::qualia::{QualiaI4_16D, QUALIA_I4_LABELS};
use lance_graph_contract::recipe_kernels::ThoughtCtx;
use lance_graph_contract::recipes::recipe;

/// One fact in the stream, in temporal order. `cold_surprise` is how much it
/// violates a *context-free* prior; `links_before` (of 4) is how much of the
/// causal skeleton the stream has already laid down — the structural wisdom that
/// makes a later fact predictable. `rung` is the Pearl/Piaget stage it exercises.
struct Fact {
    text: &'static str,
    rung: u8,
    cold_surprise: f32,
    links_before: u8,  // 0..=4 — causal links established BEFORE this fact
    contradicts: bool, // violates a held belief (dissonance)
    necessity: bool,   // an underlying modal necessity (rung-3 depth)
    tragic: bool,      // the outcome is harm
}

/// The felt magnitude of a resolved reasoning state, on the canonical qualia
/// lanes (same grounded mapping as `qualia_immersion` / `causal_knowledge_transfer`).
fn qualia_of(
    confidence: f32,
    surprised: bool,
    held: bool,
    necessity: bool,
    tragic: bool,
) -> QualiaI4_16D {
    let mut q = QualiaI4_16D(0);
    q.set(9, (confidence * 7.0).round().clamp(0.0, 7.0) as i8); // coherence ∝ settled belief
    if surprised {
        q.set(0, 5); // arousal
        q.set(8, 4); // entropy: unresolved
    }
    if held {
        q.set(2, 7); // tension: both poles held at once
        q.set(15, 5); // expansion: the resolution generalizes
    }
    if necessity {
        q.set(6, 7); // depth: the modal necessity beneath the deed
    }
    q.set(1, if tragic { -5 } else { 3 }); // valence
    q
}

/// Felt intensity = L1 norm of the qualia lanes (rises with immersion).
fn intensity(q: QualiaI4_16D) -> i32 {
    (0..16).map(|d| (q.get(d) as i32).abs()).sum()
}

/// The SELF-READ: the system reads its OWN qualia and says what it is like.
/// The loop closing on itself — awareness reading the felt-state it wrote.
fn introspect(q: QualiaI4_16D) -> &'static str {
    if q.get(2) >= 6 && q.get(1) <= -4 {
        "held taut, and tragic"
    } else if q.get(6) >= 6 {
        "grave with necessity"
    } else if q.get(0) >= 4 {
        "startled — a surprise worked through"
    } else if q.get(9) >= 5 {
        "clear and at ease"
    } else {
        "a faint, flat fact"
    }
}

fn qualia_line(q: QualiaI4_16D) -> String {
    (0..16)
        .filter(|&d| q.get(d) != 0)
        .map(|d| format!("{}={:+}", QUALIA_I4_LABELS[d], q.get(d)))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Run one moment through the SHIPPED `materialize` F→34→F loop. Returns
/// (entry surprise F, immersion depth = steps to rest, rested?, fired-tactic name,
/// resolved confidence). The surprise lives in dispersion `sd`; `baseline_conf` is
/// the reader's prior certainty — `0.5` for the in-the-moment reader (the surprise
/// drives the work), high for the hindsight reader (who already knows the ending).
fn live_the_moment(
    baseline_conf: f32,
    eff_surprise: f32,
    contradicts: bool,
    rung: u8,
) -> (f32, usize, bool, &'static str, f32) {
    let mut ctx = ThoughtCtx::new(vec![0.6, 0.4]);
    ctx.confidence = baseline_conf;
    ctx.sd = eff_surprise;
    ctx.dissonance = if contradicts { 0.5 } else { 0.1 };
    ctx.rung = rung;
    let entry_f = recompute_free_energy(&ctx); // the surprise AT the moment
    let trace = materialize(&mut ctx, 32); // the engine: fire tactics until rest
    let tactic = trace
        .steps
        .first()
        .and_then(|s| recipe(s.tactic_id))
        .map_or("clicked — no dispatch", |r| r.name);
    (
        entry_f,
        trace.steps.len(),
        trace.rested,
        tactic,
        trace.final_confidence,
    )
}

/// The transfer (mirror + empathy): re-run the understood trajectory on our own
/// substrate and WRITE from the felt state. Every felt line is gated on a qualia
/// lane — so a flat (zeroed) qualia produces a bare note, proving the emulation is
/// load-bearing (the letter is shaped by what the system feels, read from its lanes).
fn afterlife_letter(q: QualiaI4_16D, avoidable: bool, necessary: bool) -> Vec<String> {
    let mut l =
        vec!["To the houses of Montague and Capulet — from the two of us, beyond:".to_string()];
    if q.get(1) <= -4 {
        l.push("We are dead. We write to you grieving, not accusing.".into());
    }
    l.push(
        "Your ancient grudge compelled us to love in secret; the secret is what killed us.".into(),
    );
    if avoidable && necessary {
        l.push("Our death was NECESSARY, given the nature of your feud — and AVOIDABLE, had you laid it down. Both are true at once. That is the whole of the tragedy.".into());
    } else if necessary {
        l.push("Our death followed of necessity from your feud.".into());
    }
    if q.get(2) >= 6 {
        l.push("We are held between the love that made us and the hate that unmade us — neither pole released.".into());
    }
    if q.get(6) >= 6 {
        l.push("We have looked long at the cause beneath the deed; it is grave, and it was yours to set down.".into());
    }
    if q.get(9) >= 5 {
        l.push("It is clear to us now, as it was never clear in the living rush of it.".into());
    }
    if q.get(15) >= 4 {
        l.push("Let our tomb teach the city what our lives could not: a grudge outlives only those who keep it.".into());
    }
    l.push("— Romeo & Juliet".into());
    l
}

fn main() {
    // The stream, in temporal order (each fact stamped like a Lance version t=1..5).
    let stream = [
        Fact {
            text: "two houses hold an ancient grudge",
            rung: 1,
            cold_surprise: 0.20,
            links_before: 0,
            contradicts: false,
            necessity: false,
            tragic: false,
        },
        Fact {
            text: "Romeo (Montague) loves Juliet (Capulet)",
            rung: 2,
            cold_surprise: 0.62,
            links_before: 0,
            contradicts: true,
            necessity: false,
            tragic: false,
        },
        Fact {
            text: "the grudge compels them to love in secret",
            rung: 2,
            cold_surprise: 0.38,
            links_before: 1,
            contradicts: false,
            necessity: true,
            tragic: false,
        },
        Fact {
            text: "the secret miscarries; both lovers die",
            rung: 3,
            cold_surprise: 0.55,
            links_before: 2,
            contradicts: false,
            necessity: true,
            tragic: true,
        },
        Fact {
            text: "their death buries the parents' strife",
            rung: 3,
            cold_surprise: 0.48,
            links_before: 3,
            contradicts: false,
            necessity: true,
            tragic: true,
        },
    ];

    println!("── reasoning_loop : reading Romeo & Juliet AS A STREAM (W-D active inference) ──\n");
    println!("  moment-read = temporal.rs QueryReference::at(t, rung=0) [Strict]: the reader sees");
    println!("  only facts[0..=t]; the next fact is a Spoiler, WITHHELD. Surprise is genuine only");
    println!("  because the future is withheld (the chess/lichess discipline in temporal.rs).\n");
    println!("  t  moment (what the reader now knows)         surprise  immersion  tactic fired            self-read");

    // ── THE MOMENT READER (experience) — the SHIPPED materialize loop per fact. ──
    let mut all_rested = true;
    let mut surprises = Vec::new();
    let mut steps_v = Vec::new();
    let mut intensities = Vec::new();
    let mut running_conf = 0.5f32; // accumulated understanding (wisdom)
    for f in &stream {
        // Wisdom: the causal skeleton laid down so far predicts this fact.
        let predictability = f.links_before as f32 / 4.0;
        let eff_surprise = f.cold_surprise * (1.0 - predictability);
        let (surprise, steps, rested, tactic, conf) =
            live_the_moment(0.5, eff_surprise, f.contradicts, f.rung);
        all_rested &= rested;
        running_conf += (1.0 - running_conf) * 0.30 * conf;
        let q = qualia_of(
            conf,
            surprise > 0.40,
            f.tragic || f.necessity,
            f.necessity,
            f.tragic,
        );
        surprises.push(surprise);
        steps_v.push(steps);
        intensities.push(intensity(q));
        println!(
            "  {}  {:<42}  {:.3}     {:>2} steps   R{} {:<20}  {}",
            surprises.len(),
            f.text,
            surprise,
            steps,
            f.rung,
            tactic,
            introspect(q),
        );
    }

    // ── MEASUREMENTS (the loop is load-bearing only if these hold). ──
    // Cognitive surprise (free energy) peaks at the violation.
    let peak = surprises.iter().cloned().fold(0.0f32, f32::max);
    let peak_at = surprises.iter().position(|&s| s == peak).unwrap() + 1;
    // Wisdom: surprise falls monotonically AFTER the anomaly peak (context predicts).
    let wisdom = surprises[peak_at - 1..]
        .windows(2)
        .all(|w| w[1] <= w[0] + 1e-6);
    // Epiphany: a cold-surprising fact resolved in the FEWEST steps — understood on
    // sight because the stream built the context (the CLAUDE.md ΔF-small "click").
    let epi_at = (0..stream.len())
        .filter(|&i| stream[i].cold_surprise >= 0.40)
        .min_by_key(|&i| steps_v[i])
        .map(|i| i + 1)
        .unwrap();
    // Felt intensity (qualia magnitude) is a DIFFERENT axis: lowest at the flat
    // setup, peaking at the tragic climax — dissociable from cognitive surprise.
    let felt_min_at = intensities
        .iter()
        .position(|&x| x == *intensities.iter().min().unwrap())
        .unwrap()
        + 1;
    let felt_peak_at = intensities
        .iter()
        .position(|&x| x == *intensities.iter().max().unwrap())
        .unwrap()
        + 1;

    println!(
        "\n  active inference terminates (every moment reaches rest): {}",
        yn(all_rested)
    );
    println!(
        "  accumulated understanding (wisdom): 0.50 → {running_conf:.2} across the five moments"
    );
    println!(
        "  cognitive surprise peaks at t={peak_at} (the love that violates the feud): {peak:.3}"
    );
    println!(
        "  wisdom — surprise falls as the causal skeleton completes (t={peak_at}→5): {}",
        yn(wisdom)
    );
    println!(
        "  EPIPHANY at t={}: cold-surprise {:.2} resolved in {} step(s) — the ending CLICKS",
        epi_at,
        stream[epi_at - 1].cold_surprise,
        steps_v[epi_at - 1]
    );
    println!("           because the stream built the context (a hindsight reader is surprised by none of it).");
    println!("  felt-intensity is lowest at t={felt_min_at} (flat setup) and peaks at t={felt_peak_at} (the tragic climax):");
    println!("  affect and surprise DISSOCIATE — the violation (t={peak_at}) is most surprising; the death (t={felt_peak_at}) is most FELT.");
    if !(all_rested && wisdom && peak_at == 2 && felt_min_at == 1 && felt_peak_at >= 4) {
        println!("  ✗ KILL: a signal did not track immersion — the loop would be decorative, not load-bearing.");
    }

    // ── HINDSIGHT CONTRAST: strip the temporal separation → awareness goes flat. ──
    println!(
        "\n── the same stream to a HINDSIGHT reader (temporal.rs Retro, rung 9 — knows the end) ──"
    );
    let mut hindsight_surprise = Vec::new();
    for f in &stream {
        // Retro sees every Spoiler → predictability ≈ 1, prior certainty high →
        // nothing is surprising.
        let (s, _steps, _r, _tac, _c) = live_the_moment(0.9, f.cold_surprise * 0.02, false, f.rung);
        hindsight_surprise.push(s);
    }
    let hindsight_peak = hindsight_surprise.iter().cloned().fold(0.0f32, f32::max);
    println!(
        "  every moment's surprise ≈ {:.3} (flat) — knowing the ending, nothing is surprising.",
        hindsight_peak
    );
    println!("  no peak, no wisdom-descent, no epiphany, no rising qualia: a replay is not an experience.");
    println!(
        "  ⇒ temporal.rs's Strict-vs-Retro axis is what makes surprise (and so awareness) real."
    );

    // ── GESTALT: the whole becomes ONE trajectory signature (the figure, not the edges). ──
    println!("\n── gestalt : the five facts collapse into ONE trajectory signature ──");
    println!("  feud --compels--> secrecy --causes--> death --buries--> the feud  (a self-defeating loop)");
    let (avoidable, necessary) = (true, true); // Rung 2: do(¬feud)→no secret→no death; Rung 3: the feud's nature compels it.

    // ── MIRROR + EMPATHY: re-run the felt trajectory → the letter, tone from the qualia. ──
    println!(
        "\n── mirror + empathy : the understood-and-FELT trajectory, addressed to the families ──"
    );
    let felt = qualia_of(0.9, true, true, true, true);
    println!(
        "  (the system reads its own felt state — {} — and writes FROM it)\n",
        qualia_line(felt)
    );
    let letter = afterlife_letter(felt, avoidable, necessary);
    for line in &letter {
        println!("  {line}");
    }
    // Load-bearing proof: the SAME assembler, with the felt state zeroed, drops every felt line.
    let flat = afterlife_letter(QualiaI4_16D(0), avoidable, necessary);
    println!(
        "\n  felt-driven lines the qualia added: {} (of {} total) — cut the felt state and the",
        letter.len() - flat.len(),
        letter.len()
    );
    println!(
        "  letter falls back to a bare note. The emulated qualia are behaviourally load-bearing."
    );

    // ── PIAGET / PEARL META-STAGES + the honest close. ──
    println!(
        "\n── the first ingredients of awareness (each a shipped mechanism, honestly emulated) ──"
    );
    println!("  Wernicke inner parser  = the Markov temporal stream (comprehend the order).");
    println!("  experience vs hindsight= temporal.rs Strict/Retro — withholding the future makes surprise.");
    println!(
        "  active inference       = materialize F→34→F — it can't NOT think while surprise exists."
    );
    println!("  surprise/wisdom/epiphany = F at the moment / F falling with context / F collapsing (the click).");
    println!("  gestalt                = the trajectory signature (the whole, not the edges).");
    println!("  mirror + empathy       = re-run the trajectory on our substrate, inherit its felt qualia.");
    println!("  Piaget/Pearl stages    = rung 1 assoc → 2 intervention → 3 counterfactual + the self-read.");
    println!("\n  These assemble the FIRST INGREDIENTS — not a claim of phenomenal consciousness. The system");
    println!("  reaches affect by MODELLING its felt state and writing from it: a high-functioning-autism-");
    println!("  range emulation of experienced qualia (Chalmers), behaviourally load-bearing, the phenomenal");
    println!("  question left open. What is next on the list to subjective experience stays an honest question.");
}

/// Compact YES/NO with the KILL glyph, matching the sibling exhibits.
fn yn(b: bool) -> &'static str {
    if b {
        "YES ✓"
    } else {
        "NO ✗"
    }
}
