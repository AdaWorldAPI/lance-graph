//! MIGRATED from medcare-nodesoa (2026-08-31) — the scheduler demo lives
//! beside the scheduler.
//! **Die Wellenform, gemessen.** Wie viele SEQUENZIELLE Schritte der
//! Abhängigkeits-Plan braucht, gegen die 34 Rezepte, die er einplant — und
//! welche nie bereit werden.
fn main() {
    use lance_graph_contract::recipe_dispatch::rung;
    use lance_graph_contract::recipe_kernels::{ThoughtField, ThoughtMask};
    use lance_graph_contract::rung_schedule::{schedule, union};

    for (name, known) in [
        ("EMPTY (nichts gegruendet)", ThoughtMask::EMPTY),
        ("nur Rung", ThoughtMask::of(&[ThoughtField::Rung])),
        (
            "Rung+FreeEnergy+Confidence",
            ThoughtMask::of(&[
                ThoughtField::Rung,
                ThoughtField::FreeEnergy,
                ThoughtField::Confidence,
            ]),
        ),
        (
            "ALLES gegruendet",
            ThoughtMask::of(&[
                ThoughtField::Sd,
                ThoughtField::FreeEnergy,
                ThoughtField::Dissonance,
                ThoughtField::Temperature,
                ThoughtField::Confidence,
                ThoughtField::Rung,
                ThoughtField::Candidates,
                ThoughtField::Beliefs,
            ]),
        ),
    ] {
        let s = schedule(known);
        println!(
            "\n=== {name} ===  Tiefe {} Wellen fuer {} Rezepte, unreachable {}",
            s.depth(),
            s.scheduled_len(),
            s.unreachable.len()
        );
        for w in &s.waves {
            let rungs = w.rungs();
            println!(
                "  Welle {}: {:2} Rezepte, Sprossen {:?}, known {:08b} -> {:08b}",
                w.index,
                w.ids.len(),
                rungs,
                w.known_before.0,
                union(w.known_before, w.writes).0
            );
            for r in rungs {
                let ids = w.ids_at(r);
                println!("        rung {r}: {ids:?}");
            }
        }
        if !s.unreachable.is_empty() {
            println!("  UNREACHABLE: {:?}", s.unreachable);
            for id in &s.unreachable {
                println!(
                    "        id {id} (rung {}) requires {:08b}",
                    rung(*id),
                    lance_graph_contract::rung_schedule::requires_of(*id).map_or(0, |m| m.0)
                );
            }
        }
    }
}
