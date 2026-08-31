//! ⚠ MIGRATED FROM `medcare-rs/crates/medcare-nodesoa/src/rung_schedule.rs`
//! (2026-08-31, operator-ruled: scheduling over the recipe catalogue is
//! THINKING substrate and belongs beside the catalogue, not in a consumer).
//!
//! **Der Wellen-Scheduler** — die 34 NARS-Rezepte in der Reihenfolge ihrer
//! ABHÄNGIGKEITEN losgeschickt, nicht in der ihrer Tiefe.
//!
//! Operator, 2026-08-31: *„thinking styles rung 0-9 parallel vom scheduler in
//! der reihenfolge der abhängigkeiten losgeschickt"*.
//!
//! # Warum das nicht `dispatch_order()` ist
//!
//! [`recipe_dispatch::dispatch_order`] sortiert die 34 Rezepte nach
//! `(rung, id)` und [`recipe_dispatch::ladder`] läuft sie **streng
//! sequenziell** ab. Das ist eine **Tiefen**-Ordnung: sie sagt, wie tief ein
//! Rezept greift, nicht, wovon es abhängt. Zwei Rezepte auf derselben Sprosse,
//! von denen eines das Feld schreibt, das das andere braucht, laufen dort in
//! **id-Reihenfolge** — also nach einer Zahl, die mit der Abhängigkeit nichts
//! zu tun hat.
//!
//! Die Abhängigkeit steht längst als Daten da und wurde nur nie zum Planen
//! benutzt: jeder Kernel deklariert
//! [`Tactic::requires`](crate::recipe_kernels::Tactic::requires)
//! und [`Tactic::writes`](crate::recipe_kernels::Tactic::writes)
//! als [`ThoughtMask`] über acht [`ThoughtField`]. Das IST der DAG.
//!
//! # Die Regel, in einem Satz
//!
//! Ein Rezept ist **bereit**, wenn `requires().covered_by(known)` — genau die
//! Deckungsregel, die der Contract selbst `E-RELIABILITY-IS-CHECKLIST-COVERAGE`
//! nennt. Alle gleichzeitig bereiten Rezepte bilden **eine Welle** und dürfen
//! parallel laufen, weil keines ein Feld liest, das ein anderes derselben
//! Welle erst schreibt. Nach der Welle wächst `known` um die Vereinigung ihrer
//! `writes()`, und die nächste Welle wird gebildet.
//!
//! # Was NICHT still verschwindet
//!
//! Ein Rezept, dessen `requires()` nie gedeckt wird, landet in
//! [`Schedule::unreachable`] — mit Namen. Ein Scheduler, der es einfach nicht
//! aufruft, sähe von außen identisch aus wie einer, der fertig ist; das ist
//! genau die Stille, die dieser Ledger sonst als Defekt führt.
//!
//! # Determinismus
//!
//! Innerhalb einer Welle wird nach `(rung, id)` sortiert. Die Welle ist eine
//! MENGE (die Reihenfolge darin ist bedeutungslos, sonst wäre sie keine
//! Welle) — die Sortierung existiert also nicht für die Semantik, sondern
//! damit zwei Läufe dieselbe Ausgabe erzeugen und ein Test überhaupt etwas
//! festnageln kann.

use crate::recipe_dispatch::rung;
use crate::recipe_kernels::{all_kernels, kernel, ThoughtField, ThoughtMask};

/// Die Sprossen des Rezept-Ladders: `1..=9` (`recipe_dispatch::rung`).
pub const MAX_RUNG: u8 = 9;

/// Ebenen im Tunnel: Sprosse `0` (= nicht gesetzt / vor jeder Ableitung) plus
/// `1..=9`. **Zehn** — die Zahl, die der Alpha-Kanal als Spuren führt.
pub const LEVELS: usize = 10;

/// Die Anzahl der Rezepte im Katalog.
pub const RECIPES: u8 = 34;

/// Eine **Welle**: die Rezepte, deren Eingaben zum selben Zeitpunkt gedeckt
/// sind. Sie dürfen parallel laufen — nicht als Optimierung, sondern weil
/// keines ein Feld liest, das ein anderes derselben Welle erst schreibt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Wave {
    /// Position der Welle im Ablauf (0 ist die erste).
    pub index: usize,
    /// Die Rezept-Ids, nach `(rung, id)` sortiert — siehe Determinismus im
    /// Modulkopf.
    pub ids: Vec<u8>,
    /// Was VOR dieser Welle bekannt war.
    pub known_before: ThoughtMask,
    /// Die Vereinigung der `writes()` dieser Welle — was sie hinzufügt.
    pub writes: ThoughtMask,
}

impl Wave {
    /// Die Sprossen, die in dieser Welle vorkommen, aufsteigend und ohne
    /// Wiederholung. Das ist die Landkarte auf die Tunnel-Spuren:
    /// **eine Spur je Sprosse**.
    #[must_use]
    pub fn rungs(&self) -> Vec<u8> {
        let mut r: Vec<u8> = self.ids.iter().map(|&id| rung(id)).collect();
        r.sort_unstable();
        r.dedup();
        r
    }

    /// Die Ids dieser Welle, die auf `rung` fallen — der Inhalt EINER Spur.
    #[must_use]
    pub fn ids_at(&self, rung_level: u8) -> Vec<u8> {
        self.ids
            .iter()
            .copied()
            .filter(|&id| rung(id) == rung_level)
            .collect()
    }
}

/// Der fertige Plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schedule {
    /// Die Wellen, in Ausführungsreihenfolge.
    pub waves: Vec<Wave>,
    /// Rezepte, deren `requires()` **nie** gedeckt wurde — benannt, nicht
    /// verschwiegen (siehe Modulkopf).
    pub unreachable: Vec<u8>,
    /// Was am Ende bekannt ist.
    pub known_after: ThoughtMask,
}

impl Schedule {
    /// Wie viele Rezepte überhaupt eingeplant wurden.
    #[must_use]
    pub fn scheduled_len(&self) -> usize {
        self.waves.iter().map(|w| w.ids.len()).sum()
    }

    /// Die Wellen-Tiefe — wie viele SEQUENZIELLE Schritte der Plan braucht.
    /// Gegen `scheduled_len()` gehalten ist das der Parallelitäts-Gewinn.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.waves.len()
    }
}

/// Alle acht Felder — die Reihenfolge ist die Ordinal-Reihenfolge von
/// [`ThoughtField`], nicht eine zweite Konvention.
const FIELDS: [ThoughtField; 8] = [
    ThoughtField::Sd,
    ThoughtField::FreeEnergy,
    ThoughtField::Dissonance,
    ThoughtField::Temperature,
    ThoughtField::Confidence,
    ThoughtField::Rung,
    ThoughtField::Candidates,
    ThoughtField::Beliefs,
];

/// Vereinigung zweier Masken. `ThoughtMask` ist ein `pub u8`-Tupel ohne
/// `BitOr`; das hier ist die eine Stelle, an der geodert wird, statt es an
/// jeder Aufrufstelle zu wiederholen.
#[must_use]
pub fn union(a: ThoughtMask, b: ThoughtMask) -> ThoughtMask {
    ThoughtMask(a.0 | b.0)
}

/// Eine Maske aus den Feldern bauen, die in `ctx` **gegründet** sind.
///
/// Der Contract hat die Umkehrung schon —
/// [`recipe_dispatch::nan_disqualifier`] nennt das erste UNGEDECKTE Feld eines
/// Rezepts. Hier wird dieselbe Frage einmal für den ganzen Kontext gestellt,
/// damit der Plan sie nicht 34-mal wiederholt.
#[must_use]
pub fn known_from(ctx: &crate::recipe_kernels::ThoughtCtx) -> ThoughtMask {
    let mut m = ThoughtMask::EMPTY;
    for f in FIELDS {
        let grounded = match f {
            ThoughtField::Sd => !ctx.sd.is_nan(),
            ThoughtField::FreeEnergy => !ctx.free_energy.is_nan(),
            ThoughtField::Dissonance => !ctx.dissonance.is_nan(),
            ThoughtField::Temperature => !ctx.temperature.is_nan(),
            ThoughtField::Confidence => !ctx.confidence.is_nan(),
            // `Rung` ist ein `u8` und nie NaN — der Contract sagt es selbst
            // (`field_is_undefined`), also ist es immer gegründet.
            ThoughtField::Rung => true,
            ThoughtField::Candidates => !ctx.candidates.is_empty(),
            ThoughtField::Beliefs => !ctx.beliefs.is_empty(),
        };
        if grounded {
            m = ThoughtMask(m.0 | ThoughtMask::of(&[f]).0);
        }
    }
    m
}

/// **Den Plan rechnen.**
///
/// Fixpunkt-Iteration über die Deckungsregel: bereit = `requires().covered_by(
/// known)`; nach jeder Welle wächst `known` um deren `writes()`. Terminiert,
/// weil `known` monoton wächst und über acht Bits läuft — höchstens neun
/// Wellen sind also möglich, und der Beweis dafür ist die Maske selbst, nicht
/// ein Zähler.
#[must_use]
pub fn schedule(initial_known: ThoughtMask) -> Schedule {
    let mut done = [false; RECIPES as usize + 1];
    let mut known = initial_known;
    let mut waves: Vec<Wave> = Vec::new();

    loop {
        let mut ids: Vec<u8> = (1..=RECIPES)
            .filter(|&id| !done[id as usize])
            .filter(|&id| kernel(id).is_some_and(|k| k.requires().covered_by(known)))
            .collect();
        if ids.is_empty() {
            break;
        }
        // NUR fuer den Determinismus, nicht fuer die Semantik (Modulkopf).
        ids.sort_by_key(|&id| (rung(id), id));

        let writes = ids
            .iter()
            .filter_map(|&id| kernel(id))
            .fold(ThoughtMask::EMPTY, |acc, k| union(acc, k.writes()));

        for &id in &ids {
            done[id as usize] = true;
        }
        waves.push(Wave {
            index: waves.len(),
            ids,
            known_before: known,
            writes,
        });
        known = union(known, writes);
    }

    let unreachable: Vec<u8> = (1..=RECIPES).filter(|&id| !done[id as usize]).collect();
    Schedule {
        waves,
        unreachable,
        known_after: known,
    }
}

/// Der Plan für einen **gegründeten** Kontext — der übliche Einstieg.
#[must_use]
pub fn schedule_for(ctx: &crate::recipe_kernels::ThoughtCtx) -> Schedule {
    schedule(known_from(ctx))
}

/// Die `requires()`-Maske eines Rezepts, für Aufrufer, die den Katalog nicht
/// selbst anfassen wollen.
#[must_use]
pub fn requires_of(id: u8) -> Option<ThoughtMask> {
    kernel(id).map(crate::recipe_kernels::Tactic::requires)
}

/// Wie viele Rezepte der Katalog kennt — gegen [`RECIPES`] geprüft, damit die
/// Konstante nicht still von der Quelle abdriftet.
#[must_use]
pub fn catalogue_len() -> usize {
    all_kernels().len()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Keine Id geht verloren.** Eingeplant + unerreichbar == der Katalog,
    /// ohne Wiederholung. Ein Scheduler, der eine Id still verschluckt, sähe
    /// von außen aus wie einer, der fertig ist.
    #[test]
    fn keine_id_geht_verloren() {
        for known in [
            ThoughtMask::EMPTY,
            ThoughtMask(0b0011_0010),
            ThoughtMask(0xFF),
        ] {
            let s = schedule(known);
            let mut all: Vec<u8> = s.waves.iter().flat_map(|w| w.ids.clone()).collect();
            all.extend(&s.unreachable);
            all.sort_unstable();
            let mut expect: Vec<u8> = (1..=RECIPES).collect();
            expect.sort_unstable();
            assert_eq!(all, expect, "known {:08b}: Katalog vollstaendig", known.0);
        }
        assert_eq!(
            catalogue_len(),
            RECIPES as usize,
            "die Konstante folgt der Quelle"
        );
    }

    /// **Die Deckungsregel diskriminiert wirklich.** Ohne gegründete Felder
    /// sind fast alle Rezepte disqualifiziert; mit allen Feldern keines.
    ///
    /// Ohne die zweite Hälfte wäre die erste mit einem Scheduler erfüllt, der
    /// nie etwas einplant — und ohne die erste mit einem, der alles einplant.
    #[test]
    fn die_deckungsregel_diskriminiert() {
        let leer = schedule(ThoughtMask::EMPTY);
        assert!(
            leer.unreachable.len() > RECIPES as usize / 2,
            "ohne Gruendung bleibt das meiste liegen: {} unerreichbar",
            leer.unreachable.len()
        );
        assert!(
            leer.scheduled_len() > 0,
            "aber die requires-freien Rezepte laufen — sonst startet nichts je"
        );

        let voll = schedule(ThoughtMask(0xFF));
        assert!(
            voll.unreachable.is_empty(),
            "voll gegruendet disqualifiziert keines"
        );
        assert_eq!(voll.scheduled_len(), RECIPES as usize);
    }

    /// **DER GEMESSENE BEFUND: der Plan ist FLACH — Tiefe 1, immer.**
    ///
    /// Nicht weil der Scheduler keine Wellen bilden könnte, sondern weil der
    /// Katalog **keine Kante hat**: jedes Rezept, das bereit ist, schreibt nur
    /// Felder, die es selbst schon voraussetzt (`writes ⊆ known_before`).
    /// `writes()` ist ein READ-MODIFY-WRITE-Vokabular, kein
    /// Produzent/Konsument-Vokabular — ein Rezept ändert ein Feld des
    /// `ThoughtCtx`, es *stellt es nicht für ein anderes bereit*.
    ///
    /// Epistemisch ist das richtig: **keine Menge Schlussfolgern erzeugt eine
    /// Messung.** Die Felder kommen aus der Wahrnehmung, nicht aus dem Ladder.
    ///
    /// CAN FIRE — und das ist der ganze Zweck: schreibt je ein Kernel ein Feld,
    /// das seine Welle NICHT schon kannte, wächst die Tiefe über 1 und dieser
    /// Test fällt. Dann ist der DAG echt geworden und die Wellenform gehört neu
    /// vermessen, statt still eine andere zu sein.
    #[test]
    fn der_plan_ist_flach_weil_kein_rezept_fuer_ein_anderes_produziert() {
        for known in [
            ThoughtMask::EMPTY,
            ThoughtMask(0b0010_0000),
            ThoughtMask(0b0011_0010),
            ThoughtMask(0xFF),
        ] {
            let s = schedule(known);
            assert_eq!(
                s.depth(),
                1,
                "known {:08b}: Tiefe {} — der Katalog hat eine Kante bekommen, \
                 die Wellenform gehoert neu vermessen",
                known.0,
                s.depth()
            );
            for w in &s.waves {
                assert_eq!(
                    union(w.known_before, w.writes),
                    w.known_before,
                    "Welle {} schreibt ein Feld, das sie nicht schon kannte",
                    w.index
                );
            }
            assert_eq!(s.known_after, known, "known waechst nicht");
        }
    }

    /// **Die Sprossen einer Welle sind die Spuren des Tunnels.** Auf einem voll
    /// gegründeten Kontext deckt eine einzige Welle acht verschiedene Sprossen
    /// ab — also acht Spuren, die parallel laufen dürfen, weil keine ein Feld
    /// liest, das eine andere erst schreibt.
    #[test]
    fn eine_welle_faechert_ueber_mehrere_sprossen() {
        let s = schedule(ThoughtMask(0xFF));
        let w = &s.waves[0];
        let rungs = w.rungs();
        assert!(rungs.len() > 1, "mehr als eine Spur: {rungs:?}");
        assert!(
            rungs.iter().all(|&r| (1..=MAX_RUNG).contains(&r)),
            "Sprossen 1..=9: {rungs:?}"
        );
        assert!(
            (rungs.len()) < LEVELS,
            "Spur 0 bleibt reserviert, nie belegt"
        );
        // Die Sprossen-Gruppen partitionieren die Welle — keine Id doppelt,
        // keine verloren.
        let sum: usize = rungs.iter().map(|&r| w.ids_at(r).len()).sum();
        assert_eq!(sum, w.ids.len(), "die Gruppen partitionieren die Welle");
    }
}
