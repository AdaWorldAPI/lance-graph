//! ⚠ MIGRATED FROM `medcare-rs/crates/medcare-nodesoa/src/alpha_tunnel.rs`
//! (2026-08-31, operator-ruled — verbatim: *"das schlimmste handrolling ist
//! alpha tunnel in medcare. sowas sollte in lance-graph der normalzustand
//! sein."*). Ten lanes over ONE reservation, one per rung — now the normal
//! state, here.
//!
//! **Der Split-Tunnel** — zehn Spuren über EINER Reservierung, eine je Sprosse.
//!
//! Operator, 2026-08-31: *„und dann 10 level parallel über alpha channel split
//! tunnel table via reserve dont claim"*.
//!
//! # Split-Tunnel heißt: Lesen und Schreiben nehmen verschiedene Wege
//!
//! Gelesen wird der **gebackene Spine** — read-only, von allen zehn Spuren
//! gleichzeitig, ohne Sperre, weil `&[NodeRow]` geteilt werden darf.
//! Geschrieben wird der **Overlay** an denselben Adressen. Der Basis-Spine
//! erfährt nie etwas davon; [`alpha`](crate::alpha) macht diese Richtung zu
//! einer Compile-Zeit-Eigenschaft, nicht zu einer Prüfung.
//!
//! # Reserve, don't claim — und warum die Reservierung EINE ist
//!
//! Reservieren kostet **null Zeilen**: der Adressraum ist der des Spine, und
//! er ist schon da. Zehn Spuren dürfen also nicht zehn Adressmengen bedeuten —
//! sonst kostete „reservieren" plötzlich das Zehnfache von nichts.
//! Deshalb hält der Tunnel **eine** [`AlphaAllocation`] und reicht sie allen
//! Spuren als Borrow ([`AlphaOverlay::over_shared`]); eine Spur kostet einen
//! leeren `Vec`.
//!
//! # Ein Schreiber je Spur — kein geteiltes `&mut`
//!
//! Zehn parallele Spuren teilen sich **nichts** Veränderliches. Das ist keine
//! Vorsichtsmaßnahme, sondern dieselbe Regel, die eine Etage tiefer
//! `SoaEnvelope::mailbox_owner` durchsetzt und die der Workspace als
//! *ONE WRITER PER FILE* führt: ein gemeinsamer Anhänge-Puffer ist kein Kanal,
//! sondern ein verlorener Schreibvorgang mit extra Schritten.
//!
//! Weil jede Spur ihr eigenes `&mut` hat, ist die Parallelität **strukturell**
//! und braucht keine Sperre — `std::thread::scope`, keine neue Abhängigkeit.
//!
//! # Determinismus ist die Zusicherung, nicht die Geschwindigkeit
//!
//! [`AlphaTunnel::run_wave`] und [`AlphaTunnel::run_wave_parallel`] müssen
//! **byte-identisch** dasselbe ergeben. Das ist der Falsifikator, den dieses
//! Modul schuldet: ein paralleler Lauf, der ein anderes Ergebnis liefert als
//! der sequenzielle, ist kein schnellerer Lauf, sondern ein anderer.
//!
//! # Die Verschmelzung: wer zuerst hinsah, behält seinen Platz
//!
//! [`AlphaTunnel::merge`] faltet die zehn Spuren in **Sprossen-Reihenfolge**.
//! Landet dieselbe Adresse auf mehreren Sprossen, gehört `rung`/`cycle` der
//! FLACHSTEN — attention returning does not change where it had been, sagt
//! [`AlphaStamp`] über sich selbst — und die Rückkehr wird als `visits`
//! gezählt. Genau die Regression, die der Stamp als „die beste Diagnose, die
//! der Kanal hat" führt, nur eine Ebene höher: **quer über die Sprossen**.
//!
//! `seq` wird beim Verschmelzen NEU vergeben, denn die Spur-lokalen `seq`
//! kollidieren zwangsläufig (jede Spur zählt bei 0 los). Die globale Ordnung
//! ist `(rung, seq_in_lane)` — deterministisch, unabhängig davon, in welcher
//! Reihenfolge die Threads fertig wurden.

use crate::canonical_node::NodeRow;

use crate::alpha::{stamp_of, AlphaAddr, AlphaAllocation, AlphaOverlay, AlphaStamp};
use crate::rung_schedule::{Wave, LEVELS};

/// Zehn Spuren über einer Reservierung — eine je Sprosse `0..=9`.
///
/// Spur `0` ist die Sprosse „nicht gesetzt": der Rezept-Ladder vergibt
/// `1..=9` ([`recipe_dispatch::rung`](crate::recipe_dispatch::rung)),
/// und `0` bleibt für alles, was vor der ersten Ableitung hinsieht. Sie wird
/// **reserviert, nicht gestrichen** — dieselbe Regel wie überall sonst im
/// Register.
pub struct AlphaTunnel<'a> {
    lanes: Vec<AlphaOverlay<'a>>,
    cycle: u32,
}

impl<'a> AlphaTunnel<'a> {
    /// Zehn leere Spuren über `alloc`. Kostet zehn leere `Vec` — **nicht**
    /// zehn Adressmengen (siehe Modulkopf).
    #[must_use]
    pub fn over(alloc: &'a AlphaAllocation<'a>, cycle: u32) -> Self {
        Self {
            lanes: (0..LEVELS)
                .map(|_| AlphaOverlay::over_shared(alloc, cycle))
                .collect(),
            cycle,
        }
    }

    /// Die Sprossen-Spur — `None` für `rung >= LEVELS`.
    #[must_use]
    pub fn lane(&self, rung: u8) -> Option<&AlphaOverlay<'a>> {
        self.lanes.get(rung as usize)
    }

    /// Die Sprossen-Spur, beschreibbar — `None` für `rung >= LEVELS`.
    ///
    /// Die Ein-Schreiber-Regel bleibt strukturell: `&mut self` ist exklusiv,
    /// also hält genau ein Aufrufer genau eine Spur. Gebraucht von
    /// horizont-gegateten Claims (lance-graph-planner `rung_horizon`), die
    /// außerhalb eines `run_wave`-Laufs in eine bestimmte Spur schreiben.
    pub fn lane_mut(&mut self, rung: u8) -> Option<&mut AlphaOverlay<'a>> {
        self.lanes.get_mut(rung as usize)
    }

    /// Wie viele Zeilen der ganze Tunnel materialisiert hat.
    #[must_use]
    pub fn claimed_len(&self) -> usize {
        self.lanes.iter().map(AlphaOverlay::claimed_len).sum()
    }

    /// Der Zyklus, für den dieser Tunnel gilt.
    #[must_use]
    pub const fn cycle(&self) -> u32 {
        self.cycle
    }

    /// Eine Welle **sequenziell** fahren — je Sprosse der Welle einmal `f`.
    ///
    /// `f` bekommt die Sprosse, ihre Rezept-Ids und **ihre eigene Spur**. Sie
    /// kann nur diese eine Spur beschreiben; das ist die Ein-Schreiber-Regel
    /// als Typ, nicht als Verabredung.
    pub fn run_wave<F>(&mut self, wave: &Wave, mut f: F)
    where
        F: FnMut(u8, &[u8], &mut AlphaOverlay<'a>),
    {
        for rung in wave.rungs() {
            let ids = wave.ids_at(rung);
            if let Some(lane) = self.lanes.get_mut(rung as usize) {
                f(rung, &ids, lane);
            }
        }
    }

    /// Dieselbe Welle **parallel** — eine Spur je Thread, `std::thread::scope`.
    ///
    /// Keine Sperre und keine neue Abhängigkeit: die Spuren teilen sich nichts
    /// Veränderliches, also ist die Nebenläufigkeit strukturell. Was sie
    /// teilen, ist der Basis-Spine, und der wird nur gelesen.
    ///
    /// # Zusicherung
    ///
    /// Das Ergebnis ist **byte-identisch** zu [`run_wave`](Self::run_wave).
    /// Nicht „meistens" — die Spuren berühren einander nicht, und
    /// [`merge`](Self::merge) ordnet nach `(rung, seq)` statt nach
    /// Fertigstellung. Der Falsifikator dazu ist
    /// `parallel_und_sequenziell_sind_byte_identisch`.
    pub fn run_wave_parallel<F>(&mut self, wave: &Wave, f: F)
    where
        F: Fn(u8, &[u8], &mut AlphaOverlay<'a>) + Sync,
    {
        // Je Sprosse der Welle genau eine Spur — Paare aus `&mut`, die
        // einander nicht überlappen, weil `rungs()` dedupliziert ist.
        let mut jobs: Vec<(u8, Vec<u8>, &mut AlphaOverlay<'a>)> = Vec::new();
        let wanted = wave.rungs();
        for (i, lane) in self.lanes.iter_mut().enumerate() {
            let r = u8::try_from(i).unwrap_or(u8::MAX);
            if wanted.contains(&r) {
                jobs.push((r, wave.ids_at(r), lane));
            }
        }
        let fref = &f;
        std::thread::scope(|s| {
            for (r, ids, lane) in jobs {
                s.spawn(move || fref(r, &ids, lane));
            }
        });
    }

    /// **Die Verschmelzung** — zehn Spuren zu einem Scanpfad, deterministisch.
    ///
    /// Ordnung: `(rung, seq_in_lane)`. Die flachste Sprosse, die eine Adresse
    /// beansprucht hat, behält `rung` und `cycle`; jede weitere Beanspruchung
    /// erhöht `visits`. `seq` wird global neu vergeben, weil die Spur-lokalen
    /// `seq` kollidieren (jede Spur zählt bei 0 los).
    ///
    /// Rückgabe in Scanpfad-Reihenfolge — der Pfad IST der Index, wie im
    /// Alpha-Kanal.
    #[must_use]
    pub fn merge(&self) -> Vec<(AlphaAddr, AlphaStamp)> {
        let mut all: Vec<(u8, u32, AlphaAddr, AlphaStamp)> = Vec::new();
        for (i, lane) in self.lanes.iter().enumerate() {
            let r = u8::try_from(i).unwrap_or(u8::MAX);
            for row in lane.rows() {
                let st = stamp_of(row);
                all.push((r, st.seq, row.key, st));
            }
        }
        // KEIN `sort` — die Traversierung IST die Ordnung: die Spuren laufen
        // nach Index (== Sprosse), und `rows()` gibt je Spur die Anspruechse in
        // `seq`-Reihenfolge. Ein `sort_by_key((rung, seq))` stand hier und war
        // **beweisbar tot**: der Disable-Lauf (Sortierung entfernt) blieb
        // gruen, weil er nichts umordnen KANN. Statt ihn als Beruhigung
        // stehenzulassen, steht hier die Zusicherung als Assert — der faellt,
        // wenn jemand die Traversierung aendert (etwa auf eine HashMap).
        debug_assert!(
            all.windows(2).all(|w| (w[0].0, w[0].1) <= (w[1].0, w[1].1)),
            "die Traversierung muss (rung, seq)-geordnet sein"
        );

        let mut out: Vec<(AlphaAddr, AlphaStamp)> = Vec::new();
        let mut at: std::collections::HashMap<AlphaAddr, usize> = std::collections::HashMap::new();
        for (_, _, addr, st) in all {
            if let Some(&i) = at.get(&addr) {
                // Eine Rueckkehr quer ueber die Sprossen. `rung`/`cycle`/`seq`
                // der flachsten bleiben stehen — nur der Zaehler bewegt sich,
                // exakt wie `AlphaOverlay::claim` es innerhalb einer Spur tut.
                out[i].1.visits = out[i].1.visits.saturating_add(st.visits);
                continue;
            }
            let seq = u32::try_from(out.len()).unwrap_or(u32::MAX);
            at.insert(addr, out.len());
            out.push((addr, AlphaStamp { seq, ..st }));
        }
        out
    }

    /// Die verschmolzenen Adressen als kanonische Zeilen — dieselbe Form, die
    /// [`AlphaOverlay`] schreibt, damit der Tunnel-Ausgang und der
    /// Einzel-Overlay dieselbe Tabelle sind.
    #[must_use]
    pub fn merged_rows(&self) -> Vec<NodeRow> {
        self.merge()
            .into_iter()
            .map(|(addr, st)| {
                let mut row = NodeRow {
                    key: addr,
                    edges: crate::canonical_node::EdgeBlock::default(),
                    value: [0u8; 480],
                };
                row.value[crate::alpha::ALPHA_STAMP_OFFSET
                    ..crate::alpha::ALPHA_STAMP_OFFSET + crate::alpha::ALPHA_STAMP_BYTES]
                    .copy_from_slice(&st.to_le_slot());
                row
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical_node::NodeGuid;
    use crate::recipe_kernels::ThoughtMask;
    use crate::rung_schedule::{schedule, Wave};

    /// Ein synthetischer Spine — bewusst NICHT der OBO-Bake (nach einem
    /// Container-Reset leer), damit der Test hermetisch ist.
    fn tiny_base(n: u32) -> Vec<NodeRow> {
        (0..n)
            .map(|i| NodeRow {
                key: NodeGuid::new(0x0C0C_0000 + i, 1, 2, 3, 0x22, i + 1),
                edges: crate::canonical_node::EdgeBlock::default(),
                value: [0u8; 480],
            })
            .collect()
    }

    /// Eine Welle mit mehreren Sprossen, aus dem ECHTEN Katalog — nicht von
    /// Hand gebaut, damit der Test den Scheduler mitprüft und nicht nur sich
    /// selbst.
    fn real_wave() -> Wave {
        let s = schedule(ThoughtMask(0xFF));
        assert!(!s.waves.is_empty(), "der volle Kontext plant etwas ein");
        s.waves[0].clone()
    }

    /// **Reservieren kostet keine Zeile.** Zehn Spuren über EINER Reservierung:
    /// jede sieht den ganzen Adressraum, keine hat eine Zeile.
    ///
    /// Das ist die Behauptung, die dem Modul seinen Namen gibt; ohne sie wäre
    /// „reserve, don't claim" bei zehn Spuren zehnmal so teuer wie nichts.
    #[test]
    fn reservieren_kostet_keine_zeile() {
        let rows = tiny_base(64);
        let alloc = AlphaAllocation::over(&rows);
        let t = AlphaTunnel::over(&alloc, 7);
        assert_eq!(t.claimed_len(), 0, "nichts beansprucht");
        for r in 0..LEVELS {
            let lane = t.lane(u8::try_from(r).unwrap()).expect("Spur existiert");
            assert_eq!(
                lane.allocated_len(),
                rows.len(),
                "jede Spur sieht ALLE Adressen"
            );
            assert_eq!(lane.claimed_len(), 0);
        }
    }

    /// **Der Falsifikator des Moduls: parallel == sequenziell, byte-identisch.**
    ///
    /// Zwei Tunnel, dieselbe Welle, dieselbe Arbeit — einmal in einem Thread,
    /// einmal in zehn. Die verschmolzenen Zeilen müssen Byte für Byte gleich
    /// sein, sonst ist der parallele Lauf kein schnellerer, sondern ein anderer.
    ///
    /// Jede Sprosse beansprucht bewusst ÜBERLAPPENDE Adressen, damit die
    /// Verschmelzung wirklich etwas zu entscheiden hat.
    #[test]
    fn parallel_und_sequenziell_sind_byte_identisch() {
        let rows = tiny_base(64);
        let alloc = AlphaAllocation::over(&rows);
        let w = real_wave();
        let work = |rung: u8, ids: &[u8], lane: &mut AlphaOverlay<'_>| {
            for (k, &id) in ids.iter().enumerate() {
                // ueberlappend ueber die Sprossen: der Index haengt an der id,
                // nicht an der Sprosse.
                let idx = (usize::from(id) * 3 + k) % 64;
                lane.claim(rows[idx].key, rung).expect("allokiert");
            }
        };

        let mut seq = AlphaTunnel::over(&alloc, 7);
        seq.run_wave(&w, work);
        let mut par = AlphaTunnel::over(&alloc, 7);
        par.run_wave_parallel(&w, work);

        assert!(seq.claimed_len() > 0, "der Lauf muss etwas getan haben");
        // ANTI-VAKUUM: ohne mehrere Sprossen liefe nur ein Thread, und ohne
        // Kollisionen haette die Verschmelzung nichts zu entscheiden. Beides
        // wird gemessen, nicht gehofft.
        assert!(
            w.rungs().len() > 1,
            "die Welle muss mehrere Spuren fuellen: {:?}",
            w.rungs()
        );
        let merged = seq.merge();
        assert!(
            merged.iter().any(|(_, st)| st.visits > 1),
            "keine Adresse quer ueber die Sprossen — die Verschmelzung entscheidet nichts"
        );
        assert!(
            merged.len() < seq.claimed_len(),
            "ohne Ueberlappung ist die Verschmelzung eine Konkatenation"
        );

        let (a, b) = (seq.merged_rows(), par.merged_rows());
        assert_eq!(a.len(), b.len(), "gleich viele Zeilen");
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.key, y.key, "gleiche Adresse an gleicher Stelle");
            assert_eq!(x.value, y.value, "byte-identischer Stamp");
        }

        // Und der parallele Lauf ist mit SICH SELBST reproduzierbar — sonst
        // waere die Gleichheit oben ein einzelner gluecklicher Scheduler-Lauf.
        for _ in 0..8 {
            let mut again = AlphaTunnel::over(&alloc, 7);
            again.run_wave_parallel(&w, work);
            let c = again.merged_rows();
            assert_eq!(
                c.len(),
                b.len(),
                "paralleler Lauf reproduziert sich (Laenge)"
            );
            for (x, y) in c.iter().zip(b.iter()) {
                assert!(
                    x.key == y.key && x.value == y.value,
                    "paralleler Lauf reproduziert sich nicht byte-identisch"
                );
            }
        }
    }

    /// **Eine Rückkehr quer über die Sprossen wird gezählt, nicht überschrieben.**
    ///
    /// Dieselbe Adresse auf Sprosse 2 und auf Sprosse 7: verschmolzen ist es
    /// EINE Zeile, sie gehört der flachsten Sprosse, und `visits` steht auf 2.
    ///
    /// Disable-verifiziert: ersetzt man die `visits`-Addition in [`merge`]
    /// durch ein Überschreiben, fällt dieser Test.
    ///
    /// ⊘ **Korrektur an einer Behauptung, die hier stand.** Der Kommentar
    /// versprach zusätzlich, dass „ohne die Sprossen-Sortierung" die
    /// `seq`-Behauptung falle. Der Disable-Lauf sagt: **nein** — der Test
    /// blieb grün, weil die Sortierung nichts umordnen kann (die Spuren laufen
    /// ohnehin nach Sprossen-Index, die Zeilen ohnehin nach `seq`). Die
    /// Sortierung war tot und ist entfernt; die Ordnung ist jetzt als
    /// `debug_assert` über die Traversierung gesichert. Dass eine
    /// Disable-Behauptung im Doc-Kommentar stand, bevor sie gefahren war, ist
    /// genau der Fehler, den `.claude/`-Regeln als vakuumen Falsifikator
    /// führen — hier gefangen, weil der Lauf trotzdem gemacht wurde.
    #[test]
    fn eine_rueckkehr_quer_ueber_die_sprossen_wird_gezaehlt() {
        let rows = tiny_base(8);
        let alloc = AlphaAllocation::over(&rows);
        let mut t = AlphaTunnel::over(&alloc, 3);

        // absichtlich TIEF zuerst beansprucht — die Verschmelzung muss nach
        // Sprosse ordnen, nicht nach Aufrufreihenfolge.
        t.lanes[7].claim(rows[4].key, 7).unwrap();
        t.lanes[2].claim(rows[4].key, 2).unwrap();
        t.lanes[2].claim(rows[1].key, 2).unwrap();

        let m = t.merge();
        assert_eq!(m.len(), 2, "zwei verschiedene Adressen, nicht drei Zeilen");
        let hit = m.iter().find(|(a, _)| *a == rows[4].key).expect("da");
        assert_eq!(hit.1.rung, 2, "die FLACHSTE Sprosse behaelt die Adresse");
        assert_eq!(hit.1.visits, 2, "die Rueckkehr wird gezaehlt");
        assert_eq!(m[0].1.seq, 0, "seq wird global neu vergeben");
        assert_eq!(m[1].1.seq, 1);
    }
}
