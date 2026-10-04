# Labeling brief: German Wechsel-preposition phrases → TIME / PLACE / FIG

You are labeling German sentences. In each one, a prepositional phrase is
marked with [[ ... ]]. Its preposition is an, auf, hinter, in, neben, über,
unter, vor or zwischen. Give the marked phrase exactly one label.

- **TIME**: the phrase locates or measures something in time. Examples:
  - in 12 Monaten
  - vor der Fahrt
  - an dem Tag
  - über das Wochenende
  - in den letzten Jahren
  - vor zwei Wochen
  - im Jahr 2000
  - in Zukunft
  - in der Nacht
- **PLACE**: a literal physical location or direction in the physical world.
  - The test: it answers wo? or wohin? with a real place you could stand in, point at or travel to.
  - Geographic names count: in Deutschland, nach … ins Ausland.
  - So do buildings, rooms, objects, physical surfaces and vehicles: auf dem Tisch, in die Stadt, vor dem Haus, im Werk.
- **FIG**: everything else.
  - The verb governs the preposition: setzen auf, warten auf, rechnen mit, reagieren auf, denken an.
  - Metaphorical or abstract "places": in der Lage, unter Druck, auf dem Markt, im Internet, in der Branche, in Höhe von, auf Basis von, in diesem Fall, im Rahmen, unter den Anbietern, an der Börse (as an institution).
  - Manner, cause or topic: über das Thema, in bar, in Euro.

Decide from the whole sentence. When a phrase is clearly time or clearly physical place, use that label. When you are unsure between PLACE and FIG, ask whether a physical location is meant.

## Rules

- Read ONLY your input file. Do not open any other file in the repository.
- Do not run cargo, git, or any code. You read the sentences yourself.
- Label EVERY line. Do not skip any.
- Write your output file as TSV, one line per item: `id<TAB>LABEL`, where LABEL is exactly TIME, PLACE or FIG.
- Write incrementally:
  - after each batch of 25 items, append the lines to your output file with `cat >> FILE <<'EOF'` (use a heredoc, not a script);
  - when you have finished, verify with `wc -l` that the output has the same number of lines as the input.
- Reply with the line count, your label distribution, and 3 items you found hardest, with their ids. Nothing else.
