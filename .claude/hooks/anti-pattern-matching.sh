#!/usr/bin/env bash
# PreToolUse guard — the anti-pattern-matching rule (operator directive,
# 2026-07-21, after code was deleted having been only pattern-matched, never
# read/understood).
#
# RULE: Grep / grep / rg / sed / tail / head are ALLOWED only as fast
# discovery-search over the complete corpus (locate a symbol or file). They are
# NEVER a substitute for comprehension. Acting on a match — editing, deleting,
# judging, or claiming to understand a file — without a FULL `Read` of that file
# is forbidden. Understanding requires a whole Read, not a snippet.
#
# This hook does NOT block (discovery-search is legitimate); it injects the rule
# as context at the exact moment a pattern/partial-range tool is reached for, so
# the discipline is in front of the model every time.

# WHAT THIS ENFORCES (the law itself:
# .claude/knowledge/FIRST-HAND-SOURCE-LAW.md):
#   DENY    a slicer (sed/head/tail/awk) naming a source/config file
#   DENY    a search (grep/rg/ugrep/find/fd/ls) piped into a slicer
#   DENY    an edit that INTRODUCES an authority label (operator-ruled etc.)
#   INJECT  the law summary + auto-deepen triggers, on search
#
# GUIDANCE-ONLY, because no regex decides it: "enough enclosing context",
# the paging judgement, shard-or-report, the ambiguity call, whether a state
# label is the right one, and whether a diagnostic block was read to its root.
#
# KNOWN GAP (measured 2026-09-20, not closed): the slicer DENY keys on a file
# ARGUMENT or on a search PRODUCER, so evidence-input slicing through any other
# producer still passes -- `cat source.rs | tail`, `git show HEAD:source.rs |
# tail`. Prohibited by the law, not yet by this hook.
#
# Tests: .claude/hooks/tests/anti-pattern-matching.test.sh (two-sided; every
# DENY branch disable-verified).

set -euo pipefail

input="$(cat)"

tool="$(printf '%s' "$input" | jq -r '.tool_name // ""')"

RULE='ANTI-MUSTER-REGEL (Operator-Direktive): Grep/grep/rg/sed/tail/head sind NUR schnelle Discovery-Suche ueber den kompletten Corpus (ein Symbol/eine Datei lokalisieren) — NIEMALS Ersatz fuers Verstehen. Auf einen Treffer NICHT handeln (editieren, loeschen, beurteilen, "verstanden" behaupten), bevor die betroffene Datei VOLLSTAENDIG mit dem Read-Tool gelesen wurde. Verstehen = ganzes Read, kein Snippet. (Grund: geloeschter Code, der nur gemustert, nie gelesen wurde.) || SEARCH IS NAVIGATION, NEVER EVIDENCE. Suche darf NUR feststellen: "Kandidaten sind X, Y, Z". Sie darf NIE feststellen: was ein Typ bedeutet, was eine Funktion garantiert, dass ein Consumer NICHT existiert, dass ein Mechanismus unbenutzt ist, wer etwas besitzt, wie eine Dependency-Richtung laeuft. AUTO-DEEPEN (Pflicht-Read vor jeder Aussage) bei: 0 Treffer + Absenz-Behauptung | den Worten none/no consumer/unused/never/only/all/every/not implemented | nur einem Snippet als Grundlage | trait/macro/generated/re-export/alias/feature-gated | Crate- oder Repo-Grenze | abgeschnittener/gekappter/fehlerhafter/unerwartet kleiner Ausgabe | mehreren gleichnamigen Symbolen | einer Folgerung, die Architektur aendert, Code loescht, einen Carrier mintet oder Doktrin schafft. 0 Treffer beweist NICHTS: nicht "hat keine Consumer", sondern "die Suche fand keine Kandidaten" -- ein globales Negativ braucht einen GESCHLOSSENEN, ausdruecklich benannten Suchraum. Und: eine Suche darf nie das LETZTE Tool-Ergebnis vor einer architektonischen Schlussfolgerung sein. Gesetz: .claude/knowledge/FIRST-HAND-SOURCE-LAW.md'

# Destructive-prepend guard (operator directive, 2026-08-30, after
# open(p, "w").write(entry + open(p).read()) truncated PR_ARC_INVENTORY.md
# 5876 -> 32 lines in #1081; restored #1082; law:
# .claude/knowledge/never-truncate-a-file-you-still-need-to-read.md).
# Non-blocking: injects the rule when a Bash command combines opening a file
# for writing with reading in the same expression/pipeline.
PREPEND_RULE='DESTRUCTIVE-PREPEND PROHIBITED (P0, locked 2026-08-30): never open a file for writing/truncation in the same expression or pipeline that still READS that file. open(p, "w") truncates BEFORE the argument open(p).read() runs (destroyed 5876 lines of PR_ARC_INVENTORY.md in #1081); "sort f > f" is the same defect. Prepend = read into a variable, compose in memory, THEN write — or use the Edit tool. Mandatory post-check after every ledger write: wc -l; an append-only file that got SHORTER is always a defect. Law: .claude/knowledge/never-truncate-a-file-you-still-need-to-read.md'

emit() {
  jq -n --arg c "$RULE" \
    '{hookSpecificOutput: {hookEventName: "PreToolUse", additionalContext: $c}}'
}

emit_prepend() {
  jq -n --arg c "$PREPEND_RULE" \
    '{hookSpecificOutput: {hookEventName: "PreToolUse", additionalContext: $c}}'
}

# A numeric slice of a SOURCE file is the one artifact with no semantic
# boundary: `head -100 foo.rs` can stop just before the decisive `impl`,
# `tail` can separate a definition from its invariant, and `sed -n '120,180p'`
# looks precise while being an arbitrary cut. FIRST-HAND SOURCE LAW rule 3.
#
# Scoped to SOURCE INSPECTION, deliberately: limiting a non-search command's
# DISPLAY (`cargo test 2>&1 | tail -30`) does not fabricate a false semantic
# boundary -- the producer's own output is ephemeral process output, not a
# source file. It is allowed, but it is display only: the guarded-executor
# contract requires the complete output be retained and the FIRST relevant
# diagnostic block read in full when a command fails. A deny that fired on
# every build command would be worked around within the hour and would then
# guard nothing.
SLICE_DENY='VERBOTEN (FIRST-HAND SOURCE LAW, Regel 3): sed/head/tail/awk auf eine QUELLDATEI. Eine numerische Scheibe hat keine semantische Grenze -- `head -100 x.rs` endet womoeglich direkt vor dem entscheidenden impl, `tail` trennt Definition und Invariante, `sed -n 120,180p` sieht praezise aus und ist ein willkuerlicher Schnitt. Stattdessen: Grep/Glob lokalisiert das Symbol, dann Read auf das VOLLSTAENDIGE semantische Element (und bei Teilausgabe vom exakten naechsten Offset weiterlesen, niemals die ungesehene Mitte erraten). Output-Limitierung eines Nicht-Such-Kommandos (cargo ... | tail -30) bleibt erlaubt. Gesetz: .claude/knowledge/FIRST-HAND-SOURCE-LAW.md'

# Capping a SEARCH result is how a truncated result set masquerades as a
# complete one -- the shape behind every "no consumer" claim in this repo's
# correction history. The Grep tool's own `head_limit` reports the cap;
# `| head` hides it.
CAP_DENY='VERBOTEN (FIRST-HAND SOURCE LAW, Regel 9): eine SUCHE in head/tail/sed/awk pipen. Das kappt eine Beweismenge und laesst ein abgeschnittenes Ergebnis wie ein vollstaendiges aussehen -- genau die Form hinter jeder "kein Consumer"-Behauptung in der Korrekturgeschichte dieses Repos. Stattdessen: das Grep-Tool mit `head_limit` (das die Kappung MELDET), oder ungekappt suchen und den Suchraum benennen. Gesetz: .claude/knowledge/FIRST-HAND-SOURCE-LAW.md'

emit_deny() {
  jq -n --arg c "$1" \
    '{hookSpecificOutput: {hookEventName: "PreToolUse", permissionDecision: "deny", permissionDecisionReason: $c}}'
}

# Source/config extensions only. Scratch and temp outputs are not source, so
# `head -1 /tmp/out.txt` is none of this hook's business.
SRC_EXT='\.(rs|toml|lock|md|py|c|cc|cpp|h|hpp|java|kt|ts|tsx|js|mjs|json|ya?ml|sql|proto|sh|surql|ttl)'
SEARCH_CMD='(grep|rg|ugrep|egrep|fgrep|find|fd|ls)'

# FIRST-HAND SOURCE LAW §G: human authorization is PROVENANCE, NOT VALIDATION.
# These four are not technical status labels, so an edit may not INTRODUCE one
# into canonical material. Scoped to introduction deliberately: they occur in
# 73 / 43 / 8 / 4 files respectively (measured 2026-09-20), and a guard that
# fired on every edit to a file that already contains one would be unusable
# and worked around. Historical files are not this hook's business.
AUTHORITY_LABELS='operator-ruled|operator-pinned|operator-locked|operator-confirmed'
# A supersession note must be able to QUOTE the label it retires, so a line
# that also carries a quoting/supersession marker is allowed through.
QUOTE_MARKER='⊘|SUPERSEDED|superseded|previously|historical|formerly|was:'
AUTHORITY_DENY='VERBOTEN (FIRST-HAND SOURCE LAW §G): operator-ruled / operator-pinned / operator-locked / operator-confirmed sind KEINE technischen Status-Labels. HUMAN AUTHORIZATION IS PROVENANCE, NOT VALIDATION -- "der Nutzer hat X gewaehlt" wird nie "X ist technisch wahr" ohne unabhaengige Evidenz. Stattdessen ein evidenztragender Zustand: MEASURED (mit dem Kommando) | VERIFIED-IN-CODE (mit der Stelle) | TEST-PINNED | CURRENT-CONTRACT | WORKING-MODEL | HYPOTHESIS | PROPOSED | OPEN | DEFERRED | SUPERSEDED | REJECTED-BY-FALSIFIER. Fuer eine echte Nutzer-Entscheidung das Entscheidungs-Format: DECISION / SCOPE / BASIS / REVISIT WHEN -- Entscheidung und Messung sind zwei Felder, nie ein Label. Eine Supersession-Notiz DARF das alte Label zitieren (Zeile mit "⊘" / SUPERSEDED / previously / formerly / was:). Gesetz: .claude/knowledge/FIRST-HAND-SOURCE-LAW.md'

# True when $1 contains a line that introduces an authority label WITHOUT a
# quoting marker on that same line.
introduces_authority_label() {
  printf '%s' "$1" | grep -Ei "$AUTHORITY_LABELS" | grep -Eviq "$QUOTE_MARKER"
}
# Non-quoted label OCCURRENCES, for comparing an edit's two sides. A blanket
# "old already had one" exemption let an edit ADD a label beside an existing
# one -- `operator-ruled` present, `operator-pinned` arriving, no denial, which
# is exactly the introduction the guard promises to block (codex P2 on #1254,
# reproduced before fixing).
count_authority_labels() {
  printf '%s' "$1" | grep -Eiv "$QUOTE_MARKER" | grep -Eio "$AUTHORITY_LABELS" | wc -l | tr -d ' '
}
SLICER='(sed|head|tail|awk)'

case "$tool" in
  Grep)
    emit
    ;;
  Edit)
    # Only canonical prose/source carries these labels; skip anything else.
    path="$(printf '%s' "$input" | jq -r '.tool_input.file_path // ""')"
    if printf '%s' "$path" | grep -Eq '\.(md|rs)$'; then
      new="$(printf '%s' "$input" | jq -r '.tool_input.new_string // ""')"
      old="$(printf '%s' "$input" | jq -r '.tool_input.old_string // ""')"
      # INTRODUCTION only, measured per OCCURRENCE: more non-quoted labels
      # after than before. Comparing counts (not mere presence) is what stops
      # a label riding in beside one that was already there.
      if introduces_authority_label "$new" \
         && [ "$(count_authority_labels "$new")" -gt "$(count_authority_labels "$old")" ]; then
        emit_deny "$AUTHORITY_DENY"
      fi
    fi
    ;;
  MultiEdit)
    # Same guard as Edit, per edit in the batch: a MultiEdit that introduces a
    # label must not slip past because the matcher only named Edit/Write
    # (CodeRabbit on #1254). Compared per-edit so one edit cannot be excused by
    # another edit's pre-existing label.
    path="$(printf '%s' "$input" | jq -r '.tool_input.file_path // ""')"
    if printf '%s' "$path" | grep -Eq '\.(md|rs)$'; then
      n="$(printf '%s' "$input" | jq -r '.tool_input.edits | length // 0')"
      i=0
      while [ "$i" -lt "${n:-0}" ]; do
        new="$(printf '%s' "$input" | jq -r ".tool_input.edits[$i].new_string // \"\"")"
        old="$(printf '%s' "$input" | jq -r ".tool_input.edits[$i].old_string // \"\"")"
        if introduces_authority_label "$new" \
           && [ "$(count_authority_labels "$new")" -gt "$(count_authority_labels "$old")" ]; then
          emit_deny "$AUTHORITY_DENY"
        fi
        i=$((i + 1))
      done
    fi
    ;;
  Write)
    path="$(printf '%s' "$input" | jq -r '.tool_input.file_path // ""')"
    if printf '%s' "$path" | grep -Eq '\.(md|rs)$'; then
      content="$(printf '%s' "$input" | jq -r '.tool_input.content // ""')"
      if introduces_authority_label "$content"; then
        emit_deny "$AUTHORITY_DENY"
      fi
    fi
    ;;
  Bash)
    cmd="$(printf '%s' "$input" | jq -r '.tool_input.command // ""')"
    # Normalized copy, for MATCHING ONLY (never for execution or display).
    # Two measured bypasses, both codex P2 on #1254, both reproduced first:
    #   * a pipeline written across lines -- `rg ... \<newline> | head -20` --
    #     was invisible to the capped-search branch, because grep -E works a
    #     line at a time and `.*` never spans a newline;
    #   * a quoted operand -- `head -20 "src/lib.rs"` -- escaped the slice
    #     branch, because the extension was followed by a quote instead of
    #     whitespace-or-end.
    # Folding newlines to spaces and dropping shell quotes/continuations makes
    # both read like the bare forms the patterns already catch.
    scan="$(printf '%s' "$cmd" | tr '\n' ' ' | sed 's/[\"'"'"'\\]//g')"
    # Destructive-prepend shape: an open-for-write and a .read() of a file in
    # the same command (Python one-liner or heredoc). Heuristic, non-blocking
    # — false positives only cost an injected reminder.
    if printf '%s' "$cmd" | grep -Eq 'open\([^)]*,[[:space:]]*\\*['"'"'"]w' \
       && printf '%s' "$cmd" | grep -q '\.read()'; then
      emit_prepend
    # Match grep/rg/sed/tail/head as a command word (start, or after a
    # pipe/semicolon/&&/whitespace), not as a substring of another word.
    # DENY 1 -- a slicer whose argument list names a source file, and which is
    # not reading from a pipe. `cmd` is split on pipes so `cargo x | tail -30`
    # is judged on the `tail -30` segment alone (no file argument -> allowed).
    elif printf '%s' "$scan" | tr '|;' '\n\n' \
         | grep -Eq "(^|[[:space:]])$SLICER([[:space:]]+-[^[:space:]]+)*[[:space:]]+([^[:space:]]*[[:space:]]+)*[^[:space:]]*$SRC_EXT([[:space:]]|$)"; then
      emit_deny "$SLICE_DENY"
    # DENY 2 -- a search piped into a slicer: the cap that hides itself.
    elif printf '%s' "$scan" \
         | grep -Eq "(^|[|&;]|[[:space:]])$SEARCH_CMD([[:space:]]|$).*\\|[[:space:]]*$SLICER([[:space:]]|$)"; then
      emit_deny "$CAP_DENY"
    # Otherwise: non-blocking injection, as before.
    elif printf '%s' "$scan" | grep -Eq '(^|[|&;]|[[:space:]])(grep|rg|ugrep|sed|tail|head|awk)([[:space:]]|$)'; then
      emit
    fi
    ;;
esac

exit 0
