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

set -euo pipefail

input="$(cat)"

tool="$(printf '%s' "$input" | jq -r '.tool_name // ""')"

RULE='ANTI-MUSTER-REGEL (Operator-Direktive): Grep/grep/rg/sed/tail/head sind NUR schnelle Discovery-Suche ueber den kompletten Corpus (ein Symbol/eine Datei lokalisieren) — NIEMALS Ersatz fuers Verstehen. Auf einen Treffer NICHT handeln (editieren, loeschen, beurteilen, "verstanden" behaupten), bevor die betroffene Datei VOLLSTAENDIG mit dem Read-Tool gelesen wurde. Verstehen = ganzes Read, kein Snippet. (Grund: geloeschter Code, der nur gemustert, nie gelesen wurde.)'

emit() {
  jq -n --arg c "$RULE" \
    '{hookSpecificOutput: {hookEventName: "PreToolUse", additionalContext: $c}}'
}

case "$tool" in
  Grep)
    emit
    ;;
  Bash)
    cmd="$(printf '%s' "$input" | jq -r '.tool_input.command // ""')"
    # Match grep/rg/sed/tail/head as a command word (start, or after a
    # pipe/semicolon/&&/whitespace), not as a substring of another word.
    if printf '%s' "$cmd" | grep -Eq '(^|[|&;]|[[:space:]])(grep|rg|sed|tail|head)([[:space:]]|$)'; then
      emit
    fi
    ;;
esac

exit 0
