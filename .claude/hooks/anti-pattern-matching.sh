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

case "$tool" in
  Grep)
    emit
    ;;
  Bash)
    cmd="$(printf '%s' "$input" | jq -r '.tool_input.command // ""')"
    # Destructive-prepend shape: an open-for-write and a .read() of a file in
    # the same command (Python one-liner or heredoc). Heuristic, non-blocking
    # — false positives only cost an injected reminder.
    if printf '%s' "$cmd" | grep -Eq 'open\([^)]*,[[:space:]]*\\*['"'"'"]w' \
       && printf '%s' "$cmd" | grep -q '\.read()'; then
      emit_prepend
    # Match grep/rg/sed/tail/head as a command word (start, or after a
    # pipe/semicolon/&&/whitespace), not as a substring of another word.
    elif printf '%s' "$cmd" | grep -Eq '(^|[|&;]|[[:space:]])(grep|rg|sed|tail|head)([[:space:]]|$)'; then
      emit
    fi
    ;;
esac

exit 0
