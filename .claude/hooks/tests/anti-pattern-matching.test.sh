#!/usr/bin/env bash
# Two-sided test for .claude/hooks/anti-pattern-matching.sh.
#
# A guard that cannot fire and a guard that fires on everything are equally
# useless, so every case below is asserted in BOTH directions: the DENY rows
# prove the guard bites, the INJECT/SILENT rows prove it discriminates.
#
# Run:  bash .claude/hooks/tests/anti-pattern-matching.test.sh
# Exit: 0 all green, 1 any mismatch.
#
# Disable-verified (2026-09-20): removing the DENY-1 branch turns all five
# source-slice rows INJECT; removing DENY-2 turns all three capped-search rows
# INJECT. Re-run those two disables after any edit to the regexes.
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
HOOK=.claude/hooks/anti-pattern-matching.sh
fails=0

classify() {
  printf '%s' "{\"tool_name\":\"$1\",\"tool_input\":{\"command\":$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$2")}}" \
    | bash "$HOOK" | python3 -c '
import json,sys
raw = sys.stdin.read().strip()
if not raw:
    print("SILENT"); raise SystemExit
o = json.loads(raw)["hookSpecificOutput"]
print("DENY" if o.get("permissionDecision") == "deny" else "INJECT")'
}

t() {
  local want="$1" cmd="$2" got
  got="$(classify Bash "$cmd")"
  if [ "$got" = "$want" ]; then
    printf '  ok    %-7s %s\n' "$got" "$cmd"
  else
    printf '  FAIL  want=%s got=%s  %s\n' "$want" "$got" "$cmd"
    fails=$((fails + 1))
  fi
}

echo '### DENY -- a numeric slice of a SOURCE file (law rule 3)'
t DENY "head -100 crates/lance-graph-quack/src/lib.rs"
t DENY "sed -n '120,180p' Cargo.toml"
t DENY "tail -20 .claude/board/ISSUES.md"
t DENY "awk '/impl/' crates/foo/src/main.rs"
t DENY "sed -i 's/a/b/' Cargo.toml"

echo '### DENY -- a SEARCH capped by a slicer (law rule 9)'
t DENY "grep -rn CallMask crates/ | head -20"
t DENY "rg -l CallMask | head"
t DENY "find . -name '*.rs' | head -5"

echo '### ALLOW -- display limiting of a NON-search command (ephemeral process output)'
t INJECT "cargo test 2>&1 | tail -30"
t INJECT "cargo build --release | head -5"
t INJECT "head -1 /tmp/out.err"

echo '### ALLOW -- ordinary search: navigation is legitimate, injection only'
t INJECT "grep -rn CallMask crates/"
t INJECT "rg -l 'impl ClassView' crates/"

echo '### SILENT -- nothing to say'
t SILENT "ls crates/"
t SILENT "git log --oneline -1"
t SILENT "cargo test -p ogar-r2il --lib"

# ---- §G: authority labels may not be INTRODUCED into canonical material ----
edit() {
  local want="$1" path="$2" old="$3" new="$4" got
  got="$(printf '%s' "{\"tool_name\":\"Edit\",\"tool_input\":{\"file_path\":$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$path"),\"old_string\":$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$old"),\"new_string\":$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$new")}}" \
    | bash "$HOOK" | python3 -c '
import json,sys
raw = sys.stdin.read().strip()
if not raw:
    print("SILENT"); raise SystemExit
o = json.loads(raw)["hookSpecificOutput"]
print("DENY" if o.get("permissionDecision") == "deny" else "INJECT")')"
  if [ "$got" = "$want" ]; then printf '  ok    %-7s %s\n' "$got" "$5"
  else printf '  FAIL  want=%s got=%s  %s\n' "$want" "$got" "$5"; fails=$((fails + 1)); fi
}
write() {
  local want="$1" path="$2" content="$3" got
  got="$(printf '%s' "{\"tool_name\":\"Write\",\"tool_input\":{\"file_path\":$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$path"),\"content\":$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$content")}}" \
    | bash "$HOOK" | python3 -c '
import json,sys
raw = sys.stdin.read().strip()
if not raw:
    print("SILENT"); raise SystemExit
o = json.loads(raw)["hookSpecificOutput"]
print("DENY" if o.get("permissionDecision") == "deny" else "INJECT")')"
  if [ "$got" = "$want" ]; then printf '  ok    %-7s %s\n' "$got" "$4"
  else printf '  FAIL  want=%s got=%s  %s\n' "$want" "$got" "$4"; fails=$((fails + 1)); fi
}

echo '### DENY -- an edit that INTRODUCES an authority label (law §G)'
# MultiEdit: `edit`'s batch sibling. Added with the #1254 review fix -- the
# matcher named only Edit/Write, so a batch could carry a label past the guard.
# A leading untouched edit is included so one edit cannot be excused by another.
multiedit() {
  local want="$1" path="$2" old="$3" new="$4" got
  got="$(printf '%s' "{\"tool_name\":\"MultiEdit\",\"tool_input\":{\"file_path\":$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$path"),\"edits\":[{\"old_string\":\"untouched\",\"new_string\":\"untouched\"},{\"old_string\":$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$old"),\"new_string\":$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$new")}]}}" \
    | bash "$HOOK" | python3 -c '
import json,sys
raw = sys.stdin.read().strip()
if not raw:
    print("SILENT"); raise SystemExit
o = json.loads(raw)["hookSpecificOutput"]
print("DENY" if o.get("permissionDecision") == "deny" else "INJECT")')"
  if [ "$got" = "$want" ]; then printf '  ok    %-7s %s\n' "$got" "$5"
  else printf '  FAIL  want=%s got=%s  %s\n' "$want" "$got" "$5"; fails=$((fails + 1)); fi
}

edit DENY x.md "Status: WORKING-MODEL" "Status: operator-ruled"            "introduce operator-ruled"
edit DENY x.md "the pin"              "the operator-locked pin"            "introduce operator-locked"
write DENY n.md "# New

Status: operator-pinned, 2026-09-20.
"          "new file with operator-pinned"

# An ALLOWED Edit/Write emits nothing: injection is a Grep/Bash-only
# behaviour, so "allowed" reads as SILENT here, never INJECT.
echo '### ALLOW (silent) -- label already present, or quoted by a supersession note'
edit SILENT x.md "operator-ruled 2026-07-02" "operator-ruled 2026-07-02, now measured" "already present: not an introduction"
edit SILENT x.md "the rule"           "⊘ previously operator-locked; now TEST-PINNED"  "quoted under a supersession marker"
edit SILENT x.md "a"                  "SUPERSEDED: the operator-confirmed wording"     "quoted under SUPERSEDED"

echo '### ALLOW -- an evidence-bearing state is the whole point'
edit SILENT x.md "Status: OPEN"       "Status: MEASURED (cargo metadata, exit 0)"      "MEASURED"
edit SILENT x.md "a"                  "DECISION: keep path form\nBASIS: offline cost" "DECISION record"

echo '### ALLOW -- not canonical prose/source'
edit DENY x.md "was operator-ruled." "was operator-ruled. New: operator-pinned too." "a label ADDED beside an existing one is still an introduction"
multiedit DENY x.md "b" "b operator-locked" "MultiEdit introducing a label"
multiedit SILENT x.md "b" "b tidied" "MultiEdit with no label"

# A hook's stdout must be exactly ONE response document. Two violating edits in
# one batch used to emit TWO, and a concatenated pair parses as neither denial
# (CodeRabbit on #1254). This asserts the COUNT, not merely that a denial
# appeared -- the pre-existing rows above could not see the defect, because they
# carry one violating edit each. Disable-verified: removing `exit 0` from
# emit_deny makes this row report docs=2.
multiedit_two_violations() {
  local got
  got="$(printf '%s' '{"tool_name":"MultiEdit","tool_input":{"file_path":"x.md","edits":[{"old_string":"a","new_string":"a operator-ruled"},{"old_string":"b","new_string":"b operator-pinned"}]}}' \
    | bash "$HOOK" | python3 -c '
import json, sys
dec = json.JSONDecoder()
raw, i, docs, denies = sys.stdin.read(), 0, 0, 0
while i < len(raw):
    while i < len(raw) and raw[i].isspace():
        i += 1
    if i >= len(raw):
        break
    o, i = dec.raw_decode(raw, i)
    docs += 1
    if o["hookSpecificOutput"].get("permissionDecision") == "deny":
        denies += 1
print(f"docs={docs} denies={denies}")')"
  if [ "$got" = "docs=1 denies=1" ]; then printf '  ok    %-7s %s\n' "$got" "two violating edits -> ONE deny document"
  else printf '  FAIL  want=docs=1 denies=1 got=%s  %s\n' "$got" "two violating edits -> ONE deny document"; fails=$((fails + 1)); fi
}
multiedit_two_violations
write SILENT c.json '{"k":"operator-ruled"}'                                           "json is out of scope"

echo '### DENY -- review findings on #1254, each reproduced before it was fixed'
# codex P2: a quoted operand escaped the slice branch (the extension was
# followed by a quote, not whitespace-or-end). Both quote styles.
t DENY 'head -20 "src/lib.rs"'
t DENY "sed -n 1,50p 'crates/x/src/lib.rs'"
# codex P2: a pipeline written across lines was invisible to the capped-search
# branch -- grep -E works one line at a time and `.*` never spans a newline.
t DENY 'rg -n CallMask crates/ \
  | head -20'

echo '### ALLOW -- the carve-out those three fixes must not eat'
t INJECT 'cargo test 2>&1 | tail -30'
t INJECT 'cargo test > /tmp/probe.log 2>&1; tail -30 /tmp/probe.log'

echo '### the Grep TOOL always carries the law'
got="$(classify Grep '')"
if [ "$got" = "INJECT" ]; then printf '  ok    %-7s %s\n' "$got" "(Grep tool)"; else
  printf '  FAIL  want=INJECT got=%s  (Grep tool)\n' "$got"; fails=$((fails + 1)); fi

echo
if [ "$fails" -eq 0 ]; then echo "ALL PASSED"; else echo "$fails FAILED"; fi
exit $((fails > 0))
