#!/usr/bin/env bash
# PreToolUse(Bash) enforcer — grep / sed / head / tail are forbidden in this workspace.
#
# Flags them ONLY in command position (the leading token of any ; | & ( ) segment),
# so filenames like `head.txt`, args like `wc -l grep.log`, and tools like `ripgrep`
# are NOT false-positives. Uses no forbidden tool itself (jq + tr + bash builtins).
#
# Returns a PreToolUse "deny" decision (JSON on stdout) when a forbidden tool is the
# command. Fail-open: if jq is missing or input is empty, the call is allowed.
set -uo pipefail

cmd=$(jq -r '.tool_input.command // empty' 2>/dev/null) || exit 0
[ -z "$cmd" ] && exit 0

hit=""
while IFS= read -r seg; do
  seg="${seg#"${seg%%[![:space:]]*}"}"   # left-trim the segment
  word="${seg%%[[:space:]]*}"            # leading token of the segment
  case "$word" in
    grep|egrep|fgrep|sed|head|tail) hit="$word"; break ;;
  esac
done < <(printf '%s\n' "$cmd" | tr ';|&()' '\n')

[ -z "$hit" ] && exit 0

printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"FORBIDDEN: %s is banned in this workspace. Use the Read tool (not cat/head/tail), Edit (not sed), and Glob/Read (not grep). This rule is hook-enforced."}}\n' "$hit"
exit 0
