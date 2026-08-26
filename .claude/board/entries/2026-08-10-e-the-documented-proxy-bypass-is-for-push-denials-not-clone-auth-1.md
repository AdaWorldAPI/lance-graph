## 2026-08-10 — E-THE-DOCUMENTED-PROXY-BYPASS-IS-FOR-PUSH-DENIALS-NOT-CLONE-AUTH-1

**Status:** FINDING `[G]` (reproduced and resolved in-session).

Cloning `AdaWorldAPI/{ecmwf-opendata,weatherbench2,arco-era5}` failed with
`fatal: could not read Username for 'https://github.com'` — which reads exactly like
a repo-scope denial, and two independent signals reinforced that misreading:

1. `mcp__claude-code-remote__{list_repos,add_repo}` genuinely are **not exposed** in
   this session (confirmed with two different ToolSearch queries), so "the repo is
   out of scope" was the available hypothesis; and
2. this workspace's own documented lesson (tesseract-rs `CLAUDE.md`, GitHub access
   matrix) says *"a 403 here is USUALLY THE PROXY — retest with the proxy bypassed"*,
   which sent the diagnosis further the wrong way. Bypassing the proxy failed too.

**Root cause: self-inflicted.** The agent proxy **already injects credentials** for
these repos. Passing an explicit `-c http.extraHeader="Authorization: Bearer $TOKEN"`
**overrode** the proxy's injected credential with a form GitHub's git endpoint
rejects. Plain `git clone`, proxy ON, **no explicit header**, works for all three.

**What broke the tie:** a REST probe — `HTTP 200` on all four repos
(`lance-graph` + the three new) proved the token had access, so the failure had to be
the *method*, not the scope.

**Rule:** the documented "bypass the proxy" reflex is for **push denials**; for
**clone auth**, adding an explicit `Authorization` header is the bug. Never hand-roll
credentials for a transport that already carries them. (Token discipline held
throughout — expanded inline via `${GH_TOKEN//\"/}`, never printed, and
`.git/config` verified free of credentials after cloning.)

Cross-ref: tesseract-rs `CLAUDE.md` § GitHub access matrix (the push-side half, which
remains correct).

