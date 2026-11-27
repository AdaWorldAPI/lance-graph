# 2026-11-27 — Codegraph: an unread body is UNKNOWN, not "calls nothing"

**Status:** TEST-PINNED (`crates/lance-graph-codegraph/tests/unread_bodies.rs`, 6 tests, 5 disable runs red). Board: EXO-229, EXO-230.

Measured on `tesseract-paperless-web` (axum; tesseract-rs `docs/PAPERLESS-KLICKWEGE-PARITY.md`):
every handler is an `async fn`, Charon `c69e231` does not translate coroutines, and ruff_charon
writes a `BODY_NOT_STRUCTURED` residual naming each such function (31 of 31 here: the 27 route
handlers and 4 async helpers). The reader kept only a per-crate count, so `callees(handler)` was
empty with no UNKNOWN, and a function called only from handlers had no callers.

**Change:** the residual subjects are kept per crate (`Lanes::unread_bodies_of`, refreshed with the
residual counts; the unchanged-file shortcut also compares the subjects) and become the
`Codegraph::unread_bodies` mask. `callers(g)` holds every unread body as UNKNOWN (it may call `g`,
itself included); `callees(f)` of an unread `f` holds every stored function as UNKNOWN;
`impact(g)` holds every unread body other than `g` and everything that reaches one
(`Codegraph::unread_reach`). `impact_report` names that part `Gap::UnreadBodies` and no longer
counts it as `DynamicCalls`. No store format change: the residuals file already carried the names.

On the paperless store: `callers valid_key` now lists `desktop::desktop` and `desktop::folder`
(its real callers) as UNKNOWN; `sweep` empty answers 532 → 271.

OPEN: route URL literals are not facts (the `REFERENCES router → handler` edges already exist,
27/27, with call-site offsets); coroutine bodies are still not translated.
