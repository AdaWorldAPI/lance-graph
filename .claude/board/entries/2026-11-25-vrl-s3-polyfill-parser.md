# S3 — the polyfill parser (`vrl-vector-openobserve-census-v1`)

**Status:** In PR (after D-VEC-ENRICH-1).

DECISION (operator, 2026-11-25): the internal VRL, the polyfill, serves the internal mapping. Its parser lowers VRL text into the low-code BindStore surface (`Stmt` → `Recipe`) and has no dependency on upstream `vrl`. That Vector runs the same VRL is a coincidence we use through `lance-graph-vector`. So S3 is read in polyfill flavour:
- the split-based parser is replaced by a lexer in the polyfill;
- upstream's parser is never the front end, only a feature-gated oracle.

## What changed

`parse()` lexes the source once into tokens with spans: paths, identifiers, integers (a minus before a digit after an operator is a negative literal), string literals, operators, and statement ends at a newline or `;` outside braces. `#` comments run to the end of the line. It then parses statements from the tokens into the same `Stmt` forms, with the same refusals.

## Measured

`tests/parse_corpus`: 92 programs (every program string in the suites, plus lexical edge cases) are pinned against a committed snapshot, and `BLESS=1` regenerates it.

Against the old parser's snapshot, 8 lines changed:
- **3 lexical bugs fixed:**
  - a `;` inside a string literal;
  - a trailing `# comment` read as part of a path;
  - `.a<3` without spaces.
- **4 refusals reworded**, all on non-programs: `format!` templates, and `.e == false` outside an `if`. The old messages were split debris.
- **`;;` now refused** as `Computes("empty statement")`, because upstream refuses it. Before, it was accepted.

`tests/parser_oracle`, under `upstream-vector` and `upstream-o2`: every corpus program the polyfill accepts parses upstream, and lowers from upstream's own AST (`vrl::parser::ast`) to identical statements. That holds for at least 60 programs on both pins. The oracle found the `;;` divergence.

## Ablations

Both went red, then were restored:
- the oracle mapping `>` as `>=`;
- the empty-statement rule turned off.

OPEN:
- None for S3.
- Q1 (`==`/`!=` over optional fields) is unchanged by the parser.
