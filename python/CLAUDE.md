## P0 — AdaWorldAPI forks ONLY, NEVER crates.io upstream

**Always depend on the AdaWorldAPI fork of any crate that has one. NEVER use the
upstream crates.io version of a forked crate.** Non-negotiable; applies to every
`Cargo.toml` and every dependency decision in this repo. Every repo in this
workspace is local — prefer the local/fork source over the registry, always.
If a fork's coordinates are unknown, STOP and ask — never fall back to crates.io.

Use the makefile for most actions:

* Build: `maturin develop`
* Test: `make test`
* Run single test: `pytest python/tests/<test_file>.py::<test_name>`
* Doctest: `make doctest`
* Lint: `make lint`
* Format: `make format`
