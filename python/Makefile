UV ?= uv
PYTHON ?= 3.11
PYTEST_ARGS ?= -vv -s

export PATH := $(HOME)/.local/bin:$(PATH)

.PHONY: venv install test lint clean publish

venv:
	$(UV) venv --python $(PYTHON) .venv

install:
	$(UV) pip install --project . --editable . --extra tests

test:
	$(UV) run --project . --with tests pytest $(PYTEST_ARGS) python/python/tests/test_graph.py

lint:
	$(UV) run --project . --with dev ruff check python
	$(UV) run --project . --with dev ruff format --check python
	$(UV) run --project . --with dev pyright

clean:
	rm -rf .venv
	rm -rf ../target

# The repo's .cargo/config.toml defaults to `native`, which would tune the
# wheel to the build host; production runs on x86-64-v4, so the wheel is
# built for exactly that. RUSTFLAGS replaces the config's flags; it is set
# only on an x86_64 host so an arm64 build keeps its own baseline.
#
# `--target <host triple>` is load-bearing: without it RUSTFLAGS also reaches
# build scripts and proc-macros (serde, pyo3, tokio), which run on this host,
# so a publishing machine without AVX-512 would die with SIGILL. With an
# explicit target the flags reach only the wheel's own code.
IS_X86_64 := $(filter x86_64,$(shell uname -m))
HOST_TRIPLE := $(shell rustc -vV | sed -n 's/^host: //p')
PORTABLE_RUSTFLAGS := $(if $(IS_X86_64),RUSTFLAGS="-C target-cpu=x86-64-v4",)
PORTABLE_TARGET := $(if $(IS_X86_64),--target $(HOST_TRIPLE),)

publish:
	$(PORTABLE_RUSTFLAGS) maturin build --release --out dist --sdist $(PORTABLE_TARGET)
