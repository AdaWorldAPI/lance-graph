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
PORTABLE_RUSTFLAGS := $(if $(filter x86_64,$(shell uname -m)),RUSTFLAGS="-C target-cpu=x86-64-v4",)

publish:
	$(PORTABLE_RUSTFLAGS) maturin build --release --out dist --sdist
