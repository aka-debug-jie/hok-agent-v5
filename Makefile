PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

.PHONY: install lint typecheck test check accept

install:
	$(PYTHON) -m pip install -e '.[dev]'

lint:
	$(PYTHON) -m ruff check src tests

typecheck:
	$(PYTHON) -m mypy src

test:
	$(PYTHON) -m pytest

check: lint typecheck test
	$(PYTHON) -m hok_agent check

accept:
	$(PYTHON) -m hok_agent accept-minimal-v1 --seed 101
