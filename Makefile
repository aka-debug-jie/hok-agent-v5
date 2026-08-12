.PHONY: install lint typecheck test check accept accept-v2
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

install:
	$(PYTHON) -m pip install -e '.[dev,bc]'

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

accept-v2:
	$(PYTHON) -m hok_agent accept-minimal-v2-bc --output-dir /tmp/hok-agent-minimal-v2-bc
