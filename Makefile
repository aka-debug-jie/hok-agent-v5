SYSTEM_PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
PYTHON ?= $(if $(wildcard $(VENV_PYTHON)),$(VENV_PYTHON),$(SYSTEM_PYTHON))

.PHONY: install lint typecheck test safety check validate integrity env-smoke env-benchmark preflight

install:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check src tests

typecheck:
	$(PYTHON) -m mypy src

test:
	$(PYTHON) -m pytest

safety:
	$(PYTHON) -m hok_agent safety-scan --root .

check: lint typecheck test safety integrity

validate:
	$(PYTHON) -m hok_agent validate-config --config-dir configs

env-smoke:
	$(PYTHON) -m hok_agent env-smoke --config configs/run_smoke_v1.yaml

env-benchmark:
	$(PYTHON) -m hok_agent env-benchmark --episodes 100

preflight:
	$(PYTHON) -m hok_agent preflight --probe-upstream

integrity:
	$(PYTHON) -m hok_agent package-integrity --root .
