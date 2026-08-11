PYTHON ?= python

.PHONY: install lint typecheck test safety check env-smoke env-benchmark preflight

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

check: lint typecheck test safety

env-smoke:
	$(PYTHON) -m hok_agent env-smoke --config configs/run_smoke_v1.yaml

env-benchmark:
	$(PYTHON) -m hok_agent env-benchmark --episodes 100

preflight:
	$(PYTHON) -m hok_agent preflight --probe-upstream
