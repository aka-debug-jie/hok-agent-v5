.PHONY: install lint typecheck test check accept accept-v2 pixel-smoke accept-v3
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
RUN_PYTHON = env -u LD_LIBRARY_PATH $(PYTHON)

install:
	$(PYTHON) -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
	$(RUN_PYTHON) -m pip install -e '.[dev,bc,vision]'

lint:
	$(RUN_PYTHON) -m ruff check src tests

typecheck:
	$(RUN_PYTHON) -m mypy src

test:
	$(RUN_PYTHON) -m pytest

check: lint typecheck test
	$(RUN_PYTHON) -m hok_agent check

accept:
	$(RUN_PYTHON) -m hok_agent accept-minimal-v1 --seed 101

accept-v2:
	$(RUN_PYTHON) -m hok_agent accept-minimal-v2-bc --output-dir /tmp/hok-agent-minimal-v2-bc

pixel-smoke:
	$(RUN_PYTHON) -m hok_agent accept-pixel-v3 --smoke --device cpu

accept-v3:
	$(RUN_PYTHON) -m hok_agent accept-pixel-v3 --device cuda --output-dir runs/pixel-v3-v1
