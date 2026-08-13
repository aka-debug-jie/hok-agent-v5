.PHONY: install lint typecheck test check storage-show storage-preflight storage-init accept accept-v2 pixel-smoke accept-v3 shadow-live-smoke alignment-smoke temporal-smoke rich-smoke accept-v7
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
RUN_PYTHON = env -u LD_LIBRARY_PATH $(PYTHON)
WZRY_DATA_ROOT ?= /media/hgdl1012/E/wzry-data
HOK_LARGE_ROOT ?= $(WZRY_DATA_ROOT)/hok-agent-v5
HOK_DATASETS_ROOT ?= $(HOK_LARGE_ROOT)/datasets
HOK_CHECKPOINTS_ROOT ?= $(HOK_LARGE_ROOT)/checkpoints
HOK_RUNS_ROOT ?= $(HOK_LARGE_ROOT)/runs
HOK_CACHE_ROOT ?= $(HOK_LARGE_ROOT)/cache
HOK_AUDIT_ROOT ?= $(HOK_LARGE_ROOT)/audit
HOK_STAGING_ROOT ?= $(HOK_LARGE_ROOT)/staging

install:
	$(PYTHON) -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
	$(RUN_PYTHON) -m pip install -e '.[dev,bc,vision,shadow]'

lint:
	$(RUN_PYTHON) -m ruff check src tests

typecheck:
	$(RUN_PYTHON) -m mypy src

test:
	$(RUN_PYTHON) -m pytest

check: lint typecheck test
	$(RUN_PYTHON) -m hok_agent check

storage-show:
	@echo "WZRY_DATA_ROOT=$(WZRY_DATA_ROOT)"
	@echo "HOK_LARGE_ROOT=$(HOK_LARGE_ROOT)"
	@echo "datasets=$(HOK_DATASETS_ROOT)"
	@echo "checkpoints=$(HOK_CHECKPOINTS_ROOT)"
	@echo "runs=$(HOK_RUNS_ROOT)"
	@echo "cache=$(HOK_CACHE_ROOT)"
	@echo "audit=$(HOK_AUDIT_ROOT)"
	@echo "staging=$(HOK_STAGING_ROOT)"

storage-preflight:
	@findmnt -rn -O rw -T "$(WZRY_DATA_ROOT)" >/dev/null || { echo "storage is not mounted read-write: $(WZRY_DATA_ROOT)" >&2; exit 2; }
	@test -d "$(WZRY_DATA_ROOT)" || { echo "storage root is missing: $(WZRY_DATA_ROOT)" >&2; exit 2; }
	@test -w "$(WZRY_DATA_ROOT)" || { echo "storage root is not writable: $(WZRY_DATA_ROOT)" >&2; exit 2; }

storage-init: storage-preflight
	mkdir -p "$(HOK_DATASETS_ROOT)" "$(HOK_CHECKPOINTS_ROOT)" "$(HOK_RUNS_ROOT)" "$(HOK_CACHE_ROOT)" "$(HOK_AUDIT_ROOT)" "$(HOK_STAGING_ROOT)"

accept:
	$(RUN_PYTHON) -m hok_agent accept-minimal-v1 --seed 101

accept-v2:
	$(RUN_PYTHON) -m hok_agent accept-minimal-v2-bc --output-dir /tmp/hok-agent-minimal-v2-bc-$$$$

pixel-smoke:
	$(RUN_PYTHON) -m hok_agent accept-pixel-v3 --smoke --device cpu

accept-v3: storage-init
	$(RUN_PYTHON) -m hok_agent accept-pixel-v3 --device cuda --output-dir "$(HOK_RUNS_ROOT)/pixel-v3-v1"

shadow-live-smoke:
	$(RUN_PYTHON) -m pytest -q tests/test_capture.py

alignment-smoke:
	$(RUN_PYTHON) -m hok_agent alignment-v5-smoke

temporal-smoke:
	$(RUN_PYTHON) -m hok_agent temporal-v6-smoke

rich-smoke:
	$(RUN_PYTHON) -m hok_agent accept-rich-v7 --smoke --device cpu

accept-v7: storage-init
	$(RUN_PYTHON) -m hok_agent accept-rich-v7 --device cuda --output-dir "$(HOK_RUNS_ROOT)/rich-v7-v1"
