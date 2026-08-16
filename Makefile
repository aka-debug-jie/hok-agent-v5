.PHONY: install lint typecheck test check storage-show storage-preflight storage-init accept accept-v2 pixel-smoke accept-v3 shadow-live-smoke mobile-testbed-smoke t8-smoke t8-data-smoke t8-shadow-smoke t8-contract-smoke t8-v2-contract-smoke t8-v2-keyboard-smoke t8-v2-live-smoke t8-v2-live-inverse-probe t8-v2-live-collect t8-v2-live-pilot-freeze t8-v2-live-freeze t8-v2-live-pilot t8-video-three-class-pilot t8-video-retrospective-materialize t8-video-retrospective-pilot t8-video-retrospective-roi-evaluate t8-retrospective-v1-verify t8-retrospective-v1-batch t8-retrospective-v2-calibrate t8-causal-video-materialize t8-causal-video-pilot t8-causal-video-diagnose t8-causal-pixel-materialize t8-causal-pixel-probe t8-visual-teacher-replay t8-visible-onset-audit t8-combat-causal-materialize t8-combat-causal-pilot t8-combat-causal-diagnostic-materialize t8-combat-causal-diagnostic-pilot t8-v25-dry-run t8-v25-probe-20 t8-v25-smoke-60 t8-v25-collect t8-v25-pilot-freeze t8-v25-freeze t8-v25-pilot t8-v26-train-seed t8-v26-select t8-v26-evaluate-offline t8-v26-shadow t8-v26-shadow-replay t8-v26-execute-probe t8-v27-calibration-pilot t8-v27-freeze t8-v3-state-materialize t8-v3-state-train t8-v3-hybrid-replay t8-v2-touch-smoke t8-v2-collect t8-v2-freeze t8-v2-adapt t8-v2-pilot alignment-smoke temporal-smoke rich-smoke accept-v7 v5-source-produce v5-build-cohort v5-ingest-zero-label v5-validate-zero-target v5-freeze-training-config v5-train-simsiam-adapted v5-model-predict v5-materialize-pseudo v5-run-mean-teacher-round v6-zero-smoke
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
RUN_PYTHON = env -u LD_LIBRARY_PATH $(PYTHON)
WZRY_DATA_ROOT ?= $(CURDIR)/.local-data
HOK_LARGE_ROOT ?= $(WZRY_DATA_ROOT)/hok-agent-v5
HOK_DATASETS_ROOT ?= $(HOK_LARGE_ROOT)/datasets
HOK_CHECKPOINTS_ROOT ?= $(HOK_LARGE_ROOT)/checkpoints
HOK_RUNS_ROOT ?= $(HOK_LARGE_ROOT)/runs
HOK_CACHE_ROOT ?= $(HOK_LARGE_ROOT)/cache
HOK_AUDIT_ROOT ?= $(HOK_LARGE_ROOT)/audit
HOK_STAGING_ROOT ?= $(HOK_LARGE_ROOT)/staging
V5_TRAINING_BATCH_SIZE ?= 768
V5_LEARNING_RATE ?= 3e-4
V5_WEIGHT_DECAY ?= 1e-4
V5_TRAINING_EPOCHS ?= 50
V5_MEAN_TEACHER_EPOCHS ?= 20
V5_SIMSIAM_DEVICE ?= cpu
V5_MEAN_TEACHER_DEVICE ?= cpu
V5_PREFETCH_SHARDS ?= 4
T8_SERIAL ?=
T8_VIDEO_NODE ?=
T8_LAYOUT ?=configs/mobile_testbed_layout.local.json
T8_V2_DATASET ?=$(HOK_DATASETS_ROOT)/t8-demonstrations-v2/formal-auto-v1
T8_V2_SESSION ?=
T8_V21_SESSION ?=
T8_V2_ADAPTER ?=
T8_V21_ADAPTER ?=
T8_TOUCH_DEVICE ?=
T8_TOUCH_CALIBRATION ?=$(HOK_DATASETS_ROOT)/t8-demonstrations-v2/touch-calibration-v2.json
T8_INVERSE_REPORT ?=$(HOK_RUNS_ROOT)/t8-policy-v2.1/inverse-probe-three-class-v1-1786784473/report.json
T8_RETRO_BASELINE ?=$(HOK_AUDIT_ROOT)/t8-retrospective-v1
T8_RETRO_TARGET ?=$(HOK_DATASETS_ROOT)/v5-target-file-atomic-v2
T8_RETRO_SPLIT ?=train
T8_RETRO_OUTPUT ?=$(HOK_DATASETS_ROOT)/t8-retrospective-events-v1/$(T8_RETRO_SPLIT)
T8_RETRO_CALIBRATION_OUTPUT ?=$(HOK_AUDIT_ROOT)/t8-retrospective-v2-class-thresholds-v1
T8_CAUSAL_DATASET ?=$(HOK_DATASETS_ROOT)/t8-causal-video-four-class-v1
T8_CAUSAL_RUN ?=$(HOK_RUNS_ROOT)/t8-policy-v2.1/causal-four-class-seed0-v1
T8_CAUSAL_DIAGNOSTIC ?=$(HOK_RUNS_ROOT)/t8-policy-v2.1/causal-learnability-diagnostic-v1
T8_CAUSAL_ADAPTER ?=$(HOK_CHECKPOINTS_ROOT)/t8-policy-v2/video-adapter-v1/adapter-epoch-3.safetensors
T8_CAUSAL_PIXEL_DATASET ?=$(HOK_DATASETS_ROOT)/t8-causal-pixel-v2.2
T8_CAUSAL_PIXEL_RUN ?=$(HOK_RUNS_ROOT)/t8-policy-v2.2/causal-pixel-probe-seed0-v1
T8_VISUAL_TEACHER_REPLAY ?=$(HOK_RUNS_ROOT)/t8-policy-v2.3/visual-teacher-replay-v1
T8_VISIBLE_ONSET_AUDIT ?=$(HOK_AUDIT_ROOT)/t8-visible-onset-v1
T8_COMBAT_CAUSAL_DATASET ?=$(HOK_DATASETS_ROOT)/t8-combat-causal-v2.4
T8_COMBAT_CAUSAL_RUN ?=$(HOK_RUNS_ROOT)/t8-policy-v2.4/combat-causal-seed0-v1
T8_COMBAT_CAUSAL_DIAGNOSTIC_DATASET ?=$(HOK_DATASETS_ROOT)/t8-combat-causal-v2.4-diagnostic
T8_COMBAT_CAUSAL_DIAGNOSTIC_RUN ?=$(HOK_RUNS_ROOT)/t8-policy-v2.4/combat-causal-diagnostic-seed0-v1
T8_V25_ROOT ?=$(HOK_DATASETS_ROOT)/t8-demonstrations-v2.6/rgb-conditioned-v2
T8_V25_SESSION ?=
T8_V25_TEACHER ?=$(T8_VISUAL_TEACHER_REPLAY)/report.json
T8_V25_PILOT_SPLIT ?=$(T8_V25_ROOT)/t8-v2.6-pilot-split.json
T8_V25_SPLIT ?=$(T8_V25_ROOT)/t8-v2.6-split.json
T8_V25_RUN ?=$(HOK_RUNS_ROOT)/t8-policy-v2.6/pilot-seed0-v5-gate-065
T8_V26_FORMAL_RUN_ROOT ?=$(HOK_RUNS_ROOT)/t8-policy-v2.6/formal-v1
T8_V26_SHADOW_RUN ?=$(T8_V26_FORMAL_RUN_ROOT)/shadow-v2
T8_V27_TRAIN_SESSION ?=$(T8_V25_ROOT)/diagnostics/smoke-60-1786881861
T8_V27_DEV_SESSION ?=$(T8_V25_ROOT)/diagnostics/probe-20-1786881784
T8_V27_RUN ?=$(HOK_RUNS_ROOT)/t8-policy-v2.7/current-scene-head-pilot-v1
T8_V27_FREEZE ?=$(HOK_AUDIT_ROOT)/t8-v2.7-frozen-failure-v1
T8_V3_DATASET ?=$(HOK_DATASETS_ROOT)/t8-video-state-v3
T8_V3_RUN ?=$(HOK_RUNS_ROOT)/t8-policy-v3/state-seed0-v1
T8_V3_REPLAY ?=$(T8_V3_RUN)/hybrid-replay-v1
T8_SEED ?=

install:
	$(PYTHON) -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
	$(RUN_PYTHON) -m pip install -e '.[dev,bc,vision,shadow,preingest]'

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

storage-init:
	mkdir -p "$(WZRY_DATA_ROOT)"
	@findmnt -rn -O rw -T "$(WZRY_DATA_ROOT)" >/dev/null || { echo "storage is not mounted read-write: $(WZRY_DATA_ROOT)" >&2; exit 2; }
	mkdir -p "$(HOK_DATASETS_ROOT)" "$(HOK_CHECKPOINTS_ROOT)" "$(HOK_RUNS_ROOT)" "$(HOK_CACHE_ROOT)" "$(HOK_AUDIT_ROOT)" "$(HOK_STAGING_ROOT)"

accept:
	$(RUN_PYTHON) -m hok_agent accept-minimal-v1 --seed 101

accept-v2:
	$(RUN_PYTHON) -m hok_agent accept-minimal-v2-bc --output-dir /tmp/hok-agent-minimal-v2-bc-$$(date +%s%N)

pixel-smoke:
	$(RUN_PYTHON) -m hok_agent accept-pixel-v3 --smoke --device cpu

accept-v3: storage-init
	$(RUN_PYTHON) -m hok_agent accept-pixel-v3 --device cuda --output-dir "$(HOK_RUNS_ROOT)/pixel-v3-v1"

shadow-live-smoke:
	$(RUN_PYTHON) -m pytest -q tests/test_capture.py

mobile-testbed-smoke:
	$(RUN_PYTHON) -m pytest -q tests/test_mobile_testbed.py

t8-smoke:
	$(RUN_PYTHON) -m hok_agent t8-smoke

t8-data-smoke:
	$(RUN_PYTHON) -m pytest -q tests/test_mobile_testbed.py tests/test_t8.py

t8-shadow-smoke:
	$(RUN_PYTHON) -m pytest -q tests/test_t8_shadow.py

t8-contract-smoke:
	$(RUN_PYTHON) -m pytest -q tests/test_mobile_testbed.py tests/test_t8.py tests/test_t8_shadow.py

t8-v2-contract-smoke:
	$(RUN_PYTHON) -m pytest -q tests/test_mobile_testbed.py tests/test_t8.py

t8-v2-keyboard-smoke: storage-preflight
	@test -n "$(T8_SERIAL)" || { echo "set T8_SERIAL" >&2; exit 2; }
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" $(RUN_PYTHON) -m hok_agent mobile-demonstrate-keyboard-v2 --serial "$(T8_SERIAL)" --layout "$(T8_LAYOUT)" --video-node "$(T8_VIDEO_NODE)" --run-seconds 60 --output-dir "$(HOK_DATASETS_ROOT)/t8-demonstrations-v2/smoke-keyboard-v2-$$(date +%s)"

t8-v2-live-smoke: storage-preflight
	@test -n "$(T8_SERIAL)" || { echo "set T8_SERIAL" >&2; exit 2; }
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" $(RUN_PYTHON) -m hok_agent mobile-demonstrate-keyboard-v2-live --serial "$(T8_SERIAL)" --layout "$(T8_LAYOUT)" --run-seconds 20 --countdown-seconds 0 --diagnostic-control-smoke --output-dir "$(HOK_DATASETS_ROOT)/t8-demonstrations-v2.1/diagnostics/control-smoke-$$(date +%s)"

t8-v2-live-inverse-probe: storage-preflight
	@test -n "$(T8_SERIAL)" || { echo "set T8_SERIAL" >&2; exit 2; }
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" $(RUN_PYTHON) -m hok_agent mobile-demonstrate-keyboard-v2-live --serial "$(T8_SERIAL)" --layout "$(T8_LAYOUT)" --run-seconds 120 --countdown-seconds 0 --diagnostic-inverse-probe --output-dir "$(HOK_DATASETS_ROOT)/t8-demonstrations-v2.1/diagnostics/inverse-probe-$$(date +%s)"

t8-v2-live-collect: storage-preflight
	@test -n "$(T8_SERIAL)$(T8_V21_SESSION)" || { echo "set T8_SERIAL and T8_V21_SESSION=session-NNN" >&2; exit 2; }
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" $(RUN_PYTHON) -m hok_agent mobile-demonstrate-keyboard-v2-live --serial "$(T8_SERIAL)" --layout "$(T8_LAYOUT)" --run-seconds 300 --formal-session --output-dir "$(HOK_DATASETS_ROOT)/t8-demonstrations-v2.1/$(T8_V21_SESSION)"

t8-v2-live-freeze: storage-preflight
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" $(RUN_PYTHON) -m hok_agent t8-v2-live-freeze-split --dataset-root "$(HOK_DATASETS_ROOT)/t8-demonstrations-v2.1" --output "$(HOK_DATASETS_ROOT)/t8-demonstrations-v2.1/t8-v2.1-split.json"

t8-v2-live-pilot-freeze: storage-preflight
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" $(RUN_PYTHON) -m hok_agent t8-v2-live-pilot-freeze --dataset-root "$(HOK_DATASETS_ROOT)/t8-demonstrations-v2.1" --output "$(HOK_DATASETS_ROOT)/t8-demonstrations-v2.1/t8-v2.1-pilot-split.json"

t8-v2-live-pilot: storage-preflight
	@test -n "$(T8_V21_ADAPTER)" || { echo "set T8_V21_ADAPTER" >&2; exit 2; }
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" CUBLAS_WORKSPACE_CONFIG=:4096:8 $(RUN_PYTHON) -m hok_agent t8-v2-live-pilot --dataset-root "$(HOK_DATASETS_ROOT)/t8-demonstrations-v2.1" --split "$(HOK_DATASETS_ROOT)/t8-demonstrations-v2.1/t8-v2.1-pilot-split.json" --adapter-checkpoint "$(T8_V21_ADAPTER)" --output-dir "$(HOK_RUNS_ROOT)/t8-policy-v2.1/pilot-seed0-v1" --device cuda --batch-size 32

t8-video-three-class-pilot: storage-preflight
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" CUBLAS_WORKSPACE_CONFIG=:4096:8 $(RUN_PYTHON) -m hok_agent t8-video-three-class-pilot --dataset-root "$(HOK_DATASETS_ROOT)/t8-video-combat-three-class-v1" --adapter-checkpoint "$(HOK_CHECKPOINTS_ROOT)/t8-policy-v2/video-adapter-v1/adapter-epoch-3.safetensors" --output-dir "$(HOK_RUNS_ROOT)/t8-policy-v2.1/video-three-class-pilot-seed0-v1" --device cuda --batch-size 64

t8-video-retrospective-materialize: storage-preflight
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" $(RUN_PYTHON) -m hok_agent t8-video-three-class-materialize --source-dir "$(HOK_DATASETS_ROOT)/t8-video-combat-pseudolabel-v1" --inverse-report "$(T8_INVERSE_REPORT)" --output-dir "$(HOK_DATASETS_ROOT)/t8-video-combat-retrospective-three-class-v1" --retrospective

t8-video-retrospective-pilot: storage-preflight
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" CUBLAS_WORKSPACE_CONFIG=:4096:8 $(RUN_PYTHON) -m hok_agent t8-video-three-class-pilot --dataset-root "$(HOK_DATASETS_ROOT)/t8-video-combat-retrospective-three-class-v1" --adapter-checkpoint "$(HOK_CHECKPOINTS_ROOT)/t8-policy-v2/video-adapter-v1/adapter-epoch-3.safetensors" --output-dir "$(HOK_RUNS_ROOT)/t8-policy-v2.1/video-retrospective-three-class-seed0-v1" --device cuda --batch-size 64 --retrospective

t8-video-retrospective-roi-evaluate: storage-preflight
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" $(RUN_PYTHON) -m hok_agent t8-video-retrospective-roi-evaluate --dataset-root "$(HOK_DATASETS_ROOT)/t8-video-combat-retrospective-three-class-v1" --probe-report "$(HOK_RUNS_ROOT)/t8-policy-v2/video-action-probe-v7/report.json" --inverse-report "$(T8_INVERSE_REPORT)" --output-dir "$(HOK_RUNS_ROOT)/t8-policy-v2.1/video-retrospective-roi-v1"

t8-retrospective-v1-verify: storage-preflight
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" $(RUN_PYTHON) -m hok_agent t8-retrospective-baseline-verify --baseline-dir "$(T8_RETRO_BASELINE)"

t8-retrospective-v1-batch: storage-preflight
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" $(RUN_PYTHON) -m hok_agent t8-retrospective-batch --target-dir "$(T8_RETRO_TARGET)" --baseline-dir "$(T8_RETRO_BASELINE)" --layout "$(T8_LAYOUT)" --split "$(T8_RETRO_SPLIT)" --output-dir "$(T8_RETRO_OUTPUT)"

t8-retrospective-v2-calibrate: storage-preflight
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" $(RUN_PYTHON) -m hok_agent t8-retrospective-calibrate-v2 --dataset-root "$(HOK_DATASETS_ROOT)/t8-video-combat-pseudolabel-v1" --probe-report "$(HOK_RUNS_ROOT)/t8-policy-v2/video-action-probe-v7/report.json" --layout "$(T8_LAYOUT)" --baseline-dir "$(T8_RETRO_BASELINE)" --inverse-calibration "$(HOK_RUNS_ROOT)/t8-policy-v2.1/inverse-probe-v1-1786784072/inverse-probe.npz" --inverse-calibration "$(HOK_RUNS_ROOT)/t8-policy-v2.1/inverse-probe-validation-v1-1786784290/inverse-probe.npz" --inverse-holdout "$(HOK_RUNS_ROOT)/t8-policy-v2.1/inverse-probe-three-class-v1-1786784473/inverse-probe.npz" --output-dir "$(T8_RETRO_CALIBRATION_OUTPUT)"

t8-causal-video-materialize: storage-preflight
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" CUBLAS_WORKSPACE_CONFIG=:4096:8 $(RUN_PYTHON) -m hok_agent t8-causal-video-materialize --target-dir "$(T8_RETRO_TARGET)" --train-events-dir "$(HOK_DATASETS_ROOT)/t8-retrospective-events-v1/train" --dev-events-dir "$(HOK_DATASETS_ROOT)/t8-retrospective-events-v1/dev" --adapter-checkpoint "$(T8_CAUSAL_ADAPTER)" --output-dir "$(T8_CAUSAL_DATASET)" --device cuda --batch-size 512

t8-causal-video-pilot: storage-preflight
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" CUBLAS_WORKSPACE_CONFIG=:4096:8 $(RUN_PYTHON) -m hok_agent t8-causal-video-pilot --dataset-root "$(T8_CAUSAL_DATASET)" --adapter-checkpoint "$(T8_CAUSAL_ADAPTER)" --output-dir "$(T8_CAUSAL_RUN)" --device cuda --batch-size 256

t8-causal-video-diagnose: storage-preflight
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" CUBLAS_WORKSPACE_CONFIG=:4096:8 $(RUN_PYTHON) -m hok_agent t8-causal-video-diagnose --dataset-root "$(T8_CAUSAL_DATASET)" --pilot-dir "$(T8_CAUSAL_RUN)" --output-dir "$(T8_CAUSAL_DIAGNOSTIC)" --device cuda --batch-size 256

t8-causal-pixel-materialize: storage-preflight
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" $(RUN_PYTHON) -m hok_agent t8-causal-pixel-materialize --target-dir "$(T8_RETRO_TARGET)" --train-events-dir "$(HOK_DATASETS_ROOT)/t8-retrospective-events-v1/train" --dev-events-dir "$(HOK_DATASETS_ROOT)/t8-retrospective-events-v1/dev" --output-dir "$(T8_CAUSAL_PIXEL_DATASET)"

t8-causal-pixel-probe: storage-preflight
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" CUBLAS_WORKSPACE_CONFIG=:4096:8 $(RUN_PYTHON) -m hok_agent t8-causal-pixel-probe --dataset-root "$(T8_CAUSAL_PIXEL_DATASET)" --adapter-checkpoint "$(T8_CAUSAL_ADAPTER)" --output-dir "$(T8_CAUSAL_PIXEL_RUN)" --device cuda --batch-size 64

t8-visual-teacher-replay: storage-preflight
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" $(RUN_PYTHON) -m hok_agent t8-visual-teacher-replay --dataset-root "$(T8_CAUSAL_PIXEL_DATASET)" --pixel-probe-dir "$(T8_CAUSAL_PIXEL_RUN)" --layout "$(T8_LAYOUT)" --output-dir "$(T8_VISUAL_TEACHER_REPLAY)"

t8-visible-onset-audit: storage-preflight
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" $(RUN_PYTHON) -m hok_agent t8-visible-onset-audit --target-dir "$(T8_RETRO_TARGET)" --train-events-dir "$(HOK_DATASETS_ROOT)/t8-retrospective-events-v1/train" --dev-events-dir "$(HOK_DATASETS_ROOT)/t8-retrospective-events-v1/dev" --layout "$(T8_LAYOUT)" --calibration-report "$(T8_RETRO_CALIBRATION_OUTPUT)/report.json" --output-dir "$(T8_VISIBLE_ONSET_AUDIT)"

t8-combat-causal-materialize: storage-preflight
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" $(RUN_PYTHON) -m hok_agent t8-combat-causal-materialize --target-dir "$(T8_RETRO_TARGET)" --onset-audit-dir "$(T8_VISIBLE_ONSET_AUDIT)" --output-dir "$(T8_COMBAT_CAUSAL_DATASET)"

t8-combat-causal-pilot: storage-preflight
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" CUBLAS_WORKSPACE_CONFIG=:4096:8 $(RUN_PYTHON) -m hok_agent t8-combat-causal-pilot --dataset-root "$(T8_COMBAT_CAUSAL_DATASET)" --adapter-checkpoint "$(T8_CAUSAL_ADAPTER)" --output-dir "$(T8_COMBAT_CAUSAL_RUN)" --device cuda --batch-size 8

t8-combat-causal-diagnostic-materialize: storage-preflight
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" $(RUN_PYTHON) -m hok_agent t8-combat-causal-materialize --target-dir "$(T8_RETRO_TARGET)" --onset-audit-dir "$(T8_VISIBLE_ONSET_AUDIT)" --output-dir "$(T8_COMBAT_CAUSAL_DIAGNOSTIC_DATASET)" --diagnostic-only

t8-combat-causal-diagnostic-pilot: storage-preflight
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" CUBLAS_WORKSPACE_CONFIG=:4096:8 $(RUN_PYTHON) -m hok_agent t8-combat-causal-pilot --dataset-root "$(T8_COMBAT_CAUSAL_DIAGNOSTIC_DATASET)" --adapter-checkpoint "$(T8_CAUSAL_ADAPTER)" --output-dir "$(T8_COMBAT_CAUSAL_DIAGNOSTIC_RUN)" --device cuda --batch-size 8

t8-v25-dry-run: storage-preflight
	@test -n "$(T8_SERIAL)" || { echo "set T8_SERIAL" >&2; exit 2; }
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" $(RUN_PYTHON) -m hok_agent mobile-demonstrate-rgb-teacher-v25 --serial "$(T8_SERIAL)" --layout "$(T8_LAYOUT)" --video-node "$(T8_VIDEO_NODE)" --teacher-report "$(T8_V25_TEACHER)" --run-seconds 60 --output-dir "$(T8_V25_ROOT)/diagnostics/dry-run-$$(date +%s)"

t8-v25-probe-20: storage-preflight
	@test -n "$(T8_SERIAL)" || { echo "set T8_SERIAL" >&2; exit 2; }
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" $(RUN_PYTHON) -m hok_agent mobile-demonstrate-rgb-teacher-v25 --serial "$(T8_SERIAL)" --layout "$(T8_LAYOUT)" --video-node "$(T8_VIDEO_NODE)" --teacher-report "$(T8_V25_TEACHER)" --run-seconds 120 --enable-input --max-actions 20 --output-dir "$(T8_V25_ROOT)/diagnostics/probe-20-$$(date +%s)"

t8-v25-smoke-60: storage-preflight
	@test -n "$(T8_SERIAL)" || { echo "set T8_SERIAL" >&2; exit 2; }
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" $(RUN_PYTHON) -m hok_agent mobile-demonstrate-rgb-teacher-v25 --serial "$(T8_SERIAL)" --layout "$(T8_LAYOUT)" --video-node "$(T8_VIDEO_NODE)" --teacher-report "$(T8_V25_TEACHER)" --run-seconds 60 --enable-input --max-actions 300 --output-dir "$(T8_V25_ROOT)/diagnostics/smoke-60-$$(date +%s)"

t8-v25-collect: storage-preflight
	@test -n "$(T8_SERIAL)$(T8_V25_SESSION)" || { echo "set T8_SERIAL and T8_V25_SESSION=session-NNN" >&2; exit 2; }
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" $(RUN_PYTHON) -m hok_agent mobile-demonstrate-rgb-teacher-v25 --serial "$(T8_SERIAL)" --layout "$(T8_LAYOUT)" --video-node "$(T8_VIDEO_NODE)" --teacher-report "$(T8_V25_TEACHER)" --run-seconds 305 --enable-input --max-actions 300 --formal-session --output-dir "$(T8_V25_ROOT)/$(T8_V25_SESSION)"

t8-v25-pilot-freeze: storage-preflight
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" $(RUN_PYTHON) -m hok_agent t8-v25-freeze-split --dataset-root "$(T8_V25_ROOT)" --output "$(T8_V25_PILOT_SPLIT)" --pilot

t8-v25-freeze: storage-preflight
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" $(RUN_PYTHON) -m hok_agent t8-v25-freeze-split --dataset-root "$(T8_V25_ROOT)" --output "$(T8_V25_SPLIT)"

t8-v25-pilot: storage-preflight
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" CUBLAS_WORKSPACE_CONFIG=:4096:8 $(RUN_PYTHON) -m hok_agent t8-v25-pilot --dataset-root "$(T8_V25_ROOT)" --split "$(T8_V25_PILOT_SPLIT)" --adapter-checkpoint "$(T8_CAUSAL_ADAPTER)" --output-dir "$(T8_V25_RUN)" --device cuda --batch-size 8 --seed 0

t8-v26-train-seed: storage-preflight
	@test "$(T8_SEED)" = "0" -o "$(T8_SEED)" = "1" -o "$(T8_SEED)" = "2" || { echo "set T8_SEED=0, 1, or 2" >&2; exit 2; }
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" CUBLAS_WORKSPACE_CONFIG=:4096:8 $(RUN_PYTHON) -m hok_agent t8-v25-pilot --dataset-root "$(T8_V25_ROOT)" --split "$(T8_V25_SPLIT)" --adapter-checkpoint "$(T8_CAUSAL_ADAPTER)" --output-dir "$(T8_V26_FORMAL_RUN_ROOT)/seed-$(T8_SEED)" --device cuda --batch-size 8 --seed "$(T8_SEED)"

t8-v26-select: storage-preflight
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" $(RUN_PYTHON) -m hok_agent t8-v26-select --run-root "$(T8_V26_FORMAL_RUN_ROOT)" --output "$(T8_V26_FORMAL_RUN_ROOT)/selection.json"

t8-v26-evaluate-offline: storage-preflight
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" $(RUN_PYTHON) -m hok_agent t8-v26-evaluate-offline --dataset-root "$(T8_V25_ROOT)" --split "$(T8_V25_SPLIT)" --run-root "$(T8_V26_FORMAL_RUN_ROOT)" --selection "$(T8_V26_FORMAL_RUN_ROOT)/selection.json" --output "$(T8_V26_FORMAL_RUN_ROOT)/offline-test-v1.json" --device cuda --batch-size 8

t8-v26-shadow: storage-preflight
	@test -n "$(T8_SERIAL)" || { echo "set T8_SERIAL" >&2; exit 2; }
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" $(RUN_PYTHON) -m hok_agent t8-v26-shadow --serial "$(T8_SERIAL)" --model "$(T8_V26_FORMAL_RUN_ROOT)/seed-1/model-seed-1.safetensors" --offline-report "$(T8_V26_FORMAL_RUN_ROOT)/offline-test-v1.json" --split "$(T8_V25_SPLIT)" --layout "$(T8_LAYOUT)" --video-node "$(T8_VIDEO_NODE)" --output-dir "$(T8_V26_SHADOW_RUN)" --device cuda --stream-fps 30 --infer-hz 10 --run-seconds 300

t8-v26-shadow-replay: storage-preflight
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" $(RUN_PYTHON) -m hok_agent t8-v26-shadow-replay --dataset-root "$(T8_V25_ROOT)" --split "$(T8_V25_SPLIT)" --run-root "$(T8_V26_FORMAL_RUN_ROOT)" --selection "$(T8_V26_FORMAL_RUN_ROOT)/selection.json" --offline-report "$(T8_V26_FORMAL_RUN_ROOT)/offline-test-v1.json" --layout "$(T8_LAYOUT)" --output-dir "$(T8_V26_FORMAL_RUN_ROOT)/shadow-replay-v1" --device cuda

t8-v26-execute-probe: storage-preflight
	@test -n "$(T8_SERIAL)" || { echo "set T8_SERIAL" >&2; exit 2; }
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" $(RUN_PYTHON) -m hok_agent t8-v26-execute-probe --serial "$(T8_SERIAL)" --model "$(T8_V26_FORMAL_RUN_ROOT)/seed-1/model-seed-1.safetensors" --selection "$(T8_V26_FORMAL_RUN_ROOT)/selection.json" --offline-report "$(T8_V26_FORMAL_RUN_ROOT)/offline-test-v1.json" --shadow-summary "$(T8_V26_FORMAL_RUN_ROOT)/shadow-replay-v1/summary.json" --split "$(T8_V25_SPLIT)" --layout "$(T8_LAYOUT)" --video-node "$(T8_VIDEO_NODE)" --output-dir "$(T8_V26_FORMAL_RUN_ROOT)/probe-20-v1" --device cuda --stream-fps 30 --infer-hz 10 --run-seconds 60 --max-actions 20

t8-v27-calibration-pilot: storage-preflight
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" CUBLAS_WORKSPACE_CONFIG=:4096:8 $(RUN_PYTHON) -m hok_agent t8-v27-calibration-pilot --dataset-root "$(T8_V25_ROOT)" --train-session "$(T8_V27_TRAIN_SESSION)" --dev-session "$(T8_V27_DEV_SESSION)" --source-model "$(T8_V26_FORMAL_RUN_ROOT)/seed-1/model-seed-1.safetensors" --output-dir "$(T8_V27_RUN)" --device cuda --batch-size 8

t8-v27-freeze: storage-preflight
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" $(RUN_PYTHON) -m hok_agent t8-v27-freeze --report "$(HOK_RUNS_ROOT)/t8-policy-v2.7/current-scene-head-pilot-v1/report.json" --report "$(HOK_RUNS_ROOT)/t8-policy-v2.7/current-scene-head-pilot-v2-balanced/report.json" --report "$(HOK_RUNS_ROOT)/t8-policy-v2.7/current-scene-head-pilot-v3-balanced-data/report.json" --output-dir "$(T8_V27_FREEZE)"

t8-v3-state-materialize: storage-preflight
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" $(RUN_PYTHON) -m hok_agent t8-v3-state-materialize --feature-root "$(T8_CAUSAL_DATASET)" --target-root "$(T8_RETRO_TARGET)" --teacher-report "$(T8_V25_TEACHER)" --layout "$(T8_LAYOUT)" --output-dir "$(T8_V3_DATASET)"

t8-v3-state-train: storage-preflight
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" CUBLAS_WORKSPACE_CONFIG=:4096:8 $(RUN_PYTHON) -m hok_agent t8-v3-state-train --dataset-root "$(T8_V3_DATASET)" --adapter-checkpoint "$(T8_CAUSAL_ADAPTER)" --output-dir "$(T8_V3_RUN)" --device cuda --batch-size 256 --seed 0 --epochs 8

t8-v3-hybrid-replay: storage-preflight
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" $(RUN_PYTHON) -m hok_agent t8-v3-hybrid-replay --dataset-root "$(T8_V3_DATASET)" --model "$(T8_V3_RUN)/model-seed-0.safetensors" --training-report "$(T8_V3_RUN)/report.json" --output-dir "$(T8_V3_REPLAY)" --device cuda --batch-size 256

t8-v2-touch-smoke: storage-preflight
	@test -n "$(T8_SERIAL)" || { echo "set T8_SERIAL" >&2; exit 2; }
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" $(RUN_PYTHON) -m hok_agent mobile-demonstrate-touch --serial "$(T8_SERIAL)" --touch-device "$(T8_TOUCH_DEVICE)" --touch-max-slots 2 --touch-max-x 719 --touch-max-y 1599 --touch-protocol type_a --touch-calibration "$(T8_TOUCH_CALIBRATION)" --layout "$(T8_LAYOUT)" --video-node "$(T8_VIDEO_NODE)" --run-seconds 20 --max-samples 260 --semantic-smoke --output-dir "$(HOK_DATASETS_ROOT)/t8-demonstrations-v2/diagnostics/touch-smoke-$$(date +%s)"

t8-v2-collect: storage-preflight
	@test -n "$(T8_SERIAL)$(T8_V2_SESSION)" || { echo "set T8_SERIAL and T8_V2_SESSION=session-NNN" >&2; exit 2; }
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" $(RUN_PYTHON) -m hok_agent mobile-demonstrate-keyboard-v2 --serial "$(T8_SERIAL)" --layout "$(T8_LAYOUT)" --video-node "$(T8_VIDEO_NODE)" --run-seconds 300 --max-actions 600 --scripted-seed "$(patsubst session-%,%,$(T8_V2_SESSION))" --scripted-interval-seconds 1 --formal-session --output-dir "$(T8_V2_DATASET)/$(T8_V2_SESSION)"

t8-v2-freeze: storage-preflight
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" $(RUN_PYTHON) -m hok_agent t8-v2-freeze-split --dataset-root "$(T8_V2_DATASET)" --output "$(T8_V2_DATASET)/t8-v2-split.json"

t8-v2-adapt: storage-preflight
	@test -n "$(V5_SOURCE_DIR)$(V5_TARGET_DIR)" || { echo "set V5_SOURCE_DIR and V5_TARGET_DIR" >&2; exit 2; }
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" CUBLAS_WORKSPACE_CONFIG=:4096:8 $(RUN_PYTHON) -m hok_agent t8-v2-video-adapt --v5-source-dir "$(V5_SOURCE_DIR)" --target-dir "$(V5_TARGET_DIR)" --output-dir "$(HOK_CHECKPOINTS_ROOT)/t8-policy-v2/video-adapter-v1" --device cuda --batch-size 128

t8-v2-pilot: storage-preflight
	@test -n "$(T8_V2_ADAPTER)" || { echo "set T8_V2_ADAPTER" >&2; exit 2; }
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" CUBLAS_WORKSPACE_CONFIG=:4096:8 $(RUN_PYTHON) -m hok_agent t8-v2-pilot --dataset-root "$(T8_V2_DATASET)" --adapter-checkpoint "$(T8_V2_ADAPTER)" --output-dir "$(HOK_RUNS_ROOT)/t8-policy-v2/pilot-seed0-v1" --device cuda --batch-size 32

alignment-smoke:
	$(RUN_PYTHON) -m hok_agent alignment-v5-smoke

temporal-smoke:
	$(RUN_PYTHON) -m hok_agent temporal-v6-smoke

rich-smoke:
	$(RUN_PYTHON) -m hok_agent accept-rich-v7 --smoke --device cpu

v5-source-produce: storage-preflight
	@test -n "$(V5_SOURCE_OUTPUT)$(V5_SOURCE_DEVICE)" || { echo "set V5_SOURCE_OUTPUT and V5_SOURCE_DEVICE" >&2; exit 2; }
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" CUBLAS_WORKSPACE_CONFIG=:4096:8 $(RUN_PYTHON) -m hok_agent v5-source-produce --output-dir "$(V5_SOURCE_OUTPUT)" --device "$(V5_SOURCE_DEVICE)"

v5-build-cohort: storage-preflight
	@test -n "$(V5_PRE_INGEST)$(V5_COHORT_OUTPUT)" || { echo "set V5_PRE_INGEST and V5_COHORT_OUTPUT" >&2; exit 2; }
	@test "$(V5_RECORDING_OWNER_CONFIRMED)" = "1" && test "$(V5_LOCAL_RESEARCH_CONFIRMED)" = "1" && test "$(V5_ZERO_REDACTION_CONFIRMED)" = "1" || { echo "set V5_RECORDING_OWNER_CONFIRMED=1 V5_LOCAL_RESEARCH_CONFIRMED=1 V5_ZERO_REDACTION_CONFIRMED=1" >&2; exit 2; }
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" $(RUN_PYTHON) -m hok_agent v5-build-cohort --pre-ingest "$(V5_PRE_INGEST)" --output-dir "$(V5_COHORT_OUTPUT)" --recording-owner-confirmed --local-research-confirmed --zero-redaction-confirmed

v5-ingest-zero-label: storage-preflight
	@test -n "$(V5_INPUT_ROOT)$(V5_PRE_INGEST)$(V5_COHORT_DIR)$(V5_TARGET_OUTPUT)" || { echo "set V5_INPUT_ROOT, V5_PRE_INGEST, V5_COHORT_DIR and V5_TARGET_OUTPUT" >&2; exit 2; }
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" $(RUN_PYTHON) -m hok_agent v5-ingest-zero-label --input-root "$(V5_INPUT_ROOT)" --pre-ingest "$(V5_PRE_INGEST)" --cohort-dir "$(V5_COHORT_DIR)" --output-dir "$(V5_TARGET_OUTPUT)"

v5-validate-zero-target:
	@test -n "$(V5_PRE_INGEST)$(V5_COHORT_DIR)$(V5_TARGET_DIR)" || { echo "set V5_PRE_INGEST, V5_COHORT_DIR and V5_TARGET_DIR" >&2; exit 2; }
	$(RUN_PYTHON) -m hok_agent v5-validate-zero-target --pre-ingest "$(V5_PRE_INGEST)" --cohort-dir "$(V5_COHORT_DIR)" --target-dir "$(V5_TARGET_DIR)"

v5-freeze-training-config:
	@test -n "$(V5_TRAINING_CONFIG_OUTPUT)" || { echo "set V5_TRAINING_CONFIG_OUTPUT" >&2; exit 2; }
	$(RUN_PYTHON) -m hok_agent v5-freeze-training-config \
		--output "$(V5_TRAINING_CONFIG_OUTPUT)" \
		--batch-size "$(V5_TRAINING_BATCH_SIZE)" \
		--learning-rate "$(V5_LEARNING_RATE)" \
		--weight-decay "$(V5_WEIGHT_DECAY)" \
		--epochs "$(V5_TRAINING_EPOCHS)" \
		--mean-teacher-epochs "$(V5_MEAN_TEACHER_EPOCHS)"

v5-train-simsiam-adapted: storage-preflight
	@test -n "$(V5_SOURCE_DIR)$(V5_TARGET_DIR)$(V5_COHORT_DIR)$(V5_PRE_INGEST)$(V5_TRAINING_CONFIG)$(V5_SIMSIAM_ADAPTED_CHECKPOINT)" || { echo "set V5_SOURCE_DIR, V5_TARGET_DIR, V5_COHORT_DIR, V5_PRE_INGEST, V5_TRAINING_CONFIG and V5_SIMSIAM_ADAPTED_CHECKPOINT" >&2; exit 2; }
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" HOK_V5_PREFETCH_SHARDS="$(V5_PREFETCH_SHARDS)" CUBLAS_WORKSPACE_CONFIG=:4096:8 $(RUN_PYTHON) -m hok_agent v5-train-simsiam-adapted \
		--source-dir "$(V5_SOURCE_DIR)" \
		--target-dir "$(V5_TARGET_DIR)" \
		--cohort-dir "$(V5_COHORT_DIR)" \
		--pre-ingest "$(V5_PRE_INGEST)" \
		--config "$(V5_TRAINING_CONFIG)" \
		--output-checkpoint "$(V5_SIMSIAM_ADAPTED_CHECKPOINT)" \
		--device "$(V5_SIMSIAM_DEVICE)"

v5-model-predict: storage-preflight
	@test -n "$(V5_SOURCE_DIR)$(V5_TARGET_DIR)$(V5_COHORT_DIR)$(V5_PRE_INGEST)$(V5_TRAINING_CONFIG)$(V5_SIMSIAM_ADAPTED_CHECKPOINT)$(V5_PREDICTIONS_DIR)" || { echo "set V5_SOURCE_DIR, V5_TARGET_DIR, V5_COHORT_DIR, V5_PRE_INGEST, V5_TRAINING_CONFIG, V5_SIMSIAM_ADAPTED_CHECKPOINT and V5_PREDICTIONS_DIR" >&2; exit 2; }
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" CUBLAS_WORKSPACE_CONFIG=:4096:8 $(RUN_PYTHON) -m hok_agent v5-model-predict \
		--source-dir "$(V5_SOURCE_DIR)" \
		--target-dir "$(V5_TARGET_DIR)" \
		--cohort-dir "$(V5_COHORT_DIR)" \
		--pre-ingest "$(V5_PRE_INGEST)" \
		--config "$(V5_TRAINING_CONFIG)" \
		--adapted-model "$(V5_SIMSIAM_ADAPTED_CHECKPOINT)" \
		--output-dir "$(V5_PREDICTIONS_DIR)" \
		--device "$(V5_SIMSIAM_DEVICE)"

v5-materialize-pseudo: storage-preflight
	@test -n "$(V5_SOURCE_DIR)$(V5_TARGET_DIR)$(V5_COHORT_DIR)$(V5_PRE_INGEST)$(V5_TRAINING_CONFIG)$(V5_SIMSIAM_ADAPTED_CHECKPOINT)$(V5_PREDICTIONS_DIR)$(V5_PSEUDO_PATH)" || { echo "set V5_SOURCE_DIR, V5_TARGET_DIR, V5_COHORT_DIR, V5_PRE_INGEST, V5_TRAINING_CONFIG, V5_SIMSIAM_ADAPTED_CHECKPOINT, V5_PREDICTIONS_DIR and V5_PSEUDO_PATH" >&2; exit 2; }
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" CUBLAS_WORKSPACE_CONFIG=:4096:8 $(RUN_PYTHON) -m hok_agent v5-materialize-pseudo \
		--source-dir "$(V5_SOURCE_DIR)" \
		--target-dir "$(V5_TARGET_DIR)" \
		--cohort-dir "$(V5_COHORT_DIR)" \
		--pre-ingest "$(V5_PRE_INGEST)" \
		--config "$(V5_TRAINING_CONFIG)" \
		--adapted-model "$(V5_SIMSIAM_ADAPTED_CHECKPOINT)" \
		--predictions-dir "$(V5_PREDICTIONS_DIR)" \
		--output "$(V5_PSEUDO_PATH)" \
		--device "$(V5_SIMSIAM_DEVICE)"

v5-run-mean-teacher-round: storage-preflight
	@test -n "$(V5_SOURCE_DIR)$(V5_TARGET_DIR)$(V5_COHORT_DIR)$(V5_PRE_INGEST)$(V5_TRAINING_CONFIG)$(V5_PREDICTIONS_DIR)$(V5_PSEUDO_PATH)$(V5_SIMSIAM_ADAPTED_CHECKPOINT)$(V5_MEAN_TEACHER_EMA)$(V5_MEAN_TEACHER_ROUND_LEDGER)" || { echo "set V5_SOURCE_DIR, V5_TARGET_DIR, V5_COHORT_DIR, V5_PRE_INGEST, V5_TRAINING_CONFIG, V5_PREDICTIONS_DIR, V5_PSEUDO_PATH, V5_SIMSIAM_ADAPTED_CHECKPOINT, V5_MEAN_TEACHER_EMA and V5_MEAN_TEACHER_ROUND_LEDGER" >&2; exit 2; }
	HOK_LARGE_ROOT="$(HOK_LARGE_ROOT)" CUBLAS_WORKSPACE_CONFIG=:4096:8 $(RUN_PYTHON) -m hok_agent v5-run-mean-teacher-round \
		--source-dir "$(V5_SOURCE_DIR)" \
		--target-dir "$(V5_TARGET_DIR)" \
		--cohort-dir "$(V5_COHORT_DIR)" \
		--pre-ingest "$(V5_PRE_INGEST)" \
		--config "$(V5_TRAINING_CONFIG)" \
		--predictions "$(V5_PREDICTIONS_DIR)" \
		--pseudo "$(V5_PSEUDO_PATH)" \
		--adapted-checkpoint "$(V5_SIMSIAM_ADAPTED_CHECKPOINT)" \
		--ema-checkpoint "$(V5_MEAN_TEACHER_EMA)" \
		--round-ledger "$(V5_MEAN_TEACHER_ROUND_LEDGER)" \
		--device "$(V5_MEAN_TEACHER_DEVICE)"

v6-zero-smoke:
	$(RUN_PYTHON) -m hok_agent v6-zero-smoke

accept-v7: storage-init
	$(RUN_PYTHON) -m hok_agent accept-rich-v7 --device cuda --output-dir "$(HOK_RUNS_ROOT)/rich-v7-v1"
