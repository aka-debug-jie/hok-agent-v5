# ruff: noqa: E501, E701, E702
from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from time import sleep

import numpy as np
import pytest

from hok_agent import capture, pixel, shadow


def test_validate_capture_device_rejects_invalid_inputs(tmp_path: Path) -> None:
    model = tmp_path / 'model.safetensors'; model.write_bytes(b'model'); monkeypatch = pytest.MonkeyPatch(); monkeypatch.setattr(shadow, 'FROZEN_V3_MODEL_SHA256', capture._sha_file(model)); monkeypatch.setattr(pixel, 'open_rgb_predictor', lambda _path, _device: (lambda frames: ([0] * len(frames), [0.5] * len(frames)), 1))
    with pytest.raises(capture.CaptureError, match='must be'):
        capture.run_shadow_live(input_device='0', model_raw=str(model), output_dir=tmp_path / 'out', max_frames=0, frame_source=None)
    monkeypatch.undo(); symlink_target = tmp_path / 'video0'; symlink_target.write_bytes(b''); symlink = tmp_path / 'link_video'; symlink.symlink_to(symlink_target)
    with pytest.raises(capture.CaptureError, match='exactly a /dev/videoN'):
        capture._validate_capture_device(str(symlink))
def test_run_shadow_live_smoke_inject_source_emits_two_rows_and_no_raw_frame_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tmp_path / 'model.safetensors'; model.write_bytes(b'model'); output = tmp_path / 'live-output'; frame_calls = {'index': 0}
    def source() -> tuple[float, np.ndarray] | None:
        sleep(0.09 if frame_calls['index'] else 0.01); value = (time.monotonic(), np.full((128, 128, 3), 10 + 10 * frame_calls['index'], dtype=np.uint8)); frame_calls['index'] += 1; return value
    def open_predictor(model_path: Path, device_name: str) -> tuple[object, int]:
        def predict(frame_batch: np.ndarray) -> tuple[list[int], list[float]]:
            index = predict.index % 2; predict.index += 1; return ([index], [0.5 + index * 0.4])
        predict.index = 0; return (predict, 77)
    monkeypatch.setattr(shadow, 'FROZEN_V3_MODEL_SHA256', capture._sha_file(model)); monkeypatch.setattr(pixel, 'open_rgb_predictor', open_predictor); summary = capture.run_shadow_live(input_device=None, model_raw=str(model), output_dir=output, device='cpu', infer_hz=10, max_frames=2, frame_source=source); assert summary['status'] == 'SMOKE_PASSED'; assert summary['disposition'] == 'SMOKE_LIVE_SHADOW'; assert summary['source']['is_simulated'] is True; rows = [line for line in (output / 'events.jsonl').read_text(encoding='utf-8').splitlines() if line]; parsed = [json.loads(line) for line in rows]; assert len(rows) == 2; assert {path.name for path in output.iterdir()} == {'events.jsonl', 'summary.json'}; assert all(row['advisory_action'] == 'ABSTAIN' for row in parsed); assert all(row['control_output'] is False for row in parsed); assert parsed[0]['raw_model_hypothesis'] == 'wait'; assert parsed[1]['raw_model_hypothesis'] == 'forward'; assert summary['analyzed_frames'] == 2; assert summary['abstain_count'] == 2; assert summary['advisory_count'] == 0; assert summary['schedule']['target_infer_hz'] == 10; assert summary['latency']['capture_to_terminal_ms']['count'] == 2; assert summary['on_time']['on_time_ticks'] <= summary['schedule']['target_ticks']; root_entries = [p for p in output.iterdir()]; assert all(p.suffix not in {'.mp4', '.avi', '.webm', '.mov', '.mkv', '.raw'} for p in root_entries); assert all(p.name in {'events.jsonl', 'summary.json'} for p in root_entries)
def test_run_shadow_live_continuous_injection_stops_after_run_seconds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tmp_path / 'model.safetensors'; model.write_bytes(b'model'); output = tmp_path / 'continuous-output'; calls = {'count': 0}
    def source() -> tuple[float, np.ndarray]:
        calls['count'] += 1; value = calls['count'] % 256; return (time.monotonic(), np.full((128, 128, 3), value, dtype=np.uint8))
    def open_predictor(model_path: Path, device_name: str) -> tuple[object, int]:
        return (lambda frame_batch: ([0], [0.8]), 77)
    monkeypatch.setattr(shadow, 'FROZEN_V3_MODEL_SHA256', capture._sha_file(model)); monkeypatch.setattr(pixel, 'open_rgb_predictor', open_predictor); started = time.monotonic(); summary = capture.run_shadow_live(input_device=None, model_raw=str(model), output_dir=output, frame_source=source, run_seconds=0.5, infer_hz=20, max_frames=None); elapsed = time.monotonic() - started; assert summary['status'] == 'SMOKE_PASSED'; assert summary['source']['is_simulated'] is True; assert summary['analyzed_frames'] > 0; assert summary['analyzed_frames'] <= calls['count']; assert elapsed < 2.0
def test_latest_frame_queue_drops_old_frames_to_keep_length_one() -> None:
    store: queue.Queue[tuple[float, np.ndarray]] = queue.Queue(maxsize=1); frame_a = np.zeros((128, 128, 3), dtype=np.uint8); frame_b = np.ones((128, 128, 3), dtype=np.uint8); frame_c = np.full((128, 128, 3), 2, dtype=np.uint8); capture._latest_frame_put(store, (1.0, frame_a)); capture._latest_frame_put(store, (2.0, frame_b)); capture._latest_frame_put(store, (3.0, frame_c)); assert store.qsize() == 1; _, latest = capture._latest_frame_get(store); assert int(latest[0, 0, 0]) == 2
def test_capture_output_is_written_atomically_and_empty_source_persists_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tmp_path / 'model.safetensors'; model.write_bytes(b'model'); output = tmp_path / 'empty-output'; monkeypatch.setattr(shadow, 'FROZEN_V3_MODEL_SHA256', capture._sha_file(model)); monkeypatch.setattr(pixel, 'open_rgb_predictor', lambda _path, _device: (lambda _frames: (_ for _ in ()).throw(AssertionError()), 77)); summary = capture.run_shadow_live(input_device=None, model_raw=str(model), output_dir=output, frame_source=lambda: None, max_frames=0); assert summary['status'] == 'FAILED'; assert summary['disposition'] == 'LIVE_SHADOW_FAILED'; assert summary['analyzed_frames'] == 0; assert summary['captured_frames'] == 0; events = output / 'events.jsonl'; text = events.read_text(encoding='utf-8'); assert text == ''; assert {path.name for path in output.iterdir()} == {'events.jsonl', 'summary.json'}
def test_run_shadow_live_formal_summary_is_not_spoofed_by_injected_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tmp_path / 'model.safetensors'; model.write_bytes(b'model'); output = tmp_path / 'formal-output'; called = {'index': 0}
    def source() -> tuple[float, np.ndarray] | None:
        if called['index'] >= 3:
            return None
        called['index'] += 1; return (time.monotonic(), np.full((128, 128, 3), 255, dtype=np.uint8))
    monkeypatch.setattr(shadow, 'FROZEN_V3_MODEL_SHA256', capture._sha_file(model)); monkeypatch.setattr(pixel, 'open_rgb_predictor', lambda _path, _device: (lambda _frames: ([1], [0.99]), 88)); summary = capture.run_shadow_live(input_device=None, model_raw=str(model), output_dir=output, frame_source=source, capture_size='1920x1080', capture_fps=60, infer_hz=10, run_seconds=1.0, max_frames=3); assert summary['status'] == 'SMOKE_PASSED'; assert summary['disposition'] == 'SMOKE_LIVE_SHADOW'; assert summary['source']['is_simulated'] is True
def test_formal_device_metadata_cannot_replace_ten_minute_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tmp_path / 'model.safetensors'; model.write_bytes(b'model'); yielded = False
    def source() -> tuple[float, np.ndarray] | None:
        nonlocal yielded
        if yielded:
            return None
        yielded = True; return (time.monotonic(), np.zeros((128, 128, 3), dtype=np.uint8))
    def producer(_device: Path, _size: tuple[int, int], report: dict[str, object], output: queue.Queue[capture.FrameRecord], stop: threading.Event, done: threading.Event, errors: list[BaseException], _fps: int) -> threading.Thread:
        report.update(actual_width=1920, actual_height=1080, average_rate=60.0); return capture._producer_from_injection(source, output, stop, done, errors)
    monkeypatch.setattr(shadow, 'FROZEN_V3_MODEL_SHA256', capture._sha_file(model)); monkeypatch.setattr(pixel, 'open_rgb_predictor', lambda _path, _device: (lambda _frames: ([0], [0.99]), 1)); monkeypatch.setattr(capture, '_validate_capture_device', lambda _raw: Path('/dev/video42')); monkeypatch.setattr(capture, '_producer_from_device', producer); summary = capture.run_shadow_live('/dev/video42', str(model), tmp_path / 'short-device-run', capture_size='1920x1080', capture_fps=60, infer_hz=10, run_seconds=600.0, event_sink=lambda _row: None); assert summary['status'] == 'SMOKE_PASSED'; assert summary['schedule']['elapsed_seconds'] < 600.0
