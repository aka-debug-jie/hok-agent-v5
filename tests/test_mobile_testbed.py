# ruff: noqa: E501
from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import numpy as np
import pytest

from hok_agent import mobile_testbed

TEST_PACKAGE = "org.example.owner.testbed"


def _write_identity(tmp_path: Path, monkeypatch) -> str:
    payload: dict[str, object] = {
        "schema_version": mobile_testbed.MOBILE_BUILD_IDENTITY_SCHEMA,
        "package": TEST_PACKAGE,
        "version_code": 7,
        "version_name": "0.7.0",
        "signature_ids": ["deadbeef"],
        "base_apk_sha256": "a" * 64,
        "owner_attested_self_built": True,
        "attested_date": "2026-08-16",
    }
    digest = mobile_testbed.hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload["identity_sha256"] = digest
    path = tmp_path / "mobile-testbed-identity.local.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("HOK_MOBILE_IDENTITY_PATH", str(path))
    return digest


def _write_layout(tmp_path: Path, *, complete: bool = True) -> Path:
    path = tmp_path / ("layout.json" if complete else "partial-layout.json")
    path.write_text(
        json.dumps(
            {
                "schema_version": mobile_testbed.LAYOUT_SCHEMA,
                "calibration_status": "TEST_COMPLETE" if complete else "TEST_PARTIAL",
                "screen": {"width": 1600, "height": 720},
                "joystick": {
                    "center": [0.2, 0.8],
                    "radius": 0.12,
                    "forward_vector": [0.0, -1.0],
                    "move_hold_ms": 150,
                    "skill_hold_ms": 250,
                    "aim_radius": 0.18,
                },
                "buttons": {
                    "basic_attack": [0.8, 0.8],
                    "skill1": [0.7, 0.8] if complete else None,
                    "skill2": [0.75, 0.75] if complete else None,
                    "skill3": [0.8, 0.7] if complete else None,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _guard(monkeypatch) -> None:
    monkeypatch.setattr(
        mobile_testbed,
        "_open_device_guard",
        lambda _serial: mobile_testbed.DeviceGuard("test-1", TEST_PACKAGE, 1600, 720, 1),
    )
    monkeypatch.setattr(mobile_testbed.DeviceGuard, "check", lambda _self: None)
    monkeypatch.setattr(mobile_testbed, "_require_mobile_input_identity", lambda: None)


def test_mobile_input_fails_closed_without_frozen_build_identity(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOK_MOBILE_IDENTITY_PATH", str(tmp_path / "missing.json"))
    with pytest.raises(mobile_testbed.MobileTestbedError, match="build identity"):
        mobile_testbed._require_mobile_input_identity()

    sent: list[tuple[str, ...]] = []
    guarded = mobile_testbed._guarded_send(
        mobile_testbed.DeviceGuard("test-1", TEST_PACKAGE, 1600, 720, 1),
        lambda *arguments: sent.append(arguments),
    )
    with pytest.raises(mobile_testbed.MobileTestbedError, match="build identity"):
        guarded("tap", "1", "2")
    assert sent == []


def test_mobile_build_identity_matches_installed_version_and_signature(
    tmp_path: Path, monkeypatch
) -> None:
    digest = _write_identity(tmp_path, monkeypatch)
    monkeypatch.setattr(
        mobile_testbed,
        "_run_adb",
        lambda *_args, **_kwargs: (
            "versionCode=7 minSdk=22\n"
            "versionName=0.7.0\n"
            "signatures=PackageSignatures{abc signatures:[deadbeef], past signatures:[]}\n"
        ),
    )
    assert mobile_testbed._verify_mobile_build_identity("test-1") == digest


def test_mobile_build_identity_rejects_version_drift(tmp_path: Path, monkeypatch) -> None:
    _write_identity(tmp_path, monkeypatch)
    monkeypatch.setattr(
        mobile_testbed,
        "_run_adb",
        lambda *_args, **_kwargs: (
            "versionCode=8 minSdk=22\n"
            "versionName=0.8.0\n"
            "signatures=PackageSignatures{abc signatures:[deadbeef], past signatures:[]}\n"
        ),
    )
    with pytest.raises(mobile_testbed.MobileTestbedError, match="differs"):
        mobile_testbed._verify_mobile_build_identity("test-1")


def test_mobile_build_identity_rejects_signature_drift(tmp_path: Path, monkeypatch) -> None:
    _write_identity(tmp_path, monkeypatch)
    monkeypatch.setattr(
        mobile_testbed,
        "_run_adb",
        lambda *_args, **_kwargs: (
            "versionCode=7 minSdk=22\n"
            "versionName=0.7.0\n"
            "signatures=PackageSignatures{abc signatures:[feedface], past signatures:[]}\n"
        ),
    )
    with pytest.raises(mobile_testbed.MobileTestbedError, match="differs"):
        mobile_testbed._verify_mobile_build_identity("test-1")


def test_active_display_requires_the_exact_foreground_package(monkeypatch) -> None:
    monkeypatch.setattr(
        mobile_testbed,
        "_run_adb",
        lambda *_args, **_kwargs: (
            "mCurrentFocus=Window{abc u0 other.package/.Activity}\nDisplayFrames w=1600 h=720 r=1\n"
        ),
    )
    with pytest.raises(mobile_testbed.MobileTestbedError, match="not foreground"):
        mobile_testbed._active_display("test-1", TEST_PACKAGE)


def test_active_display_accepts_the_authorized_foreground_package(monkeypatch) -> None:
    monkeypatch.setattr(
        mobile_testbed,
        "_run_adb",
        lambda *_args, **_kwargs: (
            f"mCurrentFocus=Window{{abc u0 {TEST_PACKAGE}/{TEST_PACKAGE}.MainActivity}}\n"
            "DisplayFrames w=1600 h=720 r=1\n"
        ),
    )
    assert mobile_testbed._active_display("test-1", TEST_PACKAGE) == (1600, 720, 1)


def test_mobile_testbed_dry_run_never_sends_input(tmp_path: Path, monkeypatch) -> None:
    sent: list[tuple[str, ...]] = []

    def adb(_serial: str, *args: str, text: bool = False):
        sent.append(args)
        return "device\n" if text else b"png"

    monkeypatch.setattr(mobile_testbed, "_run_adb", adb)
    _guard(monkeypatch)
    monkeypatch.setattr(
        mobile_testbed, "_frame", lambda _serial: np.zeros((720, 1600, 3), dtype=np.uint8)
    )
    monkeypatch.setattr(
        mobile_testbed,
        "_open_v3_predictor",
        lambda _model, _device: (lambda _frames: ([1], [0.99]), 2),
    )
    result = mobile_testbed.run_mobile_testbed(
        serial="test-1",
        model_path=tmp_path / "model.safetensors",
        output_dir=tmp_path / "out",
        device="cpu",
        run_seconds=0.01,
        infer_hz=1000,
    )
    assert result["executed_actions"] == 0
    assert sent == []
    rows = [
        json.loads(line) for line in (tmp_path / "out" / "events.jsonl").read_text().splitlines()
    ]
    assert rows and not rows[0]["input_sent"]


def test_mobile_testbed_input_is_bounded_and_explicit(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    class Pipe:
        def __init__(self, _serial: str) -> None:
            pass

        def send(self, *args: str) -> None:
            calls.append(args)

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        mobile_testbed,
        "_run_adb",
        lambda _serial, *args, text=False: (calls.append(args), "device\n" if text else b"")[1],
    )
    monkeypatch.setattr(mobile_testbed, "AdbInputPipe", Pipe)
    _guard(monkeypatch)
    monkeypatch.setattr(
        mobile_testbed, "_frame", lambda _serial: np.zeros((720, 1600, 3), dtype=np.uint8)
    )
    monkeypatch.setattr(
        mobile_testbed,
        "_open_v3_predictor",
        lambda _model, _device: (lambda _frames: ([3], [0.99]), 2),
    )
    result = mobile_testbed.run_mobile_testbed(
        serial="test-1",
        model_path=tmp_path / "model.safetensors",
        output_dir=tmp_path / "out",
        device="cpu",
        run_seconds=1,
        infer_hz=1000,
        enable_input=True,
        max_actions=1,
    )
    assert result["executed_actions"] == 1
    assert any(call[:1] == ("tap",) for call in calls)


def test_layout_drives_factorized_movement_and_attack(tmp_path: Path, monkeypatch) -> None:
    layout_path = tmp_path / "layout.json"
    layout_path.write_text(
        json.dumps(
            {
                "schema_version": mobile_testbed.LAYOUT_SCHEMA,
                "screen": {"width": 1600, "height": 720},
                "joystick": {
                    "center": [0.2, 0.8],
                    "radius": 0.1,
                    "forward_vector": [0.0, -1.0],
                    "move_hold_ms": 120,
                    "skill_hold_ms": 240,
                    "aim_radius": 0.2,
                },
                "buttons": {
                    "basic_attack": [0.8, 0.8],
                    "skill1": [0.7, 0.8],
                    "skill2": [0.75, 0.75],
                    "skill3": [0.8, 0.7],
                },
            }
        ),
        encoding="utf-8",
    )
    layout, digest = mobile_testbed.load_layout(layout_path)
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        mobile_testbed, "_run_adb", lambda _serial, *args, text=False: calls.append(args) or b""
    )
    forward = mobile_testbed._intent("forward", layout)
    attack = mobile_testbed._intent("attack_tower", layout)
    assert digest and forward.movement == (0.0, -1.0) and attack.target == "tower"
    assert mobile_testbed._input("test-1", forward, layout, 1600, 720)
    assert mobile_testbed._input("test-1", attack, layout, 1600, 720)
    assert calls[0][-5:] == ("320", "576", "320", "504", "120")
    assert calls[1][-2:] == ("1280", "576")


def test_scrcpy_mode_requires_one_explicit_video_node(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mobile_testbed, "_run_adb", lambda *_args, **_kwargs: "device\n")
    _guard(monkeypatch)
    with pytest.raises(mobile_testbed.MobileTestbedError, match="video-node"):
        mobile_testbed.run_mobile_testbed(
            serial="test-1",
            model_path=tmp_path / "model.safetensors",
            output_dir=tmp_path / "out",
            device="cpu",
            capture_mode="scrcpy-v4l2",
        )


def test_keyboard_actions_are_factorized_and_layout_bound() -> None:
    controller = mobile_testbed.KeyboardDemonstrator(mobile_testbed.DEFAULT_LAYOUT)
    assert controller.feed("h") == (None, False)
    assert controller.feed("1") == (None, False)
    skill, stopped = controller.feed("e")
    assert not stopped
    assert skill == mobile_testbed.FactorizedAction(
        ability="skill1", aim="north_east", target="none", hold_ms=250
    )
    move, stopped = controller.feed("a")
    assert not stopped
    assert move == mobile_testbed.FactorizedAction(movement="west", hold_ms=150)
    upper, stopped = controller.feed("W")
    assert not stopped
    assert upper == mobile_testbed.FactorizedAction(movement="north", hold_ms=150)


def test_terminal_keyboard_reads_unbuffered_fd(monkeypatch) -> None:
    keyboard = mobile_testbed.TerminalKeyboard()
    keyboard._fd = 7
    monkeypatch.setattr(mobile_testbed.select, "select", lambda *_args: ([7], [], []))
    monkeypatch.setattr(mobile_testbed.os, "read", lambda fd, size: b"W")
    assert keyboard.read(0.1) == "W"


def test_focused_keyboard_requires_confirmed_desktop_focus() -> None:
    calls: list[str] = []

    class Root:
        def lift(self) -> None:
            calls.append("lift")

        def focus_force(self) -> None:
            calls.append("focus")

        def update(self) -> None:
            calls.append("update")

        def focus_displayof(self) -> object:
            return self

        def grab_set(self) -> None:
            calls.append("grab")

    keyboard = mobile_testbed.FocusedKeyboardWindow()
    keyboard._root = Root()
    keyboard.wait_for_focus()
    assert keyboard.has_focus()
    assert calls == ["lift", "focus", "update", "grab"]


def test_touch_discovery_and_type_b_decoder_keep_slots_separate(monkeypatch) -> None:
    descriptors = mobile_testbed._parse_touch_descriptors(
        """add device 1: /dev/input/event4\n  name:     "touch"\n  events:\n    ABS (0003): ABS_MT_SLOT : value 0, min 0, max 9\n                ABS_MT_POSITION_X : value 0, min 0, max 1079\n                ABS_MT_POSITION_Y : value 0, min 0, max 2399\n                ABS_MT_TRACKING_ID : value 0, min 0, max 20\n"""
    )
    assert descriptors == (
        mobile_testbed.TouchDescriptor("/dev/input/event4", "touch", 10, 1079, 2399),
    )
    monkeypatch.setattr(mobile_testbed, "_validate_serial", lambda serial: serial)
    observer = mobile_testbed.TouchObserver("test-1", descriptors[0])
    observer._decode(
        StringIO(
            "[ 1.0] /dev/input/event4: EV_ABS ABS_MT_SLOT 00000000\n"
            "[ 1.0] /dev/input/event4: EV_ABS ABS_MT_TRACKING_ID 00000011\n"
            "[ 1.0] /dev/input/event4: EV_ABS ABS_MT_POSITION_X 00000064\n"
            "[ 1.0] /dev/input/event4: EV_ABS ABS_MT_POSITION_Y 000000c8\n"
            "[ 1.0] /dev/input/event4: EV_ABS ABS_MT_SLOT 00000001\n"
            "[ 1.0] /dev/input/event4: EV_ABS ABS_MT_TRACKING_ID 00000012\n"
            "[ 1.0] /dev/input/event4: EV_ABS ABS_MT_POSITION_X 0000012c\n"
            "[ 1.0] /dev/input/event4: EV_ABS ABS_MT_POSITION_Y 00000190\n"
            "[ 1.0] /dev/input/event4: EV_SYN SYN_REPORT 00000000\n"
        )
    )
    first, second = observer.read(0.0), observer.read(0.0)
    assert (first.slot, first.tracking_id, first.x, first.y) == (0, 17, 100, 200)  # type: ignore[union-attr]
    assert (second.slot, second.tracking_id, second.x, second.y) == (1, 18, 300, 400)  # type: ignore[union-attr]


def test_type_a_decoder_assigns_stable_logical_slots(monkeypatch) -> None:
    monkeypatch.setattr(mobile_testbed, "_validate_serial", lambda serial: serial)
    descriptor = mobile_testbed.TouchDescriptor(
        "/dev/input/event7", "touch", 2, 719, 1599, "type_a"
    )
    observer = mobile_testbed.TouchObserver("test-1", descriptor)
    observer._decode(
        StringIO("[ 1.0] /dev/input/event7: EV_ABS ABS_MT_TRACKING_ID 00000000\n[ 1.0] /dev/input/event7: EV_ABS ABS_MT_POSITION_X 00000090\n[ 1.0] /dev/input/event7: EV_ABS ABS_MT_POSITION_Y 000004ff\n[ 1.0] /dev/input/event7: EV_SYN SYN_MT_REPORT 00000000\n[ 1.0] /dev/input/event7: EV_SYN SYN_REPORT 00000000\n[ 1.0] /dev/input/event7: EV_ABS ABS_MT_TRACKING_ID 00000000\n[ 1.0] /dev/input/event7: EV_ABS ABS_MT_POSITION_X 00000230\n[ 1.0] /dev/input/event7: EV_ABS ABS_MT_POSITION_Y 000004ff\n[ 1.0] /dev/input/event7: EV_SYN SYN_MT_REPORT 00000000\n[ 1.0] /dev/input/event7: EV_ABS ABS_MT_TRACKING_ID 00000000\n[ 1.0] /dev/input/event7: EV_ABS ABS_MT_POSITION_X 0000012c\n[ 1.0] /dev/input/event7: EV_ABS ABS_MT_POSITION_Y 000004b0\n[ 1.0] /dev/input/event7: EV_SYN SYN_MT_REPORT 00000000\n[ 1.0] /dev/input/event7: EV_SYN SYN_REPORT 00000000\n")
    )
    packets = [item for item in iter(lambda: observer.read(0.0), None)]
    assert [(item.slot, item.tracking_id, item.x, item.y) for item in packets] == [(0, 0, 144, 1279), (0, 0, 300, 1200), (1, 0, 560, 1279)]


def test_touch_action_mapper_derives_parallel_move_and_skill() -> None:
    layout = mobile_testbed.Layout(
        1000,
        1000,
        (0.2, 0.8),
        0.15,
        (0.0, -1.0),
        150,
        250,
        0.2,
        {"basic_attack": (0.8, 0.8), "skill1": (0.7, 0.8), "skill2": None, "skill3": None},
    )
    descriptor = mobile_testbed.TouchDescriptor("/dev/input/event4", "touch", 10, 1000, 1000)
    calibration = mobile_testbed.TouchCalibration(
        descriptor.sha256, "a" * 64, False, False, False, 5.0, 0.08, 0.1
    )
    mapper = mobile_testbed._TouchActionMapper(descriptor, calibration, layout)
    mapper.feed(mobile_testbed.TouchPacket(1_000_000_000, 0, 1, 300, 800))
    mapper.feed(mobile_testbed.TouchPacket(1_000_000_000, 1, 2, 700, 730))
    mapper.feed(mobile_testbed.TouchPacket(1_200_000_000, 0, 1, 300, 600))
    mapper.feed(mobile_testbed.TouchPacket(1_200_000_000, 1, 2, 700, 580))
    assert mapper.action(1_300_000_000) == mobile_testbed.FactorizedAction(
        movement="north", ability="skill1", aim="north", hold_ms=300
    )
    assert mapper.parallel is True


def test_touch_calibration_uses_only_fixed_buttons_not_the_joystick() -> None:
    layout = mobile_testbed.Layout(
        1000,
        1000,
        (0.2, 0.8),
        0.15,
        (0.0, -1.0),
        150,
        250,
        0.2,
        {
            "basic_attack": (0.83, 0.84),
            "skill1": (0.69, 0.87),
            "skill2": (0.75, 0.69),
            "skill3": (0.83, 0.59),
        },
    )
    descriptor = mobile_testbed.TouchDescriptor(
        "/dev/input/event7", "touch", 2, 1000, 1000, "type_a"
    )
    raw_points = {
        name: (round(point[1] * 1000), round(point[0] * 1000))
        for name, point in layout.buttons.items()
    }
    calibration = mobile_testbed.calibrate_touch_transform(
        descriptor=descriptor, layout=layout, layout_sha256="a" * 64, raw_points=raw_points
    )
    assert calibration.affine is not None
    assert calibration.transform(*raw_points["skill2"], descriptor) == pytest.approx(
        layout.buttons["skill2"]
    )


def test_touch_coverage_reports_missing_factors_without_external_inspection() -> None:
    coverage = mobile_testbed._touch_factor_coverage(
        [
            mobile_testbed.FactorizedAction(),
            mobile_testbed.FactorizedAction(movement="north", hold_ms=200),
            mobile_testbed.FactorizedAction(ability="skill1", aim="east", hold_ms=500),
        ]
    )
    assert coverage["complete"] is False
    assert "south" in coverage["missing"]["movement"]
    assert coverage["counts"]["combat"]["skill1"] == 1


def test_scripted_reader_is_seeded_and_covers_every_factor(monkeypatch) -> None:
    clock = iter(float(index) * 2.0 for index in range(80))
    monkeypatch.setattr(mobile_testbed.time, "monotonic", lambda: next(clock))
    reader = mobile_testbed.scripted_key_reader(seed=7, interval_seconds=1.5)
    controller = mobile_testbed.KeyboardDemonstrator(mobile_testbed.DEFAULT_LAYOUT)
    actions = []
    for _ in range(58):
        action, _stopped = controller.feed(reader(0.0) or "")
        if action is not None:
            actions.append(action)
    assert {action.movement for action in actions} == set(mobile_testbed.MOVEMENTS)
    assert {action.ability for action in actions} == set(mobile_testbed.ABILITIES)
    assert {action.aim for action in actions} == set(mobile_testbed.AIMS)


def test_demonstrator_writes_derived_shards_and_actual_events(tmp_path: Path, monkeypatch) -> None:
    commands: list[tuple[str, ...]] = []

    class Pipe:
        def __init__(self, _serial: str) -> None:
            pass

        def send(self, *args: str) -> None:
            commands.append(args)

        def close(self) -> None:
            pass

    class Stream:
        def __init__(self, *_: object) -> None:
            pass

        def start(self) -> None:
            pass

        def frame(self) -> np.ndarray:
            return np.zeros((720, 1600, 3), dtype=np.uint8)

        def close(self) -> None:
            pass

    keys = iter(("w", "f", "\x1b"))
    root = tmp_path / "large"
    root.mkdir()
    monkeypatch.setenv("HOK_LARGE_ROOT", str(root))
    _guard(monkeypatch)
    monkeypatch.setattr(mobile_testbed, "AdbInputPipe", Pipe)
    monkeypatch.setattr(mobile_testbed, "ScrcpyV4L2", Stream)
    layout = tmp_path / "layout.json"
    layout.write_text(
        json.dumps(
            {
                "schema_version": mobile_testbed.LAYOUT_SCHEMA,
                "screen": {"width": 1600, "height": 720},
                "joystick": {
                    "center": [0.2, 0.8],
                    "radius": 0.1,
                    "forward_vector": [0, -1],
                    "move_hold_ms": 150,
                    "skill_hold_ms": 250,
                    "aim_radius": 0.2,
                },
                "buttons": {
                    "basic_attack": [0.8, 0.8],
                    "skill1": [0.7, 0.8],
                    "skill2": [0.75, 0.75],
                    "skill3": [0.8, 0.7],
                },
            }
        ),
        encoding="utf-8",
    )
    result = mobile_testbed.run_mobile_demonstrate(
        serial="test-1",
        output_dir=root / "datasets" / "demo",
        run_seconds=2,
        max_actions=3,
        layout_path=layout,
        capture_mode="scrcpy-v4l2",
        video_node=Path("/dev/video42"),
        key_reader=lambda _timeout: next(keys),
        event_source=mobile_testbed.SCRIPTED_DEMONSTRATION_SOURCE,
    )
    assert result["samples"] == 2
    assert result["executed_actions"] == 2
    assert result["derived_rgb_frames_persisted"] is True
    with np.load(root / "datasets" / "demo" / "samples-00000.npz", allow_pickle=False) as shard:
        assert shard["frames"].shape == (2, 8, 128, 128, 3)
        assert shard["movement"].tolist() == [1, 0]
        assert shard["ability"].tolist() == [0, 1]
    rows = [
        json.loads(line)
        for line in (root / "datasets" / "demo" / "events.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2 and all(row["input_sent"] for row in rows)
    assert all(row["source"] == mobile_testbed.SCRIPTED_DEMONSTRATION_SOURCE for row in rows)
    summary = json.loads((root / "datasets" / "demo" / "summary.json").read_text())
    assert summary["event_source"] == mobile_testbed.SCRIPTED_DEMONSTRATION_SOURCE
    assert commands[0][0] == "swipe" and commands[1][0] == "tap"


def test_keyboard_v2_writer_binds_causal_windows_and_dispatch_source(
    tmp_path: Path, monkeypatch
) -> None:
    commands: list[tuple[str, ...]] = []

    class Pipe:
        def __init__(self, _serial: str) -> None:
            pass

        def send(self, *args: str) -> None:
            commands.append(args)

        def close(self) -> None:
            pass

    class Stream:
        def __init__(self, *_: object) -> None:
            pass

        def start(self) -> None:
            pass

        def frame(self) -> np.ndarray:
            return np.zeros((720, 1600, 3), dtype=np.uint8)

        def close(self) -> None:
            pass

    root = tmp_path / "large"
    root.mkdir()
    monkeypatch.setenv("HOK_LARGE_ROOT", str(root))
    _guard(monkeypatch)
    monkeypatch.setattr(mobile_testbed, "AdbInputPipe", Pipe)
    monkeypatch.setattr(mobile_testbed, "ScrcpyV4L2", Stream)
    layout = tmp_path / "layout.json"
    layout.write_text(
        json.dumps(
            {
                "schema_version": mobile_testbed.LAYOUT_SCHEMA,
                "screen": {"width": 1600, "height": 720},
                "joystick": {
                    "center": [0.2, 0.8],
                    "radius": 0.1,
                    "forward_vector": [0, -1],
                    "move_hold_ms": 150,
                    "skill_hold_ms": 250,
                    "aim_radius": 0.2,
                },
                "buttons": {
                    "basic_attack": [0.8, 0.8],
                    "skill1": [0.7, 0.8],
                    "skill2": [0.75, 0.75],
                    "skill3": [0.8, 0.7],
                },
            }
        ),
        encoding="utf-8",
    )
    keys = iter(("1", "w", "k", "f", "s", "\x1b"))
    output = root / "datasets" / "t8-demonstrations-v2" / "smoke"
    result = mobile_testbed.run_mobile_demonstrate_keyboard_v2(
        serial="test-1",
        output_dir=output,
        layout_path=layout,
        video_node=Path("/dev/video42"),
        run_seconds=2,
        key_reader=lambda _timeout: next(keys),
    )
    assert result["event_source"] == mobile_testbed.EXECUTED_ACTION_SOURCE
    with np.load(output / "samples-00000.npz", allow_pickle=False) as shard:
        assert shard["frames"].shape == (3, 16, 128, 128, 3)
        assert shard["label_source"].tolist() == [1, 1, 1]
        assert shard["input_sent"].tolist() == [1, 1, 0]
        assert shard["hold_bucket"].tolist() == [1, 2, 0]
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert [row["input_sent"] for row in events] == [True, True, False]
    assert all(row["source"] == mobile_testbed.EXECUTED_ACTION_SOURCE for row in events)
    assert all(command[0] == "swipe" for command in commands)


def test_guided_calibration_writes_new_complete_layout(tmp_path: Path, monkeypatch) -> None:
    source = _write_layout(tmp_path, complete=False)
    commands: list[tuple[str, ...]] = []

    class Pipe:
        def __init__(self, _serial: str) -> None:
            pass

        def send(self, *args: str) -> None:
            commands.append(args)

        def close(self) -> None:
            pass

    class Stream:
        def __init__(self, *_: object) -> None:
            pass

        def start(self) -> None:
            pass

        def frame(self) -> np.ndarray:
            return np.zeros((720, 1600, 3), dtype=np.uint8)

        def close(self) -> None:
            pass

    points = {
        "joystick_center": (0.20, 0.80),
        "joystick_north_endpoint": (0.20, 0.68),
        **{name: (0.55, 0.50) for name in mobile_testbed.ABILITIES[2:]},
    }
    _guard(monkeypatch)
    monkeypatch.setattr(mobile_testbed, "AdbInputPipe", Pipe)
    monkeypatch.setattr(mobile_testbed, "ScrcpyV4L2", Stream)
    confirmations = iter((None, *([True] * 12)))
    result = mobile_testbed.run_layout_calibration(
        serial="test-1",
        layout_path=source,
        output_path=tmp_path / "candidate.json",
        video_node=Path("/dev/video42"),
        stream_fps=30,
        point_provider=points.__getitem__,
        confirmer=lambda _name: next(confirmations),
    )
    payload = json.loads((tmp_path / "candidate.json").read_text(encoding="utf-8"))
    assert result["control_actions"] == 12
    assert payload["calibration_status"].startswith("COMPLETE")
    assert all(value is not None for value in payload["buttons"].values())
    assert payload["joystick"]["center"] == [0.2, 0.8]
    assert payload["joystick"]["radius"] == pytest.approx(0.12)
    assert payload["joystick"]["forward_vector"] == pytest.approx([0.0, -1.0])
    assert "targets" not in payload
    assert len(commands) >= 13


def _live_layout() -> mobile_testbed.Layout:
    return mobile_testbed.Layout(
        1600,
        720,
        (0.2, 0.8),
        0.1,
        (0.0, -1.0),
        150,
        250,
        0.2,
        {
            "basic_attack": (0.8, 0.8),
            "skill1": (0.7, 0.8),
            "skill2": (0.75, 0.75),
            "skill3": (0.8, 0.7),
        },
    )


def test_scrcpy_touch_message_matches_v125_wire_contract(monkeypatch) -> None:
    monkeypatch.setattr(mobile_testbed, "_require_mobile_input_identity", lambda: None)
    messages: list[bytes] = []

    class ControlSocket:
        def sendall(self, message: bytes) -> None:
            messages.append(message)

    session = mobile_testbed.ScrcpyControlSession("test-1", 30)
    session._control_socket = ControlSocket()
    operation = mobile_testbed.TouchOperation(
        mobile_testbed.ANDROID_ACTION_DOWN, mobile_testbed.COMBAT_POINTER_ID, 1200, 600
    )
    assert session.touch(operation, 1600, 720) > 0
    assert len(messages[0]) == 28
    assert messages[0][:2] == bytes(
        (mobile_testbed.SCRCPY_CONTROL_TOUCH, mobile_testbed.ANDROID_ACTION_DOWN)
    )
    assert int.from_bytes(messages[0][2:10], "big") == mobile_testbed.COMBAT_POINTER_ID
    assert int.from_bytes(messages[0][10:14], "big") == 1200
    assert int.from_bytes(messages[0][14:18], "big") == 600
    assert int.from_bytes(messages[0][18:20], "big") == 1600
    assert int.from_bytes(messages[0][20:22], "big") == 720
    assert messages[0][22:24] == b"\xff\xff"


def test_live_keyboard_uses_continuous_two_pointer_lifecycles() -> None:
    controller = mobile_testbed.LiveKeyboardController(_live_layout(), 1600, 720)
    movement, changed, conflict = controller.transition("w", True, 1_000_000_000)
    assert changed and not conflict
    assert [item.action for item in movement] == [
        mobile_testbed.ANDROID_ACTION_DOWN,
        mobile_testbed.ANDROID_ACTION_MOVE,
    ]
    assert all(item.pointer_id == mobile_testbed.JOYSTICK_POINTER_ID for item in movement)

    combat, changed, conflict = controller.transition("1", True, 1_100_000_000)
    assert changed and not conflict and len(combat) == 1
    assert combat[0].pointer_id == mobile_testbed.COMBAT_POINTER_ID
    assert controller.action(1_300_000_000) == mobile_testbed.FactorizedAction(
        movement="north", ability="skill1", hold_ms=200
    )

    aim, changed, conflict = controller.transition("Right", True, 1_310_000_000)
    assert changed and not conflict and aim[0].action == mobile_testbed.ANDROID_ACTION_MOVE
    assert aim[0].pointer_id == mobile_testbed.COMBAT_POINTER_ID
    assert controller.action(1_400_000_000).aim == "east"

    ignored, changed, conflict = controller.transition("f", True, 1_410_000_000)
    assert ignored == [] and not changed and conflict and controller.conflict_count == 1

    stopped, changed, conflict = controller.transition("w", False, 1_500_000_000)
    assert changed and not conflict
    assert stopped[0] == mobile_testbed.TouchOperation(
        mobile_testbed.ANDROID_ACTION_UP,
        mobile_testbed.JOYSTICK_POINTER_ID,
        movement[-1].x,
        movement[-1].y,
    )
    released, changed, conflict = controller.transition("1", False, 1_600_000_000)
    assert changed and not conflict and released[0].action == mobile_testbed.ANDROID_ACTION_UP
    assert controller.action(1_700_000_000) == mobile_testbed.FactorizedAction()


def test_live_keyboard_direction_change_sends_move_without_new_down() -> None:
    controller = mobile_testbed.LiveKeyboardController(_live_layout(), 1600, 720)
    controller.transition("w", True, 1)
    operations, changed, conflict = controller.transition("d", True, 2)
    assert changed and not conflict
    assert len(operations) == 1
    assert operations[0].action == mobile_testbed.ANDROID_ACTION_MOVE
    assert controller.action(3).movement == "north_east"


def test_v21_writer_streams_shards_and_keeps_semantic_contract(tmp_path: Path) -> None:
    output = tmp_path / "dataset" / "session-001"
    output.parent.mkdir()
    contract, digest = mobile_testbed._keyboard_v21_contract("layout-hash")
    assert contract["scrcpy_server_version"] == mobile_testbed.SCRCPY_SERVER_VERSION
    assert contract["scrcpy_server_sha256"] == mobile_testbed.SCRCPY_SERVER_SHA256
    assert contract["pointer_roles"] == {"joystick": 0, "combat": 1}
    diagnostic, _ = mobile_testbed._keyboard_v21_contract(
        "layout-hash", mobile_testbed.DIAGNOSTIC_CONTROL_SOURCE
    )
    assert diagnostic["source"] == mobile_testbed.DIAGNOSTIC_CONTROL_SOURCE
    inverse, _ = mobile_testbed._keyboard_v21_contract(
        "layout-hash", mobile_testbed.DIAGNOSTIC_INVERSE_SOURCE
    )
    assert inverse["source"] == mobile_testbed.DIAGNOSTIC_INVERSE_SOURCE
    writer = mobile_testbed.KeyboardV21Writer(output, contract, shard_size=2)
    frame = np.zeros((16, 128, 128, 3), dtype=np.uint8)
    for sequence in range(3):
        writer.add_sample(
            mobile_testbed.LiveSample(
                frame,
                mobile_testbed.FactorizedAction(movement="north"),
                sequence + 10,
                sequence,
                sequence + 5,
                True,
            )
        )
    writer.add_event(
        {
            "schema_version": mobile_testbed.KEYBOARD_V21_SCHEMA,
            "source": mobile_testbed.SCRCPY_EXECUTED_ACTION_SOURCE,
        }
    )
    writer.finalize(
        {
            "schema_version": mobile_testbed.KEYBOARD_V21_SCHEMA,
            "action_contract_sha256": digest,
        }
    )
    assert sorted(path.name for path in output.glob("samples-*.npz")) == [
        "samples-00000.npz",
        "samples-00001.npz",
    ]
    with np.load(output / "samples-00001.npz", allow_pickle=False) as shard:
        assert shard["frames"].shape == (1, 16, 128, 128, 3)
        assert shard["transition_sequence"].tolist() == [2]
    summary = json.loads((output / "summary.json").read_text())
    assert summary["schema_version"] == mobile_testbed.KEYBOARD_V21_SCHEMA


def test_inverse_probe_schedule_is_combat_only_balanced_and_bounded() -> None:
    events = mobile_testbed.inverse_probe_events(60)
    presses = [key for _offset, key, pressed in events if pressed]
    releases = [key for _offset, key, pressed in events if not pressed]
    assert presses == releases
    assert set(presses) == {"f", "1", "2", "3"}
    assert max(presses.count(key) for key in set(presses)) - min(
        presses.count(key) for key in set(presses)
    ) <= 1
    assert all(0 < offset < 60 for offset, _key, _pressed in events)


@pytest.mark.parametrize("run_seconds", [float("inf"), float("nan")])
def test_v21_rejects_nonfinite_duration_before_device_access(
    tmp_path: Path, run_seconds: float
) -> None:
    with pytest.raises(mobile_testbed.MobileTestbedError, match="bounds"):
        mobile_testbed.run_mobile_demonstrate_keyboard_v21(
            serial="test-1",
            output_dir=tmp_path / "out",
            layout_path=tmp_path / "layout.json",
            run_seconds=run_seconds,
        )


def test_scrcpy_session_rejects_stale_or_ended_video() -> None:
    session = mobile_testbed.ScrcpyControlSession("test-1", 30)
    session._frame = np.zeros((720, 1600, 3), dtype=np.uint8)
    session._frame_timestamp_ns = 1
    with pytest.raises(mobile_testbed.MobileTestbedError, match="fresh"):
        session.frame()
    session._error = EOFError()
    with pytest.raises(mobile_testbed.MobileTestbedError, match="ended"):
        session.frame()


def test_v21_incomplete_writer_can_publish_only_as_diagnostic(tmp_path: Path) -> None:
    requested = tmp_path / "dataset" / "session-001"
    requested.parent.mkdir()
    diagnostic = requested.parent / "diagnostics" / "session-001-attempt-1"
    contract, _digest = mobile_testbed._keyboard_v21_contract("layout-hash")
    writer = mobile_testbed.KeyboardV21Writer(requested, contract, shard_size=2)
    writer.add_sample(
        mobile_testbed.LiveSample(
            np.zeros((16, 128, 128, 3), dtype=np.uint8),
            mobile_testbed.FactorizedAction(),
            10,
            0,
            0,
            True,
        )
    )
    writer.finalize(
        {
            "schema_version": mobile_testbed.KEYBOARD_V21_SCHEMA,
            "status": "INCOMPLETE_FORMAL_ATTEMPT",
            "published_as_formal": False,
        },
        diagnostic,
    )
    assert not requested.exists()
    assert (diagnostic / "summary.json").is_file()


def test_rgb_teacher_uses_activity_button_margin_and_abstains(tmp_path: Path) -> None:
    layout, _digest = mobile_testbed.load_layout(_write_layout(tmp_path))
    calibration = mobile_testbed.RGBTeacherCalibration(
        "a" * 64,
        "b" * 64,
        0.10,
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
    )
    history = np.zeros((3, 128, 128, 3), dtype=np.uint8)
    current = history.copy()
    current[1] = 64
    point = layout.buttons["skill1"]
    assert point is not None
    center_x = round((point[0] - 0.52) / 0.48 * 127)
    center_y = round((point[1] - 0.30) / 0.70 * 127)
    current[2, center_y - 3 : center_y + 4, center_x - 3 : center_x + 4, 0] = 255
    decision = mobile_testbed.rgb_teacher_decision(
        current, history, layout, calibration
    )
    assert decision.combat_id == mobile_testbed.ABILITIES.index("skill1")
    assert decision.activity >= calibration.activity_threshold
    assert decision.margin >= mobile_testbed.RGB_TEACHER_MARGIN
    low_activity = current.copy()
    low_activity[1] = 8
    live_decision = mobile_testbed.rgb_teacher_decision(
        low_activity, history, layout, calibration
    )
    assert mobile_testbed.RGB_TEACHER_LIVE_ACTIVITY_THRESHOLD <= live_decision.activity < 0.10
    assert live_decision.combat_id == mobile_testbed.ABILITIES.index("skill1")
    enemy = current.copy()
    enemy[1] = 0
    enemy[0, 50, 30:55, 0] = 255
    enemy_decision = mobile_testbed.rgb_teacher_decision(
        enemy, history, layout, calibration
    )
    assert enemy_decision.enemy_cue is True
    assert enemy_decision.enemy_red_row_max >= mobile_testbed.RGB_TEACHER_ENEMY_RED_ROW_MAX
    assert enemy_decision.combat_id == mobile_testbed.ABILITIES.index("skill1")
    near_enemy = current.copy()
    near_enemy[1] = 0
    near_enemy[0, 50, 30:40, 0] = 255
    assert (
        mobile_testbed.rgb_teacher_decision(
            near_enemy, history, layout, calibration
        ).enemy_cue
        is False
    )
    assert (
        mobile_testbed.rgb_teacher_decision(current, current, layout, calibration).combat_id
        == 0
    )


def test_rgb_teacher_minimap_navigation_tracks_nearest_red_target() -> None:
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    frame[9:12, 15:18, 1] = 180
    frame[10, 22:25, 0] = 180
    navigation = mobile_testbed.rgb_teacher_minimap_navigation(frame)
    assert navigation is not None
    assert navigation.movement == "east"
    assert navigation.player_yx == (10.0, 10.0)
    assert navigation.target_yx == (10, 16)


def test_rgb_teacher_writer_stores_one_frame_stream_not_repeated_windows(
    tmp_path: Path,
) -> None:
    output = tmp_path / "session"
    frames = [
        (index * 100_000_000, np.full((128, 128, 3), index, dtype=np.uint8))
        for index in range(52)
    ]
    samples = [
        mobile_testbed.RGBTeacherSample(51, 31, 0, 5_100_000_000, 5_200_000_000, -1, 0.0, False),
        mobile_testbed.RGBTeacherSample(
            51,
            31,
            2,
            5_100_000_000,
            5_300_000_000,
            5_400_000_000,
            0.9,
            True,
        ),
    ]
    calibration = mobile_testbed.RGBTeacherCalibration(
        "a" * 64, "b" * 64, 0.1, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    )
    contract, _digest = mobile_testbed._rgb_teacher_contract("b" * 64, calibration)
    assert contract["activity_threshold"] == mobile_testbed.RGB_TEACHER_LIVE_ACTIVITY_THRESHOLD
    assert (
        contract["global_dispatch_interval_ms"]
        == mobile_testbed.RGB_TEACHER_GLOBAL_DISPATCH_INTERVAL_MS
    )
    assert contract["minimum_formal_decision_coverage"] == 0.90
    mobile_testbed._publish_rgb_teacher_session(
        output,
        frames,
        samples,
        [],
        {"schema_version": mobile_testbed.RGB_TEACHER_SCHEMA},
        contract,
        32,
    )
    with np.load(output / "frames-00000.npz", allow_pickle=False) as shard:
        assert shard["frames"].shape == (32, 128, 128, 3)
    with np.load(output / "samples-00000.npz", allow_pickle=False) as shard:
        assert "views" not in shard.files
        assert shard["observation_index"].tolist() == [51, 51]
        assert shard["shifted_observation_index"].tolist() == [31, 31]


def test_rgb_teacher_dry_run_records_decisions_and_never_opens_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Clock:
        value = 0.0

        @classmethod
        def monotonic(cls) -> float:
            return cls.value

        @classmethod
        def monotonic_ns(cls) -> int:
            return int(cls.value * 1_000_000_000)

        @classmethod
        def sleep(cls, seconds: float) -> None:
            cls.value += max(0.0, seconds)

    class Stream:
        def __init__(self, *_args: object) -> None:
            pass

        def start(self) -> None:
            pass

        def frame_with_timestamp(self) -> tuple[int, np.ndarray]:
            return Clock.monotonic_ns(), np.zeros((720, 1600, 3), dtype=np.uint8)

        def close(self) -> None:
            pass

    class ForbiddenPipe:
        def __init__(self, _serial: str) -> None:
            raise AssertionError("dry-run opened an input pipe")

    root = tmp_path / "large"
    root.mkdir()
    monkeypatch.setenv("HOK_LARGE_ROOT", str(root))
    _guard(monkeypatch)
    monkeypatch.setattr(mobile_testbed, "ScrcpyV4L2", Stream)
    monkeypatch.setattr(mobile_testbed, "AdbInputPipe", ForbiddenPipe)
    monkeypatch.setattr(mobile_testbed.time, "monotonic", Clock.monotonic)
    monkeypatch.setattr(mobile_testbed.time, "monotonic_ns", Clock.monotonic_ns)
    monkeypatch.setattr(mobile_testbed.time, "sleep", Clock.sleep)
    layout = _write_layout(tmp_path)
    _layout, layout_sha = mobile_testbed.load_layout(layout)
    teacher = tmp_path / "teacher.json"
    teacher.write_text(
        json.dumps(
            {
                "schema_version": mobile_testbed.RGB_TEACHER_CALIBRATION_SCHEMA,
                "status": "OFFLINE_TEACHER_READY",
                "offline_teacher_ready": True,
                "training_eligible": False,
                "live_execution_allowed": False,
                "video_test_accessed": False,
                "layout_sha256": layout_sha,
                "combat_vocabulary": list(mobile_testbed.ABILITIES[:4]),
                "activity_threshold": 0.1,
                "button_score_medians": [0.0, 0.0, 0.0],
                "button_score_iqr": [1.0, 1.0, 1.0],
            }
        ),
        encoding="utf-8",
    )
    output = root / "datasets" / "dry-run"
    result = mobile_testbed.run_mobile_demonstrate_rgb_teacher_v25(
        serial="test-1",
        output_dir=output,
        layout_path=layout,
        video_node=Path("/dev/video42"),
        teacher_report_path=teacher,
        run_seconds=0.4,
    )
    assert result["input_enabled"] is False
    assert result["executed_actions"] == 0
    assert result["samples"] >= 1
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert events and all(not event["input_sent"] for event in events)
    with pytest.raises(mobile_testbed.MobileTestbedError, match="bounds"):
        mobile_testbed.run_mobile_demonstrate_rgb_teacher_v25(
            serial="test-1",
            output_dir=root / "datasets" / "invalid-warmup",
            layout_path=layout,
            video_node=Path("/dev/video42"),
            teacher_report_path=teacher,
            run_seconds=0.4,
            warmup_basic_attack=True,
        )
    with pytest.raises(mobile_testbed.MobileTestbedError, match="bounds"):
        mobile_testbed.run_mobile_demonstrate_rgb_teacher_v25(
            serial="test-1",
            output_dir=root / "datasets" / "invalid-patrol",
            layout_path=layout,
            video_node=Path("/dev/video42"),
            teacher_report_path=teacher,
            run_seconds=0.4,
            patrol=True,
        )
    with pytest.raises(mobile_testbed.MobileTestbedError, match="bounds"):
        mobile_testbed.run_mobile_demonstrate_rgb_teacher_v25(
            serial="test-1",
            output_dir=root / "datasets" / "invalid-balanced",
            layout_path=layout,
            video_node=Path("/dev/video42"),
            teacher_report_path=teacher,
            run_seconds=0.4,
            balanced_actions=True,
        )
