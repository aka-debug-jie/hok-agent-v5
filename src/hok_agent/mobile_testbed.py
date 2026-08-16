# ruff: noqa: E501

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import re
import select
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import termios
import threading
import time
import tty
from collections import Counter, deque
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from queue import Empty, Queue
from typing import Any, BinaryIO, TextIO, cast

import av
import numpy as np
from av.container import InputContainer

SCHEMA = "hok-agent-mobile-testbed-v1"
DEMONSTRATOR_SCHEMA = "hok-agent-mobile-demonstrate-v1"
DEMONSTRATOR_DATA_SCHEMA = "hok-agent-mobile-demonstrate-data-v3"
DEMONSTRATOR_SESSION_SCHEMA = "hok-agent-mobile-demonstrate-session-v3"
TOUCH_DEMONSTRATOR_SCHEMA = "hok-agent-mobile-demonstrate-touch-v3"
TOUCH_DEMONSTRATOR_DATA_SCHEMA = "hok-agent-mobile-demonstrate-touch-data-v2"
TOUCH_DEMONSTRATOR_SESSION_SCHEMA = "hok-agent-mobile-demonstrate-touch-session-v2"
KEYBOARD_V2_SCHEMA = "hok-agent-mobile-demonstrate-keyboard-v2"
KEYBOARD_V2_DATA_SCHEMA = "hok-agent-mobile-demonstrate-keyboard-data-v1"
KEYBOARD_V2_SESSION_SCHEMA = "hok-agent-mobile-demonstrate-keyboard-session-v1"
KEYBOARD_V21_SCHEMA = "hok-agent-mobile-demonstrate-keyboard-v2.1"
KEYBOARD_V21_DATA_SCHEMA = "hok-agent-mobile-demonstrate-keyboard-data-v2.1"
KEYBOARD_V21_SESSION_SCHEMA = "hok-agent-mobile-demonstrate-keyboard-session-v2.1"
RGB_TEACHER_SCHEMA = "hok-agent-mobile-demonstrate-rgb-teacher-v2.5.1"
RGB_TEACHER_DATA_SCHEMA = "hok-agent-mobile-demonstrate-rgb-teacher-data-v2"
RGB_TEACHER_SESSION_SCHEMA = "hok-agent-mobile-demonstrate-rgb-teacher-session-v2"
RGB_TEACHER_CALIBRATION_SCHEMA = "hok-agent-t8-v2.3-visual-teacher-replay-v1"
TOUCH_CALIBRATION_SCHEMA = "hok-agent-mobile-touch-calibration-v2"
LAYOUT_SCHEMA = "hok-agent-mobile-layout-v3"
MOBILE_BUILD_IDENTITY_SCHEMA = "hok-agent-mobile-build-identity-v1"
MOBILE_BUILD_IDENTITY_DEFAULT_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "mobile_testbed_identity.local.json"
)
ANDROID_PACKAGE_RE = re.compile(
    r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)+"
)
DEMONSTRATOR_WINDOW_FRAMES = 8
DEMONSTRATOR_SAMPLE_HZ = 10
TOUCH_WINDOW_FRAMES = 16
TOUCH_SAMPLE_HZ = 10
TOUCH_HOLD_BUCKETS = ("none", "short", "medium", "long")
TOUCH_BUTTON_TOLERANCE_MULTIPLIER = 2.0
CALIBRATION_MOVE_HOLD_MS = 800
TERMINAL_DEMONSTRATION_SOURCE = "terminal_keyboard"
SCRIPTED_DEMONSTRATION_SOURCE = "bounded_scripted_controller_v1"
OBSERVED_TOUCH_DEMONSTRATION_SOURCE = "observed_touch_action"
EXECUTED_ACTION_SOURCE = "executed_action"
SCRCPY_EXECUTED_ACTION_SOURCE = "scrcpy_control_executed_action"
DIAGNOSTIC_CONTROL_SOURCE = "bounded_scrcpy_control_smoke_v1"
DIAGNOSTIC_INVERSE_SOURCE = "bounded_scrcpy_inverse_probe_v1"
RGB_TEACHER_SOURCE = "rgb_conditioned_executed_action_v2"
DEMONSTRATION_SOURCES = (
    TERMINAL_DEMONSTRATION_SOURCE,
    SCRIPTED_DEMONSTRATION_SOURCE,
)
ACTIONS = ("wait", "forward", "backward", "attack_hero", "attack_tower", "attack_crystal")
MOVEMENTS = (
    "wait",
    "north",
    "north_east",
    "east",
    "south_east",
    "south",
    "south_west",
    "west",
    "north_west",
)
ABILITIES = ("none", "basic_attack", "skill1", "skill2", "skill3")
AIMS = ("none", *MOVEMENTS[1:])
TARGETS = ("none",)
KEY_TO_MOVEMENT = dict(zip("wedcxzaq", MOVEMENTS[1:], strict=True))
KEY_TO_ABILITY = {"f": "basic_attack", "1": "skill1", "2": "skill2", "3": "skill3"}
KEY_TO_HOLD_MS = {"j": 200, "k": 500, "l": 900}
KEYBOARD_V2_MIN_FORMAL_SAMPLES = 180
TOUCH_MIN_FORMAL_SAMPLES = 2850
TOUCH_MAX_FORMAL_SAMPLES = 3300
RGB_TEACHER_WINDOW_FRAMES = 32
RGB_TEACHER_SAMPLE_HZ = 10
RGB_TEACHER_DECISION_HZ = 5
RGB_TEACHER_HISTORY_FRAMES = 20
RGB_TEACHER_EXECUTION_LAG_MS = 100
RGB_TEACHER_MARGIN = 0.25
RGB_TEACHER_LIVE_ACTIVITY_THRESHOLD = 0.02
RGB_TEACHER_GLOBAL_DISPATCH_INTERVAL_MS = 1_000
RGB_TEACHER_MIN_FORMAL_SAMPLES = 1400
RGB_TEACHER_MIN_FORMAL_DECISION_COVERAGE = 0.90
RGB_TEACHER_PATROL_INTERVAL_SECONDS = 5.0
RGB_TEACHER_PATROL_HOLD_MS = 1_000
RGB_TEACHER_PATROL_DIRECTIONS = (
    *(("north",) * 6),
    *(("east",) * 6),
    *(("south",) * 6),
    *(("west",) * 6),
)
RGB_TEACHER_ENEMY_RED_PIXELS = 400
RGB_TEACHER_ENEMY_RED_ROW_MAX = 11
RGB_TEACHER_CLASS_COOLDOWN_MS = (0, 1_000, 8_000, 8_000)
SCRCPY_FRAME_STALE_NS = 2_000_000_000
SCRCPY_SERVER_VERSION = "1.25"
SCRCPY_SERVER_PATH = Path("/usr/share/scrcpy/scrcpy-server")
SCRCPY_SERVER_SHA256 = "29f50cc567e295859d01b86026c33962622ab3467a62ffd6521d48f9770807ac"
SCRCPY_CONTROL_TOUCH = 2
ANDROID_ACTION_DOWN = 0
ANDROID_ACTION_UP = 1
ANDROID_ACTION_MOVE = 2
JOYSTICK_POINTER_ID = 0
COMBAT_POINTER_ID = 1
LIVE_MOVEMENT_KEYS = {"w": "north", "d": "east", "s": "south", "a": "west"}
LIVE_COMBAT_KEYS = {"f": "basic_attack", "1": "skill1", "2": "skill2", "3": "skill3"}
LIVE_AIM_KEYS = {"Up": "north", "Right": "east", "Down": "south", "Left": "west"}
DIAGNOSTIC_SMOKE_EVENTS = (
    (0.5, "w", True),
    (1.5, "1", True),
    (1.8, "Right", True),
    (2.8, "Right", False),
    (3.0, "1", False),
    (3.3, "f", True),
    (3.6, "f", False),
    (4.0, "w", False),
    (4.5, "d", True),
    (5.5, "2", True),
    (5.8, "Up", True),
    (6.8, "Up", False),
    (7.0, "2", False),
    (7.5, "3", True),
    (8.5, "3", False),
    (9.0, "d", False),
)


def inverse_probe_events(run_seconds: float) -> tuple[tuple[float, str, bool], ...]:
    events: list[tuple[float, str, bool]] = []
    offset = 1.0
    ability_keys = ("f", "1", "2", "3")
    while offset + 1.0 < run_seconds:
        for key in ability_keys:
            if offset + 1.0 >= run_seconds:
                break
            events.extend(((offset, key, True), (offset + 0.2, key, False)))
            offset += 3.0
    return tuple(events)
SERIAL_RE = re.compile(r"[A-Za-z0-9._-]+")
TOUCH_DEVICE_RE = re.compile(r"/dev/input/event[0-9]+")
GETEVENT_LINE_RE = re.compile(
    r"^\[\s*([0-9]+\.[0-9]+)\]\s+(?:/dev/input/event[0-9]+:\s+)?"
    r"([A-Z_]+)\s+([A-Z0-9_]+)\s+([0-9a-fA-F]+)$"
)


class MobileTestbedError(ValueError):
    pass


@dataclass(frozen=True)
class DeviceGuard:
    serial: str
    package: str
    width: int
    height: int
    rotation: int

    def check(self) -> None:
        _validate_serial(self.serial)
        width, height, rotation = _active_display(self.serial, self.package)
        if (width, height, rotation) != (self.width, self.height, self.rotation):
            raise MobileTestbedError("test device display changed during the bounded run")


class GuardWatchdog:

    def __init__(self, guard: DeviceGuard, interval_seconds: float = 0.1) -> None:
        self._guard = guard
        self._interval_seconds = interval_seconds
        self._lock = threading.Lock()
        self._allowed = True
        self._checked_ns = time.monotonic_ns()
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._guard.check()
        with self._lock:
            self._checked_ns = time.monotonic_ns()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            try:
                self._guard.check()
            except BaseException as exc:
                with self._lock:
                    self._allowed = False
                    self._error = exc
                    self._checked_ns = time.monotonic_ns()
                return
            with self._lock:
                self._allowed = True
                self._checked_ns = time.monotonic_ns()

    def ensure_fresh(self, maximum_age_ms: int = 500) -> None:
        with self._lock:
            allowed = self._allowed
            checked_ns = self._checked_ns
            error = self._error
        if not allowed:
            raise MobileTestbedError("mobile device guard stopped the live session") from error
        if time.monotonic_ns() - checked_ns > maximum_age_ms * 1_000_000:
            raise MobileTestbedError("mobile device guard snapshot is stale")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)


@dataclass(frozen=True)
class Layout:
    width: int
    height: int
    joystick_center: tuple[float, float]
    joystick_radius: float
    forward_vector: tuple[float, float]
    move_hold_ms: int
    skill_hold_ms: int
    aim_radius: float
    buttons: dict[str, tuple[float, float] | None]

    @property
    def basic_attack(self) -> tuple[float, float]:
        point = self.buttons["basic_attack"]
        if point is None:
            raise MobileTestbedError("layout action basic_attack is not calibrated")
        return point


@dataclass(frozen=True)
class Intent:
    movement: tuple[float, float] | None
    attack: bool
    target: str | None


@dataclass(frozen=True)
class FactorizedAction:
    movement: str = "wait"
    ability: str = "none"
    aim: str = "none"
    target: str = "none"
    hold_ms: int = 0


@dataclass(frozen=True)
class TouchOperation:
    action: int
    pointer_id: int
    x: int
    y: int


@dataclass(frozen=True)
class LiveSample:
    frames: np.ndarray
    action: FactorizedAction
    timestamp_ns: int
    transition_sequence: int
    last_dispatch_ns: int
    input_sent: bool


@dataclass(frozen=True)
class RGBTeacherCalibration:
    report_sha256: str
    layout_sha256: str
    activity_threshold: float
    medians: tuple[float, float, float]
    scales: tuple[float, float, float]


@dataclass(frozen=True)
class RGBTeacherDecision:
    combat_id: int
    activity: float
    normalized_scores: tuple[float, float, float]
    margin: float
    enemy_red_pixels: int
    enemy_red_row_max: int
    enemy_cue: bool


@dataclass(frozen=True)
class MinimapNavigation:
    movement: str
    player_yx: tuple[float, float]
    target_yx: tuple[int, int]


@dataclass(frozen=True)
class RGBTeacherSample:
    observation_index: int
    shifted_observation_index: int
    combat_id: int
    observation_end_timestamp_ns: int
    decision_timestamp_ns: int
    execution_timestamp_ns: int
    confidence: float
    input_sent: bool


@dataclass(frozen=True)
class TouchDescriptor:
    path: str
    name: str
    max_slots: int
    max_x: int
    max_y: int
    protocol: str = "type_b"

    def __post_init__(self) -> None:
        _touch_device(self.path)
        if (
            self.protocol not in {"type_a", "type_b"}
            or self.max_slots < 2
            or self.max_x <= 0
            or self.max_y <= 0
        ):
            raise MobileTestbedError(
                "touch descriptor must support recognized multitouch coordinates"
            )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "name": self.name,
                    "max_slots": self.max_slots,
                    "max_x": self.max_x,
                    "max_y": self.max_y,
                    "protocol": self.protocol,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()


@dataclass(frozen=True)
class TouchPacket:
    received_ns: int
    slot: int
    tracking_id: int | None
    x: int | None
    y: int | None


@dataclass(frozen=True)
class TouchCalibration:
    descriptor_sha256: str
    layout_sha256: str
    swap_axes: bool
    flip_x: bool
    flip_y: bool
    joystick_start_radius: float
    button_radius: float
    dead_zone_fraction: float
    affine: tuple[float, float, float, float, float, float] | None = None

    def transform(self, x: int, y: int, descriptor: TouchDescriptor) -> tuple[float, float]:
        if self.affine is not None:
            if len(self.affine) != 6:
                raise MobileTestbedError("touch calibration affine transform is invalid")
            raw_x, raw_y = x / descriptor.max_x, y / descriptor.max_y
            a, b, c, d, e, f = self.affine
            return (a * raw_x + b * raw_y + c, d * raw_x + e * raw_y + f)
        first, second = (y, x) if self.swap_axes else (x, y)
        first_max, second_max = (
            (descriptor.max_y, descriptor.max_x)
            if self.swap_axes
            else (descriptor.max_x, descriptor.max_y)
        )
        if first_max <= 0 or second_max <= 0:
            raise MobileTestbedError("touch descriptor coordinate range is invalid")
        norm_x, norm_y = first / first_max, second / second_max
        return (
            1.0 - norm_x if self.flip_x else norm_x,
            1.0 - norm_y if self.flip_y else norm_y,
        )


class KeyboardDemonstrator:

    def __init__(self, layout: Layout) -> None:
        self._layout = layout
        self._armed = "none"

    def feed(self, key: str) -> tuple[FactorizedAction | None, bool]:
        if len(key) == 1 and key.isascii():
            key = key.lower()
        if key == "\x1b":
            return (None, True)
        if key in KEY_TO_ABILITY and key != "f":
            self._armed = KEY_TO_ABILITY[key]
            return (None, False)
        if key in KEY_TO_MOVEMENT:
            direction = KEY_TO_MOVEMENT[key]
            if self._armed != "none":
                ability = self._armed
                self._armed = "none"
                return (
                    FactorizedAction(
                        ability=ability,
                        aim=direction,
                        hold_ms=self._layout.skill_hold_ms,
                    ),
                    False,
                )
            return (FactorizedAction(movement=direction, hold_ms=self._layout.move_hold_ms), False)
        if key in {" ", "s"}:
            if self._armed != "none":
                ability = self._armed
                self._armed = "none"
                return (
                    FactorizedAction(ability=ability, hold_ms=self._layout.skill_hold_ms),
                    False,
                )
            return (FactorizedAction(), False)
        if key == "f":
            return (
                FactorizedAction(ability="basic_attack", hold_ms=self._layout.skill_hold_ms),
                False,
            )
        return (None, False)


class KeyboardV2Demonstrator:

    def __init__(self) -> None:
        self._armed = "none"
        self._hold_ms = KEY_TO_HOLD_MS["j"]

    def feed(self, key: str) -> tuple[FactorizedAction | None, bool]:
        if len(key) == 1 and key.isascii():
            key = key.lower()
        if key == "\x1b":
            return (None, True)
        if key in KEY_TO_HOLD_MS:
            self._hold_ms = KEY_TO_HOLD_MS[key]
            return (None, False)
        if key in KEY_TO_ABILITY and key != "f":
            self._armed = KEY_TO_ABILITY[key]
            return (None, False)
        action: FactorizedAction | None = None
        if key in KEY_TO_MOVEMENT:
            direction = KEY_TO_MOVEMENT[key]
            action = (
                FactorizedAction(ability=self._armed, aim=direction, hold_ms=self._hold_ms)
                if self._armed != "none"
                else FactorizedAction(movement=direction, hold_ms=self._hold_ms)
            )
            self._armed = "none"
        elif key in {" ", "s"}:
            if self._armed != "none":
                action = FactorizedAction(ability=self._armed, hold_ms=self._hold_ms)
                self._armed = "none"
            else:
                action = FactorizedAction()
        elif key == "f":
            action = FactorizedAction(ability="basic_attack", hold_ms=self._hold_ms)
        if action is not None:
            self._hold_ms = KEY_TO_HOLD_MS["j"]
        return (action, False)


class TerminalKeyboard:

    def __init__(self) -> None:
        self._fd: int | None = None
        self._state: list[int | list[bytes | int]] | None = None

    def __enter__(self) -> TerminalKeyboard:
        if not sys.stdin.isatty():
            raise MobileTestbedError("mobile demonstration requires an interactive terminal")
        self._fd = sys.stdin.fileno()
        self._state = cast(list[int | list[bytes | int]], termios.tcgetattr(self._fd))
        tty.setcbreak(self._fd)
        return self

    def read(self, timeout: float) -> str | None:
        if self._fd is None:
            raise MobileTestbedError("terminal keyboard is unavailable")
        readable, _, _ = select.select([self._fd], [], [], timeout)
        return os.read(self._fd, 1).decode("ascii", errors="ignore") if readable else None

    def __exit__(self, *_: object) -> None:
        if self._fd is not None and self._state is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._state)


class FocusedKeyboardWindow:

    def __init__(self, release_debounce_ms: int = 30) -> None:
        self._release_debounce_ms = release_debounce_ms
        self._root: Any | None = None
        self._events: deque[tuple[str, bool, int]] = deque()
        self._pressed: set[str] = set()
        self._pending_release: dict[str, object] = {}

    def __enter__(self) -> FocusedKeyboardWindow:
        try:
            import tkinter as tk

            root = tk.Tk()
        except Exception as exc:
            raise MobileTestbedError("focused desktop keyboard window is unavailable") from exc
        root.title("T8-v2.1 live demonstration")
        root.geometry("620x160")
        root.resizable(False, False)
        root.attributes("-topmost", True)
        tk.Label(
            root,
            text=(
                "WASD movement | F attack | 1/2/3 skills | arrows aim\n"
                "Hold keys for continuous control. Escape stops safely."
            ),
            font=("TkDefaultFont", 13),
            padx=20,
            pady=28,
        ).pack(fill="both", expand=True)
        root.bind("<KeyPress>", self._on_press)
        root.bind("<KeyRelease>", self._on_release)
        root.protocol("WM_DELETE_WINDOW", self._request_stop)
        self._root = root
        root.deiconify()
        root.lift()
        root.focus_force()
        root.update()
        return self

    def has_focus(self) -> bool:
        return self._root is not None and self._root.focus_displayof() is not None

    def wait_for_focus(self, timeout_seconds: float = 5.0) -> None:
        if self._root is None:
            raise MobileTestbedError("focused keyboard window is unavailable")
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            self._root.lift()
            self._root.focus_force()
            self._root.update()
            if self.has_focus():
                self._root.grab_set()
                return
            time.sleep(0.05)
        raise MobileTestbedError("focused keyboard window did not receive desktop focus")

    @staticmethod
    def _supported(key: str) -> bool:
        return key in {
            *LIVE_MOVEMENT_KEYS,
            *LIVE_COMBAT_KEYS,
            *LIVE_AIM_KEYS,
            "Escape",
        }

    def _on_press(self, event: object) -> None:
        key = str(getattr(event, "keysym", ""))
        if not self._supported(key):
            return
        root = self._root
        pending = self._pending_release.pop(key, None)
        if pending is not None and root is not None:
            root.after_cancel(pending)
        if key in self._pressed:
            return
        self._pressed.add(key)
        self._events.append((key, True, time.monotonic_ns()))

    def _on_release(self, event: object) -> None:
        key = str(getattr(event, "keysym", ""))
        if key not in self._pressed or self._root is None:
            return

        def release() -> None:
            self._pending_release.pop(key, None)
            if key in self._pressed:
                self._pressed.remove(key)
                self._events.append((key, False, time.monotonic_ns()))

        self._pending_release[key] = self._root.after(
            self._release_debounce_ms, release
        )

    def _request_stop(self) -> None:
        self._events.append(("Escape", True, time.monotonic_ns()))

    def poll(self) -> tuple[str, bool, int] | None:
        if self._root is None:
            raise MobileTestbedError("focused keyboard window is unavailable")
        try:
            self._root.update()
        except Exception as exc:
            raise MobileTestbedError("focused keyboard window closed unexpectedly") from exc
        return self._events.popleft() if self._events else None

    def __exit__(self, *_: object) -> None:
        if self._root is not None:
            with suppress(Exception):
                self._root.destroy()
            self._root = None


class LiveKeyboardController:

    def __init__(self, layout: Layout, width: int, height: int) -> None:
        self._layout = layout
        self._width = width
        self._height = height
        self._movement_keys: set[str] = set()
        self._aim_keys: set[str] = set()
        self._movement = "wait"
        self._aim = "none"
        self._combat_key: str | None = None
        self._combat_started_ns = 0
        self._joystick_position = _point(width, height, *layout.joystick_center)
        self._combat_position = (0, 0)
        self.conflict_count = 0

    @staticmethod
    def _direction(keys: set[str], mapping: dict[str, str]) -> str:
        directions = {mapping[key] for key in keys}
        vertical = int("north" in directions) - int("south" in directions)
        horizontal = int("east" in directions) - int("west" in directions)
        return {
            (0, 0): "wait",
            (1, 0): "north",
            (1, 1): "north_east",
            (0, 1): "east",
            (-1, 1): "south_east",
            (-1, 0): "south",
            (-1, -1): "south_west",
            (0, -1): "west",
            (1, -1): "north_west",
        }[(vertical, horizontal)]

    def _movement_point(self, direction: str) -> tuple[int, int]:
        vector = _direction_vector(direction, self._layout)
        return _point(
            self._width,
            self._height,
            self._layout.joystick_center[0] + self._layout.joystick_radius * vector[0],
            self._layout.joystick_center[1] + self._layout.joystick_radius * vector[1],
        )

    def _combat_point(self, ability: str, aim: str) -> tuple[int, int]:
        button = self._layout.buttons[ability]
        if button is None:
            raise MobileTestbedError(f"layout action {ability} is not calibrated")
        if aim == "none" or ability == "basic_attack":
            return _point(self._width, self._height, *button)
        vector = _direction_vector(aim, self._layout)
        return _point(
            self._width,
            self._height,
            button[0] + self._layout.aim_radius * vector[0],
            button[1] + self._layout.aim_radius * vector[1],
        )

    def transition(
        self, key: str, pressed: bool, timestamp_ns: int
    ) -> tuple[list[TouchOperation], bool, bool]:
        operations: list[TouchOperation] = []
        changed = False
        conflict = False
        if key in LIVE_MOVEMENT_KEYS:
            self._movement_keys.discard(key)
            if pressed:
                self._movement_keys.add(key)
            movement = self._direction(self._movement_keys, LIVE_MOVEMENT_KEYS)
            if movement != self._movement:
                center = _point(self._width, self._height, *self._layout.joystick_center)
                if self._movement == "wait" and movement != "wait":
                    operations.append(TouchOperation(ANDROID_ACTION_DOWN, JOYSTICK_POINTER_ID, *center))
                if movement == "wait":
                    operations.append(
                        TouchOperation(ANDROID_ACTION_UP, JOYSTICK_POINTER_ID, *self._joystick_position)
                    )
                    self._joystick_position = center
                else:
                    self._joystick_position = self._movement_point(movement)
                    operations.append(
                        TouchOperation(
                            ANDROID_ACTION_MOVE,
                            JOYSTICK_POINTER_ID,
                            *self._joystick_position,
                        )
                    )
                self._movement = movement
                changed = True
        elif key in LIVE_COMBAT_KEYS:
            if pressed and self._combat_key is None:
                self._combat_key = key
                self._combat_started_ns = timestamp_ns
                ability = LIVE_COMBAT_KEYS[key]
                self._combat_position = self._combat_point(ability, self._aim)
                operations.append(
                    TouchOperation(ANDROID_ACTION_DOWN, COMBAT_POINTER_ID, *self._combat_position)
                )
                changed = True
            elif pressed and key != self._combat_key:
                self.conflict_count += 1
                conflict = True
            elif not pressed and key == self._combat_key:
                operations.append(
                    TouchOperation(ANDROID_ACTION_UP, COMBAT_POINTER_ID, *self._combat_position)
                )
                self._combat_key = None
                self._combat_started_ns = 0
                self._aim = "none"
                self._aim_keys.clear()
                changed = True
        elif key in LIVE_AIM_KEYS:
            self._aim_keys.discard(key)
            if pressed:
                self._aim_keys.add(key)
            aim = self._direction(self._aim_keys, LIVE_AIM_KEYS)
            aim = "none" if aim == "wait" else aim
            if aim != self._aim:
                self._aim = aim
                if self._combat_key is not None:
                    ability = LIVE_COMBAT_KEYS[self._combat_key]
                    if ability != "basic_attack":
                        self._combat_position = self._combat_point(ability, aim)
                        operations.append(
                            TouchOperation(
                                ANDROID_ACTION_MOVE,
                                COMBAT_POINTER_ID,
                                *self._combat_position,
                            )
                        )
                        changed = True
        return operations, changed, conflict

    def action(self, timestamp_ns: int) -> FactorizedAction:
        if self._combat_key is None:
            return FactorizedAction(movement=self._movement)
        ability = LIVE_COMBAT_KEYS[self._combat_key]
        return FactorizedAction(
            movement=self._movement,
            ability=ability,
            aim=self._aim if ability != "basic_attack" else "none",
            hold_ms=max(1, (timestamp_ns - self._combat_started_ns) // 1_000_000),
        )

    def release_all(self) -> list[TouchOperation]:
        operations: list[TouchOperation] = []
        if self._movement != "wait":
            operations.append(
                TouchOperation(ANDROID_ACTION_UP, JOYSTICK_POINTER_ID, *self._joystick_position)
            )
        if self._combat_key is not None:
            operations.append(
                TouchOperation(ANDROID_ACTION_UP, COMBAT_POINTER_ID, *self._combat_position)
            )
        self._movement_keys.clear()
        self._aim_keys.clear()
        self._movement = "wait"
        self._aim = "none"
        self._combat_key = None
        self._combat_started_ns = 0
        return operations


def scripted_key_reader(
    *, seed: int, interval_seconds: float = 1.5
) -> Callable[[float], str | None]:
    if type(seed) is not int or not 0.25 <= interval_seconds <= 30.0:
        raise MobileTestbedError("scripted demonstrator bounds are invalid")
    templates = [*(tuple(key) for key in KEY_TO_MOVEMENT), ("s",), ("f",), ("k", "f"), ("l", "1", "w")]
    templates.extend(
        (ability, direction) for ability in ("1", "2", "3") for direction in KEY_TO_MOVEMENT
    )
    randomizer = random.Random(seed)
    scheduled: deque[tuple[str, ...]] = deque()
    pending: deque[str] = deque()
    next_due: float | None = None

    def read(timeout: float) -> str | None:
        nonlocal next_due
        if pending:
            return pending.popleft()
        now = time.monotonic()
        if next_due is None:
            next_due = now
        if now < next_due:
            time.sleep(min(max(timeout, 0.0), next_due - now))
            return None
        if not scheduled:
            cycle = list(templates)
            randomizer.shuffle(cycle)
            scheduled.extend(cycle)
        pending.extend(scheduled.popleft())
        next_due = max(next_due + interval_seconds, now)
        return pending.popleft()

    return read


class AdbInputPipe:

    def __init__(self, serial: str) -> None:
        self._process = subprocess.Popen(
            ("adb", "-s", serial, "shell"),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._stdin = cast(BinaryIO, self._process.stdin)

    def send(self, *arguments: str) -> None:
        self._stdin.write(("input " + " ".join(arguments) + "\n").encode())
        self._stdin.flush()

    def close(self) -> None:
        self._stdin.close()
        if self._process.poll() is None:
            self._process.terminate()
            self._process.wait(timeout=5)


DEFAULT_LAYOUT = Layout(
    1600,
    720,
    (0.19, 0.80),
    0.12,
    (-0.59, -0.81),
    150,
    250,
    0.18,
    {"basic_attack": (0.83, 0.84), "skill1": None, "skill2": None, "skill3": None},
)


def _run_adb(serial: str, *arguments: str, text: bool = False) -> str | bytes:
    result = subprocess.run(
        ("adb", "-s", serial, *arguments),
        check=False,
        capture_output=True,
        text=text,
        timeout=10,
    )
    if result.returncode:
        raise MobileTestbedError("ADB command failed")
    return str(result.stdout) if text else bytes(result.stdout)


def _validate_serial(serial: str) -> str:
    if not SERIAL_RE.fullmatch(serial):
        raise MobileTestbedError("serial must be one explicit USB device identifier")
    output = _run_adb(serial, "get-state", text=True)
    if not isinstance(output, str) or output.strip() != "device":
        raise MobileTestbedError("explicit test device is not authorized")
    return serial


def _active_display(serial: str, package: str) -> tuple[int, int, int]:
    output = _run_adb(serial, "shell", "dumpsys", "window", "displays", text=True)
    if not isinstance(output, str):
        raise MobileTestbedError("test device display is unavailable")
    focus = re.search(r"mCurrentFocus=Window\{[^}]*\s([A-Za-z0-9._]+)/[^}]+\}", output)
    display = re.search(r"DisplayFrames w=(\d+) h=(\d+) r=(\d+)", output)
    if focus is None or focus.group(1) != package:
        raise MobileTestbedError("owner-authorized target package is not foreground")
    if display is None:
        raise MobileTestbedError("test device display identity is unavailable")
    return (int(display.group(1)), int(display.group(2)), int(display.group(3)))


def _mobile_build_identity() -> dict[str, object]:
    configured = os.environ.get("HOK_MOBILE_IDENTITY_PATH")
    path = Path(configured) if configured else MOBILE_BUILD_IDENTITY_DEFAULT_PATH
    try:
        if path.is_symlink() or not path.is_file():
            raise OSError
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MobileTestbedError("frozen mobile build identity is unavailable") from exc
    if not isinstance(payload, dict):
        raise MobileTestbedError("frozen mobile build identity is invalid")
    supplied = dict(payload)
    digest = supplied.pop("identity_sha256", None)
    expected_fields = {
        "schema_version",
        "package",
        "version_code",
        "version_name",
        "signature_ids",
        "base_apk_sha256",
        "owner_attested_self_built",
        "attested_date",
    }
    if (
        set(supplied) != expected_fields
        or supplied.get("schema_version") != MOBILE_BUILD_IDENTITY_SCHEMA
        or not ANDROID_PACKAGE_RE.fullmatch(str(supplied.get("package")))
        or supplied.get("owner_attested_self_built") is not True
        or not isinstance(supplied.get("version_code"), int)
        or not isinstance(supplied.get("version_name"), str)
        or not isinstance(supplied.get("attested_date"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", str(supplied.get("base_apk_sha256")))
        or not (
            isinstance(supplied.get("signature_ids"), list)
            and supplied["signature_ids"]
            and all(re.fullmatch(r"[0-9a-f]+", str(item)) for item in supplied["signature_ids"])
        )
        or digest
        != hashlib.sha256(
            json.dumps(supplied, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    ):
        raise MobileTestbedError("frozen mobile build identity is invalid")
    return payload


def _verify_mobile_build_identity(
    serial: str, identity: dict[str, object] | None = None
) -> str:
    identity = _mobile_build_identity() if identity is None else identity
    package = cast(str, identity["package"])
    output = _run_adb(serial, "shell", "dumpsys", "package", package, text=True)
    if not isinstance(output, str):
        raise MobileTestbedError("installed mobile build identity is unavailable")
    version_code = re.search(r"\bversionCode=(\d+)\b", output)
    version_name = re.search(r"\bversionName=([^\s]+)", output)
    signatures = re.search(r"signatures:\[([^\]]+)\]", output)
    observed_signatures = (
        sorted(item.strip().lower() for item in signatures.group(1).split(","))
        if signatures
        else []
    )
    if (
        version_code is None
        or version_name is None
        or int(version_code.group(1)) != identity["version_code"]
        or version_name.group(1) != identity["version_name"]
        or observed_signatures != sorted(cast(list[str], identity["signature_ids"]))
    ):
        raise MobileTestbedError("installed mobile build differs from the frozen identity")
    return cast(str, identity["identity_sha256"])


def _open_device_guard(serial: str) -> DeviceGuard:
    identity = _mobile_build_identity()
    package = cast(str, identity["package"])
    active_serial = _validate_serial(serial)
    _verify_mobile_build_identity(active_serial, identity)
    width, height, rotation = _active_display(active_serial, package)
    return DeviceGuard(active_serial, package, width, height, rotation)


def _touch_device(path: str) -> str:
    if not TOUCH_DEVICE_RE.fullmatch(path):
        raise MobileTestbedError("touch device must be one explicit /dev/input/eventN path")
    return path


def _parse_touch_descriptors(output: str) -> tuple[TouchDescriptor, ...]:
    blocks = re.split(r"(?=add device [0-9]+: /dev/input/event[0-9]+)", output)
    result: list[TouchDescriptor] = []
    for block in blocks:
        path_match = re.search(r"add device [0-9]+: (/dev/input/event[0-9]+)", block)
        name_match = re.search(r'name:\s+"([^"]+)"', block)
        slot_match = re.search(r"ABS_MT_SLOT\s+:\s+value [0-9]+, min 0, max ([0-9]+)", block)
        x_match = re.search(r"ABS_MT_POSITION_X\s+:\s+value [0-9]+, min 0, max ([0-9]+)", block)
        y_match = re.search(r"ABS_MT_POSITION_Y\s+:\s+value [0-9]+, min 0, max ([0-9]+)", block)
        tracking_match = re.search(r"ABS_MT_TRACKING_ID\s+:", block)
        if None in (path_match, name_match, x_match, y_match) or tracking_match is None:
            continue
        protocol = "type_b" if slot_match is not None else "type_a"
        descriptor = TouchDescriptor(
            _touch_device(cast(re.Match[str], path_match).group(1)),
            cast(re.Match[str], name_match).group(1),
            int(slot_match.group(1)) + 1 if slot_match else 2,
            int(cast(re.Match[str], x_match).group(1)),
            int(cast(re.Match[str], y_match).group(1)),
            protocol,
        )
        if descriptor.max_slots >= 2 and descriptor.max_x > 0 and descriptor.max_y > 0:
            result.append(descriptor)
    return tuple(result)


def discover_touch_devices(serial: str) -> tuple[TouchDescriptor, ...]:
    guard = _open_device_guard(serial)
    output = _run_adb(guard.serial, "exec-out", "getevent", "-pl", text=True)
    if not isinstance(output, str):
        raise MobileTestbedError("touch descriptor discovery is unavailable")
    descriptors = _parse_touch_descriptors(output)
    if not descriptors:
        raise MobileTestbedError("no Type-B multitouch descriptor was found")
    return descriptors


class TouchObserver:

    def __init__(self, serial: str, descriptor: TouchDescriptor) -> None:
        self._serial, self.descriptor = _validate_serial(serial), descriptor
        self._process: subprocess.Popen[str] | None = None
        self._packets: Queue[TouchPacket] = Queue()
        self._error: BaseException | None = None

    def start(self) -> None:
        _touch_device(self.descriptor.path)
        self._process = subprocess.Popen(
            (
                "adb",
                "-s",
                self._serial,
                "shell",
                "-t",
                "getevent",
                "-lt",
                self.descriptor.path,
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        if self._process.stdout is None:
            self.close()
            raise MobileTestbedError("read-only touch observer has no stdout")
        threading.Thread(target=self._decode, args=(self._process.stdout,), daemon=True).start()

    def _decode(self, stream: TextIO) -> None:
        if self.descriptor.protocol == "type_a":
            self._decode_type_a(stream)
        else:
            self._decode_type_b(stream)

    def _decode_type_b(self, stream: TextIO) -> None:
        slot = 0
        contacts: dict[int, tuple[int | None, int | None, int | None]] = {}
        changed: set[int] = set()
        try:
            for line in stream:
                parsed = GETEVENT_LINE_RE.fullmatch(line.strip())
                if parsed is None:
                    continue
                event_type, code, raw = parsed.group(2), parsed.group(3), parsed.group(4)
                value = int(raw, 16)
                if event_type == "EV_ABS" and code == "ABS_MT_SLOT":
                    slot = value
                elif event_type == "EV_ABS" and code == "ABS_MT_TRACKING_ID":
                    _previous, x, y = contacts.get(slot, (None, None, None))
                    contacts[slot] = (None if value == 0xFFFFFFFF else value, x, y)
                    changed.add(slot)
                elif event_type == "EV_ABS" and code == "ABS_MT_POSITION_X":
                    tracking_id, _, y = contacts.get(slot, (None, None, None))
                    contacts[slot] = (tracking_id, value, y)
                    changed.add(slot)
                elif event_type == "EV_ABS" and code == "ABS_MT_POSITION_Y":
                    tracking_id, x, _ = contacts.get(slot, (None, None, None))
                    contacts[slot] = (tracking_id, x, value)
                    changed.add(slot)
                elif event_type == "EV_SYN" and code == "SYN_REPORT":
                    received_ns = time.monotonic_ns()
                    for changed_slot in sorted(changed):
                        tracking_id, x, y = contacts[changed_slot]
                        self._packets.put(TouchPacket(received_ns, changed_slot, tracking_id, x, y))
                        if tracking_id is None:
                            contacts.pop(changed_slot)
                    changed.clear()
        except BaseException as exc:  # pragma: no cover - hardware stream failure
            self._error = exc

    def _decode_type_a(self, stream: TextIO) -> None:
        active: dict[int, tuple[int | None, int, int]] = {}
        report: list[tuple[int, int, int]] = []
        tracking_id = x = y = None

        def finish_contact() -> None:
            nonlocal tracking_id, x, y
            if tracking_id is not None and x is not None and y is not None:
                report.append((tracking_id, x, y))
            tracking_id, x, y = None, None, None

        try:
            for line in stream:
                parsed = GETEVENT_LINE_RE.fullmatch(line.strip())
                if parsed is None:
                    continue
                event_type, code, raw = parsed.group(2), parsed.group(3), parsed.group(4)
                value = int(raw, 16)
                if event_type == "EV_ABS" and code == "ABS_MT_TRACKING_ID":
                    tracking_id = None if value == 0xFFFFFFFF else value
                elif event_type == "EV_ABS" and code == "ABS_MT_POSITION_X":
                    x = value
                elif event_type == "EV_ABS" and code == "ABS_MT_POSITION_Y":
                    y = value
                elif event_type == "EV_SYN" and code == "SYN_MT_REPORT":
                    finish_contact()
                elif event_type == "EV_SYN" and code == "SYN_REPORT":
                    finish_contact()
                    received_ns = time.monotonic_ns()
                    assigned: dict[int, tuple[int | None, int, int]] = {}
                    unused = set(active)
                    for identifier, current_x, current_y in report:
                        slot = min(unused, key=lambda item: (active[item][1] - current_x) ** 2 + (active[item][2] - current_y) ** 2, default=None)
                        if slot is not None and ((active[slot][1] - current_x) / self.descriptor.max_x) ** 2 + ((active[slot][2] - current_y) / self.descriptor.max_y) ** 2 > 0.16:
                            slot = None
                        if slot is None:
                            slot = next((item for item in range(self.descriptor.max_slots) if item not in assigned and item not in active), None)
                        if slot is None:
                            raise MobileTestbedError("Type-A report exceeds confirmed touch slots")
                        unused.discard(slot)
                        assigned[slot] = (identifier, current_x, current_y)
                    for slot in sorted(unused):
                        _, old_x, old_y = active[slot]
                        self._packets.put(TouchPacket(received_ns, slot, None, old_x, old_y))
                    for slot, (current_identifier, current_x, current_y) in sorted(assigned.items()):
                        self._packets.put(
                            TouchPacket(
                                received_ns, slot, current_identifier, current_x, current_y
                            )
                        )
                    active, report = assigned, []
        except BaseException as exc:  # pragma: no cover - hardware stream failure
            self._error = exc

    def read(self, timeout: float) -> TouchPacket | None:
        try:
            return self._packets.get(timeout=max(timeout, 0.0))
        except Empty:
            return None

    def close(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            self._process.wait(timeout=5)


def touch_probe_report(
    *, serial: str, descriptor: TouchDescriptor, run_seconds: float = 15.0
) -> dict[str, object]:
    if not 1.0 <= run_seconds <= 15.0:
        raise MobileTestbedError("touch probe duration is invalid")
    guard, observer = _open_device_guard(serial), TouchObserver(serial, descriptor)
    observer.start()
    started, packets, slots, first, last = time.monotonic(), 0, set(), None, None
    try:
        while time.monotonic() - started < run_seconds:
            guard.check()
            packet = observer.read(0.25)
            if packet is None:
                continue
            packets += 1
            slots.add(packet.slot)
            first = packet.received_ns if first is None else first
            last = packet.received_ns
    finally:
        observer.close()
    return {
        "schema_version": TOUCH_CALIBRATION_SCHEMA,
        "status": "PASSED" if packets and len(slots) >= 2 else "FAILED",
        "descriptor_sha256": descriptor.sha256,
        "packet_count": packets,
        "distinct_slots": len(slots),
        "timestamps_monotonic": first is not None and last is not None and last >= first,
        "raw_touch_events_persisted": False,
        "touch_device_path_persisted": False,
        "input_commands_sent": 0,
    }


def collect_touch_calibration_points(
    *, serial: str, descriptor: TouchDescriptor, prompt: Callable[[str], None]
) -> dict[str, tuple[int, int]]:
    guard, observer = _open_device_guard(serial), TouchObserver(serial, descriptor)
    names = ("basic_attack", "skill1", "skill2", "skill3")
    result: dict[str, tuple[int, int]] = {}
    observer.start()
    try:
        for name in names:
            prompt(name)
            while observer.read(0.0) is not None:
                pass
            deadline = time.monotonic() + 2.0
            observed: list[tuple[int, int]] = []
            while time.monotonic() < deadline:
                guard.check()
                packet = observer.read(0.10)
                if (
                    packet is not None
                    and packet.tracking_id is not None
                    and packet.x is not None
                    and packet.y is not None
                ):
                    observed.append((packet.x, packet.y))
            if not observed:
                raise MobileTestbedError(f"touch calibration did not observe {name}")
            result[name] = (
                int(round(float(np.median([item[0] for item in observed])))),
                int(round(float(np.median([item[1] for item in observed])))),
            )
    finally:
        observer.close()
    return result


def _new_output(path: Path) -> Path:
    if os.path.lexists(path):
        raise MobileTestbedError("output directory already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _new_large_output(path: Path) -> Path:
    root_text = os.environ.get("HOK_LARGE_ROOT")
    if not root_text:
        raise MobileTestbedError("HOK_LARGE_ROOT is required for T8 data")
    try:
        root = Path(root_text).resolve(strict=True)
    except OSError as exc:
        raise MobileTestbedError("HOK_LARGE_ROOT is unavailable") from exc
    target = path.resolve()
    if root.is_symlink() or not root.is_dir() or target == root or root not in target.parents:
        raise MobileTestbedError("T8 output must be a new directory below HOK_LARGE_ROOT")
    return _new_output(path)


def _guarded_send(guard: DeviceGuard, send: Callable[..., None]) -> Callable[..., None]:
    def guarded(*arguments: str) -> None:
        _require_mobile_input_identity()
        guard.check()
        send(*arguments)

    return guarded


def _require_mobile_input_identity() -> None:
    _mobile_build_identity()


def _pair(value: object, name: str) -> tuple[float, float]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, (int, float)) for item in value)
    ):
        raise MobileTestbedError(f"layout {name} must be a two-number array")
    return (float(value[0]), float(value[1]))


def _point_map(
    section: dict[str, object], names: tuple[str, ...], name: str
) -> dict[str, tuple[float, float] | None]:
    if set(section) != set(names):
        raise MobileTestbedError(f"layout {name} fields are invalid")
    result: dict[str, tuple[float, float] | None] = {}
    for key in names:
        value = section[key]
        result[key] = None if value is None else _pair(value, f"{name} {key}")
    return result


def load_layout(path: Path) -> tuple[Layout, str]:
    try:
        data = path.read_bytes()
        raw = json.loads(data)
    except (OSError, json.JSONDecodeError) as exc:
        raise MobileTestbedError("mobile layout is unavailable") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != LAYOUT_SCHEMA:
        raise MobileTestbedError("mobile layout schema is invalid")
    sections = (raw.get("screen"), raw.get("joystick"), raw.get("buttons"))
    if not all(isinstance(value, dict) for value in sections):
        raise MobileTestbedError("mobile layout sections are invalid")
    screen, joystick, buttons = (cast(dict[str, object], value) for value in sections)
    width, height = screen.get("width"), screen.get("height")
    move_hold, skill_hold = joystick.get("move_hold_ms"), joystick.get("skill_hold_ms")
    radius, aim_radius = joystick.get("radius"), joystick.get("aim_radius")
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise MobileTestbedError("mobile layout screen is invalid")
    if (
        not isinstance(move_hold, int)
        or not isinstance(skill_hold, int)
        or not 1 <= move_hold <= 1000
        or not 1 <= skill_hold <= 1000
        or not isinstance(radius, (int, float))
        or not isinstance(aim_radius, (int, float))
        or not 0 < radius <= 1
        or not 0 < aim_radius <= 1
    ):
        raise MobileTestbedError("mobile layout joystick is invalid")
    layout = Layout(
        width,
        height,
        _pair(joystick.get("center"), "joystick center"),
        float(radius),
        _pair(joystick.get("forward_vector"), "forward vector"),
        move_hold,
        skill_hold,
        float(aim_radius),
        _point_map(buttons, ABILITIES[1:], "buttons"),
    )
    return (layout, hashlib.sha256(data).hexdigest())


def _touch_options() -> tuple[tuple[bool, bool, bool], ...]:
    return tuple(
        (swap_axes, flip_x, flip_y)
        for swap_axes in (False, True)
        for flip_x in (False, True)
        for flip_y in (False, True)
    )


def calibrate_touch_transform(
    *,
    descriptor: TouchDescriptor,
    layout: Layout,
    layout_sha256: str,
    raw_points: dict[str, tuple[int, int]],
) -> TouchCalibration:
    expected = {
        "basic_attack": layout.basic_attack,
        **{name: cast(tuple[float, float], layout.buttons[name]) for name in ABILITIES[2:]},
    }
    if set(raw_points) != set(expected):
        raise MobileTestbedError("touch calibration points are incomplete")
    source = np.asarray(
        [
            (raw_points[name][0] / descriptor.max_x, raw_points[name][1] / descriptor.max_y, 1.0)
            for name in expected
        ],
        dtype=np.float64,
    )
    target = np.asarray([expected[name] for name in expected], dtype=np.float64)
    coefficients, _residuals, rank, _singular = np.linalg.lstsq(source, target, rcond=None)
    predicted = source @ coefficients
    rmse = float(np.sqrt(np.mean(np.square(predicted - target))))
    determinant = float(
        coefficients[0, 0] * coefficients[1, 1] - coefficients[1, 0] * coefficients[0, 1]
    )
    if rank != 3 or abs(determinant) < 0.01 or rmse > 0.04:
        unique_points = len(set(raw_points.values()))
        raw_span_x = float(np.ptp(source[:, 0]))
        raw_span_y = float(np.ptp(source[:, 1]))
        raise MobileTestbedError(
            "touch coordinate affine transform is inaccurate "
            f"(normalized_rmse={rmse:.4f}, determinant={determinant:.4f}, rank={rank}, "
            f"unique_points={unique_points}, raw_span=({raw_span_x:.4f},{raw_span_y:.4f}))"
        )
    affine = tuple(float(value) for value in coefficients.T.reshape(-1))
    return TouchCalibration(
        descriptor.sha256,
        layout_sha256,
        False,
        False,
        False,
        1.25,
        0.06,
        0.20,
        cast(tuple[float, float, float, float, float, float], affine),
    )


def _touch_calibration_payload(value: TouchCalibration) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": TOUCH_CALIBRATION_SCHEMA,
        "descriptor_sha256": value.descriptor_sha256,
        "layout_sha256": value.layout_sha256,
        "swap_axes": value.swap_axes,
        "flip_x": value.flip_x,
        "flip_y": value.flip_y,
        "joystick_start_radius": value.joystick_start_radius,
        "button_radius": value.button_radius,
        "dead_zone_fraction": value.dead_zone_fraction,
        "affine": list(value.affine) if value.affine is not None else None,
    }
    payload["calibration_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def load_touch_calibration(
    path: Path, descriptor: TouchDescriptor, layout_sha256: str
) -> TouchCalibration:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MobileTestbedError("touch calibration is unavailable") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != TOUCH_CALIBRATION_SCHEMA:
        raise MobileTestbedError("touch calibration schema is invalid")
    expected_fields = set(
        _touch_calibration_payload(
            TouchCalibration(
                descriptor.sha256,
                layout_sha256,
                False,
                False,
                False,
                1.25,
                0.06,
                0.20,
                (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            )
        )
    )
    if (
        set(payload) != expected_fields
        or payload.get("descriptor_sha256") != descriptor.sha256
        or payload.get("layout_sha256") != layout_sha256
    ):
        raise MobileTestbedError("touch calibration identity is invalid")
    supplied = dict(payload)
    digest = supplied.pop("calibration_sha256", None)
    if (
        digest
        != hashlib.sha256(
            json.dumps(supplied, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    ):
        raise MobileTestbedError("touch calibration hash is invalid")
    values = (
        supplied.get("swap_axes"),
        supplied.get("flip_x"),
        supplied.get("flip_y"),
        supplied.get("joystick_start_radius"),
        supplied.get("button_radius"),
        supplied.get("dead_zone_fraction"),
    )
    affine = supplied.get("affine")
    if (
        not all(isinstance(item, (bool, float)) for item in values)
        or any(isinstance(item, bool) for item in values[3:])
        or not (
            isinstance(affine, list)
            and len(affine) == 6
            and all(
                isinstance(item, (int, float)) and not isinstance(item, bool) for item in affine
            )
        )
    ):
        raise MobileTestbedError("touch calibration values are invalid")
    return TouchCalibration(
        descriptor.sha256,
        layout_sha256,
        cast(bool, values[0]),
        cast(bool, values[1]),
        cast(bool, values[2]),
        float(cast(float, values[3])),
        float(cast(float, values[4])),
        float(cast(float, values[5])),
        cast(
            tuple[float, float, float, float, float, float], tuple(float(item) for item in affine)
        ),
    )


def _calibration_point(value: object, name: str) -> tuple[float, float]:
    point = _pair(list(value) if isinstance(value, tuple) else value, name)
    if not all(0 < item < 1 for item in point):
        raise MobileTestbedError(f"calibration {name} must be inside the normalized screen")
    return point


def run_layout_calibration(
    *,
    serial: str,
    layout_path: Path,
    output_path: Path,
    video_node: Path,
    stream_fps: int,
    point_provider: Callable[[str], tuple[float, float]],
    confirmer: Callable[[str], bool | None],
) -> dict[str, object]:
    if os.path.lexists(output_path) or not 1 <= stream_fps <= 60:
        raise MobileTestbedError("calibration output or stream settings are invalid")
    _new_output(output_path)
    guard = _open_device_guard(serial)
    try:
        raw = json.loads(layout_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MobileTestbedError("calibration source layout is invalid") from exc
    if not isinstance(raw, dict):
        raise MobileTestbedError("calibration source layout is invalid")
    layout, source_hash = load_layout(layout_path)
    if (guard.width, guard.height) != (layout.width, layout.height):
        raise MobileTestbedError("calibration layout does not match the active display")
    center = _calibration_point(point_provider("joystick_center"), "joystick_center")
    north_endpoint = _calibration_point(
        point_provider("joystick_north_endpoint"), "joystick_north_endpoint"
    )
    delta = (north_endpoint[0] - center[0], north_endpoint[1] - center[1])
    radius = (delta[0] ** 2 + delta[1] ** 2) ** 0.5
    if not 0.02 <= radius <= 0.30:
        raise MobileTestbedError("calibration joystick radius must be between 0.02 and 0.30")
    forward = (delta[0] / radius, delta[1] / radius)
    buttons = dict(layout.buttons)
    for name in ABILITIES[2:]:
        point = _calibration_point(point_provider(name), name)
        buttons[name] = point
    candidate = replace(
        layout,
        joystick_center=center,
        joystick_radius=radius,
        forward_vector=forward,
        buttons=buttons,
    )
    stream, pipe = ScrcpyV4L2(guard.serial, video_node, stream_fps), AdbInputPipe(guard.serial)
    verified: list[str] = []
    try:
        stream.start()
        frame = stream.frame()
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise MobileTestbedError("calibration stream did not provide RGB")
        actions = [
            *(
                FactorizedAction(movement=name, hold_ms=CALIBRATION_MOVE_HOLD_MS)
                for name in MOVEMENTS[1:]
            ),
            FactorizedAction(ability="basic_attack", hold_ms=candidate.skill_hold_ms),
            *(
                FactorizedAction(ability=name, aim="north", hold_ms=candidate.skill_hold_ms)
                for name in ABILITIES[2:]
            ),
        ]
        names = [*MOVEMENTS[1:], "basic_attack", *ABILITIES[2:]]
        for name, action in zip(names, actions, strict=True):
            while True:
                _execute_action(
                    action,
                    candidate,
                    frame.shape[1],
                    frame.shape[0],
                    _guarded_send(guard, pipe.send),
                )
                decision = confirmer(name)
                if decision is None:
                    continue
                if not decision:
                    raise MobileTestbedError(f"calibration {name} was not confirmed")
                break
            verified.append(name)
    finally:
        pipe.close()
        stream.close()
    screen, joystick = raw.get("screen"), raw.get("joystick")
    if not isinstance(screen, dict) or not isinstance(joystick, dict):
        raise MobileTestbedError("calibration source layout is invalid")
    candidate_joystick = dict(joystick)
    candidate_joystick.update(
        {
            "center": list(center),
            "radius": radius,
            "forward_vector": list(forward),
        }
    )
    payload = {
        "schema_version": LAYOUT_SCHEMA,
        "calibration_status": "COMPLETE: owner_confirmed_static_controls_v2",
        "screen": screen,
        "joystick": candidate_joystick,
        "buttons": {
            name: None if point is None else list(point) for name, point in buttons.items()
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    _, candidate_hash = load_layout(output_path)
    return {
        "status": "PASSED",
        "disposition": "OWNER_CONFIRMED_GUIDED_LAYOUT_CALIBRATION",
        "source_layout_sha256": source_hash,
        "layout_sha256": candidate_hash,
        "verified_actions": verified,
        "control_actions": len(verified),
        "raw_frames_persisted": False,
    }


def _frame(serial: str) -> np.ndarray:
    raw = _run_adb(serial, "exec-out", "screencap", "-p")
    if not isinstance(raw, bytes) or not raw:
        raise MobileTestbedError("test device returned no screen frame")
    try:
        with cast(InputContainer, av.open(io.BytesIO(raw), format="png_pipe")) as container:
            frame = next(container.decode(video=0))
            rgb = frame.to_ndarray(format="rgb24")
    except (av.FFmpegError, StopIteration, ValueError) as exc:
        raise MobileTestbedError("test device screen frame is invalid") from exc
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise MobileTestbedError("test device screen must be RGB")
    return rgb


def _open_v3_predictor(
    model_path: Path, device: str
) -> tuple[Callable[[np.ndarray], tuple[list[int], list[float]]], int]:
    from hok_agent.pixel import open_rgb_predictor

    return open_rgb_predictor(model_path, device)


def _model_frame(frame: np.ndarray) -> np.ndarray:
    rows = np.linspace(0, frame.shape[0] - 1, 128).astype(np.int64)
    cols = np.linspace(0, frame.shape[1] - 1, 128).astype(np.int64)
    return frame[rows[:, None], cols[None, :], :]


def load_rgb_teacher_calibration(
    path: Path, layout_sha256: str
) -> RGBTeacherCalibration:
    try:
        data = path.read_bytes()
        report = json.loads(data)
    except (OSError, json.JSONDecodeError) as exc:
        raise MobileTestbedError("RGB teacher calibration is unavailable") from exc
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != RGB_TEACHER_CALIBRATION_SCHEMA
        or report.get("status") != "OFFLINE_TEACHER_READY"
        or report.get("offline_teacher_ready") is not True
        or report.get("training_eligible") is not False
        or report.get("live_execution_allowed") is not False
        or report.get("video_test_accessed") is not False
        or report.get("layout_sha256") != layout_sha256
        or report.get("combat_vocabulary") != list(ABILITIES[:4])
    ):
        raise MobileTestbedError("RGB teacher calibration contract is invalid")
    threshold = report.get("activity_threshold")
    medians = report.get("button_score_medians")
    scales = report.get("button_score_iqr")
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not 0 < float(threshold) < 1
        or not isinstance(medians, list)
        or not isinstance(scales, list)
        or len(medians) != 3
        or len(scales) != 3
        or not all(
            isinstance(item, (int, float)) and not isinstance(item, bool)
            for item in (*medians, *scales)
        )
        or not all(float(item) > 0 for item in scales)
    ):
        raise MobileTestbedError("RGB teacher calibration values are invalid")
    return RGBTeacherCalibration(
        hashlib.sha256(data).hexdigest(),
        layout_sha256,
        float(threshold),
        cast(tuple[float, float, float], tuple(float(item) for item in medians)),
        cast(tuple[float, float, float], tuple(float(item) for item in scales)),
    )


def _rgb_teacher_crop(
    frame: np.ndarray, box: tuple[int, int, int, int]
) -> np.ndarray:
    x0, y0, x1, y1 = box
    rows = np.linspace(y0, y1 - 1, 128).astype(np.int64)
    cols = np.linspace(x0, x1 - 1, 128).astype(np.int64)
    return frame[rows[:, None], cols[None, :], :]


def _rgb_teacher_views(frame: np.ndarray) -> np.ndarray:
    normalized = _model_frame(frame).astype(np.uint8, copy=False)
    return np.stack(
        (
            normalized,
            _rgb_teacher_crop(normalized, (0, 0, 92, 128)),
            _rgb_teacher_crop(normalized, (67, 38, 128, 128)),
        )
    )


def rgb_teacher_decision(
    current: np.ndarray,
    history: np.ndarray,
    layout: Layout,
    calibration: RGBTeacherCalibration,
) -> RGBTeacherDecision:
    if current.shape != (3, 128, 128, 3) or history.shape != current.shape:
        raise MobileTestbedError("RGB teacher views are invalid")
    activity = float(
        np.mean(
            np.abs(current[1].astype(np.float32) - history[1].astype(np.float32))
        )
        / 255.0
    )
    hud = current[2].astype(np.float32) / 255.0
    scores: list[float] = []
    for name in ABILITIES[1:4]:
        point = layout.buttons[name]
        if point is None:
            raise MobileTestbedError("RGB teacher layout lacks a combat button")
        center_x = round((point[0] - 0.52) / 0.48 * 127)
        center_y = round((point[1] - 0.30) / 0.70 * 127)
        x0, x1 = max(0, center_x - 6), min(128, center_x + 7)
        y0, y1 = max(0, center_y - 5), min(128, center_y + 6)
        if x0 >= x1 or y0 >= y1:
            raise MobileTestbedError("RGB teacher button ROI is invalid")
        patch = hud[y0:y1, x0:x1]
        maximum, minimum = patch.max(axis=2), patch.min(axis=2)
        scores.append(
            float(0.55 * maximum.mean() + 0.45 * (maximum - minimum).mean())
        )
    normalized = tuple(
        (score - calibration.medians[index]) / calibration.scales[index]
        for index, score in enumerate(scores)
    )
    order = sorted(range(3), key=lambda index: normalized[index], reverse=True)
    margin = float(normalized[order[0]] - normalized[order[1]])
    scene = current[0, 8:108, 15:108]
    red, green, blue = (scene[..., index].astype(np.int16) for index in range(3))
    red_mask = (red > 140) & (red - green > 45) & (red - blue > 25)
    enemy_red_pixels = int(red_mask.sum())
    enemy_red_row_max = int(red_mask.sum(axis=1).max())
    enemy_cue = bool(
        enemy_red_pixels >= RGB_TEACHER_ENEMY_RED_PIXELS
        or enemy_red_row_max >= RGB_TEACHER_ENEMY_RED_ROW_MAX
    )
    combat_id = (
        order[0] + 1
        if (
            activity
            >= min(calibration.activity_threshold, RGB_TEACHER_LIVE_ACTIVITY_THRESHOLD)
            and margin >= RGB_TEACHER_MARGIN
        )
        or enemy_cue
        else 0
    )
    return RGBTeacherDecision(
        combat_id,
        activity,
        cast(tuple[float, float, float], normalized),
        margin,
        enemy_red_pixels,
        enemy_red_row_max,
        enemy_cue,
    )


def rgb_teacher_minimap_navigation(frame: np.ndarray) -> MinimapNavigation | None:
    if frame.shape != (128, 128, 3):
        raise MobileTestbedError("RGB teacher minimap frame is invalid")
    roi = frame[:34, 6:26].astype(np.int16)
    red, green, blue = (roi[..., index] for index in range(3))
    player_mask = (green > 85) & (green - red > 18) & (green - blue > 10)
    target_mask = (red > 105) & (red - green > 28) & (red - blue > 18)
    player_y, player_x = np.where(player_mask)
    target_y, target_x = np.where(target_mask)
    if not 3 <= len(player_y) <= 50 or not len(target_y):
        return None
    player = np.asarray((player_y.mean(), player_x.mean()))
    targets = np.stack((target_y, target_x), axis=1)
    squared_distance = np.square(targets - player).sum(axis=1)
    selected = targets[int(np.argmin(squared_distance))]
    if float(np.min(squared_distance)) < 4.0:
        return None
    delta_y, delta_x = selected - player
    sector = round(math.atan2(float(delta_x), float(-delta_y)) / (math.pi / 4)) % 8
    movement = MOVEMENTS[1:][sector]
    return MinimapNavigation(
        movement,
        (round(float(player[0]), 4), round(float(player[1]), 4)),
        (int(selected[0]), int(selected[1])),
    )


def pick_layout_points(
    serial: str, video_node: Path, stream_fps: int, names: tuple[str, ...]
) -> dict[str, tuple[float, float]]:
    try:
        import tkinter as tk
    except ImportError as exc:
        raise MobileTestbedError("Tk is required for graphical layout picking") from exc
    guard = _open_device_guard(serial)
    stream = ScrcpyV4L2(guard.serial, video_node, stream_fps)
    try:
        stream.start()
        frame = stream.frame()
    finally:
        stream.close()
    scale = min(1.0, 960 / frame.shape[1], 720 / frame.shape[0])
    height, width = round(frame.shape[0] * scale), round(frame.shape[1] * scale)
    rows = np.linspace(0, frame.shape[0] - 1, height).astype(np.int64)
    cols = np.linspace(0, frame.shape[1] - 1, width).astype(np.int64)
    shown = np.ascontiguousarray(frame[rows[:, None], cols[None, :], :], dtype=np.uint8)
    root = tk.Tk()
    root.title("T8 layout calibration")
    prompt = tk.StringVar(value=f"Click {names[0]} on the current test-app screen")
    tk.Label(root, textvariable=prompt).pack()
    image = tk.PhotoImage(
        data=b"P6\n%d %d\n255\n" % (width, height) + shown.tobytes(), format="PPM"
    )
    canvas = tk.Canvas(root, width=width, height=height, highlightthickness=0)
    canvas.create_image(0, 0, image=image, anchor="nw")
    canvas.pack()
    points: dict[str, tuple[float, float]] = {}

    def clicked(event: object) -> None:
        if not hasattr(event, "x") or not hasattr(event, "y"):
            return
        name = names[len(points)]
        points[name] = (float(event.x) / width, float(event.y) / height)
        if len(points) == len(names):
            root.quit()
        else:
            prompt.set(f"Click {names[len(points)]} on the current test-app screen")

    canvas.bind("<Button-1>", clicked)
    root.mainloop()
    root.destroy()
    if len(points) != len(names):
        raise MobileTestbedError("layout picking was cancelled before every point was selected")
    return points


def _point(width: int, height: int, x: float, y: float) -> tuple[int, int]:
    return (round(width * x), round(height * y))


def _intent(action: str, layout: Layout) -> Intent:
    if action == "forward":
        return Intent(layout.forward_vector, False, None)
    if action == "backward":
        return Intent((-layout.forward_vector[0], -layout.forward_vector[1]), False, None)
    if action.startswith("attack_"):
        return Intent(None, True, action.removeprefix("attack_"))
    return Intent(None, False, None)


def _direction_vector(direction: str, layout: Layout) -> tuple[float, float]:
    if direction not in MOVEMENTS[1:]:
        raise MobileTestbedError("action direction is invalid")
    forward_x, forward_y = layout.forward_vector
    right_x, right_y = -forward_y, forward_x
    components = {
        "north": (1, 0),
        "north_east": (1, 1),
        "east": (0, 1),
        "south_east": (-1, 1),
        "south": (-1, 0),
        "south_west": (-1, -1),
        "west": (0, -1),
        "north_west": (1, -1),
    }
    front, right = components[direction]
    x, y = front * forward_x + right * right_x, front * forward_y + right * right_y
    scale = max((x * x + y * y) ** 0.5, 1e-9)
    return (x / scale, y / scale)


def _action_intent(action: FactorizedAction, layout: Layout) -> Intent:
    if (
        action.movement not in MOVEMENTS
        or action.ability not in ABILITIES
        or action.aim not in AIMS
        or action.target not in TARGETS
    ):
        raise MobileTestbedError("factorized action is invalid")
    if action.movement != "wait":
        return Intent(_direction_vector(action.movement, layout), False, None)
    return Intent(
        None, action.ability != "none", None if action.target == "none" else action.target
    )


def _input(
    serial: str,
    intent: Intent,
    layout: Layout,
    width: int,
    height: int,
    send: Callable[..., None] | None = None,
) -> bool:
    emit = send or (lambda *args: _run_adb(serial, "shell", "input", *args))
    if intent.movement is not None:
        start = _point(width, height, *layout.joystick_center)
        end = _point(
            width,
            height,
            layout.joystick_center[0] + layout.joystick_radius * intent.movement[0],
            layout.joystick_center[1] + layout.joystick_radius * intent.movement[1],
        )
        emit("swipe", *map(str, (*start, *end, layout.move_hold_ms)))
        return True
    if intent.attack:
        point = _point(width, height, *layout.basic_attack)
        emit("tap", *map(str, point))
        return True
    return False


def _execute_action(
    action: FactorizedAction,
    layout: Layout,
    width: int,
    height: int,
    send: Callable[..., None],
) -> bool:
    intent = _action_intent(action, layout)
    if intent.movement is not None:
        start = _point(width, height, *layout.joystick_center)
        end = _point(
            width,
            height,
            layout.joystick_center[0] + layout.joystick_radius * intent.movement[0],
            layout.joystick_center[1] + layout.joystick_radius * intent.movement[1],
        )
        send("swipe", *map(str, (*start, *end, action.hold_ms)))
        return True
    if action.ability == "none":
        return False
    button = layout.buttons[action.ability]
    if button is None:
        raise MobileTestbedError(f"layout action {action.ability} is not calibrated")
    start = _point(width, height, *button)
    if action.aim == "none":
        send("tap", *map(str, start))
    else:
        vector = _direction_vector(action.aim, layout)
        end = _point(
            width,
            height,
            button[0] + layout.aim_radius * vector[0],
            button[1] + layout.aim_radius * vector[1],
        )
        send("swipe", *map(str, (*start, *end, action.hold_ms)))
    return True


class ScrcpyV4L2:

    def __init__(self, serial: str, node: Path, max_fps: int) -> None:
        self._serial, self._node, self._max_fps = serial, node, max_fps
        self._process: subprocess.Popen[bytes] | None = None
        self._frame: np.ndarray | None = None
        self._frame_timestamp_ns = 0
        self._error: BaseException | None = None
        self._lock, self._ready = threading.Lock(), threading.Event()

    def start(self) -> None:
        try:
            node_stat = self._node.lstat()
        except OSError as exc:
            raise MobileTestbedError(
                "scrcpy stream requires an explicit existing /dev/videoN sink"
            ) from exc
        if (
            not re.fullmatch(r"/dev/video[0-9]+", str(self._node))
            or self._node.is_symlink()
            or not stat.S_ISCHR(node_stat.st_mode)
        ):
            raise MobileTestbedError("scrcpy stream requires an explicit existing /dev/videoN sink")
        command = (
            "scrcpy",
            "--serial",
            self._serial,
            "--no-control",
            "--no-display",
            "--lock-video-orientation=1",
            "--max-fps",
            str(self._max_fps),
            "--v4l2-sink",
            str(self._node),
        )
        self._process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        threading.Thread(target=self._decode, daemon=True).start()
        if not self._ready.wait(10):
            self.close()
            raise MobileTestbedError("scrcpy stream did not provide a frame")

    def _decode(self) -> None:
        deadline, last_error = time.monotonic() + 10, None
        while self._process is not None and self._process.poll() is None:
            try:
                with av.open(str(self._node), format="v4l2") as container:
                    for decoded in container.decode(video=0):
                        with self._lock:
                            self._frame = decoded.to_ndarray(format="rgb24")
                            self._frame_timestamp_ns = time.monotonic_ns()
                        self._ready.set()
            except av.FFmpegError as exc:
                last_error = exc
            if not self._ready.is_set() and time.monotonic() >= deadline:
                self._error = last_error
                self._ready.set()
                return
            time.sleep(0.1)
        self._error = last_error
        self._ready.set()

    def frame(self) -> np.ndarray:
        return self.frame_with_timestamp()[1]

    def frame_with_timestamp(self) -> tuple[int, np.ndarray]:
        with self._lock:
            if (
                self._frame is not None
                and time.monotonic_ns() - self._frame_timestamp_ns <= SCRCPY_FRAME_STALE_NS
            ):
                return self._frame_timestamp_ns, self._frame.copy()
        raise MobileTestbedError("scrcpy stream has no fresh frame") from self._error

    def close(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            self._process.wait(timeout=5)


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise MobileTestbedError("scrcpy socket closed during handshake")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class ScrcpyControlSession:

    def __init__(
        self,
        serial: str,
        max_fps: int,
        server_path: Path = SCRCPY_SERVER_PATH,
    ) -> None:
        self._serial = serial
        self._max_fps = max_fps
        self._server_path = server_path
        self._server: subprocess.Popen[bytes] | None = None
        self._video_socket: socket.socket | None = None
        self._control_socket: socket.socket | None = None
        self._video_file: BinaryIO | None = None
        self._frame: np.ndarray | None = None
        self._frame_timestamp_ns = 0
        self._frame_size: tuple[int, int] | None = None
        self._server_digest: str | None = None
        self._error: BaseException | None = None
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._port: int | None = None

    @property
    def server_sha256(self) -> str:
        if self._server_digest is None:
            raise MobileTestbedError("scrcpy server identity was not verified")
        return self._server_digest

    @property
    def frame_size(self) -> tuple[int, int]:
        if self._frame_size is None:
            raise MobileTestbedError("scrcpy session has no display metadata")
        return self._frame_size

    def _validate_server(self) -> None:
        try:
            server_stat = self._server_path.lstat()
            digest = hashlib.sha256(self._server_path.read_bytes()).hexdigest()
        except OSError as exc:
            raise MobileTestbedError("pinned scrcpy server is unavailable") from exc
        if (
            self._server_path.is_symlink()
            or not stat.S_ISREG(server_stat.st_mode)
            or digest != SCRCPY_SERVER_SHA256
        ):
            raise MobileTestbedError("pinned scrcpy 1.25 server identity differs")
        self._server_digest = digest

    def _remove_forward(self) -> None:
        if self._port is None:
            return
        subprocess.run(
            ("adb", "-s", self._serial, "forward", "--remove", f"tcp:{self._port}"),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        self._port = None

    def start(self) -> None:
        self._validate_server()
        _run_adb(self._serial, "push", str(self._server_path), "/data/local/tmp/scrcpy-server.jar")
        forwarded = _run_adb(
            self._serial, "forward", "tcp:0", "localabstract:scrcpy", text=True
        )
        match = re.search(r"([0-9]+)", cast(str, forwarded))
        if match is None:
            raise MobileTestbedError("ADB did not allocate a scrcpy tunnel")
        self._port = int(match.group(1))
        try:
            self._server = subprocess.Popen(
                (
                    "adb",
                    "-s",
                    self._serial,
                    "shell",
                    "CLASSPATH=/data/local/tmp/scrcpy-server.jar",
                    "app_process",
                    "/",
                    "com.genymobile.scrcpy.Server",
                    SCRCPY_SERVER_VERSION,
                    "log_level=error",
                    "bit_rate=8000000",
                    f"max_fps={self._max_fps}",
                    "lock_video_orientation=1",
                    "tunnel_forward=true",
                    "send_frame_meta=false",
                    "clipboard_autosync=false",
                    "power_on=false",
                ),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            self._remove_forward()
            raise MobileTestbedError("scrcpy server could not start") from exc
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            candidate: socket.socket | None = None
            try:
                candidate = socket.create_connection(("127.0.0.1", self._port), timeout=0.5)
                if _recv_exact(candidate, 1) != b"\x00":
                    raise MobileTestbedError("scrcpy handshake byte is invalid")
                self._video_socket = candidate
                break
            except (OSError, MobileTestbedError):
                if candidate is not None:
                    candidate.close()
                time.sleep(0.1)
        if self._video_socket is None:
            self.close()
            raise MobileTestbedError("scrcpy server connection timed out")
        try:
            self._control_socket = socket.create_connection(
                ("127.0.0.1", self._port), timeout=2
            )
            metadata = _recv_exact(self._video_socket, 68)
        except (OSError, MobileTestbedError) as exc:
            self.close()
            raise MobileTestbedError("scrcpy control connection failed") from exc
        finally:
            self._remove_forward()
        width, height = struct.unpack(">HH", metadata[64:])
        time.sleep(0.1)
        if self._server is None or self._server.poll() is not None:
            self.close()
            raise MobileTestbedError("pinned scrcpy server did not own the live session")
        if width <= 0 or height <= 0:
            self.close()
            raise MobileTestbedError("scrcpy display metadata is invalid")
        self._frame_size = (width, height)
        self._video_socket.settimeout(None)
        self._control_socket.settimeout(None)
        self._video_file = cast(BinaryIO, self._video_socket.makefile("rb", buffering=0))
        threading.Thread(target=self._decode, daemon=True).start()
        if not self._ready.wait(10):
            self.close()
            raise MobileTestbedError("scrcpy socket stream did not provide a frame")
        if self._error is not None:
            self.close()
            raise MobileTestbedError("scrcpy socket stream failed") from self._error

    def _decode(self) -> None:
        try:
            assert self._video_file is not None
            with av.open(self._video_file, format="h264", mode="r") as container:
                for decoded in container.decode(video=0):
                    with self._lock:
                        self._frame = decoded.to_ndarray(format="rgb24")
                        self._frame_timestamp_ns = time.monotonic_ns()
                    self._ready.set()
        except BaseException as exc:
            self._error = exc
            self._ready.set()

    def frame(self) -> tuple[int, np.ndarray]:
        with self._lock:
            if self._error is not None:
                raise MobileTestbedError("scrcpy socket stream ended") from self._error
            if (
                self._frame is not None
                and time.monotonic_ns() - self._frame_timestamp_ns <= 500_000_000
            ):
                return self._frame_timestamp_ns, self._frame.copy()
        raise MobileTestbedError("scrcpy socket stream has no fresh frame")

    def touch(self, operation: TouchOperation, width: int, height: int) -> int:
        _require_mobile_input_identity()
        if (
            self._control_socket is None
            or operation.action not in {ANDROID_ACTION_DOWN, ANDROID_ACTION_UP, ANDROID_ACTION_MOVE}
            or operation.pointer_id not in {JOYSTICK_POINTER_ID, COMBAT_POINTER_ID}
            or not 0 <= operation.x < width
            or not 0 <= operation.y < height
        ):
            raise MobileTestbedError("scrcpy touch operation is invalid")
        active = operation.action != ANDROID_ACTION_UP
        message = struct.pack(
            ">BBqiiHHHi",
            SCRCPY_CONTROL_TOUCH,
            operation.action,
            operation.pointer_id,
            operation.x,
            operation.y,
            width,
            height,
            0xFFFF if active else 0,
            1 if active else 0,
        )
        try:
            self._control_socket.sendall(message)
        except OSError as exc:
            raise MobileTestbedError("scrcpy touch dispatch failed") from exc
        return time.monotonic_ns()

    def close(self) -> None:
        self._remove_forward()
        if self._video_file is not None:
            with suppress(OSError):
                self._video_file.close()
            self._video_file = None
        for connection in (self._control_socket, self._video_socket):
            if connection is not None:
                with suppress(OSError):
                    connection.close()
        self._control_socket = self._video_socket = None
        if self._server is not None and self._server.poll() is None:
            self._server.terminate()
            try:
                self._server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._server.kill()
                self._server.wait(timeout=5)


def _publish(output: Path, rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        (staging / "events.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.rename(output)


def _action_record(action: FactorizedAction) -> dict[str, object]:
    return {
        "movement": action.movement,
        "ability": action.ability,
        "aim": action.aim,
        "target": action.target,
        "hold_ms": action.hold_ms,
    }


def write_touch_calibration(path: Path, calibration: TouchCalibration) -> dict[str, object]:
    if os.path.lexists(path):
        raise MobileTestbedError("touch calibration output already exists")
    payload = _touch_calibration_payload(calibration)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return payload


@dataclass
class _Pointer:
    down_ns: int
    start: tuple[float, float]
    current: tuple[float, float]
    role: str


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return float(((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5)


def _direction_from_vector(vector: tuple[float, float], layout: Layout) -> str:
    x, y = vector
    magnitude = (x * x + y * y) ** 0.5
    if magnitude <= 1e-9:
        return "wait"
    forward_x, forward_y = layout.forward_vector
    right_x, right_y = -forward_y, forward_x
    front, right = (
        (x * forward_x + y * forward_y) / magnitude,
        (x * right_x + y * right_y) / magnitude,
    )
    choices = {
        "north": (1, 0),
        "north_east": (1, 1),
        "east": (0, 1),
        "south_east": (-1, 1),
        "south": (-1, 0),
        "south_west": (-1, -1),
        "west": (0, -1),
        "north_west": (1, -1),
    }
    return max(choices, key=lambda name: front * choices[name][0] + right * choices[name][1])


def _hold_bucket(hold_ms: int) -> int:
    if hold_ms <= 0:
        return 0
    if hold_ms <= 250:
        return 1
    if hold_ms <= 750:
        return 2
    return 3


def _touch_factor_coverage(
    actions: list[FactorizedAction], *, parallel_samples: int = 0, conflict_samples: int = 0
) -> dict[str, object]:
    factors: dict[str, list[str]] = {
        "movement": [item.movement for item in actions],
        "combat": [item.ability for item in actions],
        "aim": [item.aim for item in actions],
        "hold": [TOUCH_HOLD_BUCKETS[_hold_bucket(item.hold_ms)] for item in actions],
    }
    required = {"movement": MOVEMENTS, "combat": ABILITIES, "aim": AIMS[1:], "hold": TOUCH_HOLD_BUCKETS[1:]}
    missing = {name: [value for value in required[name] if value not in factors[name]] for name in required}
    core = {
        "wait": "wait" in factors["movement"],
        "movement": any(item != "wait" for item in factors["movement"]),
        "combat": any(item != "none" for item in factors["combat"]),
        "all_combat_buttons": set(ABILITIES[1:]).issubset(factors["combat"]),
        "aim": any(item != "none" for item in factors["aim"]),
        "hold": any(item != "none" for item in factors["hold"]),
        "parallel_move_and_combat": parallel_samples > 0,
        "conflict_free": conflict_samples == 0,
    }
    return {
        "counts": {name: {value: values.count(value) for value in dict.fromkeys(values)} for name, values in factors.items()},
        "missing": missing,
        "complete": not any(missing.values()),
        "core": core,
        "core_complete": all(core.values()),
        "parallel_samples": parallel_samples,
        "conflict_samples": conflict_samples,
    }


def _live_factor_coverage(
    counts: dict[str, Counter[str]], *, parallel_samples: int, conflict_events: int
) -> dict[str, object]:
    required = {
        "movement": MOVEMENTS,
        "combat": ABILITIES,
        "aim": AIMS[1:],
        "hold": TOUCH_HOLD_BUCKETS[1:],
    }
    missing = {
        name: [value for value in values if counts[name][value] == 0]
        for name, values in required.items()
    }
    core = {
        "wait": counts["movement"]["wait"] > 0,
        "movement": sum(counts["movement"][value] for value in MOVEMENTS[1:]) > 0,
        "combat": sum(counts["combat"][value] for value in ABILITIES[1:]) > 0,
        "all_combat_buttons": all(counts["combat"][value] > 0 for value in ABILITIES[1:]),
        "aim": sum(counts["aim"][value] for value in AIMS[1:]) > 0,
        "hold": sum(counts["hold"][value] for value in TOUCH_HOLD_BUCKETS[1:]) > 0,
        "parallel_move_and_combat": parallel_samples > 0,
        "conflict_free": conflict_events == 0,
    }
    return {
        "counts": {name: dict(values) for name, values in counts.items()},
        "missing": missing,
        "complete": not any(missing.values()),
        "core": core,
        "core_complete": all(core.values()),
        "parallel_samples": parallel_samples,
        "conflict_events": conflict_events,
    }


def _touch_semantic_state(action: FactorizedAction) -> dict[str, object]:
    return dict(zip(("movement", "ability", "aim", "target", "hold_bucket"), _touch_semantic_key(action), strict=True))


def _touch_semantic_key(action: FactorizedAction) -> tuple[str, str, str, str, int]:
    return action.movement, action.ability, action.aim, action.target, _hold_bucket(action.hold_ms)


class _TouchActionMapper:
    def __init__(
        self, descriptor: TouchDescriptor, calibration: TouchCalibration, layout: Layout
    ) -> None:
        self._descriptor, self._calibration, self._layout = descriptor, calibration, layout
        self._pointers: dict[int, _Pointer] = {}
        self._conflict_samples = 0

    @property
    def conflict_samples(self) -> int:
        return self._conflict_samples

    @property
    def conflicted(self) -> bool:
        joysticks = sum(pointer.role == "joystick" for pointer in self._pointers.values())
        combat = sum(pointer.role in ABILITIES[1:] for pointer in self._pointers.values())
        return joysticks > 1 or combat > 1

    @property
    def parallel(self) -> bool:
        roles = {pointer.role for pointer in self._pointers.values()}
        return "joystick" in roles and any(role in ABILITIES[1:] for role in roles)

    def _role(self, point: tuple[float, float]) -> str:
        distances = sorted(
            (
                (_distance(point, button), name)
                for name, button in self._layout.buttons.items()
                if button is not None
            ),
            key=lambda item: (item[0], item[1]),
        )
        if (
            distances
            and distances[0][0]
            <= self._calibration.button_radius * TOUCH_BUTTON_TOLERANCE_MULTIPLIER
        ):
            return distances[0][1]
        if (
            _distance(point, self._layout.joystick_center)
            <= self._layout.joystick_radius * self._calibration.joystick_start_radius
        ):
            return "joystick"
        return "unknown"

    def feed(self, packet: TouchPacket) -> None:
        if packet.tracking_id is None:
            self._pointers.pop(packet.slot, None)
            return
        if packet.x is None or packet.y is None:
            raise MobileTestbedError("active touch packet has no coordinates")
        point = self._calibration.transform(packet.x, packet.y, self._descriptor)
        if not 0.0 <= point[0] <= 1.0 or not 0.0 <= point[1] <= 1.0:
            raise MobileTestbedError("touch coordinate transform is outside the normalized layout")
        existing = self._pointers.get(packet.slot)
        if existing is None:
            role = self._role(point)
            self._pointers[packet.slot] = _Pointer(packet.received_ns, point, point, role)
        else:
            existing.current = point

    def action(self, timestamp_ns: int) -> FactorizedAction:
        joysticks = [pointer for pointer in self._pointers.values() if pointer.role == "joystick"]
        combat = [
            (pointer.role, pointer)
            for pointer in self._pointers.values()
            if pointer.role in ABILITIES[1:]
        ]
        if len(combat) > 1:
            self._conflict_samples += 1
            return FactorizedAction()
        joystick = joysticks[0] if len(joysticks) == 1 else None
        if len(joysticks) > 1:
            self._conflict_samples += 1
            return FactorizedAction()
        movement = "wait"
        if joystick is not None:
            vector = (
                joystick.current[0] - joystick.start[0],
                joystick.current[1] - joystick.start[1],
            )
            if (
                vector[0] ** 2 + vector[1] ** 2
            ) ** 0.5 >= self._layout.joystick_radius * self._calibration.dead_zone_fraction:
                movement = _direction_from_vector(vector, self._layout)
        ability, aim, owner = "none", "none", joystick
        if combat:
            ability, owner = combat[0]
            if ability != "basic_attack":
                vector = (owner.current[0] - owner.start[0], owner.current[1] - owner.start[1])
                if (
                    vector[0] ** 2 + vector[1] ** 2
                ) ** 0.5 >= self._layout.aim_radius * self._calibration.dead_zone_fraction:
                    aim = _direction_from_vector(vector, self._layout)
        hold_ms = 0 if owner is None else max(0, int((timestamp_ns - owner.down_ns) / 1_000_000))
        return FactorizedAction(movement=movement, ability=ability, aim=aim, hold_ms=hold_ms)


@dataclass(frozen=True)
class _TouchSample:
    frame: np.ndarray
    action: FactorizedAction
    timestamp_ns: int
    semantic_event_index: int


def _write_touch_shard(path: Path, samples: list[_TouchSample]) -> None:
    frames = np.stack([item.frame for item in samples]).astype(np.uint8, copy=False)
    actions = [item.action for item in samples]
    np.savez_compressed(
        path,
        frames=frames,
        movement=np.asarray([MOVEMENTS.index(item.movement) for item in actions], dtype=np.int8),
        combat=np.asarray([ABILITIES.index(item.ability) for item in actions], dtype=np.int8),
        aim=np.asarray([AIMS.index(item.aim) for item in actions], dtype=np.int8),
        target=np.zeros(len(actions), dtype=np.int8),
        hold_bucket=np.asarray([_hold_bucket(item.hold_ms) for item in actions], dtype=np.int8),
        hold_ms=np.asarray([item.hold_ms for item in actions], dtype=np.uint32),
        timestamp_ns=np.asarray([item.timestamp_ns for item in samples], dtype=np.int64),
        semantic_event_index=np.asarray(
            [item.semantic_event_index for item in samples], dtype=np.int32
        ),
        label_source=np.zeros(len(actions), dtype=np.uint8),
    )


def _publish_touch_demonstration(
    output: Path,
    samples: list[_TouchSample],
    rows: list[dict[str, object]],
    summary: dict[str, object],
    shard_size: int,
) -> None:
    if output.exists() or output.is_symlink():
        raise MobileTestbedError("output directory already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        for offset in range(0, len(samples), shard_size):
            _write_touch_shard(
                staging / f"samples-{offset // shard_size:05d}.npz",
                samples[offset : offset + shard_size],
            )
        (staging / "events.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
        (staging / "summary.json").write_text(
            json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        manifest = {
            "schema_version": TOUCH_DEMONSTRATOR_SESSION_SCHEMA,
            "summary_sha256": hashlib.sha256((staging / "summary.json").read_bytes()).hexdigest(),
            "events_sha256": hashlib.sha256((staging / "events.jsonl").read_bytes()).hexdigest(),
            "shards": [
                {"name": item.name, "sha256": hashlib.sha256(item.read_bytes()).hexdigest()}
                for item in sorted(staging.glob("samples-*.npz"))
            ],
        }
        manifest["session_sha256"] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        (staging / "session-manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)


def _write_demonstrator_shard(
    path: Path, samples: list[tuple[np.ndarray, FactorizedAction, int, bool]]
) -> None:
    frames = np.stack([frames for frames, _, _, _ in samples]).astype(np.uint8, copy=False)
    actions = [action for _, action, _, _ in samples]
    np.savez_compressed(
        path,
        frames=frames,
        movement=np.asarray(
            [MOVEMENTS.index(action.movement) for action in actions], dtype=np.int8
        ),
        ability=np.asarray([ABILITIES.index(action.ability) for action in actions], dtype=np.int8),
        aim=np.asarray([AIMS.index(action.aim) for action in actions], dtype=np.int8),
        target=np.asarray([TARGETS.index(action.target) for action in actions], dtype=np.int8),
        hold_ms=np.asarray([action.hold_ms for action in actions], dtype=np.uint16),
        timestamp_ns=np.asarray([timestamp for _, _, timestamp, _ in samples], dtype=np.int64),
        input_sent=np.asarray([sent for _, _, _, sent in samples], dtype=np.uint8),
    )


def _publish_demonstration(
    output: Path,
    samples: list[tuple[np.ndarray, FactorizedAction, int, bool]],
    rows: list[dict[str, object]],
    summary: dict[str, object],
    shard_size: int,
) -> None:
    if output.exists() or output.is_symlink():
        raise MobileTestbedError("output directory already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        for offset in range(0, len(samples), shard_size):
            _write_demonstrator_shard(
                staging / f"samples-{offset // shard_size:05d}.npz",
                samples[offset : offset + shard_size],
            )
        (staging / "events.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest = {
            "schema_version": DEMONSTRATOR_SESSION_SCHEMA,
            "summary_sha256": hashlib.sha256((staging / "summary.json").read_bytes()).hexdigest(),
            "events_sha256": hashlib.sha256((staging / "events.jsonl").read_bytes()).hexdigest(),
            "shards": [
                {
                    "name": item.name,
                    "sha256": hashlib.sha256(item.read_bytes()).hexdigest(),
                }
                for item in sorted(staging.glob("samples-*.npz"))
            ],
        }
        manifest["session_sha256"] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        (staging / "session-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.rename(output)


def _keyboard_v2_contract(layout_sha256: str) -> tuple[dict[str, object], str]:
    contract: dict[str, object] = {
        "schema_version": KEYBOARD_V2_DATA_SCHEMA,
        "source": EXECUTED_ACTION_SOURCE,
        "layout_sha256": layout_sha256,
        "window_frames": TOUCH_WINDOW_FRAMES,
        "sample_hz": TOUCH_SAMPLE_HZ,
        "observation_end_lag_ms": 100,
        "movement_keys": KEY_TO_MOVEMENT,
        "ability_keys": KEY_TO_ABILITY,
        "hold_keys_ms": KEY_TO_HOLD_MS,
        "executor": "foreground_guarded_adb_single_swipe_v1",
    }
    digest = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return contract, digest


def _keyboard_v21_contract(
    layout_sha256: str, source: str = SCRCPY_EXECUTED_ACTION_SOURCE
) -> tuple[dict[str, object], str]:
    if source not in {
        SCRCPY_EXECUTED_ACTION_SOURCE,
        DIAGNOSTIC_CONTROL_SOURCE,
        DIAGNOSTIC_INVERSE_SOURCE,
    }:
        raise MobileTestbedError("keyboard v2.1 source is invalid")
    contract: dict[str, object] = {
        "schema_version": KEYBOARD_V21_DATA_SCHEMA,
        "source": source,
        "layout_sha256": layout_sha256,
        "window_frames": TOUCH_WINDOW_FRAMES,
        "sample_hz": TOUCH_SAMPLE_HZ,
        "observation_end_lag_ms": 100,
        "movement_keys": LIVE_MOVEMENT_KEYS,
        "ability_keys": LIVE_COMBAT_KEYS,
        "aim_keys": LIVE_AIM_KEYS,
        "hold_source": "keydown_to_keyup_elapsed_ms",
        "executor": "pinned_scrcpy_1.25_multitouch_v1",
        "scrcpy_server_version": SCRCPY_SERVER_VERSION,
        "scrcpy_server_sha256": SCRCPY_SERVER_SHA256,
        "pointer_roles": {"joystick": JOYSTICK_POINTER_ID, "combat": COMBAT_POINTER_ID},
    }
    digest = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return contract, digest


def _write_keyboard_v2_shard(
    path: Path, samples: list[tuple[np.ndarray, FactorizedAction, int, bool]]
) -> None:
    frames = np.stack([item[0] for item in samples]).astype(np.uint8, copy=False)
    actions = [item[1] for item in samples]
    np.savez_compressed(
        path,
        frames=frames,
        movement=np.asarray([MOVEMENTS.index(item.movement) for item in actions], dtype=np.int8),
        combat=np.asarray([ABILITIES.index(item.ability) for item in actions], dtype=np.int8),
        aim=np.asarray([AIMS.index(item.aim) for item in actions], dtype=np.int8),
        target=np.zeros(len(actions), dtype=np.int8),
        hold_bucket=np.asarray([_hold_bucket(item.hold_ms) for item in actions], dtype=np.int8),
        hold_ms=np.asarray([item.hold_ms for item in actions], dtype=np.uint16),
        timestamp_ns=np.asarray([item[2] for item in samples], dtype=np.int64),
        label_source=np.ones(len(actions), dtype=np.uint8),
        input_sent=np.asarray([item[3] for item in samples], dtype=np.uint8),
    )


def _publish_keyboard_v2(
    output: Path,
    samples: list[tuple[np.ndarray, FactorizedAction, int, bool]],
    rows: list[dict[str, object]],
    summary: dict[str, object],
    contract: dict[str, object],
    shard_size: int,
) -> None:
    if output.exists() or output.is_symlink():
        raise MobileTestbedError("output directory already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        for offset in range(0, len(samples), shard_size):
            _write_keyboard_v2_shard(
                staging / f"samples-{offset // shard_size:05d}.npz",
                samples[offset : offset + shard_size],
            )
        (staging / "events.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
        (staging / "summary.json").write_text(
            json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "action-contract.json").write_text(
            json.dumps(contract, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        manifest = {
            "schema_version": KEYBOARD_V2_SESSION_SCHEMA,
            "summary_sha256": hashlib.sha256((staging / "summary.json").read_bytes()).hexdigest(),
            "events_sha256": hashlib.sha256((staging / "events.jsonl").read_bytes()).hexdigest(),
            "action_contract_file_sha256": hashlib.sha256(
                (staging / "action-contract.json").read_bytes()
            ).hexdigest(),
            "shards": [
                {"name": item.name, "sha256": hashlib.sha256(item.read_bytes()).hexdigest()}
                for item in sorted(staging.glob("samples-*.npz"))
            ],
        }
        manifest["session_sha256"] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        (staging / "session-manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)


def _rgb_teacher_contract(
    layout_sha256: str,
    calibration: RGBTeacherCalibration,
    *,
    balanced_actions: bool = False,
) -> tuple[dict[str, object], str]:
    contract: dict[str, object] = {
        "schema_version": RGB_TEACHER_DATA_SCHEMA,
        "source": RGB_TEACHER_SOURCE,
        "layout_sha256": layout_sha256,
        "teacher_report_sha256": calibration.report_sha256,
        "window_frames": RGB_TEACHER_WINDOW_FRAMES,
        "sample_hz": RGB_TEACHER_SAMPLE_HZ,
        "decision_hz": RGB_TEACHER_DECISION_HZ,
        "history_ms": RGB_TEACHER_HISTORY_FRAMES * 100,
        "execution_lag_ms": RGB_TEACHER_EXECUTION_LAG_MS,
        "margin_threshold": RGB_TEACHER_MARGIN,
        "activity_threshold": min(
            calibration.activity_threshold, RGB_TEACHER_LIVE_ACTIVITY_THRESHOLD
        ),
        "global_dispatch_interval_ms": RGB_TEACHER_GLOBAL_DISPATCH_INTERVAL_MS,
        "minimum_formal_decision_coverage": RGB_TEACHER_MIN_FORMAL_DECISION_COVERAGE,
        "enemy_red_pixels_threshold": RGB_TEACHER_ENEMY_RED_PIXELS,
        "enemy_red_row_max_threshold": RGB_TEACHER_ENEMY_RED_ROW_MAX,
        "class_cooldown_ms": list(RGB_TEACHER_CLASS_COOLDOWN_MS),
        "combat_vocabulary": list(ABILITIES[:4]),
        "executor": "foreground_guarded_adb_single_swipe_v1",
        "dispatch_policy": (
            "least_executed_eligible_class_v1"
            if balanced_actions
            else "highest_score_eligible_class_v1"
        ),
        "decision_input": "rgb_only_enemy_red_cue_activity_button_appearance_and_cooldown",
    }
    return contract, hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_rgb_teacher_frame_shard(
    path: Path, frames: list[tuple[int, np.ndarray]]
) -> None:
    np.savez_compressed(
        path,
        frames=np.stack([frame for _timestamp, frame in frames]).astype(
            np.uint8, copy=False
        ),
        timestamp_ns=np.asarray([timestamp for timestamp, _frame in frames], dtype=np.int64),
    )


def _write_rgb_teacher_shard(path: Path, samples: list[RGBTeacherSample]) -> None:
    np.savez_compressed(
        path,
        observation_index=np.asarray(
            [sample.observation_index for sample in samples], dtype=np.int32
        ),
        shifted_observation_index=np.asarray(
            [sample.shifted_observation_index for sample in samples], dtype=np.int32
        ),
        combat_id=np.asarray([sample.combat_id for sample in samples], dtype=np.int8),
        observation_end_timestamp_ns=np.asarray(
            [sample.observation_end_timestamp_ns for sample in samples], dtype=np.int64
        ),
        decision_timestamp_ns=np.asarray(
            [sample.decision_timestamp_ns for sample in samples], dtype=np.int64
        ),
        execution_timestamp_ns=np.asarray(
            [sample.execution_timestamp_ns for sample in samples], dtype=np.int64
        ),
        confidence=np.asarray([sample.confidence for sample in samples], dtype=np.float32),
        input_sent=np.asarray([sample.input_sent for sample in samples], dtype=np.uint8),
    )


def _publish_rgb_teacher_session(
    output: Path,
    frames: list[tuple[int, np.ndarray]],
    samples: list[RGBTeacherSample],
    rows: list[dict[str, object]],
    summary: dict[str, object],
    contract: dict[str, object],
    shard_size: int,
) -> None:
    if output.exists() or output.is_symlink():
        raise MobileTestbedError("RGB teacher output directory already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        for offset in range(0, len(frames), shard_size):
            _write_rgb_teacher_frame_shard(
                staging / f"frames-{offset // shard_size:05d}.npz",
                frames[offset : offset + shard_size],
            )
        for offset in range(0, len(samples), shard_size):
            _write_rgb_teacher_shard(
                staging / f"samples-{offset // shard_size:05d}.npz",
                samples[offset : offset + shard_size],
            )
        (staging / "events.jsonl").write_text(
            "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        (staging / "summary.json").write_text(
            json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "action-contract.json").write_text(
            json.dumps(contract, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        manifest: dict[str, object] = {
            "schema_version": RGB_TEACHER_SESSION_SCHEMA,
            "summary_sha256": hashlib.sha256(
                (staging / "summary.json").read_bytes()
            ).hexdigest(),
            "events_sha256": hashlib.sha256(
                (staging / "events.jsonl").read_bytes()
            ).hexdigest(),
            "action_contract_file_sha256": hashlib.sha256(
                (staging / "action-contract.json").read_bytes()
            ).hexdigest(),
            "frame_shards": [
                {"name": item.name, "sha256": hashlib.sha256(item.read_bytes()).hexdigest()}
                for item in sorted(staging.glob("frames-*.npz"))
            ],
            "shards": [
                {"name": item.name, "sha256": hashlib.sha256(item.read_bytes()).hexdigest()}
                for item in sorted(staging.glob("samples-*.npz"))
            ],
        }
        manifest["session_sha256"] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        (staging / "session-manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)


def _write_keyboard_v21_shard(path: Path, samples: list[LiveSample]) -> None:
    actions = [sample.action for sample in samples]
    np.savez_compressed(
        path,
        frames=np.stack([sample.frames for sample in samples]).astype(np.uint8, copy=False),
        movement=np.asarray([MOVEMENTS.index(item.movement) for item in actions], dtype=np.int8),
        combat=np.asarray([ABILITIES.index(item.ability) for item in actions], dtype=np.int8),
        aim=np.asarray([AIMS.index(item.aim) for item in actions], dtype=np.int8),
        target=np.zeros(len(actions), dtype=np.int8),
        hold_bucket=np.asarray([_hold_bucket(item.hold_ms) for item in actions], dtype=np.int8),
        hold_ms=np.asarray([item.hold_ms for item in actions], dtype=np.uint32),
        timestamp_ns=np.asarray([sample.timestamp_ns for sample in samples], dtype=np.int64),
        transition_sequence=np.asarray(
            [sample.transition_sequence for sample in samples], dtype=np.int64
        ),
        last_dispatch_ns=np.asarray(
            [sample.last_dispatch_ns for sample in samples], dtype=np.int64
        ),
        label_source=np.ones(len(actions), dtype=np.uint8),
        input_sent=np.asarray([sample.input_sent for sample in samples], dtype=np.uint8),
    )


class KeyboardV21Writer:

    def __init__(
        self,
        output: Path,
        contract: dict[str, object],
        shard_size: int,
    ) -> None:
        self._output = output
        self._contract = contract
        self._shard_size = shard_size
        self._temporary = tempfile.TemporaryDirectory(
            prefix=f".{output.name}-", dir=output.parent
        )
        self._staging = Path(self._temporary.name)
        self._events = (self._staging / "events.jsonl").open("w", encoding="utf-8")
        self._buffer: list[LiveSample] = []
        self._shards = 0
        self.samples = 0
        self.events = 0
        self._finished = False

    def add_event(self, row: dict[str, object]) -> None:
        self._events.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        self._events.flush()
        self.events += 1

    def add_sample(self, sample: LiveSample) -> None:
        self._buffer.append(sample)
        self.samples += 1
        if len(self._buffer) >= self._shard_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        _write_keyboard_v21_shard(
            self._staging / f"samples-{self._shards:05d}.npz", self._buffer
        )
        self._buffer = []
        self._shards += 1

    def finalize(self, summary: dict[str, object], output: Path | None = None) -> None:
        self._flush()
        self._events.close()
        (self._staging / "summary.json").write_text(
            json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        (self._staging / "action-contract.json").write_text(
            json.dumps(self._contract, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        manifest: dict[str, object] = {
            "schema_version": KEYBOARD_V21_SESSION_SCHEMA,
            "summary_sha256": hashlib.sha256(
                (self._staging / "summary.json").read_bytes()
            ).hexdigest(),
            "events_sha256": hashlib.sha256(
                (self._staging / "events.jsonl").read_bytes()
            ).hexdigest(),
            "action_contract_file_sha256": hashlib.sha256(
                (self._staging / "action-contract.json").read_bytes()
            ).hexdigest(),
            "shards": [
                {"name": item.name, "sha256": hashlib.sha256(item.read_bytes()).hexdigest()}
                for item in sorted(self._staging.glob("samples-*.npz"))
            ],
        }
        manifest["session_sha256"] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        (self._staging / "session-manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        published = output or self._output
        published.parent.mkdir(parents=True, exist_ok=True)
        if published.exists() or published.is_symlink():
            raise MobileTestbedError("published keyboard v2.1 directory already exists")
        self._staging.rename(published)
        self._finished = True
        self._temporary.cleanup()

    def abort(self) -> None:
        if self._finished:
            return
        if not self._events.closed:
            self._events.close()
        self._temporary.cleanup()


def _execute_action_v2(
    action: FactorizedAction,
    layout: Layout,
    width: int,
    height: int,
    send: Callable[..., None],
) -> bool:
    if action.movement != "wait":
        return _execute_action(action, layout, width, height, send)
    if action.ability == "none":
        return False
    button = layout.buttons[action.ability]
    if button is None:
        raise MobileTestbedError(f"layout action {action.ability} is not calibrated")
    start = _point(width, height, *button)
    end = start
    if action.aim != "none":
        vector = _direction_vector(action.aim, layout)
        end = _point(
            width,
            height,
            button[0] + layout.aim_radius * vector[0],
            button[1] + layout.aim_radius * vector[1],
        )
    send("swipe", *map(str, (*start, *end, action.hold_ms)))
    return True


def run_mobile_demonstrate_rgb_teacher_v25(
    *,
    serial: str,
    output_dir: Path,
    layout_path: Path,
    video_node: Path,
    teacher_report_path: Path,
    run_seconds: float,
    enable_input: bool = False,
    max_actions: int = 0,
    shard_size: int = 256,
    stream_fps: int = 30,
    formal_session: bool = False,
    warmup_basic_attack: bool = False,
    patrol: bool = False,
    balanced_actions: bool = False,
) -> dict[str, object]:
    if (
        not math.isfinite(run_seconds)
        or run_seconds <= 0
        or enable_input != (max_actions > 0)
        or not 1 <= shard_size <= 1024
        or not 10 <= stream_fps <= 60
        or (formal_session and (not enable_input or run_seconds < 300))
        or (warmup_basic_attack and (not formal_session or not enable_input))
        or (patrol and not enable_input)
        or (balanced_actions and (not enable_input or formal_session))
    ):
        raise MobileTestbedError("RGB teacher demonstrator bounds are invalid")
    _new_large_output(output_dir)
    guard = _open_device_guard(serial)
    layout, layout_sha256 = load_layout(layout_path)
    if (guard.width, guard.height) != (layout.width, layout.height) or any(
        layout.buttons[name] is None for name in ABILITIES[1:4]
    ):
        raise MobileTestbedError("RGB teacher requires the calibrated combat layout")
    calibration = load_rgb_teacher_calibration(teacher_report_path, layout_sha256)
    contract, contract_sha256 = _rgb_teacher_contract(
        layout_sha256, calibration, balanced_actions=balanced_actions
    )
    stream = ScrcpyV4L2(guard.serial, video_node, stream_fps)
    pipe = AdbInputPipe(guard.serial) if enable_input else None
    guarded_send = _guarded_send(guard, pipe.send) if pipe is not None else None
    history: deque[tuple[int, int, np.ndarray]] = deque(
        maxlen=RGB_TEACHER_WINDOW_FRAMES + RGB_TEACHER_HISTORY_FRAMES + 2
    )
    frame_stream: list[tuple[int, np.ndarray]] = []
    samples: list[RGBTeacherSample] = []
    rows: list[dict[str, object]] = []
    executed = 0
    warmup_input_commands_sent = 0
    warmup_execution_timestamp_ns = -1
    environment_actions: list[dict[str, object]] = []
    status = "COMPLETED"
    last_dispatch: dict[int, int] = {}
    dispatch_counts = {index: 0 for index in range(1, 4)}
    last_any_dispatch = 0
    stream.start()
    started = 0.0
    try:
        warmup = RGB_TEACHER_WINDOW_FRAMES + RGB_TEACHER_HISTORY_FRAMES
        next_sample = time.monotonic()
        while len(history) < warmup:
            guard.check()
            now = time.monotonic()
            if now < next_sample:
                time.sleep(next_sample - now)
            timestamp_ns, frame = stream.frame_with_timestamp()
            normalized = _model_frame(frame).astype(np.uint8, copy=False)
            frame_stream.append((timestamp_ns, normalized))
            history.append(
                (timestamp_ns, len(frame_stream) - 1, _rgb_teacher_views(normalized))
            )
            next_sample += 1.0 / RGB_TEACHER_SAMPLE_HZ
        if warmup_basic_attack:
            decision_timestamp_ns = time.monotonic_ns()
            deadline_ns = decision_timestamp_ns + RGB_TEACHER_EXECUTION_LAG_MS * 1_000_000
            remaining = (deadline_ns - time.monotonic_ns()) / 1_000_000_000
            if remaining > 0:
                time.sleep(remaining)
            assert guarded_send is not None
            sent = _execute_action_v2(
                FactorizedAction(ability="basic_attack"),
                layout,
                guard.width,
                guard.height,
                guarded_send,
            )
            warmup_execution_timestamp_ns = time.monotonic_ns()
            if not sent or warmup_execution_timestamp_ns < deadline_ns:
                raise MobileTestbedError("RGB teacher warmup action was not guarded")
            warmup_input_commands_sent = 1
            last_dispatch[1] = warmup_execution_timestamp_ns
            last_any_dispatch = warmup_execution_timestamp_ns
        started = next_decision = time.monotonic()
        next_sample = started + 1.0 / RGB_TEACHER_SAMPLE_HZ
        next_patrol = started
        while time.monotonic() - started < run_seconds:
            guard.check()
            now = time.monotonic()
            if now >= next_sample:
                timestamp_ns, frame = stream.frame_with_timestamp()
                normalized = _model_frame(frame).astype(np.uint8, copy=False)
                frame_stream.append((timestamp_ns, normalized))
                history.append(
                    (timestamp_ns, len(frame_stream) - 1, _rgb_teacher_views(normalized))
                )
                while next_sample <= now:
                    next_sample += 1.0 / RGB_TEACHER_SAMPLE_HZ
            if now < next_decision:
                time.sleep(min(next_decision - now, max(0.0, next_sample - now)))
                continue
            values = list(history)
            current_values = values[-RGB_TEACHER_WINDOW_FRAMES :]
            shifted_values = values[
                -RGB_TEACHER_WINDOW_FRAMES
                - RGB_TEACHER_HISTORY_FRAMES : -RGB_TEACHER_HISTORY_FRAMES
            ]
            if len(current_values) != RGB_TEACHER_WINDOW_FRAMES or len(
                shifted_values
            ) != RGB_TEACHER_WINDOW_FRAMES:
                raise MobileTestbedError("RGB teacher causal history is incomplete")
            current = np.stack([value[2] for value in current_values])
            shifted = np.stack([value[2] for value in shifted_values])
            raw_decision = rgb_teacher_decision(
                current[-1], shifted[-1], layout, calibration
            )
            decision_timestamp_ns = time.monotonic_ns()
            candidate_id = raw_decision.combat_id
            if raw_decision.enemy_cue:
                eligible = [
                    index
                    for index in range(1, 4)
                    if decision_timestamp_ns - last_dispatch.get(index, 0)
                    >= RGB_TEACHER_CLASS_COOLDOWN_MS[index] * 1_000_000
                ]
                if balanced_actions and eligible:
                    least_executed = min(dispatch_counts.values())
                    eligible = [
                        index
                        for index in eligible
                        if dispatch_counts[index] == least_executed
                    ]
                candidate_id = (
                    max(
                        eligible,
                        key=lambda index: raw_decision.normalized_scores[index - 1],
                    )
                    if eligible
                    else 0
                )
            combat_id = candidate_id
            minimum_interval_ns = 500_000_000 if combat_id == 1 else 1_000_000_000
            if combat_id and (
                decision_timestamp_ns - last_dispatch.get(combat_id, 0)
                < minimum_interval_ns
            ):
                combat_id = 0
            if combat_id and (
                decision_timestamp_ns - last_any_dispatch
                < RGB_TEACHER_GLOBAL_DISPATCH_INTERVAL_MS * 1_000_000
            ):
                combat_id = 0
            action = FactorizedAction(
                ability=ABILITIES[combat_id],
                hold_ms=layout.skill_hold_ms if combat_id >= 2 else 0,
            )
            sent = False
            execution_timestamp_ns = -1
            if combat_id and enable_input:
                deadline_ns = decision_timestamp_ns + RGB_TEACHER_EXECUTION_LAG_MS * 1_000_000
                remaining = (deadline_ns - time.monotonic_ns()) / 1_000_000_000
                if remaining > 0:
                    time.sleep(remaining)
                assert guarded_send is not None
                sent = _execute_action_v2(
                    action, layout, guard.width, guard.height, guarded_send
                )
                execution_timestamp_ns = time.monotonic_ns()
                if not sent or execution_timestamp_ns < deadline_ns:
                    raise MobileTestbedError("RGB teacher action was not causally dispatched")
                last_dispatch[combat_id] = execution_timestamp_ns
                dispatch_counts[combat_id] += 1
                last_any_dispatch = execution_timestamp_ns
                executed += 1
            confidence = float(
                min(
                    1.0,
                    max(
                        0.0,
                        raw_decision.margin,
                        raw_decision.enemy_red_pixels / RGB_TEACHER_ENEMY_RED_PIXELS
                        if raw_decision.enemy_cue
                        else 0.0,
                    ),
                )
            )
            observation_end_ns = current_values[-1][0]
            samples.append(
                RGBTeacherSample(
                    current_values[-1][1],
                    shifted_values[-1][1],
                    combat_id,
                    observation_end_ns,
                    decision_timestamp_ns,
                    execution_timestamp_ns,
                    confidence,
                    sent,
                )
            )
            rows.append(
                {
                    "schema_version": RGB_TEACHER_SCHEMA,
                    "sequence": len(rows),
                    "observation_end_timestamp_ns": observation_end_ns,
                    "decision_timestamp_ns": decision_timestamp_ns,
                    "execution_timestamp_ns": execution_timestamp_ns,
                    "frame_sha256": hashlib.sha256(current[-1, 0].tobytes()).hexdigest(),
                    "candidate_combat": ABILITIES[candidate_id],
                    "combat": ABILITIES[combat_id],
                    "activity": round(raw_decision.activity, 8),
                    "normalized_scores": [round(value, 8) for value in raw_decision.normalized_scores],
                    "margin": round(raw_decision.margin, 8),
                    "confidence": round(confidence, 8),
                    "enemy_red_pixels": raw_decision.enemy_red_pixels,
                    "enemy_red_row_max": raw_decision.enemy_red_row_max,
                    "enemy_cue": raw_decision.enemy_cue,
                    "source": RGB_TEACHER_SOURCE,
                    "input_enabled": enable_input,
                    "input_sent": sent,
                }
            )
            next_decision += 1.0 / RGB_TEACHER_DECISION_HZ
            if next_decision < time.monotonic():
                next_decision = time.monotonic()
            if enable_input and executed >= max_actions:
                status = "ACTION_CAP_REACHED"
                break
            if patrol and time.monotonic() >= next_patrol:
                navigation = rgb_teacher_minimap_navigation(current[-1, 0])
                movement = (
                    navigation.movement
                    if navigation is not None
                    else RGB_TEACHER_PATROL_DIRECTIONS[
                        len(environment_actions) % len(RGB_TEACHER_PATROL_DIRECTIONS)
                    ]
                )
                patrol_decision_ns = time.monotonic_ns()
                assert guarded_send is not None
                patrol_sent = _execute_action_v2(
                    FactorizedAction(
                        movement=movement,
                        hold_ms=RGB_TEACHER_PATROL_HOLD_MS,
                    ),
                    layout,
                    guard.width,
                    guard.height,
                    guarded_send,
                )
                patrol_execution_ns = time.monotonic_ns()
                if not patrol_sent:
                    raise MobileTestbedError("RGB teacher patrol action was not guarded")
                environment_actions.append(
                    {
                        "source": (
                            "guarded_rgb_minimap_navigation_v1"
                            if navigation is not None
                            else "guarded_deterministic_patrol_v1"
                        ),
                        "movement": movement,
                        "hold_ms": RGB_TEACHER_PATROL_HOLD_MS,
                        "observation_end_timestamp_ns": current_values[-1][0],
                        "frame_sha256": hashlib.sha256(current[-1, 0].tobytes()).hexdigest(),
                        "player_yx": list(navigation.player_yx) if navigation is not None else None,
                        "target_yx": list(navigation.target_yx) if navigation is not None else None,
                        "decision_timestamp_ns": patrol_decision_ns,
                        "execution_timestamp_ns": patrol_execution_ns,
                    }
                )
                next_patrol += RGB_TEACHER_PATROL_INTERVAL_SECONDS
    except KeyboardInterrupt:
        status = "STOPPED"
    finally:
        if pipe is not None:
            pipe.close()
        stream.close()
    if not samples:
        raise MobileTestbedError("RGB teacher demonstration captured no decisions")
    duration = time.monotonic() - started
    counts = {
        name: sum(sample.combat_id == index for sample in samples)
        for index, name in enumerate(ABILITIES[:4])
    }
    valid_dispatches = all(
        (sample.combat_id == 0 and not sample.input_sent)
        or (sample.combat_id > 0 and sample.input_sent)
        for sample in samples
    )
    cycle_coverage = len(samples) / max(run_seconds * RGB_TEACHER_DECISION_HZ, 1.0)
    formal_complete = bool(
        formal_session
        and status == "COMPLETED"
        and duration >= 300
        and len(samples) >= RGB_TEACHER_MIN_FORMAL_SAMPLES
        and all(counts[name] > 0 for name in ABILITIES[:4])
        and valid_dispatches
        and cycle_coverage >= RGB_TEACHER_MIN_FORMAL_DECISION_COVERAGE
    )
    if formal_session and not formal_complete:
        status = "INCOMPLETE_FORMAL_ATTEMPT"
    summary: dict[str, object] = {
        "schema_version": RGB_TEACHER_SCHEMA,
        "dataset_schema_version": RGB_TEACHER_DATA_SCHEMA,
        "status": status,
        "testbed_owner_authorized": True,
        "serial_sha256": hashlib.sha256(guard.serial.encode()).hexdigest(),
        "layout_sha256": layout_sha256,
        "teacher_report_sha256": calibration.report_sha256,
        "action_contract_sha256": contract_sha256,
        "capture_mode": "scrcpy-v4l2-no-control",
        "event_source": RGB_TEACHER_SOURCE,
        "duration_seconds": round(duration, 6),
        "samples": len(samples),
        "executed_actions": executed,
        "combat_counts": counts,
        "window_frames": RGB_TEACHER_WINDOW_FRAMES,
        "sample_hz": RGB_TEACHER_SAMPLE_HZ,
        "decision_hz": RGB_TEACHER_DECISION_HZ,
        "execution_lag_ms": RGB_TEACHER_EXECUTION_LAG_MS,
        "activity_threshold": min(
            calibration.activity_threshold, RGB_TEACHER_LIVE_ACTIVITY_THRESHOLD
        ),
        "global_dispatch_interval_ms": RGB_TEACHER_GLOBAL_DISPATCH_INTERVAL_MS,
        "decision_cycle_coverage": cycle_coverage,
        "input_enabled": enable_input,
        "max_actions": max_actions,
        "balanced_actions": balanced_actions,
        "warmup_basic_attack": warmup_basic_attack,
        "warmup_input_commands_sent": warmup_input_commands_sent,
        "warmup_execution_timestamp_ns": warmup_execution_timestamp_ns,
        "environment_driver": (
            "guarded_rgb_minimap_navigation_v1_with_deterministic_fallback"
            if patrol
            else "none"
        ),
        "environment_actions": environment_actions,
        "environment_input_commands_sent": len(environment_actions),
        "patrol_interval_seconds": RGB_TEACHER_PATROL_INTERVAL_SECONDS if patrol else 0,
        "patrol_hold_ms": RGB_TEACHER_PATROL_HOLD_MS if patrol else 0,
        "formal_session": formal_session,
        "published_as_formal": formal_complete,
        "training_eligible": formal_complete,
        "manual_annotation_required": False,
        "derived_rgb_frames_persisted": True,
        "raw_frames_persisted": False,
        "raw_video_or_device_paths_persisted": False,
        "video_test_accessed": False,
        "model_checkpoint_loaded": False,
    }
    published = (
        output_dir
        if status != "INCOMPLETE_FORMAL_ATTEMPT"
        else output_dir.parent
        / "diagnostics"
        / f"{output_dir.name}-attempt-{time.monotonic_ns()}"
    )
    _publish_rgb_teacher_session(
        published, frame_stream, samples, rows, summary, contract, shard_size
    )
    return summary


def run_mobile_demonstrate_keyboard_v2(
    *,
    serial: str,
    output_dir: Path,
    layout_path: Path,
    video_node: Path,
    run_seconds: float = 300.0,
    max_actions: int = 600,
    shard_size: int = 256,
    stream_fps: int = 30,
    key_reader: Callable[[float], str | None] | None = None,
    formal_session: bool = False,
) -> dict[str, object]:
    if (
        not 1 <= run_seconds <= 300
        or not 1 <= max_actions <= 2000
        or not 1 <= shard_size <= 1024
        or not 1 <= stream_fps <= 60
        or (formal_session and (key_reader is None or run_seconds != 300))
    ):
        raise MobileTestbedError("keyboard v2 demonstrator bounds are invalid")
    output_dir = _new_large_output(output_dir)
    guard = _open_device_guard(serial)
    layout, layout_sha256 = load_layout(layout_path)
    if (guard.width, guard.height) != (layout.width, layout.height) or any(
        point is None for point in layout.buttons.values()
    ):
        raise MobileTestbedError("keyboard v2 requires the complete calibrated layout")
    contract, contract_sha256 = _keyboard_v2_contract(layout_sha256)
    stream, keyboard, pipe = (
        ScrcpyV4L2(guard.serial, video_node, stream_fps),
        None if key_reader is not None else TerminalKeyboard(),
        AdbInputPipe(guard.serial),
    )
    controller = KeyboardV2Demonstrator()
    guarded_send = _guarded_send(guard, pipe.send)
    windows: deque[tuple[int, np.ndarray]] = deque(maxlen=TOUCH_WINDOW_FRAMES + 8)
    samples: list[tuple[np.ndarray, FactorizedAction, int, bool]] = []
    rows: list[dict[str, object]] = []
    status, started = "COMPLETED", 0.0
    stream.start()
    try:
        while len(windows) < TOUCH_WINDOW_FRAMES + 1:
            guard.check()
            windows.append((time.monotonic_ns(), _model_frame(stream.frame())))
            time.sleep(1.0 / TOUCH_SAMPLE_HZ)
        read_key: Callable[[float], str | None]
        if keyboard is not None:
            keyboard.__enter__()
            read_key = keyboard.read
            print("T8-v2 ARMED: focus this terminal and press Enter to start.", flush=True)
            while keyboard.read(0.25) not in {"\n", "\r"}:
                guard.check()
            print(
                "T8-v2 ACTIVE: directions w/e/d/c/x/z/a/q; s/space wait; f attack; "
                "1/2/3 then direction; j/k/l short/medium/long; Escape stops.",
                flush=True,
            )
        else:
            assert key_reader is not None
            read_key = key_reader
        started = next_frame = time.monotonic()
        while time.monotonic() - started < run_seconds and len(samples) < max_actions:
            guard.check()
            now = time.monotonic()
            while now >= next_frame:
                windows.append((time.monotonic_ns(), _model_frame(stream.frame())))
                next_frame += 1.0 / TOUCH_SAMPLE_HZ
            key = read_key(min(0.05, max(0.0, run_seconds - (time.monotonic() - started))))
            if key is None:
                continue
            action, stop = controller.feed(key)
            if stop:
                status = "STOPPED"
                break
            if action is None:
                continue
            sent = _execute_action_v2(
                action, layout, guard.width, guard.height, guarded_send
            )
            timestamp_ns = time.monotonic_ns()
            causal = [frame for frame_time, frame in windows if frame_time <= timestamp_ns - 100_000_000]
            if len(causal) < TOUCH_WINDOW_FRAMES:
                continue
            window = np.stack(causal[-TOUCH_WINDOW_FRAMES:])
            samples.append((window, action, timestamp_ns, sent))
            rows.append(
                {
                    "schema_version": KEYBOARD_V2_SCHEMA,
                    "sequence": len(rows),
                    "timestamp_ns": timestamp_ns,
                    "frame_sha256": hashlib.sha256(causal[-1].tobytes()).hexdigest(),
                    "action": _action_record(action),
                    "hold_bucket": _hold_bucket(action.hold_ms),
                    "source": EXECUTED_ACTION_SOURCE,
                    "input_sent": sent,
                }
            )
            if keyboard is not None:
                missing = _touch_factor_coverage([item[1] for item in samples])["missing"]
                print(f"T8-v2 ACCEPTED {len(samples)} remaining={missing}", flush=True)
    except KeyboardInterrupt:
        status = "STOPPED"
    finally:
        if keyboard is not None:
            keyboard.__exit__()
        pipe.close()
        stream.close()
    if not samples:
        raise MobileTestbedError("keyboard v2 demonstration captured no actions")
    coverage = _touch_factor_coverage([item[1] for item in samples])
    duration = time.monotonic() - started
    if len(samples) >= max_actions and duration < run_seconds:
        status = "ACTION_CAP_REACHED"
    summary: dict[str, object] = {
        "schema_version": KEYBOARD_V2_SCHEMA,
        "dataset_schema_version": KEYBOARD_V2_DATA_SCHEMA,
        "status": status,
        "testbed_owner_authorized": True,
        "serial_sha256": hashlib.sha256(guard.serial.encode()).hexdigest(),
        "layout_sha256": layout_sha256,
        "action_contract_sha256": contract_sha256,
        "capture_mode": "scrcpy-v4l2",
        "duration_seconds": round(duration, 6),
        "samples": len(samples),
        "executed_actions": sum(item[3] for item in samples),
        "window_frames": TOUCH_WINDOW_FRAMES,
        "sample_hz": TOUCH_SAMPLE_HZ,
        "observation_end_lag_ms": 100,
        "event_source": EXECUTED_ACTION_SOURCE,
        "formal_session": formal_session,
        "factor_coverage": coverage,
        "raw_frames_persisted": False,
        "derived_rgb_frames_persisted": True,
        "manual_annotation_required": False,
        "raw_video_or_device_paths_persisted": False,
    }
    valid_dispatches = all(sent or (action.movement == "wait" and action.ability == "none") for _window, action, _timestamp, sent in samples)
    if formal_session and (status != "COMPLETED" or duration < 300 or len(samples) < KEYBOARD_V2_MIN_FORMAL_SAMPLES or not coverage["complete"] or not valid_dispatches):
        status = summary["status"] = "INCOMPLETE_FORMAL_ATTEMPT"
    published = output_dir if status != "INCOMPLETE_FORMAL_ATTEMPT" else output_dir.parent / "diagnostics" / f"{output_dir.name}-attempt-{time.monotonic_ns()}"
    summary["published_as_formal"] = formal_session and published == output_dir
    _publish_keyboard_v2(published, samples, rows, summary, contract, shard_size)
    return summary


def run_mobile_demonstrate_keyboard_v21(
    *,
    serial: str,
    output_dir: Path,
    layout_path: Path,
    run_seconds: float,
    max_actions: int | None = None,
    shard_size: int = 256,
    stream_fps: int = 30,
    countdown_seconds: float = 3.0,
    formal_session: bool = False,
    diagnostic_control_smoke: bool = False,
    diagnostic_inverse_probe: bool = False,
) -> dict[str, object]:
    if (
        not math.isfinite(run_seconds)
        or run_seconds <= 0
        or (max_actions is not None and max_actions <= 0)
        or not 1 <= shard_size <= 1024
        or not 10 <= stream_fps <= 60
        or not math.isfinite(countdown_seconds)
        or not 0 <= countdown_seconds <= 10
        or (formal_session and run_seconds < 300)
        or (diagnostic_control_smoke and (run_seconds != 20 or formal_session))
        or (diagnostic_inverse_probe and (run_seconds < 20 or formal_session))
        or (diagnostic_control_smoke and diagnostic_inverse_probe)
    ):
        raise MobileTestbedError("keyboard v2.1 demonstrator bounds are invalid")
    output_dir = _new_large_output(output_dir)
    guard = _open_device_guard(serial)
    layout, layout_sha256 = load_layout(layout_path)
    if (guard.width, guard.height) != (layout.width, layout.height) or any(
        point is None for point in layout.buttons.values()
    ):
        raise MobileTestbedError("keyboard v2.1 requires the complete calibrated layout")
    source = SCRCPY_EXECUTED_ACTION_SOURCE
    if diagnostic_control_smoke:
        source = DIAGNOSTIC_CONTROL_SOURCE
    elif diagnostic_inverse_probe:
        source = DIAGNOSTIC_INVERSE_SOURCE
    contract, contract_sha256 = _keyboard_v21_contract(layout_sha256, source)
    session = ScrcpyControlSession(guard.serial, stream_fps)
    watchdog = GuardWatchdog(guard)
    diagnostic_run = diagnostic_control_smoke or diagnostic_inverse_probe
    keyboard = None if diagnostic_run else FocusedKeyboardWindow()
    controller = LiveKeyboardController(layout, guard.width, guard.height)
    writer: KeyboardV21Writer | None = None
    windows: deque[tuple[int, np.ndarray]] = deque(maxlen=TOUCH_WINDOW_FRAMES + 8)
    counts = {name: Counter[str]() for name in ("movement", "combat", "aim", "hold")}
    dispatch_latencies_ms: deque[float] = deque(maxlen=10000)
    transition_sequence = 0
    touch_messages = 0
    parallel_samples = 0
    conflict_events = 0
    last_dispatch_ns = 0
    status = "COMPLETED"
    started = 0.0
    session.start()
    try:
        if session.frame_size != (guard.width, guard.height):
            raise MobileTestbedError("scrcpy display size differs from the guarded layout")
        watchdog.start()
        last_frame_timestamp_ns = 0
        while len(windows) < TOUCH_WINDOW_FRAMES + 2:
            watchdog.ensure_fresh()
            frame_timestamp_ns, frame = session.frame()
            if frame_timestamp_ns > last_frame_timestamp_ns:
                windows.append((frame_timestamp_ns, _model_frame(frame)))
                last_frame_timestamp_ns = frame_timestamp_ns
            time.sleep(1.0 / TOUCH_SAMPLE_HZ)
        if keyboard is not None:
            keyboard.__enter__()
            keyboard.wait_for_focus()
            countdown_end = time.monotonic() + countdown_seconds
            while time.monotonic() < countdown_end:
                watchdog.ensure_fresh()
                event = keyboard.poll()
                if event is not None and event[0] == "Escape" and event[1]:
                    raise MobileTestbedError("keyboard v2.1 demonstration cancelled before start")
                time.sleep(0.01)
        writer = KeyboardV21Writer(output_dir, contract, shard_size)
        started = time.monotonic()
        next_sample = started
        diagnostic_events = deque(
            inverse_probe_events(run_seconds)
            if diagnostic_inverse_probe
            else DIAGNOSTIC_SMOKE_EVENTS
        )
        while time.monotonic() - started < run_seconds:
            watchdog.ensure_fresh()
            if keyboard is not None and not keyboard.has_focus():
                raise MobileTestbedError("focused keyboard window lost desktop focus")
            keyboard_event: tuple[str, bool, int] | None = None
            if diagnostic_run:
                if diagnostic_events and time.monotonic() - started >= diagnostic_events[0][0]:
                    _offset, key, pressed = diagnostic_events.popleft()
                    keyboard_event = (key, pressed, time.monotonic_ns())
            else:
                assert keyboard is not None
                keyboard_event = keyboard.poll()
            if keyboard_event is not None:
                key, pressed, event_timestamp_ns = keyboard_event
                if key == "Escape" and pressed:
                    status = "STOPPED"
                    break
                operations, changed, conflict = controller.transition(
                    key, pressed, event_timestamp_ns
                )
                dispatch_times: list[int] = []
                for operation in operations:
                    watchdog.ensure_fresh()
                    dispatch_times.append(session.touch(operation, guard.width, guard.height))
                    touch_messages += 1
                if dispatch_times:
                    last_dispatch_ns = dispatch_times[-1]
                    dispatch_latencies_ms.append(
                        (last_dispatch_ns - event_timestamp_ns) / 1_000_000
                    )
                if conflict:
                    conflict_events += 1
                    writer.add_event(
                        {
                            "schema_version": KEYBOARD_V21_SCHEMA,
                            "event_type": "combat_conflict",
                            "sequence": writer.events,
                            "timestamp_ns": event_timestamp_ns,
                            "source": source,
                            "input_sent": False,
                        }
                    )
                if changed:
                    transition_sequence += 1
                    action = controller.action(last_dispatch_ns or event_timestamp_ns)
                    writer.add_event(
                        {
                            "schema_version": KEYBOARD_V21_SCHEMA,
                            "event_type": "semantic_transition",
                            "sequence": writer.events,
                            "transition_sequence": transition_sequence,
                            "timestamp_ns": event_timestamp_ns,
                            "dispatch_completed_ns": last_dispatch_ns,
                            "dispatch_count": len(dispatch_times),
                            "action": _action_record(action),
                            "hold_bucket": _hold_bucket(action.hold_ms),
                            "source": source,
                            "input_sent": bool(dispatch_times),
                        }
                    )
                    if max_actions is not None and transition_sequence >= max_actions:
                        status = "ACTION_CAP_REACHED"
                        break
            now = time.monotonic()
            if now >= next_sample:
                sample_timestamp_ns = time.monotonic_ns()
                frame_timestamp_ns, frame = session.frame()
                if frame_timestamp_ns <= last_frame_timestamp_ns:
                    time.sleep(0.002)
                    continue
                windows.append((frame_timestamp_ns, _model_frame(frame)))
                last_frame_timestamp_ns = frame_timestamp_ns
                causal = [
                    item
                    for frame_time, item in windows
                    if frame_time <= sample_timestamp_ns - 100_000_000
                ]
                if len(causal) >= TOUCH_WINDOW_FRAMES:
                    action = controller.action(sample_timestamp_ns)
                    writer.add_sample(
                        LiveSample(
                            np.stack(causal[-TOUCH_WINDOW_FRAMES:]),
                            action,
                            sample_timestamp_ns,
                            transition_sequence,
                            last_dispatch_ns,
                            last_dispatch_ns > 0
                            or (action.movement == "wait" and action.ability == "none"),
                        )
                    )
                    counts["movement"][action.movement] += 1
                    counts["combat"][action.ability] += 1
                    counts["aim"][action.aim] += 1
                    counts["hold"][TOUCH_HOLD_BUCKETS[_hold_bucket(action.hold_ms)]] += 1
                    if action.movement != "wait" and action.ability != "none":
                        parallel_samples += 1
                next_sample = now + 1.0 / TOUCH_SAMPLE_HZ
            time.sleep(0.002)
    except KeyboardInterrupt:
        status = "STOPPED"
    except BaseException:
        if writer is not None:
            writer.abort()
        raise
    finally:
        try:
            watchdog.ensure_fresh()
            for operation in controller.release_all():
                session.touch(operation, guard.width, guard.height)
        except MobileTestbedError:
            pass
        if keyboard is not None:
            keyboard.__exit__()
        watchdog.stop()
        session.close()
    if writer is None or writer.samples == 0:
        if writer is not None:
            writer.abort()
        raise MobileTestbedError("keyboard v2.1 demonstration captured no causal samples")
    duration = time.monotonic() - started
    coverage = _live_factor_coverage(
        counts, parallel_samples=parallel_samples, conflict_events=conflict_events
    )
    minimum_formal_samples = int(run_seconds * TOUCH_SAMPLE_HZ * 0.95)
    summary: dict[str, object] = {
        "schema_version": KEYBOARD_V21_SCHEMA,
        "dataset_schema_version": KEYBOARD_V21_DATA_SCHEMA,
        "status": status,
        "testbed_owner_authorized": True,
        "serial_sha256": hashlib.sha256(guard.serial.encode()).hexdigest(),
        "layout_sha256": layout_sha256,
        "action_contract_sha256": contract_sha256,
        "scrcpy_server_version": SCRCPY_SERVER_VERSION,
        "scrcpy_server_sha256": session.server_sha256,
        "capture_mode": "scrcpy-socket-control",
        "duration_seconds": round(duration, 6),
        "samples": writer.samples,
        "semantic_events": transition_sequence,
        "touch_messages_sent": touch_messages,
        "window_frames": TOUCH_WINDOW_FRAMES,
        "sample_hz": TOUCH_SAMPLE_HZ,
        "observation_end_lag_ms": 100,
        "event_source": source,
        "formal_session": formal_session,
        "factor_coverage": coverage,
        "keyboard_to_socket_p95_ms": (
            round(float(np.percentile(dispatch_latencies_ms, 95)), 3)
            if dispatch_latencies_ms
            else None
        ),
        "raw_frames_persisted": False,
        "derived_rgb_frames_persisted": True,
        "manual_annotation_required": False,
        "training_eligible": not diagnostic_run,
        "diagnostic_control_smoke": diagnostic_control_smoke,
        "diagnostic_inverse_probe": diagnostic_inverse_probe,
        "inverse_probe_press_events": (
            sum(pressed for _offset, _key, pressed in inverse_probe_events(run_seconds))
            if diagnostic_inverse_probe
            else 0
        ),
        "raw_video_or_device_paths_persisted": False,
        "code_or_dataset_scale_cap": None,
    }
    if formal_session and (
        status != "COMPLETED"
        or duration < run_seconds
        or writer.samples < minimum_formal_samples
        or not cast(dict[str, object], coverage["core"])["conflict_free"]
    ):
        summary["status"] = "INCOMPLETE_FORMAL_ATTEMPT"
    published = output_dir
    if summary["status"] == "INCOMPLETE_FORMAL_ATTEMPT":
        published = (
            output_dir.parent
            / "diagnostics"
            / f"{output_dir.name}-attempt-{time.monotonic_ns()}"
        )
    summary["published_as_formal"] = formal_session and published == output_dir
    writer.finalize(summary, published)
    return summary


def run_mobile_demonstrate(
    *,
    serial: str,
    output_dir: Path,
    run_seconds: float = 300.0,
    max_actions: int = 300,
    shard_size: int = 256,
    layout_path: Path | None = None,
    capture_mode: str = "adb-png",
    video_node: Path | None = None,
    stream_fps: int = 30,
    key_reader: Callable[[float], str | None] | None = None,
    event_source: str = TERMINAL_DEMONSTRATION_SOURCE,
) -> dict[str, object]:
    if (
        not 0 < run_seconds <= 300
        or not 1 <= max_actions <= 1000
        or not 1 <= shard_size <= 1024
        or capture_mode != "scrcpy-v4l2"
        or not 1 <= stream_fps <= 60
        or layout_path is None
        or event_source not in DEMONSTRATION_SOURCES
    ):
        raise MobileTestbedError("demonstrator bounds are invalid")
    output_dir = _new_large_output(output_dir)
    if video_node is None:
        raise MobileTestbedError("scrcpy capture requires --video-node")
    guard = _open_device_guard(serial)
    layout, layout_sha256 = load_layout(layout_path)
    if (guard.width, guard.height) != (layout.width, layout.height) or any(
        point is None for point in layout.buttons.values()
    ):
        raise MobileTestbedError("demonstrator requires the complete calibrated layout")
    stream = ScrcpyV4L2(guard.serial, video_node, stream_fps)
    keyboard = None if key_reader is not None else TerminalKeyboard()
    stream.start()
    input_pipe = AdbInputPipe(guard.serial)
    controller = KeyboardDemonstrator(layout)
    samples: list[tuple[np.ndarray, FactorizedAction, int, bool]] = []
    rows: list[dict[str, object]] = []
    start, status, executed = 0.0, "COMPLETED", 0
    windows: deque[np.ndarray] = deque(maxlen=DEMONSTRATOR_WINDOW_FRAMES)
    guarded_send = _guarded_send(guard, input_pipe.send)
    try:
        while len(windows) < DEMONSTRATOR_WINDOW_FRAMES:
            windows.append(_model_frame(stream.frame()))
            time.sleep(1.0 / DEMONSTRATOR_SAMPLE_HZ)
        read_key: Callable[[float], str | None]
        if keyboard is not None:
            keyboard.__enter__()
            read_key = keyboard.read
            print(
                "T8 DEMONSTRATOR ARMED: focus this terminal and press Enter to start.",
                flush=True,
            )
            while True:
                guard.check()
                if keyboard.read(0.25) in {"\n", "\r"}:
                    break
            windows.clear()
            while len(windows) < DEMONSTRATOR_WINDOW_FRAMES:
                windows.append(_model_frame(stream.frame()))
                time.sleep(1.0 / DEMONSTRATOR_SAMPLE_HZ)
            print(
                "T8 DEMONSTRATOR ACTIVE: focus this terminal and use w/e/d/c/x/z/a/q, "
                "s/space, f, or 1/2/3 plus a direction; Escape stops.",
                flush=True,
            )
        else:
            assert key_reader is not None
            read_key = key_reader
        start = next_frame = time.monotonic()
        while time.monotonic() - start < run_seconds and executed < max_actions:
            now = time.monotonic()
            if now >= next_frame:
                windows.append(_model_frame(stream.frame()))
                next_frame += 1.0 / DEMONSTRATOR_SAMPLE_HZ
            key = read_key(min(0.05, max(0.0, run_seconds - (time.monotonic() - start))))
            if key is None:
                continue
            action, stop = controller.feed(key)
            if stop:
                status = "STOPPED"
                break
            if action is None:
                key_status = (
                    "ARMED" if key.lower() in KEY_TO_ABILITY and key.lower() != "f" else "IGNORED"
                )
                if keyboard is not None:
                    print(f"T8 KEY {key_status}: {key!r}", flush=True)
                continue
            if keyboard is not None:
                print(f"T8 KEY ACCEPTED: {key!r}", flush=True)
            timestamp_ns = time.monotonic_ns()
            frame = stream.frame()
            sent = _execute_action(action, layout, frame.shape[1], frame.shape[0], guarded_send)
            samples.append((np.stack(tuple(windows)), action, timestamp_ns, sent))
            rows.append(
                {
                    "schema_version": DEMONSTRATOR_SCHEMA,
                    "sequence": len(rows),
                    "timestamp_ns": timestamp_ns,
                    "frame_sha256": hashlib.sha256(windows[-1].tobytes()).hexdigest(),
                    "action": _action_record(action),
                    "input_sent": sent,
                    "source": event_source,
                }
            )
            executed += int(sent)
    except KeyboardInterrupt:
        status = "STOPPED"
    finally:
        if keyboard is not None:
            keyboard.__exit__()
        input_pipe.close()
        stream.close()
    if not samples:
        raise MobileTestbedError("demonstration captured no executable actions")
    if executed >= max_actions and time.monotonic() - start < run_seconds:
        status = "ACTION_CAP_REACHED"
    summary: dict[str, object] = {
        "schema_version": DEMONSTRATOR_SCHEMA,
        "dataset_schema_version": DEMONSTRATOR_DATA_SCHEMA,
        "status": status,
        "testbed_owner_authorized": True,
        "serial_sha256": hashlib.sha256(guard.serial.encode()).hexdigest(),
        "layout_sha256": layout_sha256,
        "capture_mode": capture_mode,
        "duration_seconds": round(time.monotonic() - start, 6),
        "max_actions": max_actions,
        "executed_actions": executed,
        "samples": len(samples),
        "window_frames": DEMONSTRATOR_WINDOW_FRAMES,
        "sample_hz": DEMONSTRATOR_SAMPLE_HZ,
        "shard_size": shard_size,
        "raw_frames_persisted": False,
        "derived_rgb_frames_persisted": True,
        "manual_annotation_required": False,
        "event_source": event_source,
    }
    _publish_demonstration(output_dir, samples, rows, summary, shard_size)
    return summary


def run_mobile_touch_demonstrate(
    *,
    serial: str,
    touch_descriptor: TouchDescriptor,
    touch_calibration_path: Path,
    output_dir: Path,
    run_seconds: float = 300.0,
    max_samples: int = 3000,
    shard_size: int = 256,
    layout_path: Path,
    video_node: Path,
    stream_fps: int = 30,
    formal_session: bool = False,
    semantic_smoke: bool = False,
) -> dict[str, object]:
    if (
        not 1.0 <= run_seconds <= 300.0
        or not 1 <= max_samples <= TOUCH_MAX_FORMAL_SAMPLES
        or not 1 <= shard_size <= 1024
        or not 1 <= stream_fps <= 60
        or (formal_session and run_seconds != 300)
        or (semantic_smoke and (formal_session or run_seconds != 20))
    ):
        raise MobileTestbedError("touch demonstrator bounds are invalid")
    output_dir = _new_large_output(output_dir)
    guard = _open_device_guard(serial)
    layout, layout_sha256 = load_layout(layout_path)
    if (guard.width, guard.height) != (layout.width, layout.height) or any(
        point is None for point in layout.buttons.values()
    ):
        raise MobileTestbedError("touch demonstrator requires the complete calibrated layout")
    calibration = load_touch_calibration(touch_calibration_path, touch_descriptor, layout_sha256)
    stream, observer = (
        ScrcpyV4L2(guard.serial, video_node, stream_fps),
        TouchObserver(guard.serial, touch_descriptor),
    )
    windows: deque[tuple[int, np.ndarray]] = deque(maxlen=TOUCH_WINDOW_FRAMES + 4)
    mapper = _TouchActionMapper(touch_descriptor, calibration, layout)
    samples: list[_TouchSample] = []
    rows: list[dict[str, object]] = []
    status, last_semantic, event_index, parallel_samples = "COMPLETED", None, -1, 0
    stream.start()
    observer.start()
    started = next_frame = next_sample = time.monotonic()
    try:
        while len(windows) < TOUCH_WINDOW_FRAMES + 1:
            guard.check()
            windows.append((time.monotonic_ns(), _model_frame(stream.frame())))
            time.sleep(1.0 / TOUCH_SAMPLE_HZ)
        started = next_frame = next_sample = time.monotonic()
        while time.monotonic() - started < run_seconds and len(samples) < max_samples:
            guard.check()
            now = time.monotonic()
            while now >= next_frame:
                windows.append((time.monotonic_ns(), _model_frame(stream.frame())))
                next_frame += 1.0 / TOUCH_SAMPLE_HZ
            packet = observer.read(0.01)
            if packet is not None:
                mapper.feed(packet)
            if now < next_sample:
                continue
            timestamp_ns = time.monotonic_ns()
            action = mapper.action(timestamp_ns)
            causal = [
                frame for frame_time, frame in windows if frame_time <= timestamp_ns - 100_000_000
            ]
            if len(causal) < TOUCH_WINDOW_FRAMES:
                next_sample += 1.0 / TOUCH_SAMPLE_HZ
                continue
            if mapper.conflicted:
                next_sample += 1.0 / TOUCH_SAMPLE_HZ
                continue
            semantic = _touch_semantic_key(action)
            _, frame = next((frame_time, frame) for frame_time, frame in reversed(windows) if frame_time <= timestamp_ns - 100_000_000)
            if semantic != last_semantic:
                event_index += 1
                rows.append({"schema_version": TOUCH_DEMONSTRATOR_SCHEMA, "sequence": len(rows), "timestamp_ns": timestamp_ns, "first_sample_index": len(samples), "frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(), "state": _touch_semantic_state(action), "source": OBSERVED_TOUCH_DEMONSTRATION_SOURCE, "evidence": "physical_touch_observed_not_app_acknowledged", "input_commands_sent": 0})
                last_semantic = semantic
            samples.append(_TouchSample(frame, action, timestamp_ns, event_index))
            if mapper.parallel:
                parallel_samples += 1
            next_sample += 1.0 / TOUCH_SAMPLE_HZ
    except KeyboardInterrupt:
        status = "STOPPED"
    finally:
        observer.close()
        stream.close()
    if not samples:
        raise MobileTestbedError("touch demonstration captured no observed actions")
    if len(samples) >= max_samples and time.monotonic() - started < run_seconds:
        status = "SAMPLE_CAP_REACHED"
    calibration_sha256 = _touch_calibration_payload(calibration)["calibration_sha256"]
    factor_coverage = _touch_factor_coverage([item.action for item in samples], parallel_samples=parallel_samples, conflict_samples=mapper.conflict_samples)
    duration = time.monotonic() - started
    if formal_session and (
        status != "COMPLETED"
        or duration < 300
        or len(samples) < TOUCH_MIN_FORMAL_SAMPLES
        or not any(item.action.movement != "wait" for item in samples)
        or mapper.conflict_samples > 0
    ):
        status = "INCOMPLETE_FORMAL_ATTEMPT"
    if semantic_smoke:
        status = "SMOKE_PASSED" if status == "COMPLETED" and factor_coverage["core_complete"] else "SMOKE_FAILED"
    summary = {
        "schema_version": TOUCH_DEMONSTRATOR_SCHEMA,
        "dataset_schema_version": TOUCH_DEMONSTRATOR_DATA_SCHEMA,
        "status": status,
        "testbed_owner_authorized": True,
        "serial_sha256": hashlib.sha256(guard.serial.encode()).hexdigest(),
        "layout_sha256": layout_sha256,
        "touch_calibration_sha256": calibration_sha256,
        "touch_descriptor_sha256": touch_descriptor.sha256,
        "capture_mode": "scrcpy-v4l2",
        "duration_seconds": round(duration, 6),
        "max_samples": max_samples,
        "samples": len(samples),
        "window_frames": TOUCH_WINDOW_FRAMES,
        "storage_frames_per_sample": 1,
        "sample_hz": TOUCH_SAMPLE_HZ,
        "observation_end_lag_ms": 100,
        "raw_frames_persisted": False,
        "raw_touch_events_persisted": False,
        "touch_device_path_persisted": False,
        "derived_rgb_frames_persisted": True,
        "manual_annotation_required": False,
        "event_source": OBSERVED_TOUCH_DEMONSTRATION_SOURCE,
        "input_commands_sent": 0,
        "factor_coverage": factor_coverage,
        "formal_session": formal_session,
        "semantic_smoke": semantic_smoke,
        "semantic_event_count": len(rows),
    }
    published = output_dir
    if status == "INCOMPLETE_FORMAL_ATTEMPT":
        published = (
            output_dir.parent
            / "diagnostics"
            / f"{output_dir.name}-attempt-{time.monotonic_ns()}"
        )
    summary["published_as_formal"] = formal_session and published == output_dir
    _publish_touch_demonstration(published, samples, rows, summary, shard_size)
    return summary


def run_mobile_testbed(
    *,
    serial: str,
    model_path: Path,
    output_dir: Path,
    device: str = "cuda",
    run_seconds: float = 60.0,
    infer_hz: int = 1,
    min_confidence: float = 0.90,
    enable_input: bool = False,
    max_actions: int = 0,
    layout_path: Path | None = None,
    capture_mode: str = "adb-png",
    video_node: Path | None = None,
    stream_fps: int = 30,
) -> dict[str, object]:
    if (
        not 0 < run_seconds <= 300
        or infer_hz <= 0
        or not 0 <= min_confidence <= 1
        or capture_mode not in {"adb-png", "scrcpy-v4l2"}
        or stream_fps <= 0
    ):
        raise MobileTestbedError("invalid test duration, rate, or confidence threshold")
    if enable_input != (max_actions > 0):
        raise MobileTestbedError("input requires an explicit positive max-actions bound")
    _new_output(output_dir)
    guard = _open_device_guard(serial)
    layout, layout_sha256 = load_layout(layout_path) if layout_path else (DEFAULT_LAYOUT, "default")
    if (guard.width, guard.height) != (layout.width, layout.height):
        raise MobileTestbedError("mobile layout does not match the active display")
    stream = (
        ScrcpyV4L2(guard.serial, video_node, stream_fps)
        if capture_mode == "scrcpy-v4l2" and video_node
        else None
    )
    if capture_mode == "scrcpy-v4l2" and stream is None:
        raise MobileTestbedError("scrcpy capture requires --video-node")
    if stream is not None:
        stream.start()
    predictor, model_seed = _open_v3_predictor(model_path, device)
    input_pipe = AdbInputPipe(guard.serial) if enable_input else None
    start = time.monotonic()
    interval, next_tick, next_action, executed, missed = 1.0 / infer_hz, start, start, 0, 0
    rows: list[dict[str, object]] = []
    status = "COMPLETED"
    try:
        while time.monotonic() - start < run_seconds:
            now = time.monotonic()
            if now < next_tick:
                time.sleep(next_tick - now)
            next_tick += interval
            capture_start = time.monotonic()
            frame = stream.frame() if stream is not None else _frame(guard.serial)
            capture_ms = (time.monotonic() - capture_start) * 1000.0
            inference_start = time.monotonic()
            predicted, confidence = predictor(_model_frame(frame)[None, ...])
            inference_ms = (time.monotonic() - inference_start) * 1000.0
            action, score = ACTIONS[int(predicted[0])], float(confidence[0])
            intent = _intent(action, layout)
            accepted = (
                enable_input
                and score >= min_confidence
                and executed < max_actions
                and time.monotonic() >= next_action
            )
            sent = (
                _input(
                    guard.serial,
                    intent,
                    layout,
                    frame.shape[1],
                    frame.shape[0],
                    _guarded_send(guard, input_pipe.send) if input_pipe else None,
                )
                if accepted
                else False
            )
            executed += int(sent)
            if sent:
                next_action = time.monotonic() + layout.move_hold_ms / 1000.0
            rows.append(
                {
                    "schema_version": SCHEMA,
                    "sequence": len(rows),
                    "frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
                    "model_action": action,
                    "intent": {
                        "movement": list(intent.movement) if intent.movement else None,
                        "attack": intent.attack,
                        "target": intent.target,
                    },
                    "confidence": round(score, 8),
                    "capture_ms": round(capture_ms, 4),
                    "inference_ms": round(inference_ms, 4),
                    "input_enabled": enable_input,
                    "input_sent": sent,
                    "reason": (
                        "EXECUTED"
                        if sent
                        else ("WAIT" if action == "wait" else "DRY_RUN_OR_LOW_CONFIDENCE")
                    ),
                }
            )
            if enable_input and executed >= max_actions:
                break
            missed += int(time.monotonic() > next_tick)
    except KeyboardInterrupt:
        status = "STOPPED"
    finally:
        if input_pipe is not None:
            input_pipe.close()
        if stream is not None:
            stream.close()
    summary: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": status,
        "testbed_owner_authorized": True,
        "serial_sha256": hashlib.sha256(guard.serial.encode()).hexdigest(),
        "model_seed": model_seed,
        "layout_sha256": layout_sha256,
        "capture_mode": capture_mode,
        "duration_seconds": round(time.monotonic() - start, 6),
        "inference_hz": infer_hz,
        "min_confidence": min_confidence,
        "input_enabled": enable_input,
        "max_actions": max_actions,
        "executed_actions": executed,
        "frames": len(rows),
        "achieved_hz": round(len(rows) / max(time.monotonic() - start, 1e-9), 6),
        "missed_deadlines": missed,
        "raw_frames_persisted": False,
    }
    _publish(output_dir, rows, summary)
    return summary
