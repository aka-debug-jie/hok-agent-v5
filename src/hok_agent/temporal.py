"""Compact RGB-only V6 temporal coach with a fail-closed V5 release binding."""

from __future__ import annotations

import json
import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from hok_agent import alignment

ACTION_NAMES: tuple[str, ...] = alignment.ACTION_TYPES
ABSTAIN = "ABSTAIN"


def tracking_gate(
    *,
    label_count: int,
    pck: float,
    visibility_f1: float,
    hp_mae: float,
    skill_ready_f1: float,
) -> bool:
    """Frozen V6 tracking gate; fewer than 300 sealed labels fails closed."""
    values = (pck, visibility_f1, hp_mae, skill_ready_f1)
    return (
        label_count >= 300
        and all(math.isfinite(value) for value in values)
        and pck >= 0.85
        and visibility_f1 >= 0.90
        and hp_mae <= 0.10
        and skill_ready_f1 >= 0.90
    )


def temporal_audit_gate(
    *,
    sealed_count: int,
    overall_precision: float,
    unlocked_class_precision: Mapping[str, float],
    coverage: float,
    transition_false_advice: float,
    ood_false_advice: float,
    switch_reduction: float,
    median_delay_ms: float,
    p95_delay_ms: float,
    live_hz: float,
    live_p95_ms: float,
) -> bool:
    """Frozen V6 temporal gate; fewer than 200 sealed clips fails closed."""
    values = (
        overall_precision,
        *unlocked_class_precision.values(),
        coverage,
        transition_false_advice,
        ood_false_advice,
        switch_reduction,
        median_delay_ms,
        p95_delay_ms,
        live_hz,
        live_p95_ms,
    )
    return (
        sealed_count >= 200
        and bool(unlocked_class_precision)
        and set(unlocked_class_precision).issubset(ACTION_NAMES)
        and all(math.isfinite(value) for value in values)
        and overall_precision >= 0.85
        and min(unlocked_class_precision.values()) >= 0.75
        and coverage >= 0.20
        and transition_false_advice <= 0.05
        and ood_false_advice <= 0.05
        and switch_reduction >= 0.50
        and median_delay_ms <= 300.0
        and p95_delay_ms <= 500.0
        and live_hz >= 10.0
        and live_p95_ms <= 100.0
    )


@dataclass(frozen=True)
class _ReleaseBinding:
    release_sha256: str
    allowed_classes: tuple[str, ...]
    class_thresholds: dict[str, float]


def _valid_sha(value: str | None) -> bool:
    return (
        value is not None and len(value) == 64 and all(char in "0123456789abcdef" for char in value)
    )


def _load_bound_release(
    path: Path | None,
    expected_model_sha256: str | None,
    expected_alignment_sha256: str | None,
    expected_audit_sha256: str | None,
    expected_config_sha256: str | None,
) -> tuple[_ReleaseBinding | None, str]:
    expected = (
        expected_model_sha256,
        expected_alignment_sha256,
        expected_audit_sha256,
        expected_config_sha256,
    )
    if path is None:
        return None, "NO_RELEASE"
    if (
        path.name != "release.json"
        or path.is_symlink()
        or not all(_valid_sha(value) for value in expected)
    ):
        return None, "RELEASE_BINDING"
    try:
        payload = alignment.load_release(path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None, "INVALID_RELEASE"
    fields = (
        "model_sha256",
        "alignment_sha256",
        "audit_sha256",
        "config_sha256",
    )
    if any(payload[field] != value for field, value in zip(fields, expected, strict=True)):
        return None, "RELEASE_BINDING"
    allowed = payload["allowed_classes"]
    thresholds = payload["class_thresholds"]
    if not isinstance(allowed, list) or not isinstance(thresholds, dict):
        return None, "INVALID_RELEASE"
    parsed_thresholds: dict[str, float] = {}
    for name, value in thresholds.items():
        if (
            not isinstance(name, str)
            or name not in ACTION_NAMES
            or isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or not 0.75 <= float(value) <= 1.0
        ):
            return None, "INVALID_RELEASE"
        parsed_thresholds[name] = float(value)
    if set(allowed) != set(parsed_thresholds):
        return None, "INVALID_RELEASE"
    return (
        _ReleaseBinding(
            release_sha256=str(payload["release_sha256"]),
            allowed_classes=tuple(allowed),
            class_thresholds=parsed_thresholds,
        ),
        "",
    )


@dataclass
class _Runtime:
    classes: deque[int]
    timestamps: deque[int]
    features: deque[Tensor]
    last_timestamp_ms: int | None = None
    cooldown_until_ms: int = -1


class _RGBEncoder(nn.Module):
    """Small RGB encoder with dual-hero heatmaps and a four-value HUD head."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
        )
        self.body = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.ReLU(),
        )
        self.heatmap_head = nn.Conv2d(32, 2, 1)
        self.hud_head = nn.Sequential(nn.Linear(96, 32), nn.ReLU(), nn.Linear(32, 4), nn.Sigmoid())

    def forward(self, rgb: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        stem = self.stem(rgb)
        heatmaps = torch.sigmoid(self.heatmap_head(F.adaptive_avg_pool2d(stem, (8, 8))))
        feature = F.adaptive_avg_pool2d(self.body(stem), 1).flatten(1)
        return feature, heatmaps, self.hud_head(feature)


class _CausalDepthwiseTCN(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.first = nn.Conv1d(channels, channels, 3, groups=channels)
        self.second = nn.Conv1d(channels, channels, 3, dilation=2, groups=channels)

    def forward(self, sequence: Tensor) -> Tensor:
        value = sequence.transpose(1, 2)
        value = torch.relu(self.first(F.pad(value, (2, 0))))
        value = torch.relu(self.second(F.pad(value, (4, 0))))
        return value.transpose(1, 2)


class TemporalModel(nn.Module):
    """Causal eight-frame RGB model; no legal mask or structured state enters this API."""

    def __init__(
        self,
        *,
        class_threshold: float = 0.75,
        quality_threshold: float = 0.80,
        stable_frames: int = 5,
        frame_interval_ms: int = 100,
        gap_reset_ms: int = 250,
        cooldown_ms: int = 500,
    ) -> None:
        super().__init__()
        self.seq_len = 8
        self.class_threshold = class_threshold
        self.quality_threshold = quality_threshold
        self.stable_frames = stable_frames
        self.frame_interval_ms = frame_interval_ms
        self.gap_reset_ms = gap_reset_ms
        self.cooldown_ms = cooldown_ms
        self.visual_encoder = _RGBEncoder()
        self.visual_projection = nn.Linear(96, 64)
        self.track_projection = nn.Linear(10, 64)
        self.tcn = _CausalDepthwiseTCN(64)
        self.logit_head = nn.Linear(64, len(ACTION_NAMES))
        self.ood_head = nn.Linear(64, 1)
        self._runtime: list[_Runtime] | None = None
        self._track_pos: Tensor | None = None
        self._track_vel: Tensor | None = None
        self._track_conf: Tensor | None = None
        self._last_frame: Tensor | None = None
        self._reset_generation = 0

    def reset(self, reason: str = "new_segment") -> None:
        self._runtime = None
        self._track_pos = None
        self._track_vel = None
        self._track_conf = None
        self._last_frame = None
        self._reset_generation += 1
        self._last_reset_reason = reason

    def _ensure_runtime(self, batch: int, device: torch.device) -> None:
        self._runtime = [
            _Runtime(
                deque(maxlen=self.stable_frames), deque(maxlen=self.stable_frames), deque(maxlen=8)
            )
            for _ in range(batch)
        ]
        self._track_pos = torch.zeros(batch, 2, 2, device=device)
        self._track_vel = torch.zeros(batch, 2, 2, device=device)
        self._track_conf = torch.zeros(batch, 2, device=device)

    def _reset_sample(self, index: int, reason: str) -> None:
        if (
            self._runtime is None
            or self._track_pos is None
            or self._track_vel is None
            or self._track_conf is None
        ):
            raise RuntimeError("temporal state is not initialized")
        self._runtime[index] = _Runtime(
            deque(maxlen=self.stable_frames), deque(maxlen=self.stable_frames), deque(maxlen=8)
        )
        self._track_pos[index].zero_()
        self._track_vel[index].zero_()
        self._track_conf[index].zero_()
        self._reset_generation += 1
        self._last_reset_reason = reason

    @staticmethod
    def _observations(heatmaps: Tensor) -> tuple[Tensor, Tensor]:
        flat = heatmaps.flatten(2)
        probs = flat / flat.sum(-1, keepdim=True).clamp_min(1e-6)
        axis = torch.linspace(0.0, 1.0, 8, device=heatmaps.device)
        ys, xs = torch.meshgrid(axis, axis, indexing="ij")
        xy = torch.stack([(probs * xs.flatten()).sum(-1), (probs * ys.flatten()).sum(-1)], dim=-1)
        return xy, heatmaps.mean(dim=(2, 3))

    def _tracking_update(self, observed: Tensor, visibility: Tensor, dt_ms: Tensor) -> None:
        if self._track_pos is None or self._track_vel is None or self._track_conf is None:
            raise RuntimeError("tracking state is not initialized")
        dt = dt_ms.to(self._track_pos.dtype).clamp_min(1.0)[:, None, None] / 1000.0
        predicted = self._track_pos + self._track_vel * dt
        gain = (0.65 * visibility).clamp(0.0, 0.9).unsqueeze(-1)
        innovation = observed - predicted
        self._track_pos = predicted + gain * innovation
        self._track_vel = self._track_vel + 0.35 * gain * innovation / dt
        self._track_conf = (visibility * (1.0 - 0.2 * innovation.abs().mean(-1))).clamp(0.0, 1.0)

    def _timestamps(
        self, value: Tensor | int | float | None, frames: int, device: torch.device
    ) -> Tensor:
        if value is None:
            base = 0
            if self._runtime and self._runtime[0].last_timestamp_ms is not None:
                base = self._runtime[0].last_timestamp_ms + self.frame_interval_ms
            return (
                torch.arange(frames, device=device, dtype=torch.long) * self.frame_interval_ms
                + base
            )
        if isinstance(value, Tensor):
            if value.ndim == 1 and value.numel() == frames:
                return value.to(device=device, dtype=torch.long)
            if value.ndim == 0:
                value = int(value.item())
            else:
                raise ValueError("timestamps_ms must be scalar or length-T")
        start = int(value)
        return (
            torch.arange(frames, device=device, dtype=torch.long) * self.frame_interval_ms + start
        )

    def _window(self, device: torch.device) -> Tensor:
        if self._runtime is None:
            raise RuntimeError("temporal state is not initialized")
        rows: list[Tensor] = []
        for state in self._runtime:
            history = list(state.features)
            padding = [torch.zeros(64, device=device) for _ in range(8 - len(history))]
            rows.append(torch.stack([*padding, *history]))
        return torch.stack(rows)

    def forward(
        self, rgb: Tensor, timestamps_ms: Tensor | int | float | None = None
    ) -> dict[str, Any]:
        if rgb.ndim == 4:
            rgb = rgb[:, None]
        if rgb.ndim != 5 or rgb.shape[2] != 3:
            raise ValueError("rgb must be [B,3,H,W] or [B,T,3,H,W]")
        if not rgb.is_floating_point():
            rgb = rgb.float()
        batch, frames = rgb.shape[:2]
        if self._runtime is None or len(self._runtime) != batch:
            self._ensure_runtime(batch, rgb.device)
        assert self._runtime is not None
        timestamps = self._timestamps(timestamps_ms, frames, rgb.device)
        logits_rows: list[Tensor] = []
        ood_rows: list[Tensor] = []
        quality_rows: list[Tensor] = []
        dt_rows: list[Tensor] = []
        advisory = [ABSTAIN] * batch
        reasons = ["STABILITY"] * batch
        last_heatmaps = torch.empty(batch, 2, 8, 8, device=rgb.device)
        last_visibility = torch.empty(batch, 2, device=rgb.device)
        last_hud = torch.empty(batch, 4, device=rgb.device)
        reset_count = 0

        for frame_index in range(frames):
            timestamp = int(timestamps[frame_index].item())
            current = rgb[:, frame_index]
            boundary: list[str | None] = [None] * batch
            deltas = torch.full((batch,), float(self.frame_interval_ms), device=rgb.device)
            for sample, state in enumerate(self._runtime):
                previous = state.last_timestamp_ms
                if previous is not None:
                    delta = timestamp - previous
                    if delta < 0:
                        boundary[sample] = "PTS_BACKWARD"
                    elif delta > self.gap_reset_ms:
                        boundary[sample] = "PTS_GAP"
                    else:
                        deltas[sample] = float(delta)
                previous_frame = (
                    rgb[sample, frame_index - 1]
                    if frame_index
                    else (self._last_frame[sample] if self._last_frame is not None else None)
                )
                if (
                    previous_frame is not None
                    and (current[sample] - previous_frame).abs().mean() > 0.55
                ):
                    boundary[sample] = "SCENE_CUT"
                if boundary[sample] is not None:
                    self._reset_sample(sample, boundary[sample] or "SEGMENT")
                    reset_count += 1

            feature, heatmaps, hud = self.visual_encoder(current)
            observed, visibility = self._observations(heatmaps)
            with torch.no_grad():
                self._tracking_update(observed, visibility, deltas)
            assert (
                self._track_pos is not None
                and self._track_vel is not None
                and self._track_conf is not None
            )
            track = torch.cat(
                (self._track_pos.flatten(1), self._track_vel.flatten(1), self._track_conf), dim=1
            )
            encoded = self.visual_projection(feature) + self.track_projection(track)
            for sample, state in enumerate(self._runtime):
                state.features.append(encoded[sample])
            temporal = self.tcn(self._window(rgb.device))[:, -1]
            logits = self.logit_head(temporal)
            ood = torch.sigmoid(self.ood_head(temporal).squeeze(-1))
            quality = self._track_conf.mean(1)
            classes = logits.argmax(1)
            confidence = torch.softmax(logits, 1).max(1).values

            for sample, state in enumerate(self._runtime):
                cls = int(classes[sample].item())
                state.classes.append(cls)
                state.timestamps.append(timestamp)
                reason = boundary[sample]
                stable = (
                    len(state.classes) >= self.stable_frames
                    and sum(value == cls for value in state.classes) >= self.stable_frames - 1
                    and max(state.timestamps) - min(state.timestamps) >= 300
                )
                if reason is None and confidence[sample] < self.class_threshold:
                    reason = "LOW_SCORE"
                elif reason is None and quality[sample] < self.quality_threshold:
                    reason = "LOW_QUALITY"
                elif reason is None and not stable:
                    reason = "STABILITY"
                elif reason is None and timestamp < state.cooldown_until_ms:
                    reason = "COOLDOWN"
                if reason is None:
                    advisory[sample] = ACTION_NAMES[cls]
                    reasons[sample] = ""
                    state.cooldown_until_ms = timestamp + self.cooldown_ms
                else:
                    advisory[sample] = ABSTAIN
                    reasons[sample] = reason
                state.last_timestamp_ms = timestamp
            self._last_frame = current.detach()
            logits_rows.append(logits)
            ood_rows.append(ood)
            quality_rows.append(quality)
            dt_rows.append(deltas)
            last_heatmaps, last_visibility, last_hud = heatmaps, visibility, hud

        frame_logits = torch.stack(logits_rows, 1)[:, -8:]
        frame_ood = torch.stack(ood_rows, 1)[:, -8:]
        frame_quality = torch.stack(quality_rows, 1)[:, -8:]
        frame_dt = torch.stack(dt_rows, 1)[:, -8:]
        assert self._track_pos is not None
        assert self._track_vel is not None
        assert self._track_conf is not None
        self._track_pos = self._track_pos.detach()
        self._track_vel = self._track_vel.detach()
        self._track_conf = self._track_conf.detach()
        return {
            "logits": frame_logits[:, -1],
            "ood": frame_ood[:, -1],
            "tracking_quality": frame_quality[:, -1],
            "hero_heatmaps": last_heatmaps,
            "hero_visibility": last_visibility,
            "hud": last_hud,
            "frame_logits": frame_logits,
            "frame_ood": frame_ood,
            "frame_tracking_quality": frame_quality,
            "frame_dt_ms": frame_dt,
            "advisory": advisory,
            "abstain_reason": reasons,
            "metrics": {"label_available": False, "label_based_gate_passed": False},
            "reset_generation": self._reset_generation,
            "reset_count": reset_count,
        }


class TemporalCoach:
    """Revalidates an exact on-disk V5 release before every RGB-only call."""

    def __init__(
        self,
        *,
        release_path: Path | None = None,
        expected_model_sha256: str | None = None,
        expected_alignment_sha256: str | None = None,
        expected_audit_sha256: str | None = None,
        expected_config_sha256: str | None = None,
    ) -> None:
        if release_path is not None and not isinstance(release_path, Path):
            raise TypeError("release_path must be pathlib.Path")
        self.model = TemporalModel()
        self.release_path = release_path
        self._expected = (
            expected_model_sha256,
            expected_alignment_sha256,
            expected_audit_sha256,
            expected_config_sha256,
        )
        self._active_release: tuple[str, str] | None = None

    def _release(self) -> tuple[_ReleaseBinding | None, str]:
        binding, reason = _load_bound_release(self.release_path, *self._expected)
        identity = None if binding is None else (binding.release_sha256, binding.release_sha256)
        if identity != self._active_release:
            self.model.reset("release_or_alignment_change")
            self._active_release = identity
        return binding, reason

    def __call__(
        self, rgb: Tensor, timestamps_ms: Tensor | int | float | None = None
    ) -> dict[str, Any]:
        binding, release_reason = self._release()
        output: dict[str, Any] = self.model(rgb, timestamps_ms)
        if binding is None:
            output["advisory"] = [ABSTAIN] * len(output["advisory"])
            output["abstain_reason"] = [release_reason] * len(output["advisory"])
            self.model.reset(release_reason)
            output["reset_generation"] = self.model._reset_generation
        else:
            output["advisory"] = [ABSTAIN] * len(output["advisory"])
            output["abstain_reason"] = ["V6_AUDIT_NOT_BOUND"] * len(output["advisory"])
        output["metrics"].update(
            {
                "release_binding_passed": binding is not None,
                "release_sha256": None if binding is None else binding.release_sha256,
            }
        )
        return output


def cpu_smoke() -> dict[str, object]:
    output = TemporalCoach()(torch.zeros((1, 8, 3, 64, 64)))
    return {
        "status": "PASSED",
        "disposition": "NON_PROMOTING_FAIL_CLOSED_SMOKE",
        "advisory": output["advisory"],
        "release_binding_passed": output["metrics"]["release_binding_passed"],
        "expected_advisory": ABSTAIN,
    }
