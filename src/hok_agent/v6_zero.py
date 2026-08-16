from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

ABSTAIN = "ABSTAIN"
ACTION_NAMES: tuple[str, ...] = (
    "wait",
    "forward",
    "backward",
    "attack_hero",
    "attack_tower",
    "attack_crystal",
)
SEQ_LEN = 8
ARCHITECTURE = "rgb-only-causal-depthwise-tcn-v1"
DEFAULT_QUALITY_THRESHOLD = 0.85


class V6ZeroError(ValueError):
    pass


class _Runtime:
    def __init__(self, device: torch.device) -> None:
        self.features: deque[Tensor] = deque(maxlen=SEQ_LEN)
        self.timestamps: deque[int] = deque(maxlen=SEQ_LEN)
        self.track_pos: Tensor = torch.zeros(2, 2, device=device)
        self.track_vel: Tensor = torch.zeros(2, 2, device=device)
        self.track_conf: Tensor = torch.zeros(2, device=device)
        self.last_timestamp_ms: int | None = None


class _CausalDepthwiseTCN(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layer1 = nn.Conv1d(channels, channels, 3, groups=channels)
        self.layer2 = nn.Conv1d(channels, channels, 3, dilation=2, groups=channels)

    def forward(self, sequence: Tensor) -> Tensor:
        hidden = self.layer1(F.pad(sequence, (2, 0)))
        hidden = torch.relu(hidden)
        hidden = self.layer2(F.pad(hidden, (4, 0)))
        return torch.relu(hidden)


class _RGBEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.SiLU(),
        )
        self.heatmap_head = nn.Conv2d(32, 2, 1)
        self.hud_head = nn.Sequential(nn.Linear(96, 24), nn.SiLU(), nn.Linear(24, 4), nn.Sigmoid())

    def forward(self, rgb: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        stem = self.stem[:3](rgb)
        heatmaps = torch.sigmoid(self.heatmap_head(stem))
        body = self.stem[3:](stem)
        feature = torch.flatten(F.adaptive_avg_pool2d(body, 1), 1)
        return (
            feature,
            torch.nn.functional.adaptive_avg_pool2d(heatmaps, (8, 8)),
            self.hud_head(feature),
        )


class TemporalZeroModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seq_len = SEQ_LEN
        self.quality_threshold = DEFAULT_QUALITY_THRESHOLD
        self.encoder = _RGBEncoder()
        self.visual_projection = nn.Linear(96, 64)
        self.track_projection = nn.Linear(10, 64)
        self.tcn = _CausalDepthwiseTCN(64)
        self.logit_head = nn.Linear(64, len(ACTION_NAMES))
        self.ood_head = nn.Linear(64, 1)
        self._runtime: list[_Runtime] | None = None
        self._reset_generation = 0

    @staticmethod
    def _observations(heatmaps: Tensor) -> tuple[Tensor, Tensor]:
        flat = heatmaps.flatten(2)
        probs = flat / flat.sum(-1, keepdim=True).clamp_min(1e-6)
        axis = torch.linspace(0.0, 1.0, heatmaps.shape[-1], device=heatmaps.device)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        xy = torch.stack([(probs * x.flatten()).sum(-1), (probs * y.flatten()).sum(-1)], dim=-1)
        return xy, probs.max(-1).values

    def _ensure_runtime(self, batch: int, device: torch.device) -> None:
        self._runtime = [_Runtime(device) for _ in range(batch)]

    def reset_runtime(self) -> None:
        self._runtime = None
        self._reset_generation += 1

    def _reset_sample(self, index: int) -> None:
        if self._runtime is None:
            raise RuntimeError("runtime uninitialized")
        device = self._runtime[index].track_pos.device
        self._runtime[index] = _Runtime(device)
        self._reset_generation += 1

    @staticmethod
    def _timestamps(value: Tensor | Sequence[int] | int | None, device: torch.device) -> Tensor:
        if value is None:
            return torch.arange(SEQ_LEN, device=device, dtype=torch.long) * 100
        if isinstance(value, Tensor):
            if value.ndim != 1 or value.shape[0] != SEQ_LEN:
                raise ValueError("timestamps_ms must be length-8 tensor or sequence")
            return value.to(device=device, dtype=torch.long)
        if isinstance(value, Sequence):
            if len(value) != SEQ_LEN:
                raise ValueError("timestamps_ms must contain eight entries")
            return torch.as_tensor(list(int(x) for x in value), device=device, dtype=torch.long)
        return torch.full((SEQ_LEN,), int(value), device=device, dtype=torch.long)

    def forward(
        self, rgb: Tensor, timestamps_ms: Tensor | Sequence[int] | int | None = None
    ) -> dict[str, Any]:
        if not torch.is_floating_point(rgb):
            raise ValueError("rgb must be floating-point")
        if rgb.ndim != 5 or rgb.shape[1] != SEQ_LEN or rgb.shape[2] != 3:
            raise ValueError("rgb must be [B,8,3,H,W]")
        batch, frames = rgb.shape[:2]
        device = rgb.device
        if frames != SEQ_LEN:
            raise ValueError("sequence length must be exactly 8")
        if self._runtime is None or len(self._runtime) != batch:
            self._ensure_runtime(batch, device)
        assert self._runtime is not None
        timestamps = self._timestamps(timestamps_ms, device)
        if timestamps.shape != (frames,):
            raise ValueError("timestamp shape mismatch")
        if (timestamps < 0).any():
            raise ValueError("timestamps cannot be negative")

        frame_logits: list[Tensor] = []
        frame_ood: list[Tensor] = []
        frame_quality: list[Tensor] = []
        frame_dt: list[Tensor] = []
        reset_count = 0

        for frame_idx in range(frames):
            current = rgb[:, frame_idx]
            timestamp = int(timestamps[frame_idx].item())
            deltas = []
            for sample, state in enumerate(self._runtime):
                prev = state.last_timestamp_ms
                if prev is None:
                    deltas.append(100.0)
                    continue
                delta = timestamp - prev
                if delta <= 0 or delta > 500:
                    self._reset_sample(sample)
                    deltas.append(100.0)
                    reset_count += 1
                else:
                    deltas.append(float(delta))

            feature, heatmaps, hud = self.encoder(current)
            centers, visible = self._observations(heatmaps)
            dt = torch.tensor(deltas, device=device, dtype=torch.float32).clamp_min(1.0)
            dt = dt[:, None, None] / 1000.0
            for sample, state in enumerate(self._runtime):
                predicted = state.track_pos + state.track_vel * dt[sample]
                gain = (0.65 * visible[sample]).clamp(0.0, 0.9).unsqueeze(-1)
                innovation = centers[sample] - predicted
                state.track_pos = predicted + gain * innovation
                state.track_vel = state.track_vel + 0.35 * gain * innovation / dt[sample]
                state.track_conf = (
                    visible[sample] * (1.0 - 0.2 * innovation.abs().mean(-1))
                ).clamp(0.0, 1.0)
                state.timestamps.append(timestamp)

            assert all(state.track_pos is not None for state in self._runtime)
            tracks = torch.stack(
                [
                    torch.cat((s.track_pos.flatten(), s.track_vel.flatten(), s.track_conf), dim=-1)
                    for s in self._runtime
                ],
                0,
            )
            encoded = self.visual_projection(feature) + self.track_projection(tracks)
            for sample, state in enumerate(self._runtime):
                state.features.append(encoded[sample])

            padded = []
            for state in self._runtime:
                missing = SEQ_LEN - len(state.features)
                if missing > 0:
                    pad = [torch.zeros(64, device=device) for _ in range(missing)]
                    window = torch.stack([*pad, *state.features])
                else:
                    window = torch.stack(list(state.features))
                padded.append(window)
            tcn_in = torch.stack(padded, 0).transpose(1, 2)
            tcn_out = self.tcn(tcn_in).transpose(1, 2)
            current_t = tcn_out[:, -1]
            logits = self.logit_head(current_t)
            ood = torch.sigmoid(self.ood_head(current_t).squeeze(-1))
            quality = torch.stack([state.track_conf.mean() for state in self._runtime], 0)
            frame_logits.append(logits)
            frame_ood.append(ood)
            frame_quality.append(quality)
            frame_dt.append(torch.tensor(deltas, device=device))
            for _sample, state in enumerate(self._runtime):
                state.last_timestamp_ms = timestamp

        frame_logits_t = torch.stack(frame_logits, 1)
        frame_ood_t = torch.stack(frame_ood, 1)
        frame_quality_t = torch.stack(frame_quality, 1)
        frame_dt_t = torch.stack(frame_dt, 1)
        final_logits = frame_logits_t[:, -1]
        final_ood = frame_ood_t[:, -1]
        final_quality = frame_quality_t[:, -1]
        probs = torch.softmax(final_logits, -1)
        confidence = probs.max(-1)
        class_ids = probs.argmax(-1)
        stability = frame_quality_t.mean(1) > self.quality_threshold
        return {
            "frame_logits": frame_logits_t,
            "frame_ood": frame_ood_t,
            "frame_tracking_quality": frame_quality_t,
            "frame_dt_ms": frame_dt_t,
            "raw_hypothesis_id": class_ids,
            "raw_confidence": confidence.values,
            "raw_hypothesis": [ACTION_NAMES[int(ix)] for ix in class_ids.tolist()],
            "tracking_quality": final_quality,
            "tracking_stability": stability,
            "ood": final_ood,
            "advisory_action": [ABSTAIN] * batch,
            "control_output": False,
            "reset_count": reset_count,
            "possible_timestamps": bool((timestamps[1:] > timestamps[:-1]).all()),
        }


class TemporalZeroCoach:
    """Public V6 route: fail closed until a V5-derived training bundle exists."""

    def __init__(self) -> None:
        self._reset_generation = 0

    def _failure(self, batch: int, reason: str) -> dict[str, Any]:
        self._reset_generation += 1
        return {
            "advisory_action": [ABSTAIN] * batch,
            "control_output": False,
            "metrics": {"v6_zero_release_binding_passed": False, "reason": reason},
            "reset_generation": self._reset_generation,
        }

    def __call__(
        self, rgb: Tensor, timestamps_ms: Tensor | Sequence[int] | int | None = None
    ) -> dict[str, Any]:
        del timestamps_ms
        batch = int(rgb.shape[0]) if isinstance(rgb, Tensor) and rgb.ndim else 0
        return self._failure(batch, "V5_ZERO_LABEL_BASE_NOT_AVAILABLE")


def cpu_smoke() -> dict[str, object]:
    sample = torch.zeros((1, SEQ_LEN, 3, 64, 64), dtype=torch.float32)
    output = TemporalZeroCoach()(sample)
    return {
        "status": "NON_PROMOTING_FRAMEWORK_SMOKE",
        "advisory": output["advisory_action"],
        "release_binding_passed": False,
    }
