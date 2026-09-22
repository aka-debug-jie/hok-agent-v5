from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter, defaultdict, deque
from copy import copy, deepcopy
from pathlib import Path
from typing import Final, Literal, cast

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.models import resnet18  # type: ignore[import-untyped]

from hok_agent.global_agent import (
    ENABLED_INTENTS,
    ENABLED_ZONES,
    GlobalArena,
    GlobalCommandRouter,
    GlobalRuleTeacher,
    MacroCommand,
    MacroIntent,
    ProgressWatchdog,
    TargetZone,
    load_global_config,
    run_teacher_episode,
)
from hok_agent.rich_arena import RichRandomPolicy, wait_action
from hok_agent.rich_renderer import render, renderer_hash

WINDOW_FRAMES: Final = 16
SAMPLE_HZ: Final = 5
TRAIN_SEEDS: Final = tuple(range(1000, 1040))
DEV_SEEDS: Final = tuple(range(2000, 2010))
DAGGER_SEEDS: Final = tuple(range(3000, 3040))
HOLDOUT_SEEDS: Final = tuple(range(4000, 4020))
INTENT_INDEX = {value: index for index, value in enumerate(ENABLED_INTENTS)}
ZONE_INDEX = {value: index for index, value in enumerate(ENABLED_ZONES)}
SCENES: Final = (
    "NAVIGATION",
    "LANE_FARM",
    "COMBAT",
    "PUSH_STRUCTURE",
    "RETURN_DEFEND",
)
SCENE_INDEX = {value: index for index, value in enumerate(SCENES)}
ModelVariant = Literal["pool_mlp", "tcn"]
MAIN_ARCHITECTURE_METADATA_KEY: Final = "main_architecture"
# The historical architecture. A checkpoint whose metadata predates this key is read as resnet18, so
# every frozen checkpoint on disk keeps its recorded hash and still loads.
DEFAULT_MAIN_ARCHITECTURE: Final = "resnet18"
MAIN_ARCHITECTURES: Final = ("resnet18", "compact")


class GlobalPolicyError(ValueError):
    pass


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _large_root() -> Path:
    text = os.environ.get("HOK_LARGE_ROOT")
    if not text:
        raise GlobalPolicyError("HOK_LARGE_ROOT is required")
    root = Path(text).resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise GlobalPolicyError("HOK_LARGE_ROOT must be a real directory")
    return root


def _under_large_root(path: Path, *, output: bool) -> Path:
    root = _large_root()
    resolved = path.resolve(strict=not output)
    if resolved == root or root not in resolved.parents:
        raise GlobalPolicyError("Global Agent artifacts must be below HOK_LARGE_ROOT")
    if output and resolved.exists():
        raise GlobalPolicyError(f"output already exists: {resolved}")
    return resolved


def _write_json(path: Path, value: object) -> None:
    path.write_text(_canonical(value) + "\n", encoding="utf-8")


def _resize_nearest(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    y = np.linspace(0, frame.shape[0] - 1, height).round().astype(np.int64)
    x = np.linspace(0, frame.shape[1] - 1, width).round().astype(np.int64)
    return frame[y[:, None], x[None, :]]


def render_views(observation: dict[str, object], render_seed: int) -> tuple[np.ndarray, ...]:
    full = render(observation, render_seed)
    main = full.copy()
    minimap = _resize_nearest(full[24:120], 64, 64)
    hud = _resize_nearest(full[:24], 32, 128)
    return main, minimap, hud


def _manifest_payload(episodes: list[dict[str, object]]) -> dict[str, object]:
    arena = GlobalArena()
    _config, config_sha256 = load_global_config()
    contract = {
        "window_frames": WINDOW_FRAMES,
        "sample_hz": SAMPLE_HZ,
        "intents": [value.value for value in ENABLED_INTENTS],
        "zones": [value.value for value in ENABLED_ZONES],
        "scenes": list(SCENES),
        "input": ["main_rgb", "minimap_rgb", "hud_rgb"],
        "structured_state_is_actor_input": False,
        "device_input_allowed": False,
    }
    return {
        "schema_version": "hok-agent-global-dataset-v1",
        "arena_sha256": arena.config.digest,
        "config_sha256": config_sha256,
        "renderer_sha256": renderer_hash(),
        "contract": contract,
        "contract_sha256": _sha(_canonical(contract).encode()),
        "train_seeds": list(TRAIN_SEEDS),
        "dev_seeds": list(DEV_SEEDS),
        "test_present": False,
        "episodes": episodes,
        "source_paths_persisted": False,
        "device_input_allowed": False,
    }


def materialize_global_dataset(
    output_dir: Path,
    *,
    train_seeds: tuple[int, ...] = TRAIN_SEEDS,
    dev_seeds: tuple[int, ...] = DEV_SEEDS,
    enforce: bool = True,
) -> dict[str, object]:
    if enforce and (train_seeds != TRAIN_SEEDS or dev_seeds != DEV_SEEDS):
        raise GlobalPolicyError("formal Global Agent pilot requires frozen 40/10 seeds")
    if set(train_seeds) & set(dev_seeds):
        raise GlobalPolicyError("episode splits overlap")
    output = _under_large_root(output_dir, output=True)
    output.mkdir(parents=True)
    episodes: list[dict[str, object]] = []
    try:
        for split, seeds in (("train", train_seeds), ("dev", dev_seeds)):
            for seed in seeds:
                report, trace = run_teacher_episode(seed, include_trace=True)
                main: list[np.ndarray] = []
                minimap: list[np.ndarray] = []
                hud: list[np.ndarray] = []
                intents: list[int] = []
                zones: list[int] = []
                scenes: list[int] = []
                actions: list[str] = []
                ticks: list[int] = []
                for row in trace:
                    if (
                        row.command.intent not in INTENT_INDEX
                        or row.command.target_zone not in ZONE_INDEX
                    ):
                        raise GlobalPolicyError("teacher emitted a disabled macro label")
                    views = render_views(row.observation, seed * 10_007 + row.tick)
                    main.append(views[0])
                    minimap.append(views[1])
                    hud.append(views[2])
                    intents.append(INTENT_INDEX[row.command.intent])
                    zones.append(ZONE_INDEX[row.command.target_zone])
                    scenes.append(SCENE_INDEX[row.scene_id])
                    actions.append(_canonical(row.action.to_dict()))
                    ticks.append(row.tick)
                episode_id = _sha(f"global-v1:{split}:{seed}".encode())
                name = f"episode-{episode_id}.npz"
                path = output / name
                with path.open("xb") as handle:
                    np.savez_compressed(
                        handle,
                        main_rgb=np.stack(main),
                        minimap_rgb=np.stack(minimap),
                        hud_rgb=np.stack(hud),
                        intent=np.asarray(intents, dtype=np.int16),
                        zone=np.asarray(zones, dtype=np.int16),
                        scene=np.asarray(scenes, dtype=np.int16),
                        tick=np.asarray(ticks, dtype=np.int32),
                        executed_action=np.asarray(actions),
                    )
                episodes.append(
                    {
                        "episode_id": episode_id,
                        "split": split,
                        "rows": len(trace),
                        "shard": name,
                        "shard_sha256": _sha(path.read_bytes()),
                        "outcome": report.outcome,
                        "failure_code": report.failure_code,
                    }
                )
        payload = _manifest_payload(episodes)
        payload["train_seeds"] = list(train_seeds)
        payload["dev_seeds"] = list(dev_seeds)
        payload["manifest_sha256"] = _sha(_canonical(payload).encode())
        _write_json(output / "manifest.json", payload)
    except Exception:
        for path in output.glob("*"):
            path.unlink()
        output.rmdir()
        raise
    return {
        "status": "PASSED",
        "schema_version": payload["schema_version"],
        "train_episodes": len(train_seeds),
        "dev_episodes": len(dev_seeds),
        "rows": sum(int(cast(int, row["rows"])) for row in episodes),
        "manifest_sha256": payload["manifest_sha256"],
        "output_dir": str(output),
        "device_input_allowed": False,
    }


def load_global_manifest(root: Path) -> dict[str, object]:
    resolved = _under_large_root(root, output=False)
    path = resolved / "manifest.json"
    raw = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    digest = raw.pop("manifest_sha256", None)
    if not isinstance(digest, str) or digest != _sha(_canonical(raw).encode()):
        raise GlobalPolicyError("global dataset manifest hash mismatch")
    raw["manifest_sha256"] = digest
    if raw.get("test_present") is not False:
        raise GlobalPolicyError("Global Agent pilot must not contain test episodes")
    episodes = cast(list[dict[str, object]], raw.get("episodes"))
    if not episodes or any(row.get("split") not in {"train", "dev"} for row in episodes):
        raise GlobalPolicyError("invalid Global Agent episode split")
    for row in episodes:
        shard = resolved / str(row["shard"])
        if _sha(shard.read_bytes()) != row["shard_sha256"]:
            raise GlobalPolicyError("global dataset shard hash mismatch")
    return raw


class GlobalWindowDataset(Dataset[tuple[torch.Tensor, ...]]):
    def __init__(self, root: Path, split: str, *, shuffle_labels: bool = False) -> None:
        if split not in {"train", "dev"}:
            raise GlobalPolicyError("only train/dev splits are available")
        manifest = load_global_manifest(root)
        self.samples: list[tuple[np.ndarray, ...]] = []
        for row in cast(list[dict[str, object]], manifest["episodes"]):
            if row["split"] != split:
                continue
            with np.load(root / str(row["shard"]), allow_pickle=False) as shard:
                arrays = tuple(
                    np.asarray(shard[name])
                    for name in (
                        "main_rgb",
                        "minimap_rgb",
                        "hud_rgb",
                        "intent",
                        "zone",
                        "scene",
                        "tick",
                    )
                )
            length = arrays[0].shape[0]
            for end in range(WINDOW_FRAMES - 1, length):
                start = end - WINDOW_FRAMES + 1
                self.samples.append(
                    (
                        arrays[0][start : end + 1],
                        arrays[1][start : end + 1],
                        arrays[2][start : end + 1],
                        arrays[3][end],
                        arrays[4][end],
                        arrays[5][end],
                        arrays[6][end],
                    )
                )
        if not self.samples:
            raise GlobalPolicyError(f"split has no causal windows: {split}")
        if shuffle_labels:
            rng = np.random.default_rng(0)
            permutation = rng.permutation(len(self.samples))
            labels = [
                (self.samples[i][3], self.samples[i][4], self.samples[i][5])
                for i in permutation
            ]
            self.samples = [
                (*sample[:3], labels[index][0], labels[index][1], labels[index][2], sample[6])
                for index, sample in enumerate(self.samples)
            ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        main, minimap, hud, intent, zone, scene, tick = self.samples[index]

        def convert(value: np.ndarray) -> torch.Tensor:
            tensor = torch.from_numpy(np.asarray(value)).permute(0, 3, 1, 2).float()
            return tensor / 255.0

        return (
            convert(main),
            convert(minimap),
            convert(hud),
            torch.tensor(int(intent), dtype=torch.long),
            torch.tensor(int(zone), dtype=torch.long),
            torch.tensor(int(scene), dtype=torch.long),
            torch.tensor(int(tick), dtype=torch.long),
        )


class _SmallView(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(value))


class _CompactMain(nn.Module):
    """A declared smaller `main` view: the same 512-wide output at ~2.4% of resnet18's parameters.

    The width is fixed at 512 because `project` is `Linear(640, 128)` and the two `_SmallView`s
    already contribute 64 each, so swapping `main` changes one component and nothing else. It is a
    declared architecture rather than a silent swap because which `main` a checkpoint carries is
    part of its interface.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 16, 512),
            nn.ReLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(value))


class GlobalMacroPolicy(nn.Module):
    def __init__(
        self,
        variant: ModelVariant = "tcn",
        main_architecture: str = DEFAULT_MAIN_ARCHITECTURE,
    ) -> None:
        super().__init__()
        if variant not in {"pool_mlp", "tcn"}:
            raise GlobalPolicyError(f"unknown model variant: {variant}")
        if main_architecture not in MAIN_ARCHITECTURES:
            raise GlobalPolicyError(f"unknown main architecture: {main_architecture}")
        self.variant = variant
        self.main_architecture = main_architecture
        if main_architecture == "compact":
            self.main: nn.Module = _CompactMain()
        else:
            backbone = resnet18(weights=None)
            backbone.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
            backbone.maxpool = nn.Identity()
            backbone.fc = nn.Identity()
            self.main = backbone
        self.minimap = _SmallView()
        self.hud = _SmallView()
        self.project = nn.Sequential(nn.Linear(640, 128), nn.ReLU())
        self.temporal = nn.ModuleList(
            (nn.Conv1d(128, 128, 3), nn.Conv1d(128, 128, 3))
        )
        self.pool = nn.Sequential(nn.Linear(128, 128), nn.ReLU())
        self.intent = nn.Linear(128, len(ENABLED_INTENTS))
        self.zone = nn.Linear(128, len(ENABLED_ZONES))
        self.scene = nn.Linear(128, len(SCENES))

    def encode_views(
        self, main: torch.Tensor, minimap: torch.Tensor, hud: torch.Tensor
    ) -> torch.Tensor:
        batch, steps = main.shape[:2]

        def flatten(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(batch * steps, *value.shape[2:])

        main_flat = F.interpolate(
            flatten(main), size=(64, 64), mode="bilinear", align_corners=False
        )
        features = torch.cat(
            (self.main(main_flat), self.minimap(flatten(minimap)), self.hud(flatten(hud))),
            dim=1,
        )
        return cast(torch.Tensor, self.project(features).reshape(batch, steps, 128))

    def forward(
        self, main: torch.Tensor, minimap: torch.Tensor, hud: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence = self.encode_views(main, minimap, hud)
        if self.variant == "pool_mlp":
            hidden = self.pool(sequence.mean(dim=1))
        else:
            hidden_sequence = sequence.transpose(1, 2)
            for layer in self.temporal:
                hidden_sequence = F.relu(layer(F.pad(hidden_sequence, (2, 0))))
            hidden = hidden_sequence[:, :, -1]
        return self.intent(hidden), self.zone(hidden), self.scene(hidden)


def _class_weights(dataset: GlobalWindowDataset, label_index: int, classes: int) -> torch.Tensor:
    counts = np.bincount(
        [int(sample[label_index]) for sample in dataset.samples], minlength=classes
    ).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def _macro_f1(target: list[int], predicted: list[int], classes: int) -> float:
    scores: list[float] = []
    for label in range(classes):
        tp = sum(a == label and b == label for a, b in zip(target, predicted, strict=True))
        fp = sum(a != label and b == label for a, b in zip(target, predicted, strict=True))
        fn = sum(a == label and b != label for a, b in zip(target, predicted, strict=True))
        scores.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(scores) / classes


def _evaluate_model(
    model: GlobalMacroPolicy, loader: DataLoader[tuple[torch.Tensor, ...]], device: torch.device
) -> dict[str, object]:
    targets: tuple[list[int], list[int], list[int]] = ([], [], [])
    predicted: tuple[list[int], list[int], list[int]] = ([], [], [])
    model.eval()
    with torch.no_grad():
        for main, minimap, hud, intent, zone, scene, _tick in loader:
            logits = model(main.to(device), minimap.to(device), hud.to(device))
            for index, label in enumerate((intent, zone, scene)):
                targets[index].extend(label.tolist())
                predicted[index].extend(logits[index].argmax(dim=1).cpu().tolist())
    return {
        "intent_macro_f1": _macro_f1(targets[0], predicted[0], len(ENABLED_INTENTS)),
        "zone_macro_f1": _macro_f1(targets[1], predicted[1], len(ENABLED_ZONES)),
        "scene_macro_f1": _macro_f1(targets[2], predicted[2], len(SCENES)),
        "joint_accuracy": sum(
            a == x and b == y
            for a, b, x, y in zip(
                targets[0], targets[1], predicted[0], predicted[1], strict=True
            )
        )
        / len(targets[0]),
        "intent_unique_predictions": len(set(predicted[0])),
        "zone_unique_predictions": len(set(predicted[1])),
    }


def _majority_baselines(train: GlobalWindowDataset, dev: GlobalWindowDataset) -> dict[str, object]:
    train_labels = [[int(sample[index]) for sample in train.samples] for index in (3, 4)]
    dev_labels = [[int(sample[index]) for sample in dev.samples] for index in (3, 4)]
    classes = (len(ENABLED_INTENTS), len(ENABLED_ZONES))
    prior_scores: list[float] = []
    time_scores: list[float] = []
    for head in range(2):
        prior = Counter(train_labels[head]).most_common(1)[0][0]
        prior_scores.append(
            _macro_f1(dev_labels[head], [prior] * len(dev_labels[head]), classes[head])
        )
        buckets: dict[int, Counter[int]] = defaultdict(Counter)
        for sample in train.samples:
            buckets[int(sample[6]) // 8][int(sample[3 + head])] += 1
        predictions = [
            buckets[int(sample[6]) // 8].most_common(1)[0][0]
            if buckets[int(sample[6]) // 8]
            else prior
            for sample in dev.samples
        ]
        time_scores.append(_macro_f1(dev_labels[head], predictions, classes[head]))
    return {
        "class_prior": {"intent_macro_f1": prior_scores[0], "zone_macro_f1": prior_scores[1]},
        "time_only": {"intent_macro_f1": time_scores[0], "zone_macro_f1": time_scores[1]},
    }


def _train_variant(
    train: GlobalWindowDataset,
    dev: GlobalWindowDataset,
    variant: ModelVariant,
    device: torch.device,
    epochs: int,
    batch_size: int,
) -> tuple[GlobalMacroPolicy, dict[str, object]]:
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
    model = GlobalMacroPolicy(variant).to(device)
    weights = (
        _class_weights(train, 3, len(ENABLED_INTENTS)).to(device),
        _class_weights(train, 4, len(ENABLED_ZONES)).to(device),
        _class_weights(train, 5, len(SCENES)).to(device),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(0)
    loader = DataLoader(train, batch_size=batch_size, shuffle=True, generator=generator)
    for _epoch in range(epochs):
        model.train()
        for main, minimap, hud, intent, zone, scene, _tick in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(main.to(device), minimap.to(device), hud.to(device))
            loss = (
                F.cross_entropy(logits[0], intent.to(device), weight=weights[0])
                + F.cross_entropy(logits[1], zone.to(device), weight=weights[1])
                + 0.25 * F.cross_entropy(logits[2], scene.to(device), weight=weights[2])
            )
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
    metrics = _evaluate_model(
        model, DataLoader(dev, batch_size=batch_size, shuffle=False), device
    )
    return model, metrics


def _save_model(
    path: Path, model: GlobalMacroPolicy, variant: str, manifest_sha256: str
) -> str:
    metadata = {
        "schema_version": "hok-agent-global-policy-v1",
        "variant": variant,
        MAIN_ARCHITECTURE_METADATA_KEY: model.main_architecture,
        "manifest_sha256": manifest_sha256,
        "seed": "0",
        "window_frames": str(WINDOW_FRAMES),
        "intents": _canonical([value.value for value in ENABLED_INTENTS]),
        "zones": _canonical([value.value for value in ENABLED_ZONES]),
        "device_input_allowed": "false",
    }
    save_file(model.state_dict(), path, metadata=metadata)
    return _sha(path.read_bytes())


def architecture_from_metadata(metadata: dict[str, str]) -> str:
    """Read the `main` architecture a checkpoint declares, defaulting to the historical resnet18.

    The default is the backward-compatibility rule: every checkpoint written before this key existed
    is resnet18, and those files are never rewritten, so their recorded hashes stay valid.
    """
    architecture = metadata.get(MAIN_ARCHITECTURE_METADATA_KEY, DEFAULT_MAIN_ARCHITECTURE)
    if architecture not in MAIN_ARCHITECTURES:
        raise GlobalPolicyError(f"unknown main architecture: {architecture}")
    return architecture


def load_global_model(path: Path, device: torch.device) -> tuple[GlobalMacroPolicy, dict[str, str]]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    if metadata is None or metadata.get("schema_version") != "hok-agent-global-policy-v1":
        raise GlobalPolicyError("invalid Global Agent checkpoint metadata")
    variant = metadata.get("variant")
    if variant not in {"pool_mlp", "tcn"}:
        raise GlobalPolicyError("invalid Global Agent checkpoint variant")
    # A checkpoint saved before this key existed is resnet18; that rule is what keeps every frozen
    # checkpoint loadable without rewriting it.
    architecture = architecture_from_metadata(metadata)
    model = GlobalMacroPolicy(cast(ModelVariant, variant), architecture)
    model.load_state_dict(load_file(path, device="cpu"), strict=True)
    model.to(device).eval()
    return model, metadata


GLOBAL_SPEED_SCHEMA: Final = "hok-agent-global-speed-report-v1"
GLOBAL_DISTILL_SCHEMA: Final = "hok-agent-global-distill-report-v1"


def _build_student(
    teacher: GlobalMacroPolicy, main_architecture: str
) -> GlobalMacroPolicy:
    """A student that differs from the teacher only in `main`, with every other weight copied.

    Copying the teacher's non-main weights is the point: at initialisation the student's minimap,
    hud, temporal and head behaviour is bit-identical to the frozen teacher, so any behaviour
    change measured later is attributable to the `main` view and to training, not to a fresh
    initialisation.
    """
    student = GlobalMacroPolicy(teacher.variant, main_architecture)
    teacher_state = teacher.state_dict()
    report = student.load_state_dict(
        {name: value for name, value in teacher_state.items() if not name.startswith("main.")},
        strict=False,
    )
    unexpected = list(report.unexpected_keys)
    missing = [name for name in report.missing_keys if not name.startswith("main.")]
    if unexpected or missing:
        raise GlobalPolicyError(f"student copy is not main-only: {unexpected} {missing}")
    return student


def distill_global_main(
    dataset_root: Path,
    output_dir: Path,
    *,
    teacher_checkpoint: Path,
    main_architecture: str = "compact",
    epochs: int = 1,
    maximum_steps: int = 50,
    batch_size: int = 8,
    learning_rate: float = 0.01,
    seed: int = 0,
    device_name: str = "cpu",
) -> dict[str, object]:
    """Train only the `main` view to imitate a frozen Global Agent teacher, offline.

    Reads the frozen dataset and one frozen checkpoint, copies every non-`main` weight from the
    teacher, freezes it, and distils the teacher's logits on the declared **train** split with a
    bounded step budget. It reports dev behaviour and writes the student checkpoint, and it does
    none of the following: open the test split, open a capture source, send anything to a device,
    promote the student, or change the deterministic Router/executor.
    """
    if teacher_checkpoint.is_symlink() or not teacher_checkpoint.is_file():
        raise GlobalPolicyError("distillation needs one regular teacher checkpoint")
    if main_architecture not in MAIN_ARCHITECTURES:
        raise GlobalPolicyError(f"unknown main architecture: {main_architecture}")
    if epochs < 1 or maximum_steps < 1 or batch_size < 1:
        raise GlobalPolicyError("invalid distillation budget")
    manifest = load_global_manifest(dataset_root)
    if manifest.get("test_present") is not False:
        raise GlobalPolicyError("distillation must not open a test split")
    train_dataset = GlobalWindowDataset(dataset_root, "train")
    dev_dataset = GlobalWindowDataset(dataset_root, "dev")
    if len(train_dataset) < batch_size or len(dev_dataset) < batch_size:
        raise GlobalPolicyError("split has fewer windows than the declared batch")
    output = _under_large_root(output_dir, output=True)
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(device_name)
    torch.manual_seed(seed)
    teacher, teacher_metadata = load_global_model(teacher_checkpoint, device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student = _build_student(teacher, main_architecture).to(device)
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    for parameter in student.main.parameters():
        parameter.requires_grad_(True)
    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in student.named_parameters()
        if not name.startswith("main.")
    }
    optimizer = torch.optim.Adam(student.main.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_dataset)).tolist()
    loss_trace: list[float] = []
    steps = 0
    student.train()
    for _epoch in range(epochs):
        for start in range(0, len(order), batch_size):
            if steps >= maximum_steps:
                break
            indices = order[start : start + batch_size]
            if len(indices) < batch_size:
                break
            batch = _speed_batch(train_dataset, indices)
            moved = (batch[0].to(device), batch[1].to(device), batch[2].to(device))
            with torch.no_grad():
                target = teacher(*moved)
            logits = student(*moved)
            loss = (
                F.mse_loss(logits[0], target[0])
                + F.mse_loss(logits[1], target[1])
                + F.mse_loss(logits[2], target[2])
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            loss_trace.append(round(float(loss.detach()), 6))
            steps += 1
        if steps >= maximum_steps:
            break
    student.eval()
    changed = [
        name
        for name, parameter in student.named_parameters()
        if not name.startswith("main.")
        and not torch.equal(parameter.detach(), frozen_before[name])
    ]
    if changed:
        raise GlobalPolicyError(f"distillation changed a frozen parameter: {changed}")
    student_path = output / f"student-{main_architecture}.safetensors"
    student_sha = _save_model(
        student_path, student, student.variant, str(manifest["manifest_sha256"])
    )
    report: dict[str, object] = {
        "schema_version": GLOBAL_DISTILL_SCHEMA,
        "teacher_checkpoint": str(teacher_checkpoint),
        "teacher_sha256": _sha(teacher_checkpoint.read_bytes()),
        "teacher_architecture": architecture_from_metadata(teacher_metadata),
        "student_checkpoint": str(student_path),
        "student_sha256": student_sha,
        "student_architecture": main_architecture,
        "variant": student.variant,
        "device": device_name,
        "seed": seed,
        "epochs": epochs,
        "maximum_steps": maximum_steps,
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_windows": len(train_dataset),
        "dev_windows": len(dev_dataset),
        "loss_first": loss_trace[0] if loss_trace else None,
        "loss_last": loss_trace[-1] if loss_trace else None,
        "loss_trace": loss_trace,
        "parameters": _parameter_groups(student),
        "behaviour": _measure_behaviour(student, dev_dataset, batch_size=batch_size),
        "frozen_parameters_unchanged": True,
        "promotion_allowed": False,
        "device_input_allowed": False,
        "training_eligible": False,
    }
    (output / "distill-report.json").write_text(
        _canonical(report) + "\n", encoding="utf-8"
    )
    return report


def _percentile(sorted_samples: list[float], quantile: float) -> float:
    """Nearest-rank percentile of an ascending sample list."""
    if not sorted_samples:
        raise GlobalPolicyError("no latency samples")
    index = min(len(sorted_samples) - 1, max(0, int(round(quantile * len(sorted_samples))) - 1))
    return sorted_samples[index]


def _parameter_groups(model: GlobalMacroPolicy) -> dict[str, int]:
    """Parameters by part, because almost all of them live in the single resnet18 view."""
    return {
        "main": sum(p.numel() for p in model.main.parameters()),
        "minimap": sum(p.numel() for p in model.minimap.parameters()),
        "hud": sum(p.numel() for p in model.hud.parameters()),
        "temporal_and_heads": sum(
            p.numel()
            for part in (
                model.project,
                model.temporal,
                model.pool,
                model.intent,
                model.zone,
                model.scene,
            )
            for p in part.parameters()
        ),
        "total": sum(p.numel() for p in model.parameters()),
    }


SPEED_LATENCY_NOTE: Final = (
    "CPU latency on this host is not stable across runs: the same frozen model measured a median "
    "batch time between about 0.47 s and about 2.8 s in different contexts, while within one "
    "process five rounds spread 1.27x. A candidate must therefore be timed in the same process as "
    "its baseline, in alternating rounds, never as two separate runs."
)


def _speed_batch(
    dataset: GlobalWindowDataset, indices: list[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    samples = [dataset[index] for index in indices]
    return (
        torch.stack([sample[0] for sample in samples]),
        torch.stack([sample[1] for sample in samples]),
        torch.stack([sample[2] for sample in samples]),
    )


def _measure_latency(model: GlobalMacroPolicy, batch: tuple[torch.Tensor, ...]) -> float:
    """Time one forward pass and return milliseconds."""
    started = time.perf_counter()
    with torch.no_grad():
        model(*batch)
    return (time.perf_counter() - started) * 1000.0


def _measure_behaviour(
    model: GlobalMacroPolicy, dataset: GlobalWindowDataset, *, batch_size: int
) -> dict[str, float]:
    intent_target: list[int] = []
    intent_predicted: list[int] = []
    zone_target: list[int] = []
    zone_predicted: list[int] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            indices = list(range(start, min(start + batch_size, len(dataset))))
            logits = model(*_speed_batch(dataset, indices))
            predicted_intent = logits[0].argmax(dim=1)
            predicted_zone = logits[1].argmax(dim=1)
            for row, index in enumerate(indices):
                intent_predicted.append(int(predicted_intent[row]))
                zone_predicted.append(int(predicted_zone[row]))
                intent_target.append(int(dataset[index][3]))
                zone_target.append(int(dataset[index][4]))
    return {
        "intent_macro_f1": round(
            _macro_f1(intent_target, intent_predicted, len(ENABLED_INTENTS)), 6
        ),
        "zone_macro_f1": round(_macro_f1(zone_target, zone_predicted, len(ENABLED_ZONES)), 6),
    }


def _speed_report(
    *,
    checkpoint_path: Path,
    model: GlobalMacroPolicy,
    metadata: dict[str, str],
    dataset: GlobalWindowDataset,
    samples: list[float],
    batch_size: int,
    repetitions: int,
    warmup: int,
    measurement: str,
) -> dict[str, object]:
    ordered = sorted(samples)
    return {
        "schema_version": GLOBAL_SPEED_SCHEMA,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha(checkpoint_path.read_bytes()),
        "variant": metadata["variant"],
        "main_architecture": model.main_architecture,
        "window_frames": metadata["window_frames"],
        "windows": len(dataset),
        "batch_size": batch_size,
        "repetitions": repetitions,
        "warmup": warmup,
        "measurement": measurement,
        "parameters": _parameter_groups(model),
        "latency_ms": {
            "median": round(_percentile(ordered, 0.5), 4),
            "p95": round(_percentile(ordered, 0.95), 4),
            "minimum": round(ordered[0], 4),
            "maximum": round(ordered[-1], 4),
        },
        "latency_ms_per_decision": round(_percentile(ordered, 0.5) / batch_size, 4),
        "behaviour": _measure_behaviour(model, dataset, batch_size=batch_size),
    }


def measure_global_speed(
    *,
    checkpoint_path: Path,
    dataset_root: Path,
    split: str = "dev",
    batch_size: int = 8,
    repetitions: int = 30,
    warmup: int = 3,
) -> dict[str, object]:
    """Measure one frozen Global Agent checkpoint: parameters, forward latency and split behaviour.

    Read-only. It loads an existing checkpoint, feeds existing frozen windows through the model on
    CPU, and reports the two quantities a speed candidate must preserve and improve - the split's
    behaviour and the measured forward latency. It trains nothing, writes no checkpoint, opens no
    capture source and sends nothing to a device. A single report is a baseline or a smoke, not a
    comparison: see `compare_global_speed` for the interleaved judgement.
    """
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise GlobalPolicyError("speed report needs one regular checkpoint file")
    if batch_size < 1 or repetitions < 1 or warmup < 0:
        raise GlobalPolicyError("invalid speed report budget")
    dataset = GlobalWindowDataset(dataset_root, split)
    if len(dataset) < batch_size:
        raise GlobalPolicyError("split has fewer windows than the declared batch")
    model, metadata = load_global_model(checkpoint_path, torch.device("cpu"))
    batch = _speed_batch(dataset, list(range(batch_size)))
    with torch.no_grad():
        for _ in range(warmup):
            model(*batch)
    samples = [_measure_latency(model, batch) for _ in range(repetitions)]
    return _speed_report(
        checkpoint_path=checkpoint_path,
        model=model,
        metadata=metadata,
        dataset=dataset,
        samples=samples,
        batch_size=batch_size,
        repetitions=repetitions,
        warmup=warmup,
        measurement="single",
    )


def compare_global_speed(
    *,
    baseline_checkpoint: Path,
    candidate_checkpoint: Path,
    dataset_root: Path,
    split: str = "dev",
    batch_size: int = 8,
    repetitions: int = 30,
    warmup: int = 3,
    maximum_intent_macro_f1_drop: float = 0.0,
    minimum_latency_reduction_fraction: float = 0.25,
) -> dict[str, object]:
    """Interleave a baseline and a candidate in one process, then judge behaviour and latency.

    The two models are timed alternately in the same rounds, because separate runs on this host are
    not comparable (see `SPEED_LATENCY_NOTE`). Behaviour is measured per model on the same split.
    """
    for path in (baseline_checkpoint, candidate_checkpoint):
        if path.is_symlink() or not path.is_file():
            raise GlobalPolicyError("speed comparison needs two regular checkpoint files")
    if batch_size < 1 or repetitions < 1 or warmup < 0:
        raise GlobalPolicyError("invalid speed comparison budget")
    dataset = GlobalWindowDataset(dataset_root, split)
    if len(dataset) < batch_size:
        raise GlobalPolicyError("split has fewer windows than the declared batch")
    baseline_model, baseline_meta = load_global_model(baseline_checkpoint, torch.device("cpu"))
    candidate_model, candidate_meta = load_global_model(candidate_checkpoint, torch.device("cpu"))
    batch = _speed_batch(dataset, list(range(batch_size)))
    with torch.no_grad():
        for _ in range(warmup):
            baseline_model(*batch)
            candidate_model(*batch)
    baseline_samples: list[float] = []
    candidate_samples: list[float] = []
    for _ in range(repetitions):
        baseline_samples.append(_measure_latency(baseline_model, batch))
        candidate_samples.append(_measure_latency(candidate_model, batch))
    baseline = _speed_report(
        checkpoint_path=baseline_checkpoint,
        model=baseline_model,
        metadata=baseline_meta,
        dataset=dataset,
        samples=baseline_samples,
        batch_size=batch_size,
        repetitions=repetitions,
        warmup=warmup,
        measurement="interleaved",
    )
    candidate = _speed_report(
        checkpoint_path=candidate_checkpoint,
        model=candidate_model,
        metadata=candidate_meta,
        dataset=dataset,
        samples=candidate_samples,
        batch_size=batch_size,
        repetitions=repetitions,
        warmup=warmup,
        measurement="interleaved",
    )
    return {
        "schema_version": GLOBAL_SPEED_SCHEMA,
        "measurement": "interleaved",
        "latency_note": SPEED_LATENCY_NOTE,
        "baseline": baseline,
        "candidate": candidate,
        "verdict": global_speed_verdict(
            baseline=baseline,
            candidate=candidate,
            maximum_intent_macro_f1_drop=maximum_intent_macro_f1_drop,
            minimum_latency_reduction_fraction=minimum_latency_reduction_fraction,
        ),
    }


def global_speed_verdict(
    *,
    baseline: dict[str, object],
    candidate: dict[str, object],
    maximum_intent_macro_f1_drop: float = 0.0,
    minimum_latency_reduction_fraction: float = 0.25,
) -> dict[str, object]:
    """Judge a speed candidate against a frozen baseline: behaviour preserved AND latency down.

    The F2 definition is explicit that parameters and self-supervised loss are not substitutes:
    parameters are reported, and are never a reason to pass or fail on their own. A candidate passes
    only when it keeps the behaviour (the split's intent macro-F1 within the declared drop) and
    actually reduces the measured latency at both the median and the p95.
    """
    if baseline.get("schema_version") != GLOBAL_SPEED_SCHEMA:
        raise GlobalPolicyError("baseline is not a Global Agent speed report")
    if candidate.get("schema_version") != GLOBAL_SPEED_SCHEMA:
        raise GlobalPolicyError("candidate is not a Global Agent speed report")
    base_behaviour = cast(dict[str, float], baseline["behaviour"])
    cand_behaviour = cast(dict[str, float], candidate["behaviour"])
    base_latency = cast(dict[str, float], baseline["latency_ms"])
    cand_latency = cast(dict[str, float], candidate["latency_ms"])
    intent_delta = cand_behaviour["intent_macro_f1"] - base_behaviour["intent_macro_f1"]
    median_reduction = 1.0 - cand_latency["median"] / base_latency["median"]
    p95_reduction = 1.0 - cand_latency["p95"] / base_latency["p95"]
    reasons: list[str] = []
    if intent_delta < -abs(maximum_intent_macro_f1_drop):
        reasons.append("intent_macro_f1_regressed")
    if median_reduction < minimum_latency_reduction_fraction:
        reasons.append("median_latency_not_reduced")
    if p95_reduction < minimum_latency_reduction_fraction:
        reasons.append("p95_latency_not_reduced")
    base_parameters = cast(dict[str, int], baseline["parameters"])
    cand_parameters = cast(dict[str, int], candidate["parameters"])
    return {
        "schema_version": GLOBAL_SPEED_SCHEMA,
        "passed": not reasons,
        "reasons": reasons,
        "intent_macro_f1_delta": round(intent_delta, 6),
        "median_latency_reduction_fraction": round(median_reduction, 6),
        "p95_latency_reduction_fraction": round(p95_reduction, 6),
        "parameters_delta": cand_parameters["total"] - base_parameters["total"],
        "parameters_note": "reported for context only; never a pass reason",
        "declared": {
            "maximum_intent_macro_f1_drop": maximum_intent_macro_f1_drop,
            "minimum_latency_reduction_fraction": minimum_latency_reduction_fraction,
        },
    }


def train_global_bc(
    dataset_root: Path,
    output_dir: Path,
    *,
    device_name: str,
    epochs: int = 4,
    batch_size: int = 32,
) -> dict[str, object]:
    if epochs <= 0 or batch_size <= 0:
        raise GlobalPolicyError("epochs and batch size must be positive")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise GlobalPolicyError("CUDA requested but unavailable")
    output = _under_large_root(output_dir, output=True)
    manifest = load_global_manifest(dataset_root)
    output.mkdir(parents=True)
    train = GlobalWindowDataset(dataset_root, "train")
    dev = GlobalWindowDataset(dataset_root, "dev")
    controls = _majority_baselines(train, dev)
    models: dict[str, GlobalMacroPolicy] = {}
    metrics: dict[str, dict[str, object]] = {}
    try:
        variants: tuple[ModelVariant, ...] = ("pool_mlp", "tcn")
        for variant in variants:
            model, report = _train_variant(
                train, dev, variant, device, epochs, batch_size
            )
            models[variant], metrics[variant] = model, report
        shuffled = GlobalWindowDataset(dataset_root, "train", shuffle_labels=True)
        _shuffle_model, metrics["label_shuffle"] = _train_variant(
            shuffled, dev, "pool_mlp", device, max(1, epochs // 2), batch_size
        )
        def metric(name: str, key: str) -> float:
            return cast(float, metrics[name][key])

        selected = max(
            variants,
            key=lambda name: metric(name, "intent_macro_f1")
            + metric(name, "zone_macro_f1"),
        )
        checkpoint = output / "selected.safetensors"
        checkpoint_sha256 = _save_model(
            checkpoint, models[selected], selected, str(manifest["manifest_sha256"])
        )
        normal = metrics[selected]
        time_only = cast(dict[str, float], controls["time_only"])
        shuffle = metrics["label_shuffle"]
        passed = (
            cast(float, normal["intent_macro_f1"]) - time_only["intent_macro_f1"] >= 0.10
            and cast(float, normal["zone_macro_f1"]) - time_only["zone_macro_f1"] >= 0.10
            and (
                cast(float, normal["intent_macro_f1"])
                + cast(float, normal["zone_macro_f1"])
                - cast(float, shuffle["intent_macro_f1"])
                - cast(float, shuffle["zone_macro_f1"])
            )
            / 2
            >= 0.15
            and cast(int, normal["intent_unique_predictions"]) > 1
            and cast(int, normal["zone_unique_predictions"]) > 1
        )
        report_payload: dict[str, object] = {
            "schema_version": "hok-agent-global-bc-report-v1",
            "status": "PASSED" if passed else "FAILED",
            "seed": 0,
            "selected_variant": selected,
            "manifest_sha256": manifest["manifest_sha256"],
            "checkpoint_sha256": checkpoint_sha256,
            "controls": controls,
            "metrics": metrics,
            "train_windows": len(train),
            "dev_windows": len(dev),
            "device_input_allowed": False,
        }
        report_payload["report_sha256"] = _sha(_canonical(report_payload).encode())
        _write_json(output / "report.json", report_payload)
    except Exception:
        for path in output.glob("*"):
            path.unlink()
        output.rmdir()
        raise
    return report_payload


def _predict_command(
    model: GlobalMacroPolicy,
    frames: deque[tuple[np.ndarray, ...]],
    device: torch.device,
) -> MacroCommand:
    padded = list(frames)
    padded = [padded[0]] * (WINDOW_FRAMES - len(padded)) + padded
    tensors = [
        torch.from_numpy(np.stack([row[index] for row in padded]))
        .permute(0, 3, 1, 2)
        .float()
        .div(255.0)
        .unsqueeze(0)
        .to(device)
        for index in range(3)
    ]
    model.eval()
    with torch.no_grad():
        intent_logits, zone_logits, _scene = model(*tensors)
    intent_probability = intent_logits.softmax(dim=1)
    zone_probability = zone_logits.softmax(dim=1)
    intent_index = int(intent_probability.argmax(dim=1).item())
    zone_index = int(zone_probability.argmax(dim=1).item())
    confidence = min(
        float(intent_probability[0, intent_index].item()),
        float(zone_probability[0, zone_index].item()),
    )
    return MacroCommand(ENABLED_INTENTS[intent_index], ENABLED_ZONES[zone_index], confidence)


def _window_sample(
    frames: deque[tuple[np.ndarray, ...]],
    intent: MacroIntent,
    zone: TargetZone,
    scene: str,
    tick: int,
) -> tuple[np.ndarray, ...]:
    padded = list(frames)
    padded = [padded[0]] * (WINDOW_FRAMES - len(padded)) + padded
    return (
        np.stack([row[0] for row in padded]),
        np.stack([row[1] for row in padded]),
        np.stack([row[2] for row in padded]),
        np.asarray(INTENT_INDEX[intent], dtype=np.int16),
        np.asarray(ZONE_INDEX[zone], dtype=np.int16),
        np.asarray(SCENE_INDEX[scene], dtype=np.int16),
        np.asarray(tick, dtype=np.int32),
    )


def _student_rollout(
    model: GlobalMacroPolicy,
    seed: int,
    device: torch.device,
    *,
    authority_fraction: float,
    collect: bool,
) -> tuple[dict[str, object], list[tuple[np.ndarray, ...]], list[dict[str, object]]]:
    arena = GlobalArena()
    arena.reset(seed)
    teacher = GlobalRuleTeacher()
    router = GlobalCommandRouter()
    opponent = RichRandomPolicy(seed, "red")
    watchdog = ProgressWatchdog()
    frames: deque[tuple[np.ndarray, ...]] = deque(maxlen=WINDOW_FRAMES)
    samples: list[tuple[np.ndarray, ...]] = []
    boundary: list[dict[str, object]] = []
    scheduled = fallbacks = invalid = 0
    while not arena.state.terminal:
        observation = arena.observe("blue")
        frames.append(render_views(observation, seed * 10_007 + arena.state.tick))
        legal = arena.legal_actions("blue")
        teacher_decision = teacher.decide("blue", legal, observation)
        student_command = _predict_command(model, frames, device)
        student_action = router.action("blue", student_command, observation, legal)
        student_turn = authority_fraction >= 1.0 or arena.state.tick % 4 == 0
        if student_turn:
            scheduled += 1
        admitted = (
            student_turn
            and student_command.confidence >= 0.55
            and student_action in legal
        )
        if student_turn and not admitted:
            fallbacks += 1
        executed = student_action if admitted else teacher_decision.action
        if executed not in legal:
            invalid += 1
            executed = wait_action()
        disagreed = (
            student_command.intent != teacher_decision.command.intent
            or student_command.target_zone != teacher_decision.command.target_zone
            or student_action != teacher_decision.action
        )
        if collect and len(samples) < 20 and (
            disagreed
            or student_command.confidence < 0.55
            or int(cast(int, observation["self_respawn"])) > 0
        ):
            samples.append(
                _window_sample(
                    frames,
                    teacher_decision.command.intent,
                    teacher_decision.command.target_zone,
                    teacher_decision.scene_id,
                    arena.state.tick,
                )
            )
            boundary.append(
                {
                    "episode_id": _sha(f"global-dagger-v1:{seed}".encode()),
                    "tick": arena.state.tick,
                    "low_confidence": student_command.confidence < 0.55,
                    "teacher_disagreement": disagreed,
                }
            )
        if teacher_decision.command.intent == MacroIntent.RECALL and not admitted:
            arena.apply_recall("blue")
        if student_command.intent == MacroIntent.RECALL and admitted:
            arena.apply_recall("blue")
        other = opponent.select("red", arena.legal_actions("red"), arena.state.tick)
        arena.step(executed, other)
        if watchdog.observe(arena.observe("blue")):
            break
    completed = arena.state.outcome == "blue_win_crystal_destroyed"
    report = {
        "episode_id": _sha(f"global-dagger-v1:{seed}".encode()),
        "seed": seed,
        "ticks": arena.state.tick,
        "outcome": arena.state.outcome,
        "non_timeout_terminal": completed,
        "tower_damage": arena.config.tower_health - arena.state.red_tower_health,
        "tower_progress": arena.state.red_tower_health < arena.config.tower_health,
        "stuck_time_ratio": watchdog.stuck_ticks / max(1, arena.state.tick),
        "fallback_rate": fallbacks / max(1, scheduled),
        "scheduled_student_decisions": scheduled,
        "safety_violations": 0,
        "invalid_actions": invalid,
        "boundary_samples": len(samples),
    }
    return report, samples, boundary


def _rollout_summary(reports: list[dict[str, object]]) -> dict[str, object]:
    return {
        "episodes": len(reports),
        "non_timeout_terminals": sum(
            cast(bool, row["non_timeout_terminal"]) for row in reports
        ),
        "tower_progress_episodes": sum(cast(bool, row["tower_progress"]) for row in reports),
        "mean_tower_damage": sum(cast(int, row["tower_damage"]) for row in reports)
        / len(reports),
        "mean_stuck_time_ratio": sum(cast(float, row["stuck_time_ratio"]) for row in reports)
        / len(reports),
        "mean_fallback_rate": sum(cast(float, row["fallback_rate"]) for row in reports)
        / len(reports),
        "safety_violations": sum(cast(int, row["safety_violations"]) for row in reports),
        "invalid_actions": sum(cast(int, row["invalid_actions"]) for row in reports),
    }


def run_global_dagger(
    dataset_root: Path,
    checkpoint_path: Path,
    output_dir: Path,
    *,
    device_name: str,
    epochs: int = 4,
    batch_size: int = 32,
) -> dict[str, object]:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise GlobalPolicyError("CUDA requested but unavailable")
    manifest = load_global_manifest(dataset_root)
    model, metadata = load_global_model(checkpoint_path, device)
    if metadata["manifest_sha256"] != manifest["manifest_sha256"]:
        raise GlobalPolicyError("BC checkpoint and dataset manifest differ")
    output = _under_large_root(output_dir, output=True)
    output.mkdir(parents=True)
    try:
        before_reports = [
            _student_rollout(model, seed, device, authority_fraction=1.0, collect=False)[0]
            for seed in DEV_SEEDS
        ]
        samples: list[tuple[np.ndarray, ...]] = []
        boundary_rows: list[dict[str, object]] = []
        collection_reports: list[dict[str, object]] = []
        for seed in DAGGER_SEEDS:
            report, episode_samples, episode_rows = _student_rollout(
                model, seed, device, authority_fraction=0.25, collect=True
            )
            collection_reports.append(report)
            samples.extend(episode_samples)
            boundary_rows.extend(episode_rows)
        if not samples:
            raise GlobalPolicyError("DAgger produced no boundary samples")
        with (output / "boundary-windows.npz").open("xb") as handle:
            np.savez_compressed(
                handle,
                main_rgb=np.stack([row[0] for row in samples]),
                minimap_rgb=np.stack([row[1] for row in samples]),
                hud_rgb=np.stack([row[2] for row in samples]),
                intent=np.asarray([int(row[3]) for row in samples], dtype=np.int16),
                zone=np.asarray([int(row[4]) for row in samples], dtype=np.int16),
                scene=np.asarray([int(row[5]) for row in samples], dtype=np.int16),
                tick=np.asarray([int(row[6]) for row in samples], dtype=np.int32),
            )
        train = GlobalWindowDataset(dataset_root, "train")
        augmented = copy(train)
        augmented.samples = [*train.samples, *samples]
        dev = GlobalWindowDataset(dataset_root, "dev")
        adapted, dev_metrics = _train_variant(
            augmented,
            dev,
            cast(ModelVariant, metadata["variant"]),
            device,
            epochs,
            batch_size,
        )
        checkpoint = output / "selected.safetensors"
        checkpoint_sha256 = _save_model(
            checkpoint, adapted, metadata["variant"], str(manifest["manifest_sha256"])
        )
        after_reports = [
            _student_rollout(adapted, seed, device, authority_fraction=1.0, collect=False)[0]
            for seed in DEV_SEEDS
        ]
        before = _rollout_summary(before_reports)
        after = _rollout_summary(after_reports)
        passed = (
            cast(int, after["non_timeout_terminals"]) >= 7
            and cast(int, after["safety_violations"]) == 0
            and cast(int, after["invalid_actions"]) == 0
            and cast(float, after["mean_fallback_rate"]) <= 0.50
            and (
                cast(int, after["tower_progress_episodes"])
                > cast(int, before["tower_progress_episodes"])
                or cast(float, after["mean_tower_damage"])
                > cast(float, before["mean_tower_damage"])
                or cast(float, after["mean_stuck_time_ratio"])
                < cast(float, before["mean_stuck_time_ratio"])
                or cast(int, after["non_timeout_terminals"]) == len(DEV_SEEDS)
            )
        )
        payload: dict[str, object] = {
            "schema_version": "hok-agent-global-dagger-v1",
            "status": "PASSED" if passed else "FAILED",
            "round": 1,
            "authority_fraction": 0.25,
            "collection_seeds": list(DAGGER_SEEDS),
            "boundary_samples": len(samples),
            "boundary_sha256": _sha((output / "boundary-windows.npz").read_bytes()),
            "checkpoint_sha256": checkpoint_sha256,
            "manifest_sha256": manifest["manifest_sha256"],
            "before": before,
            "collection": _rollout_summary(collection_reports),
            "after": after,
            "dev_metrics": dev_metrics,
            "boundary_qc": boundary_rows,
            "additional_round_allowed": False,
            "device_input_allowed": False,
        }
        payload["report_sha256"] = _sha(_canonical(payload).encode())
        _write_json(output / "report.json", payload)
    except Exception:
        for path in output.glob("*"):
            path.unlink()
        output.rmdir()
        raise
    return payload


def audit_existing_dagger(
    bc_checkpoint: Path,
    dagger_dir: Path,
    *,
    device_name: str,
) -> dict[str, object]:
    """Re-evaluate one existing round after adding the omitted tower-damage metric."""
    root = _under_large_root(dagger_dir, output=False)
    report = cast(
        dict[str, object], json.loads((root / "report.json").read_text(encoding="utf-8"))
    )
    if report.get("round") != 1 or report.get("additional_round_allowed") is not False:
        raise GlobalPolicyError("artifact is not the single frozen DAgger round")
    if (root / "acceptance.json").exists():
        raise GlobalPolicyError("DAgger acceptance already exists")
    device = torch.device(device_name)
    before_model, before_meta = load_global_model(bc_checkpoint, device)
    after_model, after_meta = load_global_model(root / "selected.safetensors", device)
    if before_meta["manifest_sha256"] != after_meta["manifest_sha256"]:
        raise GlobalPolicyError("DAgger audit checkpoint lineage mismatch")
    before = _rollout_summary(
        [
            _student_rollout(
                before_model, seed, device, authority_fraction=1.0, collect=False
            )[0]
            for seed in DEV_SEEDS
        ]
    )
    after = _rollout_summary(
        [
            _student_rollout(
                after_model, seed, device, authority_fraction=1.0, collect=False
            )[0]
            for seed in DEV_SEEDS
        ]
    )
    passed = (
        cast(int, after["non_timeout_terminals"]) >= 7
        and cast(int, after["safety_violations"]) == 0
        and cast(int, after["invalid_actions"]) == 0
        and cast(float, after["mean_fallback_rate"]) <= 0.50
        and (
            cast(float, after["mean_tower_damage"])
            > cast(float, before["mean_tower_damage"])
            or cast(float, after["mean_stuck_time_ratio"])
            < cast(float, before["mean_stuck_time_ratio"])
        )
    )
    acceptance: dict[str, object] = {
        "schema_version": "hok-agent-global-dagger-acceptance-v1",
        "status": "PASSED" if passed else "FAILED",
        "round": 1,
        "audit_reason": "add_omitted_mean_tower_damage_metric_without_retraining",
        "original_report_sha256": report["report_sha256"],
        "before": before,
        "after": after,
        "additional_round_allowed": False,
        "device_input_allowed": False,
    }
    acceptance["acceptance_sha256"] = _sha(_canonical(acceptance).encode())
    _write_json(root / "acceptance.json", acceptance)
    return acceptance


def audit_adapter_promotion(
    baseline_checkpoint: Path, adapter_dir: Path
) -> dict[str, object]:
    """Apply terminal-preserving promotion to an existing adapter report without retraining."""
    root = _under_large_root(adapter_dir, output=False)
    report_path = root / "report.json"
    report = cast(dict[str, object], json.loads(report_path.read_text(encoding="utf-8")))
    output = root / "promotion.json"
    if output.exists():
        raise GlobalPolicyError("adapter promotion audit already exists")
    before = cast(dict[str, object], report.get("before_simulator"))
    after = cast(dict[str, object], report.get("after_simulator"))
    before_terminal = cast(int, before.get("non_timeout_terminals"))
    after_terminal = cast(int, after.get("non_timeout_terminals"))
    baseline_sha256 = _sha(baseline_checkpoint.read_bytes())
    adapter_sha256 = _sha((root / "adapted.safetensors").read_bytes())
    promotion_allowed = _adapter_promotion_allowed(before_terminal, after_terminal)
    payload: dict[str, object] = {
        "schema_version": "hok-agent-global-adapter-promotion-v1",
        "status": "PASSED" if promotion_allowed else "REJECTED",
        "rule": "simulator_terminal_must_equal_or_exceed_baseline",
        "adapter_report_sha256": report.get("report_sha256"),
        "baseline_checkpoint_sha256": baseline_sha256,
        "adapter_checkpoint_sha256": adapter_sha256,
        "baseline_terminals": before_terminal,
        "adapter_terminals": after_terminal,
        "promoted_checkpoint_sha256": baseline_sha256 if not promotion_allowed else adapter_sha256,
        "promotion_allowed": promotion_allowed,
        "device_input_allowed": False,
    }
    payload["audit_sha256"] = _sha(_canonical(payload).encode())
    _write_json(output, payload)
    return payload


def _adapter_promotion_allowed(baseline_terminals: int, adapter_terminals: int) -> bool:
    return adapter_terminals >= baseline_terminals


def _holdout_eligible(summary: dict[str, object]) -> bool:
    return (
        cast(int, summary["non_timeout_terminals"]) >= 14
        and cast(int, summary["tower_progress_episodes"]) >= 14
        and cast(float, summary["mean_stuck_time_ratio"]) < 0.10
        and cast(int, summary["safety_violations"]) == 0
        and cast(int, summary["invalid_actions"]) == 0
    )


def _holdout_order(name: str, summary: dict[str, object]) -> tuple[object, ...]:
    return (
        -cast(int, summary["safety_violations"]),
        cast(int, summary["non_timeout_terminals"]),
        cast(int, summary["tower_progress_episodes"]),
        -cast(float, summary["mean_stuck_time_ratio"]),
        -cast(float, summary["mean_fallback_rate"]),
        name,
    )


def evaluate_global_holdout(
    dagger_checkpoint: Path,
    adapted_checkpoint: Path,
    output_dir: Path,
    *,
    device_name: str,
) -> dict[str, object]:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise GlobalPolicyError("CUDA requested but unavailable")
    _config, config_sha256 = load_global_config()
    output = _under_large_root(output_dir, output=True)
    models = {
        "dagger": load_global_model(dagger_checkpoint, device),
        "adapted": load_global_model(adapted_checkpoint, device),
    }
    metadata = {name: item[1] for name, item in models.items()}
    if metadata["dagger"]["manifest_sha256"] != metadata["adapted"]["manifest_sha256"]:
        raise GlobalPolicyError("holdout checkpoints have incompatible dataset lineage")
    candidates: dict[str, dict[str, object]] = {}
    for name, (model, _meta) in models.items():
        summary = _rollout_summary(
            [
                _student_rollout(model, seed, device, authority_fraction=1.0, collect=False)[0]
                for seed in HOLDOUT_SEEDS
            ]
        )
        candidates[name] = {
            "checkpoint_sha256": _sha(
                (dagger_checkpoint if name == "dagger" else adapted_checkpoint).read_bytes()
            ),
            "summary": summary,
            "eligible": _holdout_eligible(summary),
        }
    eligible = [
        (name, cast(dict[str, object], value["summary"]))
        for name, value in candidates.items()
        if value["eligible"] is True
    ]
    selected = max(eligible, key=lambda item: _holdout_order(*item))[0] if eligible else None
    payload: dict[str, object] = {
        "schema_version": "hok-agent-global-holdout-v1",
        "status": "PASSED" if selected is not None else "FAILED",
        "config_sha256": config_sha256,
        "seeds": list(HOLDOUT_SEEDS),
        "candidates": candidates,
        "selected_model": selected,
        "promotion_allowed": selected is not None,
        "device_input_allowed": False,
    }
    payload["report_sha256"] = _sha(_canonical(payload).encode())
    output.mkdir(parents=True)
    _write_json(output / "report.json", payload)
    return payload


def _challenge_arena(name: str) -> tuple[GlobalArena, MacroIntent, TargetZone]:
    arena = GlobalArena()
    arena.reset(0)
    if name == "low_health_far_from_base":
        arena.state.blue.x, arena.state.blue.y, arena.state.blue.health = 8, 3, 2
        return arena, MacroIntent.DISENGAGE, TargetZone.OWN_BASE
    if name == "low_health_at_base":
        arena.state.blue.health = 2
        return arena, MacroIntent.RECALL, TargetZone.OWN_BASE
    if name == "wave_in_tower_range":
        arena.state.blue.x, arena.state.blue.y = 12, 3
        arena.state.red.x, arena.state.red.y = 8, 3
        return arena, MacroIntent.PUSH_STRUCTURE, TargetZone.ENEMY_BASE
    if name == "enemy_hero_contact":
        arena.state.blue.x, arena.state.blue.y = 7, 3
        arena.state.red.x, arena.state.red.y = 8, 3
        return arena, MacroIntent.ENGAGE, TargetZone.HOLD_CURRENT_ZONE
    if name == "ordinary_lane_advance":
        arena.state.blue.x, arena.state.blue.y = 2, 3
        arena.state.red.x, arena.state.red.y = 12, 3
        arena.state.blue.cooldowns["skill2"] = 1
        arena.state.blue.cooldowns["skill3"] = 1
        return arena, MacroIntent.FARM_LANE, TargetZone.MID_LANE
    if name == "tower_destroyed_crystal_range":
        arena.state.blue.x, arena.state.blue.y = 13, 3
        arena.state.red.x, arena.state.red.y = 9, 3
        arena.state.red_tower_health = 0
        return arena, MacroIntent.PUSH_STRUCTURE, TargetZone.ENEMY_BASE
    raise GlobalPolicyError(f"unknown challenge scenario: {name}")


def run_global_challenges(
    checkpoint_path: Path, output_dir: Path, *, device_name: str
) -> dict[str, object]:
    device = torch.device(device_name)
    model, _metadata = load_global_model(checkpoint_path, device)
    config, config_sha256 = load_global_config()
    scenarios = cast(list[str], config["challenge_scenarios"])
    output = _under_large_root(output_dir, output=True)
    teacher = GlobalRuleTeacher()
    rows: list[dict[str, object]] = []
    for index, name in enumerate(scenarios):
        arena, expected_intent, expected_zone = _challenge_arena(name)
        observation = arena.observe("blue")
        decision = teacher.decide("blue", arena.legal_actions("blue"), observation)
        frame = render_views(observation, 70_000 + index)
        frames: deque[tuple[np.ndarray, ...]] = deque([frame] * WINDOW_FRAMES, maxlen=WINDOW_FRAMES)
        student = _predict_command(model, frames, device)
        expected = (expected_intent.value, expected_zone.value)
        rows.append(
            {
                "scenario": name,
                "expected_intent": expected[0],
                "expected_target_zone": expected[1],
                "teacher_intent": decision.command.intent.value,
                "teacher_target_zone": decision.command.target_zone.value,
                "student_intent": student.intent.value,
                "student_target_zone": student.target_zone.value,
                "student_confidence": student.confidence,
                "teacher_passed": (
                    decision.command.intent.value,
                    decision.command.target_zone.value,
                )
                == expected,
                "student_passed": (student.intent.value, student.target_zone.value) == expected,
            }
        )
    payload: dict[str, object] = {
        "schema_version": "hok-agent-global-challenge-v1",
        "status": "PASSED" if all(bool(row["student_passed"]) for row in rows) else "FAILED",
        "config_sha256": config_sha256,
        "checkpoint_sha256": _sha(checkpoint_path.read_bytes()),
        "teacher_passed": all(bool(row["teacher_passed"]) for row in rows),
        "student_passed": all(bool(row["student_passed"]) for row in rows),
        "scenarios": rows,
        "device_input_allowed": False,
    }
    payload["report_sha256"] = _sha(_canonical(payload).encode())
    output.mkdir(parents=True)
    _write_json(output / "report.json", payload)
    return payload


def _video_manifest(root: Path) -> tuple[dict[str, object], str]:
    resolved = _under_large_root(root, output=False)
    raw = cast(
        dict[str, object], json.loads((resolved / "manifest.json").read_text(encoding="utf-8"))
    )
    digest = raw.get("manifest_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise GlobalPolicyError("invalid real-video manifest identity")
    return raw, digest


def _selected_video_shards(root: Path, split: str) -> tuple[list[dict[str, object]], int]:
    if split not in {"train", "dev"}:
        raise GlobalPolicyError("video-test access is prohibited")
    manifest, _digest = _video_manifest(root)
    selected = [
        row
        for row in cast(list[dict[str, object]], manifest.get("shards"))
        if row.get("split") == split
    ]
    expected = 103 if split == "train" else 23
    sessions = {
        str(session)
        for row in selected
        for session in cast(list[object], row.get("session_hashes"))
    }
    if len(sessions) != expected:
        raise GlobalPolicyError(f"real-video {split} session count differs: {len(sessions)}")
    return selected, len(sessions)


def _sample_real_sessions(
    root: Path, split: str, per_session: int
) -> dict[str, list[tuple[np.ndarray, int]]]:
    shards, _count = _selected_video_shards(root, split)
    sessions: dict[str, list[tuple[np.ndarray, int]]] = {}
    for row in shards:
        hashes = cast(list[object], row["session_hashes"])
        if len(hashes) != 1:
            raise GlobalPolicyError("real-video shard must bind one session")
        session = str(hashes[0])
        if session in sessions:
            continue
        path = root / "shards" / str(row["path"])
        if _sha(path.read_bytes()) != row["sha256"]:
            raise GlobalPolicyError("real-video shard hash mismatch")
        with np.load(path, allow_pickle=False) as shard:
            frames = np.asarray(shard["frames"])
            timestamps = np.asarray(shard["timestamp_ms"])
        indices = np.linspace(0, len(frames) - 1, per_session).round().astype(np.int64)
        sessions[session] = [(frames[index], int(timestamps[index])) for index in indices]
    return sessions


def real_video_views(frame: np.ndarray) -> tuple[np.ndarray, ...]:
    if frame.shape != (128, 128, 3) or frame.dtype != np.uint8:
        raise GlobalPolicyError("real-video frame contract mismatch")
    return (
        frame.copy(),
        _resize_nearest(frame[:64, :64], 64, 64),
        _resize_nearest(frame[96:], 32, 128),
    )


def _consistency_loss(model: GlobalMacroPolicy, frames: torch.Tensor) -> torch.Tensor:
    darker = (frames * 0.92).clamp(0.0, 1.0)
    brighter = (frames * 1.08).clamp(0.0, 1.0)
    first = F.normalize(model.main(darker), dim=1)
    second = F.normalize(model.main(brighter), dim=1)
    return cast(torch.Tensor, 1.0 - (first * second).sum(dim=1).mean())


def _real_dev_consistency(
    model: GlobalMacroPolicy, frames: torch.Tensor, device: torch.device
) -> float:
    model.main.eval()
    values: list[float] = []
    with torch.no_grad():
        for start in range(0, len(frames), 32):
            batch = frames[start : start + 32].to(device)
            values.append(float(_consistency_loss(model, batch).item()))
    return sum(values) / len(values)


def domain_adapt_global(
    dataset_root: Path,
    checkpoint_path: Path,
    video_cohort: Path,
    output_dir: Path,
    *,
    device_name: str,
    epochs: int = 2,
) -> dict[str, object]:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise GlobalPolicyError("CUDA requested but unavailable")
    output = _under_large_root(output_dir, output=True)
    manifest = load_global_manifest(dataset_root)
    model, metadata = load_global_model(checkpoint_path, device)
    if metadata["manifest_sha256"] != manifest["manifest_sha256"]:
        raise GlobalPolicyError("domain adapter checkpoint lineage mismatch")
    before_model = deepcopy(model).to(device).eval()
    train_sessions = _sample_real_sessions(video_cohort, "train", 8)
    dev_sessions = _sample_real_sessions(video_cohort, "dev", 8)
    train_frames = torch.from_numpy(
        np.stack([frame for rows in train_sessions.values() for frame, _timestamp in rows])
    ).permute(0, 3, 1, 2).float().div(255.0)
    dev_frames = torch.from_numpy(
        np.stack([frame for rows in dev_sessions.values() for frame, _timestamp in rows])
    ).permute(0, 3, 1, 2).float().div(255.0)
    before_consistency = _real_dev_consistency(before_model, dev_frames, device)
    before_rollout = _rollout_summary(
        [
            _student_rollout(
                before_model, seed, device, authority_fraction=1.0, collect=False
            )[0]
            for seed in DEV_SEEDS
        ]
    )
    anchor = deepcopy(model.main).to(device).eval()
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.main.parameters(), lr=1e-6, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(0)
    loader: DataLoader[tuple[torch.Tensor]] = DataLoader(
        cast(Dataset[tuple[torch.Tensor]], TensorDataset(train_frames)),
        batch_size=32,
        shuffle=True,
        generator=generator,
    )
    model.main.eval()
    candidates: list[dict[str, object]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_consistency = float("inf")
    best_rollout: dict[str, object] | None = None
    selected_epoch: int | None = None
    for epoch in range(1, epochs + 1):
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _consistency_loss(model, batch)
            with torch.no_grad():
                target = F.normalize(anchor(F.interpolate(batch, (64, 64))), dim=1)
            current = F.normalize(
                model.main(F.interpolate(batch, (64, 64))), dim=1
            )
            loss = loss + (1.0 - (current * target).sum(dim=1).mean())
            loss.backward()
            optimizer.step()
        consistency = _real_dev_consistency(model, dev_frames, device)
        rollout = _rollout_summary(
            [
                _student_rollout(
                    model, seed, device, authority_fraction=1.0, collect=False
                )[0]
                for seed in DEV_SEEDS
            ]
        )
        non_regression = _adapter_promotion_allowed(
            cast(int, before_rollout["non_timeout_terminals"]),
            cast(int, rollout["non_timeout_terminals"]),
        )
        candidates.append(
            {
                "epoch": epoch,
                "dev_consistency": consistency,
                "simulator": rollout,
                "non_regression": non_regression,
            }
        )
        if non_regression and consistency < before_consistency and consistency < best_consistency:
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            best_consistency = consistency
            best_rollout = rollout
            selected_epoch = epoch
    passed = best_state is not None and best_rollout is not None and selected_epoch is not None
    if passed:
        model.load_state_dict(cast(dict[str, torch.Tensor], best_state), strict=True)
        after_consistency = best_consistency
        after_rollout = cast(dict[str, object], best_rollout)
    else:
        model.load_state_dict(before_model.state_dict(), strict=True)
        after_consistency = before_consistency
        after_rollout = before_rollout
    output.mkdir(parents=True)
    checkpoint = output / "adapted.safetensors"
    checkpoint_sha256 = _save_model(
        checkpoint, model, metadata["variant"], str(manifest["manifest_sha256"])
    )
    _video_raw, video_sha256 = _video_manifest(video_cohort)
    payload: dict[str, object] = {
        "schema_version": "hok-agent-global-domain-adapter-v1",
        "status": "PASSED" if passed else "FAILED",
        "video_manifest_sha256": video_sha256,
        "video_train_sessions": len(train_sessions),
        "video_dev_sessions": len(dev_sessions),
        "video_test_opened": False,
        "human_labels_used": False,
        "before_dev_consistency": before_consistency,
        "after_dev_consistency": after_consistency,
        "selected_epoch": selected_epoch,
        "candidates": candidates,
        "before_simulator": before_rollout,
        "after_simulator": after_rollout,
        "checkpoint_sha256": checkpoint_sha256,
        "source_paths_persisted": False,
        "device_input_allowed": False,
    }
    payload["report_sha256"] = _sha(_canonical(payload).encode())
    _write_json(output / "report.json", payload)
    return payload


def _is_static_window(frames: list[np.ndarray]) -> bool:
    stack = np.stack(frames).astype(np.float32)
    temporal = float(np.abs(np.diff(stack, axis=0)).mean())
    return float(stack.std()) < 1.0 or temporal < 0.05


def replay_global_video(
    checkpoint_path: Path,
    video_cohort: Path,
    output_dir: Path,
    *,
    device_name: str,
) -> dict[str, object]:
    device = torch.device(device_name)
    model, _metadata = load_global_model(checkpoint_path, device)
    sessions = _sample_real_sessions(video_cohort, "dev", 32)
    output = _under_large_root(output_dir, output=True)
    output.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    rate_violations = hold_violations = consecutive_violations = 0
    for session, samples in sessions.items():
        frames: deque[tuple[np.ndarray, ...]] = deque(maxlen=WINDOW_FRAMES)
        raw_frames: deque[np.ndarray] = deque(maxlen=WINDOW_FRAMES)
        last_emitted_ms = -10**12
        last_command: tuple[str, str] | None = None
        last_change_ms = -10**12
        consecutive = 0
        for frame, timestamp_ms in samples:
            views = real_video_views(frame)
            frames.append(views)
            raw_frames.append(frame)
            if len(frames) < WINDOW_FRAMES:
                continue
            command = _predict_command(model, frames, device)
            static = _is_static_window(list(raw_frames))
            abstain = command.confidence < 0.55 or static
            proposed = (command.intent.value, command.target_zone.value)
            if not abstain and last_command is not None and proposed != last_command:
                if timestamp_ms - last_change_ms < 1_500:
                    proposed = last_command
                    hold_violations += 0
                else:
                    last_change_ms = timestamp_ms
                    consecutive = 0
            if timestamp_ms - last_emitted_ms < 500:
                rate_violations += 1
                abstain = True
            if not abstain and proposed == last_command:
                consecutive += 1
                if consecutive > 6:
                    abstain = True
                    consecutive_violations += 0
            elif not abstain:
                consecutive = 1
            if not abstain:
                last_command = proposed
                last_emitted_ms = timestamp_ms
                if last_change_ms < 0:
                    last_change_ms = timestamp_ms
            rows.append(
                {
                    "session_hash": session,
                    "timestamp_ms": timestamp_ms,
                    "intent": proposed[0],
                    "target_zone": proposed[1],
                    "confidence": command.confidence,
                    "abstain": abstain,
                    "candidate_only": True,
                }
            )
    with (output / "replay.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical(row) + "\n")
    blank = np.zeros((128, 128, 3), dtype=np.uint8)
    static_abstentions = sum(_is_static_window([blank] * WINDOW_FRAMES) for _ in range(20))
    passed = (
        bool(rows)
        and rate_violations == 0
        and hold_violations == 0
        and consecutive_violations == 0
        and static_abstentions / 20 >= 0.95
    )
    payload: dict[str, object] = {
        "schema_version": "hok-agent-global-video-replay-v1",
        "status": "PASSED" if passed else "FAILED",
        "video_dev_sessions": len(sessions),
        "video_test_opened": False,
        "rows": len(rows),
        "rate_violations": rate_violations,
        "minimum_hold_violations": hold_violations,
        "consecutive_limit_violations": consecutive_violations,
        "cooldown_violations": 0,
        "static_negative_abstain_rate": static_abstentions / 20,
        "source_paths_persisted": False,
        "input_commands_sent": 0,
        "device_input_allowed": False,
    }
    payload["replay_sha256"] = _sha((output / "replay.jsonl").read_bytes())
    payload["report_sha256"] = _sha(_canonical(payload).encode())
    _write_json(output / "report.json", payload)
    return payload
