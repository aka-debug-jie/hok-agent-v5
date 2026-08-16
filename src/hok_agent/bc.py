# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch import nn

from hok_agent.arena import ArenaConfig, FactorizedAction, PixelArena, Side, observation_hash
from hok_agent.policies import NullPolicy, RandomPolicy, ScriptedPolicy

FEATURES = ("tick", "side", "self_position", "opponent_position", "self_health", "opponent_health", "own_tower_health", "enemy_tower_health", "own_crystal_health", "enemy_crystal_health")
ACTIONS = (("wait", "none", "none"), ("move", "none", "forward"), ("move", "none", "backward"), ("attack", "enemy_hero", "none"), ("attack", "enemy_tower", "none"), ("attack", "enemy_crystal", "none"))
ACTION_INDEX = {value: index for index, value in enumerate(ACTIONS)}
TRAINING_SEEDS = (0, 1, 2)
MODEL_KEYS = {"kind", "claim_scope", "hok_capability_claim", "gamecore_equivalence_claim", "environment_identity", "config_hash", "feature_names", "action_vocabulary", "architecture", "training_seed", "weights"}


class BCError(ValueError):
    pass


@dataclass
class Sample:
    digest: str
    observation: dict[str, object]
    action: int
    legal: frozenset[int]
    split: str = ""


@dataclass
class Dataset:
    config: ArenaConfig
    samples: list[Sample]
    episode_count: int
    raw_count: int
    conflict_count: int


class StructuredActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(10, 32)
        self.output = nn.Linear(32, 6)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.output(torch.tanh(self.hidden(observation))))


def _action_key(action: FactorizedAction) -> tuple[str, str, str]:
    return action.action_type, action.target, action.direction


def _action_document(index: int) -> dict[str, object]:
    action_type, target, direction = ACTIONS[index]
    macro = "hold" if action_type == "wait" else "advance"
    if action_type == "attack":
        macro = "siege" if target in {"enemy_tower", "enemy_crystal"} else "engage"
    return FactorizedAction(macro, action_type, target=target, direction=direction, skill="basic" if action_type == "attack" else "none").to_dict()


def _feature_vector(observation: dict[str, object], config: ArenaConfig) -> list[float]:
    return [
        int(cast(int, observation["tick"])) / config.max_ticks,
        0.0 if observation["side"] == "blue" else 1.0,
        int(cast(int, observation["self_position"])) / config.lane_max,
        int(cast(int, observation["opponent_position"])) / config.lane_max,
        int(cast(int, observation["self_health"])) / config.hero_health,
        int(cast(int, observation["opponent_health"])) / config.hero_health,
        int(cast(int, observation["own_tower_health"])) / config.tower_health,
        int(cast(int, observation["enemy_tower_health"])) / config.tower_health,
        int(cast(int, observation["own_crystal_health"])) / config.crystal_health,
        int(cast(int, observation["enemy_crystal_health"])) / config.crystal_health,
    ]


def collect_dataset() -> Dataset:
    unique: dict[str, Sample] = {}
    raw_count = conflict_count = episode_count = 0
    for seed in range(128):
        for scripted_side in ("blue", "red"):
            arena = PixelArena()
            arena.reset(seed)
            scripted = ScriptedPolicy()
            other: Side = "red" if scripted_side == "blue" else "blue"
            random_policy = RandomPolicy(seed, other)
            while not arena.state.terminal:
                blue_legal = arena.legal_actions("blue")
                red_legal = arena.legal_actions("red")
                observation = arena.observe(scripted_side)
                if scripted_side == "blue":
                    blue = scripted.select("blue", blue_legal)
                    red = random_policy.select("red", red_legal)
                    selected, legal = blue, blue_legal
                else:
                    blue = random_policy.select("blue", blue_legal)
                    red = scripted.select("red", red_legal)
                    selected, legal = red, red_legal
                digest = observation_hash(observation)
                try:
                    action = ACTION_INDEX[_action_key(selected)]
                    legal_indices = frozenset(ACTION_INDEX[_action_key(item)] for item in legal)
                except KeyError as exc:
                    raise BCError("action outside fixed vocabulary") from exc
                raw_count += 1
                existing = unique.get(digest)
                if existing and existing.action != action:
                    raise BCError(f"conflicting label for observation {digest}")
                else:
                    unique[digest] = Sample(digest, observation, action, legal_indices)
                arena.step(blue, red)
            episode_count += 1
    if conflict_count:
        raise BCError(f"conflicting labels: {conflict_count}")
    by_action: dict[int, list[Sample]] = defaultdict(list)
    for sample in unique.values():
        by_action[sample.action].append(sample)
    for action, samples in by_action.items():
        samples.sort(key=lambda sample: sample.digest)
        train_end, validation_end = int(len(samples) * 0.70), int(len(samples) * 0.85)
        partitions = (("train", samples[:train_end]), ("validation", samples[train_end:validation_end]), ("test", samples[validation_end:]))
        if any(not partition for _, partition in partitions):
            raise BCError(f"action {action} cannot populate every split")
        for split, partition in partitions:
            for sample in partition:
                sample.split = split
    samples = sorted(unique.values(), key=lambda row: row.digest)
    return Dataset(PixelArena().config, samples, episode_count, raw_count, 0)


def _dataset_header(dataset: Dataset) -> dict[str, object]:
    split_counts = Counter(sample.split for sample in dataset.samples)
    class_counts = Counter(sample.action for sample in dataset.samples)
    return {
        "kind": "pixelarena_public_bc_v1",
        "claim_scope": "pixelarena_engineering",
        "hok_capability_claim": False,
        "gamecore_equivalence_claim": False,
        "environment_identity": dataset.config.identity,
        "config_hash": dataset.config.digest,
        "episode_count": dataset.episode_count,
        "raw_transition_count": dataset.raw_count,
        "unique_observation_count": len(dataset.samples),
        "duplicate_rate": 1.0 - len(dataset.samples) / dataset.raw_count,
        "label_conflicts": dataset.conflict_count,
        "feature_names": list(FEATURES),
        "action_vocabulary": [_action_document(index) for index in range(len(ACTIONS))],
        "split_counts": dict(sorted(split_counts.items())),
        "class_counts": {str(key): value for key, value in sorted(class_counts.items())},
    }


def write_dataset(path: Path, dataset: Dataset) -> None:
    documents = [_dataset_header(dataset)]
    documents.extend({"sample_hash": sample.digest, "split": sample.split, "observation": sample.observation, "action": _action_document(sample.action)} for sample in dataset.samples)
    _write(path, "".join(_json(row) + "\n" for row in documents))


def _tensors(dataset: Dataset, split: str) -> tuple[torch.Tensor, torch.Tensor, list[Sample]]:
    samples = [sample for sample in dataset.samples if sample.split == split]
    features = torch.tensor([_feature_vector(sample.observation, dataset.config) for sample in samples], dtype=torch.float32)
    labels = torch.tensor([sample.action for sample in samples], dtype=torch.long)
    return features, labels, samples


def _metrics(actor: StructuredActor, features: torch.Tensor, labels: torch.Tensor, samples: list[Sample]) -> dict[str, float]:
    with torch.no_grad():
        logits = actor(features)
        predictions = logits.argmax(dim=1)
        loss = nn.functional.cross_entropy(logits, labels).item()
    accuracy = (predictions == labels).float().mean().item()
    classes = sorted(set(labels.tolist()))
    balanced = sum((predictions[labels == label] == label).float().mean().item() for label in classes) / len(classes)
    counts = Counter(labels.tolist())
    majority = max(counts.values()) / len(labels)
    illegal = sum(int(prediction) not in sample.legal for prediction, sample in zip(predictions.tolist(), samples, strict=True)) / len(samples)
    return {"cross_entropy": loss, "exact_accuracy": accuracy, "balanced_accuracy": balanced, "majority_accuracy": majority, "illegal_top1_rate": illegal}


def train_one(dataset: Dataset, seed: int) -> tuple[StructuredActor, dict[str, object]]:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    actor = StructuredActor()
    train_x, train_y, _ = _tensors(dataset, "train")
    validation_x, validation_y, _ = _tensors(dataset, "validation")
    test_x, test_y, test_samples = _tensors(dataset, "test")
    initial_test_ce = float(nn.functional.cross_entropy(actor(test_x), test_y).item())
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-2)
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] = {}
    best_epoch = stale = 0
    for epoch in range(1, 301):
        optimizer.zero_grad()
        torch.autograd.backward(nn.functional.cross_entropy(actor(train_x), train_y))
        optimizer.step()
        with torch.no_grad():
            validation_loss = float(nn.functional.cross_entropy(actor(validation_x), validation_y))
        if validation_loss < best_loss - 1e-9:
            best_loss = validation_loss
            best_state = {key: value.detach().clone() for key, value in actor.state_dict().items()}
            best_epoch, stale = epoch, 0
        else:
            stale += 1
        if stale >= 30:
            break
    actor.load_state_dict(best_state)
    test = _metrics(actor, test_x, test_y, test_samples)
    test["initial_cross_entropy"] = initial_test_ce
    test["cross_entropy_ratio"] = test["cross_entropy"] / initial_test_ce
    passed = test["cross_entropy_ratio"] <= 0.50 and test["exact_accuracy"] >= 0.90 and test["balanced_accuracy"] >= 0.85 and test["exact_accuracy"] >= test["majority_accuracy"] + 0.20 and test["illegal_top1_rate"] <= 0.05
    return actor, {"seed": seed, "epochs": best_epoch, "validation_cross_entropy": best_loss, "test": test, "passed": passed}


def _model_document(actor: StructuredActor, dataset: Dataset, seed: int) -> dict[str, object]:
    weights = actor.state_dict()
    return {
        "kind": "pixelarena_structured_bc_mlp_v1",
        "claim_scope": "pixelarena_engineering",
        "hok_capability_claim": False,
        "gamecore_equivalence_claim": False,
        "environment_identity": dataset.config.identity,
        "config_hash": dataset.config.digest,
        "feature_names": list(FEATURES),
        "action_vocabulary": [_action_document(index) for index in range(len(ACTIONS))],
        "architecture": {"input": 10, "hidden": 32, "output": 6, "activation": "tanh"},
        "training_seed": seed,
        "weights": {key: value.detach().tolist() for key, value in weights.items()},
    }


def _numeric(value: object, shape: tuple[int, ...]) -> torch.Tensor:
    def valid(item: object) -> bool:
        if isinstance(item, list):
            return all(valid(child) for child in item)
        return type(item) in {int, float} and math.isfinite(float(cast(float, item)))

    if not valid(value):
        raise BCError("weights must contain finite JSON numbers")
    try:
        tensor = torch.tensor(value, dtype=torch.float32)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise BCError("invalid weight value") from exc
    if tuple(tensor.shape) != shape or not bool(torch.isfinite(tensor).all()):
        raise BCError(f"invalid weight shape or value: expected {shape}")
    return tensor


def _strict_equal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(cast(dict[object, object], actual)) == set(expected) and all(_strict_equal(cast(dict[object, object], actual)[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        actual_list = cast(list[object], actual)
        return len(actual_list) == len(expected) and all(_strict_equal(left, right) for left, right in zip(actual_list, expected, strict=True))
    return bool(actual == expected)


def _load_json(path: Path) -> object:
    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise BCError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    except json.JSONDecodeError as exc:
        raise BCError("invalid model JSON") from exc


def load_model(path: Path, config: ArenaConfig | None = None) -> tuple[StructuredActor, int]:
    raw = _load_json(path)
    if not isinstance(raw, dict) or set(raw) != MODEL_KEYS:
        raise BCError("invalid model fields")
    expected = config or PixelArena().config
    fixed = {
        "kind": "pixelarena_structured_bc_mlp_v1",
        "claim_scope": "pixelarena_engineering",
        "hok_capability_claim": False,
        "gamecore_equivalence_claim": False,
        "environment_identity": expected.identity,
        "config_hash": expected.digest,
        "feature_names": list(FEATURES),
        "action_vocabulary": [_action_document(index) for index in range(len(ACTIONS))],
        "architecture": {"input": 10, "hidden": 32, "output": 6, "activation": "tanh"},
    }
    if any(not _strict_equal(raw[key], value) for key, value in fixed.items()):
        raise BCError("model contract mismatch")
    if type(raw["training_seed"]) is not int:
        raise BCError("invalid training seed")
    weights = raw["weights"]
    if not isinstance(weights, dict) or set(weights) != {"hidden.weight", "hidden.bias", "output.weight", "output.bias"}:
        raise BCError("invalid weight fields")
    actor = StructuredActor()
    actor.load_state_dict({"hidden.weight": _numeric(weights["hidden.weight"], (32, 10)), "hidden.bias": _numeric(weights["hidden.bias"], (32,)), "output.weight": _numeric(weights["output.weight"], (6, 32)), "output.bias": _numeric(weights["output.bias"], (6,))})
    actor.eval()
    return actor, int(raw["training_seed"])


def _closed_loop(actor: StructuredActor, learned_side: Side, config: ArenaConfig) -> dict[str, object]:
    arena = PixelArena()
    arena.reset(101)
    null = NullPolicy()
    illegal = corrections = 0
    while not arena.state.terminal:
        legal = arena.legal_actions(learned_side)
        observation = arena.observe(learned_side)
        features = torch.tensor([_feature_vector(observation, config)], dtype=torch.float32)
        with torch.no_grad():
            logits = actor(features)[0]
        raw_index = int(logits.argmax().item())
        legal_by_index = {ACTION_INDEX[_action_key(item)]: item for item in legal}
        selected_index = max(legal_by_index, key=lambda index: float(logits[index]))
        selected = legal_by_index[selected_index]
        if raw_index not in legal_by_index:
            illegal += 1
        if selected_index != raw_index:
            corrections += 1
        if learned_side == "blue":
            blue, red = selected, null.select("red", arena.legal_actions("red"))
        else:
            blue, red = null.select("blue", arena.legal_actions("blue")), selected
        arena.step(blue, red)
    expected = f"{learned_side}_win_crystal_destroyed"
    return {
        "learned_side": learned_side,
        "ticks": arena.state.tick,
        "outcome": arena.state.outcome,
        "raw_illegal_actions": illegal,
        "mask_corrections": corrections,
        "passed": arena.state.outcome == expected and illegal == 0 and corrections == 0,
    }


def _json(document: object) -> str:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publish(temporary: Path, output: Path) -> None:
    if output.exists():
        raise BCError(f"output already exists: {output}")
    temporary.rename(output)


def accept_minimal_v2(output: Path) -> dict[str, object]:
    if output.exists():
        raise BCError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    stage = "collect"
    base: dict[str, object] = {
        "kind": "minimal_v2_structured_bc_report_v1",
        "claim_scope": "pixelarena_engineering",
        "hok_capability_claim": False,
        "gamecore_equivalence_claim": False,
        "device": "cpu",
        "parameters": {
            "episodes": 256,
            "epochs": 300,
            "patience": 30,
            "lr": 0.01,
            "seeds": [0, 1, 2],
        },
    }
    try:
        dataset = collect_dataset()
        dataset_path = temporary / "dataset.jsonl"
        write_dataset(dataset_path, dataset)
        header = _dataset_header(dataset)
        if dataset.episode_count != 256 or len(dataset.samples) < 400:
            raise BCError("dataset coverage gate failed")
        stage = "train"
        runs: list[dict[str, object]] = []
        model_files: dict[str, str] = {}
        for seed in TRAINING_SEEDS:
            actor, metrics = train_one(dataset, seed)
            model_path = temporary / f"model-seed-{seed}.json"
            _write(model_path, _json(_model_document(actor, dataset, seed)) + "\n")
            load_model(model_path, dataset.config)
            runs.append(metrics)
            model_files[model_path.name] = _sha(model_path)
        if not all(bool(run["passed"]) for run in runs):
            raise BCError("one or more training seeds failed")
        stage = "evaluate"
        best = min(runs, key=lambda run: float(cast(float, run["validation_cross_entropy"])))
        best_name = f"model-seed-{best['seed']}.json"
        best_actor, _ = load_model(temporary / best_name, dataset.config)
        closed_loop = [_closed_loop(best_actor, side, dataset.config) for side in ("blue", "red")]
        if not all(bool(scenario["passed"]) for scenario in closed_loop):
            raise BCError("closed-loop gate failed")
        stage = "report"
        report = {
            **base,
            "status": "PASSED",
            "environment_identity": dataset.config.identity,
            "config_hash": dataset.config.digest,
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "platform": platform.platform(),
                "threads": torch.get_num_threads(),
            },
            "dataset": header,
            "training_runs": runs,
            "best_model": best_name,
            "closed_loop": closed_loop,
            "files": {"dataset.jsonl": _sha(dataset_path), **model_files},
        }
        report_text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        _write(temporary / "report.json", report_text)
        _publish(temporary, output)
        return report
    except Exception as exc:
        for child in temporary.iterdir():
            if child.is_file():
                child.unlink()
        failure = {
            **base,
            "status": "FAILED",
            "stage": stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
        }
        _write(temporary / "report.json", json.dumps(failure, indent=2, sort_keys=True) + "\n")
        _publish(temporary, output)
        raise BCError(f"Minimal V2 failed at {stage}; report: {output / 'report.json'}") from exc
