# ruff: noqa: E501, E701, E702
from __future__ import annotations

import inspect
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from hok_agent import cli, pixel
from hok_agent.arena import DEFAULT_CONFIG, ArenaConfig, PixelArena


@pytest.fixture(scope='module')
def formal_data() -> pixel.PixelData:
    return pixel.collect_pixel_data()
def test_collect_pixel_data_small_range_is_dtype_shape_stable_and_renders_consistent() -> None:
    data = pixel.collect_pixel_data(range(4), variants=2, enforce=False); assert data.frames.shape[1:] == (128, 128, 3); assert data.frames.dtype == np.uint8; assert data.actions.dtype == np.uint8; assert data.group_ids.dtype == 'S64'; assert data.ticks.dtype == np.uint16; assert data.render_seeds.dtype == np.int64; assert data.splits.dtype == np.uint8; assert data.frame_hashes.dtype == 'S64'; assert data.sources.dtype == np.uint8; assert data.legal.dtype == np.bool_
    for sample_group in set(data.group_ids.tolist()):
        indices = np.where(data.group_ids == sample_group)[0]; splits = set(int(value) for value in data.splits[indices].tolist()); assert len(splits) == 1; split = splits.pop(); variant_seeds = set(int(value) for value in data.render_seeds[indices]); assert len(variant_seeds) == 2
        for render_seed in variant_seeds:
            variant_indices = indices[data.render_seeds[indices] == render_seed]; assert set(int(value) for value in data.splits[variant_indices]) == {split}
def test_formal_collection_has_256_episodes_and_split_action_coverage_and_completion(formal_data: pixel.PixelData) -> None:
    assert len(formal_data.episodes) == 256
    for split in pixel.SPLITS.values():
        split_actions = formal_data.actions[formal_data.splits == split]; counts = Counter(split_actions.tolist())
        for action in range(len(pixel.ACTIONS)):
            assert counts.get(action, 0) >= 20
    completion = sum(episode.completed for episode in formal_data.episodes) / len(formal_data.episodes); assert completion >= 0.95
def test_load_dataset_roundtrip_uses_allow_pickle_false_and_rejects_extra_fields(tmp_path: Path) -> None:
    data = pixel.collect_pixel_data(range(4), variants=1, enforce=False); dataset_path = tmp_path / 'dataset.npz'; pixel.write_dataset(dataset_path, data); loaded = pixel.load_dataset(dataset_path); assert set(loaded) == pixel.DATA_KEYS; assert all(value is not None for value in loaded.values())
    with np.load(dataset_path, allow_pickle=False) as archive:
        good = {name: archive[name] for name in archive.files}
    bad = dict(good); bad['extra'] = np.asarray([1], dtype=np.uint8); np.savez_compressed(tmp_path / 'dataset-extra.npz', **bad)
    with pytest.raises(pixel.PixelError, match='invalid pixel dataset fields'):
        pixel.load_dataset(tmp_path / 'dataset-extra.npz')
def test_load_dataset_rejects_wrong_dataset_dtype(tmp_path: Path) -> None:
    data = pixel.collect_pixel_data(range(4), variants=1, enforce=False); dataset_path = tmp_path / 'dataset-bad-dtype.npz'; dataset_valid = tmp_path / 'dataset-valid.npz'; pixel.write_dataset(dataset_valid, data); arrays = pixel.load_dataset(dataset_valid); bad = dict(arrays); bad['actions'] = bad['actions'].astype(np.int32); np.savez_compressed(dataset_path, **bad)
    with pytest.raises(pixel.PixelError, match='invalid pixel dataset dtype'):
        pixel.load_dataset(dataset_path)
    bad = dict(arrays); bad['frame_hashes'] = bad['frame_hashes'].copy(); bad['frame_hashes'][0] = b'0' * 64; np.savez_compressed(tmp_path / 'dataset-bad-hash.npz', **bad)
    with pytest.raises(pixel.PixelError, match='pixel frame hash mismatch'):
        pixel.load_dataset(tmp_path / 'dataset-bad-hash.npz')
def test_mismatched_control_pairs_every_label_with_a_different_class() -> None:
    labels = np.repeat(np.arange(len(pixel.ACTIONS), dtype=np.uint8), 2); frames = np.stack([np.full((2, 2, 1), label, dtype=np.uint8) for label in labels]); mismatched = pixel._mismatched_frames(frames, labels); observed_source_labels = mismatched[:, 0, 0, 0]; assert np.all(observed_source_labels != labels)
def test_pixel_actor_forward_contract_and_safetensors_roundtrip_and_tamper_guards(tmp_path: Path) -> None:
    signature = inspect.signature(pixel.PixelActor.forward); assert list(signature.parameters) == ['self', 'frames']; actor = pixel.PixelActor(); assert sum(param.numel() for param in actor.parameters()) <= 12000000; config = PixelArena().config; model_path = tmp_path / 'model.safetensors'; pixel.save_model(model_path, actor, config, 0); loaded, seed = pixel.load_model(model_path, config); assert isinstance(loaded, pixel.PixelActor); assert seed == 0; logits = loaded(torch.zeros((1, 3, 128, 128), dtype=torch.float32)); assert logits.shape == (1, len(pixel.ACTIONS)); base_state = load_file(model_path, device='cpu'); base_metadata = pixel._model_metadata(config, 0); assert base_metadata['training_contract_hash'] == pixel.TRAINING_HASH; metadata = dict(base_metadata); metadata['arena_config_hash'] = '0' * 64; save_file(base_state, tmp_path / 'bad-metadata.safetensors', metadata=metadata)
    with pytest.raises(pixel.PixelError, match='pixel model contract mismatch'):
        pixel.load_model(tmp_path / 'bad-metadata.safetensors', config)
    metadata = dict(base_metadata); metadata['action_vocabulary_hash'] = 'bad'; save_file(base_state, tmp_path / 'bad-action.safetensors', metadata=metadata)
    with pytest.raises(pixel.PixelError, match='pixel model contract mismatch'):
        pixel.load_model(tmp_path / 'bad-action.safetensors', config)
    raw = json.loads(Path(DEFAULT_CONFIG).read_text(encoding='utf-8')); raw['hero_health'] = raw['hero_health'] + 1; bad_config_path = tmp_path / 'config-alt.json'; bad_config_path.write_text(json.dumps(raw, sort_keys=True), encoding='utf-8')
    with pytest.raises(pixel.PixelError, match='pixel model contract mismatch'):
        pixel.load_model(model_path, ArenaConfig.load(bad_config_path))
    shape_state = dict(base_state); first_key = next((key for key, tensor in shape_state.items() if tensor.ndim > 0)); value = shape_state[first_key]; shape_state[first_key] = value[..., :max(1, value.shape[-1] - 1)]; save_file(shape_state, tmp_path / 'bad-shape.safetensors', metadata=base_metadata)
    with pytest.raises(pixel.PixelError, match='invalid pixel model tensors'):
        pixel.load_model(tmp_path / 'bad-shape.safetensors', config)
def test_accept_pixel_v3_cpu_smoke_returns_non_promoting_report() -> None:
    report = pixel.accept_pixel_v3(None, 'cpu', True); assert report['status'] == 'PASSED'; assert report['disposition'] == 'NON_PROMOTING_CPU_SMOKE'; assert report['parameters'] <= 12000000; assert set(report['dataset_arrays']) == pixel.DATA_KEYS; assert report['model_seed'] == 0
def test_pixel_cli_rejects_illegal_combinations_and_existing_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert cli.main(['accept-pixel-v3', '--device', 'cpu', '--smoke', '--output-dir', str(tmp_path / 'smoke')]) == 2; output = tmp_path / 'formal-output'; output.mkdir(); monkeypatch.setattr(pixel.torch.cuda, 'is_available', lambda: True); assert cli.main(['accept-pixel-v3', '--device', 'cuda', '--output-dir', str(output)]) == 2
def test_minimal_training_one_epoch_smoke_path() -> None:
    data = pixel.collect_pixel_data(range(4), variants=1, enforce=False); actor, metrics = pixel.train_actor(data, 0, torch.device('cpu'), epochs=1, batch_size=16, patience=1); assert isinstance(actor, pixel.PixelActor); assert 'validation' in metrics; assert isinstance(metrics['validation'], dict)
