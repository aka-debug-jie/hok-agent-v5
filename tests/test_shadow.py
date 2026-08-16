# ruff: noqa: E501, E702, I001
from __future__ import annotations

import json; import os
from pathlib import Path

import av
import numpy as np
import pytest

from hok_agent import shadow
from hok_agent.arena import PixelArena
from hok_agent.pixel import PixelActor, save_model


def _video(path: Path) -> None:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=5)
        stream.width = 128; stream.height = 128; stream.pix_fmt = "yuv420p"
        for value in (20, 100, 220):
            image = np.full((128, 128, 3), value, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_shadow_local_video_is_read_only_and_always_abstains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "recording.mp4"; model = tmp_path / "model.safetensors"; output = tmp_path / "result"
    _video(video); save_model(model, PixelActor(), PixelArena().config, 0)
    monkeypatch.setattr(shadow, "FROZEN_V3_MODEL_SHA256", shadow._sha_file(model))
    before = video.stat()

    summary = shadow.analyze_video(
        str(video), str(model), output, sample_every=1, max_frames=3
    )

    assert {path.name for path in output.iterdir()} == {"predictions.jsonl", "summary.json"}; rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]; assert len(rows) == 3
    assert all(row["advisory_action"] == "ABSTAIN" for row in rows)
    assert all(row["abstain_reason"] == "UNVALIDATED_COMMERCIAL_DOMAIN" for row in rows)
    assert all(row["control_output"] is False for row in rows)
    assert summary["abstain_count"] == len(rows); assert summary["advisory_count"] == 0; assert summary["real_domain_validated"] is False
    assert summary == json.loads((output / "summary.json").read_text()); assert str(video) not in (output / "predictions.jsonl").read_text(); after = video.stat()
    assert (before.st_ino, before.st_size, before.st_mtime_ns) == (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )


def test_shadow_rejects_non_file_inputs_before_decode(tmp_path: Path) -> None:
    video = tmp_path / "recording.mp4"; _video(video); directory = tmp_path / "directory.mp4"; directory.mkdir()
    symlink = tmp_path / "link.mp4"; symlink.symlink_to(video); fifo = tmp_path / "fifo.mp4"; os.mkfifo(fifo)
    invalid = ("0", "https://example.invalid/a.mp4", "/dev/null", str(directory), str(symlink), str(fifo))
    for raw in invalid:
        with pytest.raises(shadow.ShadowError):
            shadow.analyze_video(raw, "missing.safetensors", tmp_path / f"out-{len(raw)}")


def test_shadow_refuses_existing_output(tmp_path: Path) -> None:
    video = tmp_path / "recording.mp4"; model = tmp_path / "model.safetensors"; output = tmp_path / "existing"
    _video(video); save_model(model, PixelActor(), PixelArena().config, 1)
    output.mkdir()
    with pytest.raises(shadow.ShadowError, match="already exists"):
        shadow.analyze_video(str(video), str(model), output)


def test_shadow_rejects_non_frozen_model_and_mislabeled_playlist(tmp_path: Path) -> None:
    video = tmp_path / "recording.mp4"; model = tmp_path / "model.safetensors"
    _video(video); save_model(model, PixelActor(), PixelArena().config, 2)
    with pytest.raises(shadow.ShadowError, match="frozen promoted V3"):
        shadow.analyze_video(str(video), str(model), tmp_path / "output")

    playlist = tmp_path / "playlist.mp4"; playlist.write_text("#EXTM3U\nhttps://example.invalid/segment.ts\n", encoding="utf-8")
    with playlist.open("rb") as source, pytest.raises(shadow.ShadowError):
        shadow._decode_video(source, "mov", 1, 1)
