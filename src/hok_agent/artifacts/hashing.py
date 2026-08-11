"""Canonical content hashing for reproducible, inspectable artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path


def sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def canonical_json_bytes(value: object, *, exclude_key: str | None = None) -> bytes:
    """Encode JSON canonically, optionally excluding a self-hash field."""

    document = value
    if exclude_key is not None:
        if not isinstance(value, Mapping):
            raise TypeError("self-hashed JSON artifact must be an object")
        document = {key: item for key, item in value.items() if key != exclude_key}
    encoded = json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return encoded.encode("utf-8")


def sha256_json(value: object, *, exclude_key: str | None = None) -> str:
    return sha256_bytes(canonical_json_bytes(value, exclude_key=exclude_key))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"
