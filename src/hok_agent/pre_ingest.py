# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
import os
import stat
from bisect import bisect_left, bisect_right
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Final, cast

SCHEMA_V1: Final = 'hok-agent-v5-pre-ingest-v1'
SCHEMA: Final = 'hok-agent-v5-pre-ingest-v2'
MIN_CANDIDATES: Final = 12
SAMPLE_PERIOD_US: Final = 1000000
MAX_FINGERPRINTS: Final = 2048
MAX_HASH_WORKERS: Final = 8
RELATIONS: Final = ('exact', 'reencode', 'overlap', 'near_duplicate', 'uncertain')
READY = 'READY_FOR_COMPONENT_SPLIT'
REVIEW_STATUS = 'OPERATOR_ATTESTED_FILE_ATOMIC_COMPONENTS'
HEX = frozenset('0123456789abcdef')
ALGORITHM_SPEC_V1: Final[dict[str, object]] = {'candidate_unit': 'complete_mp4_file', 'scan_order': 'lexicographic_relative_locator_not_persisted', 'input_integrity': {'same_open_descriptor_for_hash_and_decode': True, 'stat_fields_before_hash_after_hash_after_decode': ['device', 'inode', 'mode', 'size', 'mtime_ns', 'ctime_ns'], 'raw_reread_for_integrity': False}, 'sampling': {'mode': 'online_power_of_two_temporal_stride_with_terminal_sample', 'base_period_us': SAMPLE_PERIOD_US, 'maximum_fingerprints': MAX_FINGERPRINTS, 'anchor': 'first_decoded_pts', 'selection': 'first_frame_at_or_after_next_target', 'compaction': 'at_2049_keep_indices_0_2_through_2048_then_double_period', 'next_target': 'last_retained_pts_plus_current_period', 'terminal': 'append_last_decoded_frame_or_replace_last_fingerprint_at_capacity'}, 'rotation_degrees': [0, 90, 180, 270], 'privacy_transform': {'colorspace': 'gray8', 'fixed_size': [9, 9], 'mask': 'fixed_zero_mask', 'masked_pixels': 0, 'raw_frames_persisted': False, 'audio_decoded_or_persisted': False, 'source_locators_persisted': False}, 'fingerprint': {'dhash_bits': 64, 'luma_cells': 16, 'luma_quantization': 'integer_cell_mean'}, 'thresholds': {'sample_match_dhash_hamming_max': 10, 'sample_match_luma_delta_max': 18, 'reencode_min_matches': 4, 'reencode_match_fraction_ppm': 900000, 'reencode_shorter_coverage_ppm': 900000, 'reencode_duration_delta_us': 2000000, 'overlap_min_matches': 4, 'overlap_match_fraction_ppm': 800000, 'overlap_shorter_coverage_ppm': 300000, 'near_duplicate_min_matches': 4, 'near_duplicate_match_fraction_ppm': 600000, 'uncertain_min_matches': 3, 'uncertain_match_fraction_ppm': 400000}}
ALGORITHM_SPEC: Final[dict[str, object]] = {'candidate_unit': 'complete_mp4_file', 'scan_order': 'lexicographic_relative_path', 'input_integrity': {'same_open_descriptor_for_identity': True, 'stat_fields_before_after': ['device', 'inode', 'mode', 'size', 'mtime_ns', 'ctime_ns'], 'content_hashing': False, 'decode_enabled': False}, 'relation_mode': 'operator_attested_file_atomic_no_duplicate_v1', 'raw_frames_persisted': False, 'audio_decoded_or_persisted': False}
RELATIONSHIP_MODE: Final = 'operator_attested_file_atomic_no_duplicate_v1'

class PreIngestError(ValueError):
    pass

@dataclass(frozen=True)
class PreIngestEvidence:
    pre_ingest_sha256: str
    disposition: str
    component_of: dict[str, str]

@dataclass(frozen=True)
class _Fingerprint:
    pts_us: int
    dhash: int
    luma: tuple[int, ...]

@dataclass(frozen=True)
class _Candidate:
    candidate_id: str
    content_sha256: str
    pts_range_us: tuple[int, int] | None
    samples: tuple[_Fingerprint, ...]
    blocker: str | None

def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()

def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

def _is_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and (set(value) <= HEX)

def _open_regular(path: Path) -> tuple[int, os.stat_result]:
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
            raise PreIngestError('INPUT_NOT_REGULAR')
        descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
        after = os.fstat(descriptor)
        if not stat.S_ISREG(after.st_mode) or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            os.close(descriptor)
            raise PreIngestError('INPUT_CHANGED')
        return (descriptor, after)
    except PreIngestError:
        raise
    except OSError as exc:
        raise PreIngestError('INPUT_UNREADABLE') from exc

def _stat_signature(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

def _assert_unchanged(descriptor: int, expected: os.stat_result) -> None:
    try:
        current = os.fstat(descriptor)
    except OSError as exc:
        raise PreIngestError('INPUT_CHANGED') from exc
    if not stat.S_ISREG(current.st_mode) or _stat_signature(current) != _stat_signature(expected):
        raise PreIngestError('INPUT_CHANGED')

def _read_regular(path: Path) -> bytes:
    descriptor, size = _open_regular(path)
    if size.st_size > 16 * 1024 * 1024:
        os.close(descriptor)
        raise PreIngestError('ARTIFACT_TOO_LARGE')
    try:
        with os.fdopen(descriptor, 'rb') as handle:
            return handle.read()
    except OSError as exc:
        raise PreIngestError('ARTIFACT_UNREADABLE') from exc

def _strict_json(data: bytes) -> dict[str, object]:

    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise PreIngestError('ARTIFACT_INVALID_JSON')
            result[key] = value
        return result
    try:
        value = json.loads(data.decode('utf-8'), object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(PreIngestError('ARTIFACT_INVALID_JSON')))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreIngestError('ARTIFACT_INVALID_JSON') from exc
    if type(value) is not dict:
        raise PreIngestError('ARTIFACT_INVALID_SCHEMA')
    return cast(dict[str, object], value)

def _rotation(stream: object) -> int:
    metadata = getattr(stream, 'metadata', {})
    raw = metadata.get('rotate', '0') if isinstance(metadata, dict) else '0'
    try:
        rotation = int(str(raw)) % 360
    except ValueError as exc:
        raise PreIngestError('ROTATION_UNSUPPORTED') from exc
    if rotation not in (0, 90, 180, 270):
        raise PreIngestError('ROTATION_UNSUPPORTED')
    return rotation

def _rotated_grid(grid: tuple[tuple[int, ...], ...], rotation: int) -> tuple[tuple[int, ...], ...]:
    if rotation == 0:
        return grid
    if rotation == 90:
        return tuple(tuple(grid[8 - column][row] for column in range(9)) for row in range(9))
    if rotation == 180:
        return tuple(tuple(grid[8 - row][8 - column] for column in range(9)) for row in range(9))
    return tuple(tuple(grid[column][8 - row] for column in range(9)) for row in range(9))

def _fingerprint(frame: Any, pts_us: int, rotation: int) -> _Fingerprint:
    small = frame.reformat(width=9, height=9, format='gray')
    plane = small.planes[0]
    raw = bytes(plane)
    stride = plane.line_size
    grid = tuple(tuple(raw[row * stride + column] for column in range(9)) for row in range(9))
    grid = _rotated_grid(grid, rotation)
    dhash = 0
    for row in range(8):
        for column in range(8):
            dhash = dhash << 1 | int(grid[row][column] > grid[row][column + 1])
    luma = tuple(sum(grid[row + dy][column + dx] for dy in range(2) for dx in range(2)) // 4 for row in range(0, 8, 2) for column in range(0, 8, 2))
    return _Fingerprint(pts_us, dhash, luma)

def _decode(handle: BinaryIO) -> tuple[tuple[int, int], tuple[_Fingerprint, ...]]:
    import av
    try:
        handle.seek(0)
        with av.open(handle, mode='r') as container:
            videos = list(container.streams.video)
            if len(videos) != 1:
                raise PreIngestError('VIDEO_STREAM_COUNT_UNSUPPORTED')
            stream = videos[0]
            rotation = _rotation(stream)
            first: int | None = None
            last: int | None = None
            previous: int | None = None
            next_sample: int | None = None
            period = SAMPLE_PERIOD_US
            samples: list[_Fingerprint] = []
            terminal: tuple[Any, int] | None = None
            for frame in container.decode(stream):
                if frame.pts is None or frame.time_base is None:
                    raise PreIngestError('PTS_MISSING')
                base = frame.time_base
                pts_us = int(frame.pts) * int(base.numerator) * 1000000 // int(base.denominator)
                if previous is not None and pts_us <= previous:
                    raise PreIngestError('PTS_NOT_STRICTLY_ORDERED')
                first = pts_us if first is None else first
                last = pts_us
                previous = pts_us
                terminal = (frame, pts_us)
                next_sample = pts_us if next_sample is None else next_sample
                if pts_us >= next_sample:
                    samples.append(_fingerprint(frame, pts_us, rotation))
                    next_sample = pts_us + period
                    if len(samples) > MAX_FINGERPRINTS:
                        samples = samples[::2]
                        period *= 2
                        next_sample = samples[-1].pts_us + period
            if first is None or last is None or not samples or terminal is None:
                raise PreIngestError('VIDEO_EMPTY')
            if samples[-1].pts_us != last:
                final_sample = _fingerprint(terminal[0], terminal[1], rotation)
                if len(samples) == MAX_FINGERPRINTS:
                    samples[-1] = final_sample
                else:
                    samples.append(final_sample)
            return ((first, last), tuple(samples))
    except PreIngestError:
        raise
    except Exception as exc:
        raise PreIngestError('VIDEO_DECODE_REJECTED') from exc
    raise PreIngestError('VIDEO_EMPTY')

def _scan(root: Path) -> list[Path]:
    try:
        info = root.lstat()
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise PreIngestError('INPUT_ROOT_NOT_DIRECTORY')
        found: list[Path] = []
        pending = [root]
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(path)
                    elif entry.name.lower().endswith('.mp4') and entry.is_file(follow_symlinks=False):
                        found.append(path)
        return sorted(found, key=lambda path: path.relative_to(root).as_posix())
    except PreIngestError:
        raise
    except OSError as exc:
        raise PreIngestError('INPUT_ROOT_UNREADABLE') from exc

def _candidate(path: Path, root: Path) -> _Candidate:
    descriptor, opened = _open_regular(path)
    try:
        relative = path.relative_to(root).as_posix()
        signature = _stat_signature(opened)
        identity_sha = _sha(_canonical(['file-atomic-stat-v1', signature]))
        candidate_id = _sha(_canonical(['candidate-v2-file-atomic', relative, signature]))
        _assert_unchanged(descriptor, opened)
        return _Candidate(candidate_id, identity_sha, None, (), None)
    except OSError as exc:
        raise PreIngestError('INPUT_CHANGED') from exc
    finally:
        os.close(descriptor)

def _candidate_list(paths: list[Path], root: Path) -> list[_Candidate]:
    if not paths:
        return []
    with ThreadPoolExecutor(max_workers=min(MAX_HASH_WORKERS, len(paths))) as pool:
        futures = {pool.submit(_candidate, path, root): path for path in paths}
        ordered = sorted(
            ((path, future.result()) for future, path in futures.items()), key=lambda item: item[0].as_posix()
        )
    return [candidate for _, candidate in ordered]

def _sample_distance(left: _Fingerprint, right: _Fingerprint) -> tuple[int, int]:
    hamming = (left.dhash ^ right.dhash).bit_count()
    luma = sum((abs(a - b) for a, b in zip(left.luma, right.luma, strict=True))) // 16
    return (hamming, luma)

def _period_us(samples: tuple[_Fingerprint, ...]) -> int:
    values = [right.pts_us - left.pts_us for left, right in zip(samples, samples[1:], strict=False)]
    return sorted(values)[len(values) // 2] if values else SAMPLE_PERIOD_US


def _temporal_comparable(
    left: tuple[_Fingerprint, ...], right_pts: list[int], offset: int, tolerance: int
) -> int:
    """Count deterministic one-to-one PTS pairs before inspecting their pixels."""
    used: set[int] = set()
    count = 0
    for sample in left:
        target = sample.pts_us + offset
        start = bisect_left(right_pts, target - tolerance)
        stop = bisect_right(right_pts, target + tolerance)
        choices = [(abs(right_pts[index] - target), index) for index in range(start, stop) if index not in used]
        if choices:
            used.add(min(choices)[1])
            count += 1
    return count


def _pair_metrics(left: _Candidate, right: _Candidate) -> dict[str, int]:
    """Match on decoded PTS, never on mutable sampling-array positions."""
    a, b = (left.samples, right.samples)
    if not a or not b:
        return {'compared': 0, 'matched': 0, 'coverage': 0, 'fraction': 0, 'offset': 0, 'hamming': 64, 'luma': 255}
    left_period, right_period = _period_us(a), _period_us(b)
    quantum = max(1, min(left_period, right_period) // 2)
    anchor_step = max(1, len(a) // 64)
    offsets: dict[int, int] = {}
    for sample in a[::anchor_step]:
        for other in b:
            hamming, luma = _sample_distance(sample, other)
            if hamming <= 10 and luma <= 18:
                offset = other.pts_us - sample.pts_us
                bucket = round(offset / quantum) * quantum
                offsets[bucket] = offsets.get(bucket, 0) + 1
    choices = sorted(offsets, key=lambda value: (-offsets[value], abs(value), value))[:8] or [0]
    pts = [sample.pts_us for sample in b]
    tolerance = max(left_period, right_period) // 2
    best: tuple[int, int, list[int], list[int]] | None = None
    for offset in choices:
        used, hammings, lumas = set(), [], []
        for sample in a:
            target = sample.pts_us + offset
            start, stop = bisect_left(pts, target - tolerance), bisect_right(pts, target + tolerance)
            options = [
                (sum(_sample_distance(sample, b[index])), abs(pts[index] - target), index)
                for index in range(start, stop)
                if index not in used and _sample_distance(sample, b[index])[0] <= 10 and _sample_distance(sample, b[index])[1] <= 18
            ]
            if not options:
                continue
            _, _, index = min(options)
            hamming, luma = _sample_distance(sample, b[index])
            used.add(index)
            hammings.append(hamming)
            lumas.append(luma)
        candidate = (len(used), offset, hammings, lumas)
        if best is None or (candidate[0], -abs(candidate[1]), -candidate[1]) > (best[0], -abs(best[1]), -best[1]):
            best = candidate
    assert best is not None
    matched, offset, hammings, lumas = best
    compared = min(len(a), len(b))
    comparable = _temporal_comparable(a, pts, offset, tolerance)
    ordered_hamming, ordered_luma = sorted(hammings), sorted(lumas)
    return {
        'compared': compared,
        'matched': matched,
        'coverage': matched * 1000000 // compared,
        'fraction': matched * 1000000 // comparable if comparable else 0,
        'offset': offset // max(1, min(left_period, right_period)),
        'hamming': ordered_hamming[len(ordered_hamming) // 2] if matched else 64,
        'luma': ordered_luma[len(ordered_luma) // 2] if matched else 255,
    }

def _classify(left: _Candidate, right: _Candidate) -> tuple[str | None, dict[str, object]]:
    exact = left.content_sha256 == right.content_sha256
    metrics = _pair_metrics(left, right)
    relation: str | None = None
    if exact:
        relation = 'exact'
    elif left.pts_range_us is not None and right.pts_range_us is not None:
        left_duration = left.pts_range_us[1] - left.pts_range_us[0]
        right_duration = right.pts_range_us[1] - right.pts_range_us[0]
        duration_close = abs(left_duration - right_duration) <= 2000000
        if duration_close and metrics['matched'] >= 4 and (metrics['fraction'] >= 900000) and (metrics['coverage'] >= 900000):
            relation = 'reencode'
        elif metrics['matched'] >= 4 and metrics['fraction'] >= 800000 and (metrics['coverage'] >= 300000):
            relation = 'overlap'
        elif metrics['matched'] >= 4 and metrics['fraction'] >= 600000:
            relation = 'near_duplicate'
        elif metrics['matched'] >= 3 and metrics['fraction'] >= 400000:
            relation = 'uncertain'
    private_evidence = [left.candidate_id, right.candidate_id, metrics, [[item.pts_us, item.dhash, item.luma] for item in left.samples], [[item.pts_us, item.dhash, item.luma] for item in right.samples]]
    evidence: dict[str, object] = {'whole_file_sha256_equal': exact, 'compared_samples': metrics['compared'], 'matched_samples': metrics['matched'], 'shorter_coverage_ppm': metrics['coverage'], 'match_fraction_ppm': metrics['fraction'], 'offset_samples': metrics['offset'], 'median_dhash_hamming': metrics['hamming'], 'median_luma_delta': metrics['luma'], 'evidence_sha256': _sha(_canonical(private_evidence))}
    return (relation, evidence)

class _Groups:

    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        a, b = (self.find(left), self.find(right))
        if a != b:
            self.parent[max(a, b)] = min(a, b)

def _component_map(candidate_ids: list[str], edges: list[tuple[str, str]]) -> dict[str, str]:
    groups = _Groups(candidate_ids)
    for left, right in edges:
        groups.union(left, right)
    members: dict[str, list[str]] = {}
    for candidate_id in candidate_ids:
        members.setdefault(groups.find(candidate_id), []).append(candidate_id)
    component_id = {root: _sha(_canonical(['component-v1', sorted(values)])) for root, values in members.items()}
    return {candidate_id: component_id[groups.find(candidate_id)] for candidate_id in candidate_ids}

def _write_exclusive(path: Path, data: bytes) -> None:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 384)
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise PreIngestError('OUTPUT_NOT_EXCLUSIVE') from exc

def _disposition(component_count: int, uncertain: int, blockers: list[str]) -> str:
    if uncertain:
        return 'BLOCKED_UNCERTAIN'
    if component_count < MIN_CANDIDATES:
        return 'BLOCKED_LT_12_COMPONENTS'
    return READY if not blockers else 'BLOCKED_DECODE'

def pre_ingest(input_root: Path, output: Path) -> dict[str, object]:
    paths = _scan(input_root)
    candidates = _candidate_list(paths, input_root)
    relations: list[dict[str, object]] = []
    ids = [candidate.candidate_id for candidate in candidates]
    component_of = {candidate_id: candidate_id for candidate_id in ids}
    component_count = len(component_of)
    blockers = sorted([candidate.blocker for candidate in candidates if candidate.blocker is not None])
    payload: dict[str, object] = {'schema_version': SCHEMA, 'algorithm_spec': ALGORITHM_SPEC, 'relationship_mode': RELATIONSHIP_MODE, 'candidate_count': len(candidates), 'component_count': component_count, 'candidates': [{'candidate_id': candidate.candidate_id, 'component_id': component_of[candidate.candidate_id], 'pts_range_us': list(candidate.pts_range_us) if candidate.pts_range_us else None} for candidate in candidates], 'relations': relations, 'component_of': component_of, 'uncertain_relation_count': 0, 'blockers': blockers, 'review_status': REVIEW_STATUS, 'disposition': _disposition(component_count, 0, blockers)}
    payload['pre_ingest_sha256'] = _sha(_canonical(payload))
    _write_exclusive(output, _canonical(payload) + b'\n')
    return payload
_FIELDS_V1 = {'schema_version', 'algorithm_spec', 'candidate_count', 'component_count', 'candidates', 'relations', 'component_of', 'uncertain_relation_count', 'blockers', 'review_status', 'disposition', 'pre_ingest_sha256'}
_FIELDS_V2 = {'schema_version', 'algorithm_spec', 'relationship_mode', 'candidate_count', 'component_count', 'candidates', 'relations', 'component_of', 'uncertain_relation_count', 'blockers', 'review_status', 'disposition', 'pre_ingest_sha256'}
_EVIDENCE_FIELDS = {'whole_file_sha256_equal', 'compared_samples', 'matched_samples', 'shorter_coverage_ppm', 'match_fraction_ppm', 'offset_samples', 'median_dhash_hamming', 'median_luma_delta', 'evidence_sha256'}

def load_pre_ingest(path: Path) -> PreIngestEvidence:
    data = _read_regular(path)
    payload = _strict_json(data)
    schema = payload.get('schema_version')
    fields = (
        _FIELDS_V2
        if schema == SCHEMA
        else _FIELDS_V1
        if schema == SCHEMA_V1
        else None
    )
    if fields is None or set(payload) != fields:
        raise PreIngestError('ARTIFACT_INVALID_SCHEMA')
    if schema == SCHEMA:
        if payload['algorithm_spec'] != ALGORITHM_SPEC or payload['relationship_mode'] != RELATIONSHIP_MODE:
            raise PreIngestError('ARTIFACT_INVALID_SCHEMA')
    elif payload['algorithm_spec'] != ALGORITHM_SPEC_V1:
        raise PreIngestError('ARTIFACT_INVALID_SCHEMA')
    supplied = payload['pre_ingest_sha256']
    unsigned = {key: value for key, value in payload.items() if key != 'pre_ingest_sha256'}
    if not _is_sha(supplied) or supplied != _sha(_canonical(unsigned)) or data != _canonical(payload) + b'\n':
        raise PreIngestError('ARTIFACT_SELF_HASH_MISMATCH')
    candidate_count = payload['candidate_count']
    component_count = payload['component_count']
    candidates = payload['candidates']
    component_value = payload['component_of']
    relations = payload['relations']
    blockers = payload['blockers']
    review_status = payload['review_status']
    uncertain_count = payload['uncertain_relation_count']
    if type(candidate_count) is not int or candidate_count < 0 or type(component_count) is not int or (component_count < 0) or (type(candidates) is not list) or (len(candidates) != candidate_count) or (type(component_value) is not dict) or (type(relations) is not list) or (type(blockers) is not list) or any(type(item) is not str for item in blockers) or (blockers != sorted(set(blockers))) or (type(uncertain_count) is not int) or (uncertain_count < 0) or (review_status != REVIEW_STATUS):
        raise PreIngestError('ARTIFACT_INVALID_SCHEMA')
    component_of = cast(dict[object, object], component_value)
    if any((not _is_sha(key) or not _is_sha(value) for key, value in component_of.items())):
        raise PreIngestError('ARTIFACT_INVALID_COMPONENTS')
    typed_components = cast(dict[str, str], component_value)
    ids: list[str] = []
    for value in candidates:
        if type(value) is not dict or set(value) != {'candidate_id', 'component_id', 'pts_range_us'}:
            raise PreIngestError('ARTIFACT_INVALID_CANDIDATE')
        row = cast(dict[str, object], value)
        candidate_id, component_id, pts = (row['candidate_id'], row['component_id'], row['pts_range_us'])
        if not _is_sha(candidate_id) or not _is_sha(component_id) or typed_components.get(cast(str, candidate_id)) != component_id:
            raise PreIngestError('ARTIFACT_INVALID_CANDIDATE')
        if pts is not None and (type(pts) is not list or len(pts) != 2 or any(type(item) is not int for item in pts) or (pts[0] > pts[1])):
            raise PreIngestError('ARTIFACT_INVALID_CANDIDATE')
        ids.append(cast(str, candidate_id))
    if len(set(ids)) != len(ids) or set(ids) != set(typed_components):
        raise PreIngestError('ARTIFACT_INVALID_COMPONENTS')
    if schema == SCHEMA:
        if relations:
            raise PreIngestError('ARTIFACT_INVALID_RELATION')
        if uncertain_count != 0 or component_count != len(set(typed_components)):
            raise PreIngestError('ARTIFACT_INVALID_COMPONENTS')
        if len(set(typed_components.values())) != len(typed_components):
            raise PreIngestError('ARTIFACT_INVALID_COMPONENTS')
        actual_uncertain = 0
    else:
        edges: list[tuple[str, str]] = []
        actual_uncertain = 0
        seen_pairs: set[tuple[str, str]] = set()
        for value in relations:
            if type(value) is not dict or set(value) != {'left_candidate_id', 'right_candidate_id', 'relation', 'evidence'}:
                raise PreIngestError('ARTIFACT_INVALID_RELATION')
            row = cast(dict[str, object], value)
            left, right, relation, evidence = (row['left_candidate_id'], row['right_candidate_id'], row['relation'], row['evidence'])
            if not isinstance(left, str) or not isinstance(right, str) or left >= right or (left not in typed_components) or (right not in typed_components) or (relation not in RELATIONS) or ((left, right) in seen_pairs) or (type(evidence) is not dict) or (set(evidence) != _EVIDENCE_FIELDS):
                raise PreIngestError('ARTIFACT_INVALID_RELATION')
            details = cast(dict[str, object], evidence)
            numeric = [details[key] for key in _EVIDENCE_FIELDS - {'whole_file_sha256_equal', 'evidence_sha256'}]
            if type(details['whole_file_sha256_equal']) is not bool or not _is_sha(details['evidence_sha256']) or any(type(item) is not int for item in numeric) or any(cast(int, details[key]) < 0 for key in ('compared_samples', 'matched_samples', 'shorter_coverage_ppm', 'match_fraction_ppm', 'median_dhash_hamming', 'median_luma_delta')) or (cast(int, details['shorter_coverage_ppm']) > 1000000) or (cast(int, details['match_fraction_ppm']) > 1000000) or (cast(int, details['matched_samples']) > cast(int, details['compared_samples'])):
                raise PreIngestError('ARTIFACT_INVALID_RELATION')
            if relation == 'exact' and details['whole_file_sha256_equal'] is not True:
                raise PreIngestError('ARTIFACT_INVALID_RELATION')
            seen_pairs.add((left, right))
            edges.append((left, right))
            actual_uncertain += int(relation == 'uncertain')
        if _component_map(ids, edges) != typed_components:
            raise PreIngestError('ARTIFACT_INVALID_COMPONENTS')
        if actual_uncertain != uncertain_count:
            raise PreIngestError('ARTIFACT_INVALID_COMPONENTS')
    if component_count != len(set(typed_components.values())):
        raise PreIngestError('ARTIFACT_INVALID_COMPONENTS')
    expected = _disposition(component_count, actual_uncertain, cast(list[str], blockers))
    if payload['disposition'] != expected:
        raise PreIngestError('ARTIFACT_INVALID_DISPOSITION')
    return PreIngestEvidence(supplied, expected, typed_components)
