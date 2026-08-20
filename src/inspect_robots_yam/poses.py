"""Pure JSON storage for named, wire-shaped YAM start poses."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from inspect_robots_yam.packing import TOTAL_DIM as _POSE_DIM

POSE_SCHEMA = 1
POSE_NAME_RULE = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
POSE_NAME_MAX_LENGTH = 64
_POSE_NAME_RE = re.compile(POSE_NAME_RULE)


class PoseStoreError(ValueError):
    """Report an invalid pose or a pose-store operation that cannot be completed."""


@dataclass(frozen=True)
class StartPose:
    """A portable 14-D joint pose with normalized gripper slots and provenance."""

    name: str
    joints: tuple[float, ...]
    created_at: str
    notes: str = ""
    rig: str | None = None


def validate_pose_name(name: str) -> None:
    """Reject names outside the portable filename-safe pose-name contract."""
    if (
        not isinstance(name, str)
        or len(name) > POSE_NAME_MAX_LENGTH
        or _POSE_NAME_RE.fullmatch(name) is None
    ):
        raise PoseStoreError(
            f"pose name must match {POSE_NAME_RULE} and be at most "
            f"{POSE_NAME_MAX_LENGTH} characters, got {name!r}"
        )


def pose_path(pose_dir: str | Path, name: str) -> Path:
    """Return the JSON path for a validated pose name without touching the filesystem."""
    validate_pose_name(name)
    return Path(pose_dir) / f"{name}.json"


def pose_names(pose_dir: str | Path) -> tuple[str, ...]:
    """Return sorted JSON filename stems without parsing any pose files."""
    directory = Path(pose_dir)
    if not directory.is_dir():
        return ()
    return tuple(sorted(path.stem for path in directory.glob("*.json") if path.is_file()))


def save_pose(
    pose_dir: str | Path,
    pose: StartPose,
    *,
    overwrite: bool = False,
) -> Path:
    """Validate and save one pose, refusing replacement unless ``overwrite`` is true."""
    path = pose_path(pose_dir, pose.name)
    checked = _validate_pose(pose, path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PoseStoreError(f"cannot create pose directory {path.parent}: {exc}") from exc
    if path.exists() and not overwrite:
        raise PoseStoreError(
            f"pose {pose.name!r} already exists at {path}; pass --force to replace it"
        )
    payload = {
        "schema": POSE_SCHEMA,
        "name": checked.name,
        "joints": list(checked.joints),
        "created_at": checked.created_at,
        "notes": checked.notes,
        "rig": checked.rig,
    }
    try:
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        raise PoseStoreError(f"cannot write pose file {path}: {exc}") from exc
    return path


def load_pose(pose_dir: str | Path, name: str) -> StartPose:
    """Load and fully validate one pose, including its filename-name agreement."""
    path = pose_path(pose_dir, name)
    if not path.is_file():
        raise PoseStoreError(_missing_message(pose_dir, name))
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PoseStoreError(f"cannot load pose file {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise PoseStoreError(f"invalid pose file {path}: expected a JSON object")
    schema = raw.get("schema")
    if isinstance(schema, bool) or not isinstance(schema, int) or schema != POSE_SCHEMA:
        raise PoseStoreError(
            f"invalid pose file {path}: unsupported schema {schema!r}; expected {POSE_SCHEMA}"
        )
    embedded_name = raw.get("name")
    if embedded_name != path.stem:
        raise PoseStoreError(
            f"invalid pose file {path}: embedded name {embedded_name!r} does not match "
            f"filename stem {path.stem!r}"
        )
    try:
        pose = StartPose(
            name=embedded_name,
            joints=tuple(raw["joints"]),
            created_at=raw["created_at"],
            notes=raw.get("notes", ""),
            rig=raw.get("rig"),
        )
    except (KeyError, TypeError) as exc:
        raise PoseStoreError(f"invalid pose file {path}: missing or malformed field {exc}") from exc
    return _validate_pose(pose, path)


def list_poses(pose_dir: str | Path) -> tuple[StartPose, ...]:
    """Load all poses in name order, failing on any corrupt JSON pose file."""
    loaded: list[StartPose] = []
    for name in pose_names(pose_dir):
        path = Path(pose_dir) / f"{name}.json"
        try:
            loaded.append(load_pose(pose_dir, name))
        except PoseStoreError as exc:
            raise PoseStoreError(f"cannot list pose file {path}: {exc}") from exc
    return tuple(loaded)


def delete_pose(pose_dir: str | Path, name: str) -> None:
    """Delete one named pose or report the directory's still-available names."""
    path = pose_path(pose_dir, name)
    if not path.is_file():
        raise PoseStoreError(_missing_message(pose_dir, name))
    try:
        path.unlink()
    except OSError as exc:
        raise PoseStoreError(f"cannot delete pose file {path}: {exc}") from exc


def rename_pose(pose_dir: str | Path, old: str, new: str) -> Path:
    """Rename a pose while rewriting its embedded name and preserving metadata."""
    pose = load_pose(pose_dir, old)
    target = pose_path(pose_dir, new)
    if target.exists():
        raise PoseStoreError(f"pose {new!r} already exists at {target}; choose another name")
    saved = save_pose(pose_dir, replace(pose, name=new))
    source = pose_path(pose_dir, old)
    try:
        source.unlink()
    except OSError as exc:
        with_error = f"renamed pose was written to {saved}, but could not remove {source}: {exc}"
        raise PoseStoreError(with_error) from exc
    return saved


def _validate_pose(pose: StartPose, path: Path) -> StartPose:
    """Return a normalized pose after validating the complete on-disk contract."""
    validate_pose_name(pose.name)
    if not isinstance(pose.created_at, str) or not pose.created_at:
        raise PoseStoreError(f"invalid pose file {path}: created_at must be a non-empty string")
    if not isinstance(pose.notes, str):
        raise PoseStoreError(f"invalid pose file {path}: notes must be a string")
    if pose.rig is not None and not isinstance(pose.rig, str):
        raise PoseStoreError(f"invalid pose file {path}: rig must be a string or null")
    if len(pose.joints) != _POSE_DIM:
        raise PoseStoreError(
            f"invalid pose file {path}: joints must contain exactly {_POSE_DIM} entries, "
            f"got {len(pose.joints)}"
        )
    joints: list[float] = []
    for index, value in enumerate(pose.joints):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PoseStoreError(
                f"invalid pose file {path}: joint {index} must be a finite float, got {value!r}"
            )
        converted = float(value)
        if not math.isfinite(converted):
            raise PoseStoreError(
                f"invalid pose file {path}: joint {index} must be finite, got {value!r}"
            )
        joints.append(converted)
    return replace(pose, joints=tuple(joints))


def _missing_message(pose_dir: str | Path, name: str) -> str:
    """Build a corruption-independent missing-pose diagnostic."""
    available = pose_names(pose_dir)
    listing = ", ".join(available) if available else "(none)"
    return f"pose {name!r} does not exist in {Path(pose_dir)}; available poses: {listing}"
