"""Tests for the pure named-pose JSON store."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inspect_robots_yam import poses


def _pose(name: str = "ready", *, created_at: str = "2026-08-19T12:00:00+00:00") -> poses.StartPose:
    return poses.StartPose(
        name=name,
        joints=tuple(float(index) / 20 for index in range(14)),
        created_at=created_at,
        notes="table setup",
        rig="rig-1",
    )


def _payload(name: str = "ready") -> dict[str, object]:
    return {
        "schema": poses.POSE_SCHEMA,
        "name": name,
        "joints": [float(index) for index in range(14)],
        "created_at": "2026-08-19T12:00:00+00:00",
        "notes": "note",
        "rig": "rig-1",
    }


def test_save_load_round_trip_and_overwrite_refusal(tmp_path: Path) -> None:
    pose = _pose()
    path = poses.save_pose(tmp_path / "nested", pose)
    assert path == tmp_path / "nested" / "ready.json"
    assert poses.load_pose(tmp_path / "nested", "ready") == pose
    with pytest.raises(poses.PoseStoreError, match="pass --force"):
        poses.save_pose(tmp_path / "nested", pose)
    replacement = _pose(created_at="later")
    assert poses.save_pose(tmp_path / "nested", replacement, overwrite=True) == path
    assert poses.load_pose(tmp_path / "nested", "ready") == replacement


@pytest.mark.parametrize(
    "name",
    ["../evil", "", ".hidden", "x" * 65, "with/slash", 42],
)
def test_name_rule_rejects_unsafe_or_malformed_names(name: object) -> None:
    with pytest.raises(poses.PoseStoreError, match=r"\^\[A-Za-z0-9\]"):
        poses.validate_pose_name(name)  # type: ignore[arg-type]


def test_pose_path_accepts_every_rule_character(tmp_path: Path) -> None:
    assert poses.pose_path(tmp_path, "A_1.two-three") == tmp_path / "A_1.two-three.json"


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"schema": 99}, "unsupported schema"),
        ({"schema": True}, "unsupported schema"),
        ({"joints": [0.0] * 13}, "exactly 14"),
        ({"joints": [0.0] * 13 + [float("inf")]}, "must be finite"),
        ({"joints": [0.0] * 13 + [True]}, "finite float"),
        ({"joints": [0.0] * 13 + ["bad"]}, "finite float"),
        ({"name": "other"}, "does not match filename stem"),
        ({"created_at": ""}, "created_at"),
        ({"notes": 5}, "notes must be a string"),
        ({"rig": 5}, "rig must be a string or null"),
    ],
)
def test_load_rejects_schema_dimension_nonfinite_name_and_metadata(
    tmp_path: Path, change: dict[str, object], match: str
) -> None:
    payload = _payload()
    payload.update(change)
    (tmp_path / "ready.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(poses.PoseStoreError, match=match):
        poses.load_pose(tmp_path, "ready")


def test_load_rejects_non_object_missing_field_and_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "ready.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(poses.PoseStoreError, match="JSON object"):
        poses.load_pose(tmp_path, "ready")
    path.write_text(json.dumps({"schema": 1, "name": "ready"}), encoding="utf-8")
    with pytest.raises(poses.PoseStoreError, match="missing or malformed field"):
        poses.load_pose(tmp_path, "ready")
    path.write_text("{", encoding="utf-8")
    with pytest.raises(poses.PoseStoreError, match="cannot load pose file"):
        poses.load_pose(tmp_path, "ready")


def test_list_orders_and_names_corrupt_file(tmp_path: Path) -> None:
    poses.save_pose(tmp_path, _pose("zulu"))
    poses.save_pose(tmp_path, _pose("alpha"))
    assert [pose.name for pose in poses.list_poses(tmp_path)] == ["alpha", "zulu"]
    (tmp_path / "broken.json").write_text("not-json", encoding="utf-8")
    with pytest.raises(poses.PoseStoreError, match=r"broken\.json"):
        poses.list_poses(tmp_path)
    (tmp_path / "broken.json").unlink()
    (tmp_path / ".bad.json").write_text(json.dumps(_payload(".bad")), encoding="utf-8")
    with pytest.raises(poses.PoseStoreError, match=r"\.bad\.json"):
        poses.list_poses(tmp_path)


def test_pose_names_uses_stems_without_loading_corrupt_sibling(tmp_path: Path) -> None:
    assert poses.pose_names(tmp_path / "missing") == ()
    (tmp_path / "bad.json").write_text("not-json", encoding="utf-8")
    (tmp_path / "also.json").mkdir()
    assert poses.pose_names(tmp_path) == ("bad",)
    with pytest.raises(poses.PoseStoreError, match=r"available poses: bad"):
        poses.load_pose(tmp_path, "missing")


def test_rename_rewrites_name_preserves_metadata_and_round_trips(tmp_path: Path) -> None:
    original = _pose("old")
    poses.save_pose(tmp_path, original)
    path = poses.rename_pose(tmp_path, "old", "new")
    renamed = poses.load_pose(tmp_path, "new")
    assert path == tmp_path / "new.json"
    assert renamed.name == "new"
    assert renamed.joints == original.joints
    assert renamed.created_at == original.created_at
    assert renamed.notes == original.notes
    assert renamed.rig == original.rig
    assert not (tmp_path / "old.json").exists()


def test_delete_and_rename_missing_or_existing_targets(tmp_path: Path) -> None:
    with pytest.raises(poses.PoseStoreError, match=r"available poses: \(none\)"):
        poses.delete_pose(tmp_path, "missing")
    with pytest.raises(poses.PoseStoreError, match="does not exist"):
        poses.rename_pose(tmp_path, "missing", "new")
    poses.save_pose(tmp_path, _pose("old"))
    poses.save_pose(tmp_path, _pose("new"))
    with pytest.raises(poses.PoseStoreError, match="already exists"):
        poses.rename_pose(tmp_path, "old", "new")
    poses.delete_pose(tmp_path, "old")
    assert poses.pose_names(tmp_path) == ("new",)


def test_filesystem_errors_are_wrapped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    poses.save_pose(tmp_path, _pose("old"))
    real_read = Path.read_text

    def fail_read(path: Path, *args: object, **kwargs: object) -> str:
        if path.name == "old.json":
            raise OSError("read fault")
        return real_read(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", fail_read)
    with pytest.raises(poses.PoseStoreError, match="read fault"):
        poses.load_pose(tmp_path, "old")


def test_save_directory_and_write_errors_are_wrapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("file", encoding="utf-8")
    with pytest.raises(poses.PoseStoreError, match="cannot create pose directory"):
        poses.save_pose(blocked, _pose())

    real_write = Path.write_text

    def fail_write(path: Path, *args: object, **kwargs: object) -> int:
        if path.name == "ready.json":
            raise OSError("write fault")
        return real_write(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "write_text", fail_write)
    with pytest.raises(poses.PoseStoreError, match=r"cannot write.*write fault"):
        poses.save_pose(tmp_path / "store", _pose())


def test_delete_and_rename_unlink_errors_are_wrapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    poses.save_pose(tmp_path, _pose("delete"))
    poses.save_pose(tmp_path, _pose("old"))
    real_unlink = Path.unlink

    def fail_unlink(path: Path, *args: object, **kwargs: object) -> None:
        if path.name in {"delete.json", "old.json"}:
            raise OSError("unlink fault")
        real_unlink(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "unlink", fail_unlink)
    with pytest.raises(poses.PoseStoreError, match=r"cannot delete.*unlink fault"):
        poses.delete_pose(tmp_path, "delete")
    with pytest.raises(poses.PoseStoreError, match=r"renamed pose was written.*unlink fault"):
        poses.rename_pose(tmp_path, "old", "new")
    assert (tmp_path / "new.json").exists()
