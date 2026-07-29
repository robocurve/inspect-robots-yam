"""Generate and compile-check the vendored collision-only YAM MJCF."""

from __future__ import annotations

import argparse
import math
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from inspect_robots_yam.collision import CollisionChecker, CollisionConfig

_SOURCE_RELATIVE = Path("i2rt_yam/yam.xml")
_OUTPUT = Path("src/inspect_robots_yam/assets/yam_collision.xml")


def _delete_all(spec: Any, elements: Any) -> None:
    for element in list(elements):
        spec.delete(element)


def _strip_model(mujoco: Any, source: Path) -> str:
    spec = mujoco.MjSpec.from_file(str(source.resolve()))
    for geom in list(spec.geoms):
        if geom.classname.name == "visual":
            spec.delete(geom)
    for collection in (
        spec.meshes,
        spec.textures,
        spec.materials,
        spec.lights,
        spec.cameras,
        spec.actuators,
        spec.keys,
        spec.equalities,
    ):
        _delete_all(spec, collection)
    for index, geom in enumerate(spec.geoms):
        if not geom.name:
            geom.name = f"collision_geom_{index}"
    angular_joints = {
        joint.name for joint in spec.joints if joint.type == mujoco.mjtJoint.mjJNT_HINGE
    }
    return _strip_serialized_settings(spec.to_xml(), angular_joints)


def _strip_serialized_settings(xml: str, angular_joints: set[str]) -> str:
    root = ET.fromstring(xml)
    # MjSpec serializes angular quantities in radians. Once <compiler> is
    # removed, MuJoCo's parser defaults to degrees, so preserve hinge ranges by
    # expressing the same angles in degrees. The source uses quaternions rather
    # than Euler angles, and slide-joint ranges remain linear metres.
    for joint in root.iter("joint"):
        if joint.get("name") in angular_joints and (joint_range := joint.get("range")):
            joint.set(
                "range",
                " ".join(str(math.degrees(float(value))) for value in joint_range.split()),
            )
    for tag in ("compiler", "option"):
        element = root.find(tag)
        if element is not None:
            root.remove(element)
    for parent in root.iter():
        for child in list(parent):
            if child.tag in {"position", "general"} or (
                child.tag == "default" and child.get("class") == "visual"
            ):
                parent.remove(child)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def _git_commit(checkout: Path) -> str:
    return subprocess.check_output(
        ("git", "-C", str(checkout), "rev-parse", "HEAD"),
        text=True,
    ).strip()


def _provenance(commit: str) -> str:
    return (
        "<!--\n"
        "Generated collision-only model derived from MuJoCo Menagerie.\n"
        f"Menagerie commit: {commit}\n"
        f"Source: {_SOURCE_RELATIVE.as_posix()}\n"
        "License: MIT, i2rt robotics (see MENAGERIE_LICENSE)\n"
        "Regenerate: python scripts/gen_collision_model.py /path/to/mujoco_menagerie\n"
        "-->\n"
    )


def main() -> None:
    """Strip the Menagerie model, write provenance, and compile the shipped scene."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("menagerie", type=Path, help="path to a mujoco_menagerie checkout")
    args = parser.parse_args()
    checkout = args.menagerie.resolve()
    source = checkout / _SOURCE_RELATIVE
    if not source.is_file():
        parser.error(f"missing Menagerie YAM model: {source}")

    import mujoco

    generated = _provenance(_git_commit(checkout)) + _strip_model(mujoco, source)
    _OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT.write_text(generated, encoding="utf-8")

    # Use the runtime class itself so generation cannot drift from its composition path.
    CollisionChecker(CollisionConfig(), model_xml=generated)
    print(f"wrote and compiled {_OUTPUT}")


if __name__ == "__main__":
    main()
