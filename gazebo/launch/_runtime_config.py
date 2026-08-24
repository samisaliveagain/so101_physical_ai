"""Build runtime simulation descriptions from the authoritative pose YAML."""

from __future__ import annotations

import math
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml


JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def _finite_float(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return number


def load_initial_pose(path: str | Path) -> dict[str, Any]:
    pose_path = Path(path).expanduser().resolve()
    data = yaml.safe_load(pose_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Initial pose must be a YAML mapping: {pose_path}")

    spawn = data.get("spawn", {})
    position = spawn.get("position", {})
    orientation = spawn.get("orientation_rpy", {})
    joints = data.get("joints", {})

    normalized = {
        "source": str(pose_path),
        "frame": str(spawn.get("frame", "world")),
        "position": {
            axis: _finite_float(position[axis], f"spawn.position.{axis}")
            for axis in ("x", "y", "z")
        },
        "orientation_rpy": {
            axis: _finite_float(orientation[axis], f"spawn.orientation_rpy.{axis}")
            for axis in ("roll", "pitch", "yaw")
        },
        "joints": {
            name: _finite_float(joints[name], f"joints.{name}")
            for name in JOINT_NAMES
        },
    }
    if normalized["frame"] != "world":
        raise ValueError("spawn.frame must be 'world' for this Gazebo world")
    return normalized


def _number(value: float) -> str:
    return f"{value:.12g}"


def build_robot_description(package_share: str | Path, pose: dict[str, Any]) -> str:
    urdf_path = Path(package_share) / "models/so101_dark_blue/so101_dark_blue.urdf"
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    base_joint = root.find("./joint[@name='base_to_world']")
    if base_joint is None:
        raise ValueError("URDF is missing joint base_to_world")
    origin = base_joint.find("origin")
    if origin is None:
        raise ValueError("URDF base_to_world is missing its origin")
    p = pose["position"]
    r = pose["orientation_rpy"]
    origin.set("xyz", " ".join(_number(p[a]) for a in ("x", "y", "z")))
    origin.set("rpy", " ".join(_number(r[a]) for a in ("roll", "pitch", "yaw")))

    for name in JOINT_NAMES:
        kinematic_joint = root.find(f"./joint[@name='{name}']")
        if kinematic_joint is None:
            raise ValueError(f"URDF is missing kinematic joint {name}")
        limit = kinematic_joint.find("limit")
        value = pose["joints"][name]
        if limit is not None:
            lower = float(limit.get("lower", "-inf"))
            upper = float(limit.get("upper", "inf"))
            if not lower <= value <= upper:
                raise ValueError(
                    f"Initial {name}={value} is outside [{lower}, {upper}]"
                )

        control_joint = root.find(f"./ros2_control/joint[@name='{name}']")
        if control_joint is None:
            raise ValueError(f"ros2_control is missing joint {name}")
        state = control_joint.find("./state_interface[@name='position']")
        if state is None:
            raise ValueError(f"ros2_control joint {name} has no position state")
        initial = state.find("./param[@name='initial_value']")
        if initial is None:
            initial = ET.SubElement(state, "param", {"name": "initial_value"})
        initial.text = _number(value)

    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def build_runtime_world(package_share: str | Path, pose: dict[str, Any]) -> str:
    source = Path(package_share) / "worlds/so101_dataset.sdf"
    tree = ET.parse(source)
    root = tree.getroot()
    world = root.find("world")
    if world is None:
        raise ValueError(f"World element missing from {source}")

    robot_include = None
    for include in world.findall("include"):
        if include.findtext("name") == "so101":
            robot_include = include
            break
    if robot_include is None:
        raise ValueError(f"SO-101 include missing from {source}")

    pose_element = robot_include.find("pose")
    if pose_element is None:
        pose_element = ET.SubElement(robot_include, "pose")
    p = pose["position"]
    r = pose["orientation_rpy"]
    pose_element.text = " ".join(
        _number(value)
        for value in (p["x"], p["y"], p["z"], r["roll"], r["pitch"], r["yaw"])
    )

    runtime_dir = Path(tempfile.mkdtemp(prefix="so101_gazebo_"))
    output = runtime_dir / "so101_dataset_runtime.sdf"
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return str(output)


def build_runtime_rviz(
    package_share: str | Path, pose: dict[str, Any], rviz_source: str | Path | None = None
) -> str:
    source = (
        Path(rviz_source)
        if rviz_source is not None
        else Path(package_share) / "rviz/so101_cameras.rviz"
    )
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    manager = config["Visualization Manager"]
    p = pose["position"]

    for display in manager["Displays"]:
        if display.get("Class") == "rviz_default_plugins/Grid":
            display["Offset"] = {"X": p["x"], "Y": p["y"], "Z": p["z"]}
            display["Reference Frame"] = pose["frame"]

    view = manager["Views"]["Current"]
    view["Focal Point"] = {"X": p["x"], "Y": p["y"], "Z": p["z"] + 0.12}
    view["Target Frame"] = pose["frame"]

    runtime_dir = Path(tempfile.mkdtemp(prefix="so101_rviz_"))
    output = runtime_dir / "so101_runtime.rviz"
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return str(output)
