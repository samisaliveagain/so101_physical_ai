"""Launch Gazebo and SO-101 controllers from one authoritative pose YAML."""

from pathlib import Path
import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _runtime_config import build_robot_description, build_runtime_world, load_initial_pose


def _launch_setup(context):
    package_share = get_package_share_directory("so101_gazebo_control")
    ros_gz_share = get_package_share_directory("ros_gz_sim")
    pose_path = LaunchConfiguration("pose_config").perform(context)
    gz_extra_args = LaunchConfiguration("gz_extra_args").perform(context).strip()

    initial_pose = load_initial_pose(pose_path)
    robot_description = build_robot_description(package_share, initial_pose)
    runtime_world = build_runtime_world(package_share, initial_pose)
    gz_args = f"-r {runtime_world}"
    if gz_extra_args:
        gz_args = f"{gz_args} {gz_extra_args}"

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_share, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={"gz_args": gz_args, "on_exit_shutdown": "true"}.items(),
    )

    description_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{
            "robot_description": robot_description,
            "use_sim_time": True,
            "publish_frequency": 50.0,
            "ignore_timestamp": False,
        }],
        output="screen",
    )

    clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
        output="screen",
    )

    controllers = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "arm_controller",
            "--controller-manager", "/controller_manager",
            "--controller-manager-timeout", "30",
            "--switch-timeout", "30",
            "--activate-as-group",
        ],
        output="screen",
    )

    return [
        description_publisher,
        clock_bridge,
        gazebo,
        TimerAction(period=2.0, actions=[controllers]),
    ]


def generate_launch_description():
    package_share = get_package_share_directory("so101_gazebo_control")
    return LaunchDescription([
        DeclareLaunchArgument(
            "pose_config",
            default_value=os.path.join(package_share, "config", "initial_pose.yaml"),
            description="YAML file containing the robot spawn and initial joints",
        ),
        DeclareLaunchArgument(
            "gz_extra_args",
            default_value="",
            description="Additional arguments passed to Gazebo, such as -s",
        ),
        OpaqueFunction(function=_launch_setup),
    ])
