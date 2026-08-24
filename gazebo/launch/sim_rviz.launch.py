"""Standard combined SO-101 Gazebo, ros2_control, bridge and RViz launch."""

from pathlib import Path
import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _runtime_config import build_runtime_rviz, load_initial_pose


def _launch_setup(context):
    package_share = get_package_share_directory("so101_gazebo_control")
    pose_path = LaunchConfiguration("pose_config").perform(context)
    rviz_source = LaunchConfiguration("rviz_config").perform(context)
    gz_extra_args = LaunchConfiguration("gz_extra_args").perform(context)

    initial_pose = load_initial_pose(pose_path)
    runtime_rviz = build_runtime_rviz(package_share, initial_pose, rviz_source)

    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(package_share, "launch", "controlled_world.launch.py")
        ),
        launch_arguments={
            "pose_config": pose_path,
            "gz_extra_args": gz_extra_args,
        }.items(),
    )

    camera_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/so101/camera/left/image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/so101/camera/left/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            "/so101/camera/fpv/image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/so101/camera/fpv/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
        ],
        output="screen",
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", runtime_rviz],
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(LaunchConfiguration("launch_rviz")),
        output="screen",
    )

    return [
        simulation,
        TimerAction(period=1.5, actions=[camera_bridge]),
        TimerAction(period=3.0, actions=[rviz]),
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
            "rviz_config",
            default_value=os.path.join(package_share, "rviz", "so101_cameras.rviz"),
            description="Base RViz configuration",
        ),
        DeclareLaunchArgument(
            "gz_extra_args",
            default_value="",
            description="Additional arguments passed to Gazebo",
        ),
        DeclareLaunchArgument(
            "launch_rviz",
            default_value="true",
            description="Start RViz together with Gazebo",
        ),
        OpaqueFunction(function=_launch_setup),
    ])
