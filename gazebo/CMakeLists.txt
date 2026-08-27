cmake_minimum_required(VERSION 3.16)
project(so101_gazebo_control)

find_package(ament_cmake REQUIRED)

# Direct SDF loading does not expand ROS-style $(find package) substitutions.
# Generate the installed model with the package's actual install prefix while
# retaining a portable placeholder in the source model.
set(CONTROLLERS_YAML
  "${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/config/ros2_controllers.yaml"
)
configure_file(
  models/so101_dark_blue/model.sdf
  ${CMAKE_CURRENT_BINARY_DIR}/so101_dark_blue_model.sdf
  @ONLY
)

install(
  DIRECTORY config launch models rviz worlds
  DESTINATION share/${PROJECT_NAME}
)

install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/so101_dark_blue_model.sdf
  DESTINATION share/${PROJECT_NAME}/models/so101_dark_blue
  RENAME model.sdf
)

install(
  PROGRAMS
    scripts/build_control_package.sh
    scripts/approach_nut.py
    scripts/approach_nut.sh
    scripts/collect_randomized_lerobot.py
    scripts/collect_randomized_lerobot.sh
    scripts/collect_randomized_lerobot_headless.sh
    scripts/randomize_nuts.py
    scripts/calibrated_to_urdf.py
    scripts/demo_motion.sh
    scripts/run_rviz_cameras.sh
    scripts/run_act_inference.sh
    scripts/so101_act_gazebo_bridge.py
    scripts/run_sim_rviz.sh
    scripts/run_world.sh
    scripts/send_calibrated_pose.sh
    scripts/validate_gazebo_tf_sync.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
