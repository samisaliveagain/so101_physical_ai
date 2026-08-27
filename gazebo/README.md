# SO-101 Gazebo dataset scene

This Gazebo Harmonic scene reconstructs the visual setup used by the
`shubham4413/so101_vla` recordings: a white table, textured white wall, cork
board, two red stacking parts, a dark-blue SO-101 at the right edge, one fixed
left camera, and one robot-mounted FPV camera.

The robot geometry comes from the official TheRobotStudio SO-ARM100 repository
at commit `7629d2ad9853d10fb903093a33ef6114099d97e5`
(`Simulation/SO101/so101_new_calib.urdf`). Its Apache-2.0 license is retained in
`models/so101_dark_blue/SO_ARM100_LICENSE`. The printed parts were recolored;
motor housings, calibrated joint limits, and inertia values remain from the
source model. Exact triangle-mesh collisions were replaced by 18 conservative
box proxies so contact geometry is deterministic and inexpensive.

## Start the world

From the project root:

```bash
./gazebo/scripts/run_world.sh
```

For a headless machine or HPC allocation:

```bash
./gazebo/scripts/run_world.sh -s --headless-rendering
```

The launcher builds the `so101_gazebo_control` resource package when needed,
starts `robot_state_publisher`, bridges simulation time, launches Gazebo, and
activates both `joint_state_broadcaster` and the six-joint
`JointTrajectoryController`. It sources ROS 2 Jazzy and uses Gazebo Harmonic's
default DART physics engine. DART holds the imported SO-101 joint frames
correctly; forcing Bullet caused the elbow and wrist to drift toward their
limits. Add `-v 4` for verbose Gazebo diagnostics.
It also removes Snap-specific GTK/GIO/library variables from the Gazebo child
process. This prevents Snap VS Code's Core20 `libpthread` from being mixed with
Ubuntu 24.04's native glibc; the parent terminal environment is not changed.

If the GUI still reports EGL errors, first verify the host driver outside a
container or restricted shell:

```bash
nvidia-smi
ls -l /dev/nvidia*
```

Then test software rendering to distinguish a scene problem from an NVIDIA/GLVND
problem:

```bash
LIBGL_ALWAYS_SOFTWARE=1 ./gazebo/scripts/run_world.sh
```

Software rendering is suitable for a GUI check but is too slow for camera-heavy
policy evaluation. On an HPC node, continue using `-s --headless-rendering`
inside a GPU allocation.

## Topics

The two image topics match the dataset camera roles and resolution (640x480 at
30 Hz):

```text
/so101/camera/left/image
/so101/camera/fpv/image
```

The articulated model exposes the standard ROS 2 control interfaces:

```text
/joint_states
/arm_controller/follow_joint_trajectory
/controller_manager
```

`/arm_controller/follow_joint_trajectory` is a
`control_msgs/action/FollowJointTrajectory` action, not a Gazebo transport
topic. The controller owns all six joints, including the gripper, so each VLA
action vector is applied synchronously and produces measured feedback.

The base is fixed to the world, while the six calibrated joints are dynamic and
gravity is disabled on the robot links. Its visuals retain the exact STL meshes,
while all 18 collision elements use primitive boxes. Global robot self-contact
is disabled because conservative boxes overlap at neighboring motor joints;
table and environment contacts remain active. Keep commanded trajectories
within a validated envelope. Full manipulation physics will need per-link
collision filtering or convex hulls plus identified motor and friction
parameters. The destination part remains static for stacking stability. The
grasped nut is dynamic and uses an open primitive-ring collision model.

## Move the simulated robot

To make a ground-truth Cartesian pre-grasp approach to the fixed nut, start
`run_sim_rviz.sh`, then run this in a second terminal:

```bash
./gazebo/scripts/approach_nut.sh
```

The graspable nut is at world `(0.10, 0.13, 0.631)` m, moved closer to the
robot than the original camera-validation position. The script reads the live
joint state, solves numerical IK from the URDF, opens the gripper, approaches
the centre of the nut hole, descends, and closes fully. The nut is dynamic and
its collision shape is an open six-segment rim, so this grasp can make physical
contact and lift it. `--rim-offset 0.028` restores the earlier near-rim target;
the default `0.0` targets the actual SDF model centre.
Pass `--approach-only` to retain the earlier stop-above-the-object behavior.
The default grasp descends 8 mm below the nut's top plane, leaving clearance
for the conservative jaw collision boxes, and slowly commands the gripper to
its closed limit (`-0.174` rad). Use
`--grasp-height` or `--close` for fine tuning.
The fixed gripper collision is split into a palm and a 9 x 10 mm finger, and
the moving jaw uses a 10 x 18 mm cross-section while retaining its reach;
the previous whole-gripper box incorrectly filled the jaw gap and could not
enter the nut's approximately 48 mm opening. Gripper-only trajectory tolerance
allows expected contact lag during a force-like hard-close command.
After closing, the default sequence lifts vertically, transfers to the other
red part at `(0.08, 0.02)`, places the nut on top, releases, and retreats. Use
`--grasp-only` to stop after closing or `--approach-only` to stop before the
descent. The initial lift is also the definitive check that the simulated
contact grasp is holding; this example does not create an artificial fixed
joint between the nut and gripper.
An object prevents the position-controlled finger from reaching its empty-hand
hard-close angle, so the controller can report path (`-4`) or goal (`-5`)
tolerance failure during a valid grasp. The client accepts those outcomes only
when all five arm joints reached their waypoint and the close phase produced
substantial measured finger motion; unrelated trajectory failures still stop
the sequence.
Different known object coordinates can be supplied without a camera, for
example `--x 0.08 --y 0.02 --z 0.631`. Use `--help` for safety-clearance and
duration options. This is a small position-space IK and interpolation example,
not collision-aware MoveIt planning.

## Randomized LeRobot data collection

With `run_sim_rviz.sh` running, collect randomized stacking demonstrations:

```bash
./gazebo/scripts/collect_randomized_lerobot.sh \
  --episodes 10 \
  --output data/so101_gazebo_randomized_stack
```

To launch Gazebo headlessly, collect, and shut the simulator down automatically
in one command:

```bash
./gazebo/scripts/collect_randomized_lerobot_headless.sh --episodes 10
```

To prevent repetition across collection runs, pass the earlier dataset as an
exclusion set. A candidate is rejected when both its source and destination
are within 12 mm of a previous layout; accepted layouts in the current run are
also added to that set. Each episode resets the robot and is saved only after
the final nut pose confirms a physical stack:

```bash
./gazebo/scripts/collect_randomized_lerobot_headless.sh \
  --episodes 50 \
  --seed 20260826 \
  --exclude-dataset "/media/shubhamnagar/One Touch/so101_gazebo_randomized_stack_20260824_005251"
```

Repeat `--exclude-dataset PATH` to exclude multiple earlier collections.

The OneTouch drive must be mounted and writable at
`/media/shubhamnagar/One Touch`. Each run creates a new timestamped directory,
for example
`/media/shubhamnagar/One Touch/so101_gazebo_randomized_stack_20260823_143000`,
so an existing dataset is never overwritten. Pass `--output /another/path` to
override it. Headless Gazebo output is saved under `.ros/` for diagnostics.

The launcher uses `/home/shubhamnagar/lerobot/.venv` and writes LeRobot v3
Parquet metadata/data plus AV1 videos for `observation.images.left` and
`observation.images.fpv`. `observation.state` contains measured URDF joint
positions in radians and `action` contains the trajectory controller reference.
`observation.environment_state` stores the eight ground-truth randomized spawn
values (XYZ and yaw for each part) on every frame.
The output directory must not already exist, preventing accidental overwrite.

## Run the trained ACT policy in Gazebo

Start the normal simulation in one terminal:

```bash
./gazebo/scripts/run_sim_rviz.sh
```

Each interactive launch moves both red parts to a new IK-validated layout near
the successful episode-42 reference: up to 15 mm independently in `x` and `y`,
and up to 15 degrees in yaw.  To reproduce the exact SDF layout or change the
evaluation range:

```bash
./gazebo/scripts/run_sim_rviz.sh randomize_nuts:=false
./gazebo/scripts/run_sim_rviz.sh nut_xy_jitter:=0.008 nut_yaw_jitter_deg:=5
```

In a second terminal, first run inference without moving the robot:

```bash
./gazebo/scripts/run_act_inference.sh --device cuda --duration 10
```

The bridge reads `/arm_controller/controller_state` and both 640x480 camera
topics, loads the final ACT checkpoint directly from the external SSD, and
publishes raw and safety-bounded predictions on
`/so101/act/predicted_action` and `/so101/act/commanded_action`. Without
`--execute`, it does not publish controller commands.

After inspecting the predicted actions, execute one bounded 90-second rollout:

```bash
./gazebo/scripts/run_act_inference.sh --device cuda --execute
```

The launcher executes 50 actions from each predicted 100-action ACT chunk and
then replans from new camera and joint observations. This is an inference-only
setting; it does not modify or retrain the checkpoint. The 50-action default
avoids repeatedly replaying only the low-motion start of a chunk. Use
`--action-horizon 20` for more frequent correction after confirming the policy
still makes progress, or
`--action-horizon 100` to reproduce the original open-loop behavior.

The two image streams and controller feedback are buffered and matched using
their Gazebo simulation timestamps before each inference call. The defaults
require both camera frames and the corresponding joint state to be within
50 ms. Queued ROS callbacks are drained between predictions, and a control tick
is skipped instead of combining mismatched observations. A warning is emitted
only if no valid camera/state triplet is available continuously for one second;
an occasional skipped tick between 30 Hz frames is expected.

Executed actions are clipped to the URDF joint limits and rate-limited to
1.0 rad/s for the arm and 0.8 rad/s for the gripper before being sent to
`/arm_controller/joint_trajectory`. The bridge stops commanding if either
camera or the controller state becomes stale. Override the checkpoint with
`SO101_ACT_CHECKPOINT=/path/to/pretrained_model`.

For every episode, the source nut is sampled in `x=0.075..0.135` and
`y=0.090..0.155` m. The destination is sampled in `x=0.075..0.145` and
`y=0.000..0.065` m. Layouts closer than 115 mm are rejected, and grasp, lift,
transfer, placement and retreat poses must all pass IK before Gazebo is
changed. Failed controller rollouts are discarded by default. To preview one
deterministic layout without changing the simulation:

```bash
source /opt/ros/jazzy/setup.bash
/home/shubhamnagar/lerobot/.venv/bin/python \
  gazebo/scripts/randomize_nuts.py --seed 101 --dry-run
```

With Gazebo already running, execute the visible three-waypoint demo in a second
terminal. It sends ROS 2 trajectory actions and ends in a raised pose instead
of returning to the initial pose:

```bash
./gazebo/scripts/demo_motion.sh
```

To send a specific calibrated pose, provide five LeRobot joint angles in degrees,
the dataset-style gripper percentage, and an optional duration:

```bash
./gazebo/scripts/send_calibrated_pose.sh \
  -18.77 -9.58 19.47 78.37 -5.23 9.05 4
```

The command script first looks for the live LeRobot follower file at
`~/calibration_lerobot_data_collect/calibration/robots/so_follower/my_awesome_follower_arm.json`,
then falls back to `config/so101_follower_calibration.json`. It derives each
joint's usable range from the recorded encoder endpoints and maps the first five
LeRobot degree values to URDF radians. Motor 4 (`wrist_flex`) has the opposite
CAD-axis polarity, so that coordinate alone is sign-reversed. The gripper's
0--100 value maps onto its -10--100 degree jaw range. The script waits for the
action result and returns a
failure unless the controller reports `SUCCEEDED`. Set
`SO101_CALIBRATION_FILE=/path/to/calibration.json` to test a different follower
calibration. These commands affect only the simulated arm.

The demo uses actual observations paired with the real episode-0 camera frames,
not invented joint positions:

| Real frame | Dataset time | Purpose |
|---|---:|---|
| `training_left_t00_5.jpg` | 0.5 s | folded episode-start pose |
| `training_left_t05.jpg` | 5.0 s | arm extended toward the nut |
| `training_left_t15.jpg` | 15.0 s | task-space adjustment above the nut |

To inspect the control stack directly:

```bash
source /opt/ros/jazzy/setup.bash
ros2 control list_controllers
ros2 action list -t
ros2 topic echo /joint_states --once
```

Both `joint_state_broadcaster` and `arm_controller` must report `active`.

## View both cameras in RViz

Keep Gazebo running, then launch the ROS bridge and prepared RViz configuration
from a second terminal:

```bash
./gazebo/scripts/run_rviz_cameras.sh
```

The launcher bridges both 640x480 image and camera-info topics from Gazebo to
ROS 2. RViz shows the URDF robot together with the left and FPV image panels.
It also sanitizes Snap/Core20 library variables before starting RViz.

## Calibration and initial pose

The simulated arm uses the **follower** calibration supplied at
`/home/shubhamnagar/calibration_lerobot_data_collect/calibration/robots/so_follower/my_awesome_follower_arm.json`.
A project-local snapshot is stored in
`config/so101_follower_calibration.json`. The leader calibration is not used for
the simulated follower geometry.

Dataset row 149464 was selected because it is the nearest actual observation to
the dataset median. With the follower calibration, the recorded state

```text
[-18.76923, -9.58242, 19.47253, 78.37363, -5.23077, 9.04729]
```

reconstructs the exact encoder counts `[1828, 1992, 2204, 2800, 1988, 2144]`.
The first five entries are already calibrated degrees. Both LeRobot and the
upstream `so101_new_calib` model define virtual zero at the middle of the joint
range, so no homing-offset addition is applied. The physical motor firmware
already applies `homing_offset` before LeRobot normalizes its reading. The
Gazebo CAD hinge for motor 4 points opposite to the recorded `wrist_flex`
coordinate, so only that joint is sign-reversed. `gripper.pos` is a 0--100 percentage, so
9.04729% is mapped onto the URDF jaw range of -10--100 degrees, producing
-0.04798 degrees. This places the lowest jaw visual at approximately z=0.680 m,
about 50 mm above the z=0.630 m tabletop.

The calibrated body-joint limits derived from the recorded encoder ranges are:

| Joint | LeRobot range | URDF range |
|---|---:|---:|
| shoulder_pan | ±123.120879° | ±2.148865 rad |
| shoulder_lift | ±101.274725° | ±1.767577 rad |
| elbow_flex | ±98.241758° | ±1.714642 rad |
| wrist_flex | ±105.098901° | ±1.834322 rad |
| wrist_roll | ±180° | ±3.141593 rad |

## Scene measurements

- Table: supplied 0.8 x 0.8 m STL, top surface at approximately z=0.63 m.
- Wall: supplied 2.0 x 0.2 x 1.5 m STL with the supplied wall photograph.
- Cork board: supplied 0.8 x 0.01 x 0.5 m STL with the supplied board photograph.
- Red parts: millimetre STL files imported with scale `0.001`.
- Robot base: x=0.30 m, y=0.22 m, at the far-right side of the table, rotated
  180 degrees so the gripper faces into the workspace.
- Initial joints: calibrated dataset row 149464 (`-18.77, -9.58, 19.47,
  78.37, -5.23` degrees and `9.05%` gripper), selected as a representative,
  collision-safe observation rather than the folded episode-start pose.

These poses reproduce the dataset composition approximately. Exact camera
extrinsics were not stored in the available metadata, so both cameras should be
fine-tuned after comparing rendered frames against representative dataset
frames.

## Control implementation

The control layout follows the same separation used by the ROS 2
`brukg/SO-100-arm` simulation package: a robot description publishes
`ros2_control` hardware interfaces, `gz_ros2_control/GazeboSimSystem` binds
those interfaces to Gazebo joints, and controller-manager spawners activate
joint-state and trajectory controllers. The SDF contains scene physics and
visuals; changing a link pose in the SDF is not used as a runtime motion
command.

The effective Gazebo position gain is 10 s^-1 (`0.1` multiplied by the 100 Hz
controller rate), which settles cleanly without the high-gain oscillation seen
with the earlier Gazebo-only controller.
