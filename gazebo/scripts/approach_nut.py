#!/usr/bin/env python3
"""Ground-truth Cartesian rim grasp for the nut in the SO-101 world."""

import argparse
import math
import random
import sys
import time
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import rclpy
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTrajectoryControllerState
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint


JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
CHAIN = JOINTS + ["gripper_frame_joint"]
LIMITS = [(-2.148864718609, 2.148864718609), (-1.767577405097, 1.767577405097),
          (-1.714642144267, 1.714642144267), (-1.834321864404, 1.834321864404),
          (-math.pi, math.pi)]
BASE_XYZ = (0.38, 0.22, 0.632)
BASE_RPY = (0.0, 0.0, math.pi)


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def transform(xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr, xyz[0]],
            [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr, xyz[1]],
            [-sp, cp*sr, cp*cr, xyz[2]], [0.0, 0.0, 0.0, 1.0]]


def solve_linear(a, b):
    """Solve a small dense system with pivoted Gaussian elimination."""
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(len(b)):
        pivot = max(range(col, len(b)), key=lambda row: abs(m[row][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular matrix")
        m[col], m[pivot] = m[pivot], m[col]
        scale = m[col][col]
        m[col] = [v / scale for v in m[col]]
        for row in range(len(b)):
            if row == col:
                continue
            scale = m[row][col]
            m[row] = [m[row][j] - scale * m[col][j] for j in range(len(b) + 1)]
    return [m[i][-1] for i in range(len(b))]


class Kinematics:
    def __init__(self, urdf_path):
        joints = {joint.attrib["name"]: joint for joint in ET.parse(urdf_path).getroot().findall("joint")}
        self.origins = []
        for name in CHAIN:
            origin = joints[name].find("origin")
            xyz = tuple(float(v) for v in origin.attrib.get("xyz", "0 0 0").split())
            rpy = tuple(float(v) for v in origin.attrib.get("rpy", "0 0 0").split())
            self.origins.append(transform(xyz, rpy))

    def fk(self, q):
        pose = transform(BASE_XYZ, BASE_RPY)
        for index, origin in enumerate(self.origins):
            pose = matmul(pose, origin)
            if index < len(JOINTS):
                pose = matmul(pose, transform(rpy=(0.0, 0.0, q[index])))
        return pose

    def error(self, q, target, orient_weight=0.03):
        pose = self.fk(q)
        # gripper_frame_link +Z is the approach direction; point it down.
        axis = (pose[0][2], pose[1][2], pose[2][2])
        position = [target[i] - pose[i][3] for i in range(3)]
        # cross(current_axis, world_down), scaled in metres/radian.
        orientation = [-axis[1], axis[0], 0.0]
        return position + [orient_weight * v for v in orientation]

    def ik(self, target, seed, jaw_direction=None, restarts=28):
        rng = random.Random(101)
        starts = [seed, [0.0, 0.0, 0.0, 0.0, 0.0]]
        starts += [[rng.uniform(lo, hi) for lo, hi in LIMITS] for _ in range(restarts)]
        best = None
        for start in starts:
            q = list(start)
            for _ in range(350):
                err = self.error(q, target)
                if max(abs(value) for value in err) <= 1e-5:
                    break
                # Numerical error Jacobian, dimensions 5 task errors x 5 joints.
                eps = 1e-5
                jac = [[0.0] * 5 for _ in range(5)]
                for col in range(5):
                    moved = q[:]
                    moved[col] += eps
                    moved_err = self.error(moved, target)
                    for row in range(5):
                        jac[row][col] = (moved_err[row] - err[row]) / eps
                # Gauss-Newton: (J^T J + lambda I)dq = -J^T error.
                normal = [[sum(jac[k][i] * jac[k][j] for k in range(5)) + (0.002 if i == j else 0.0)
                           for j in range(5)] for i in range(5)]
                rhs = [-sum(jac[k][i] * err[k] for k in range(5)) for i in range(5)]
                try:
                    delta = solve_linear(normal, rhs)
                except ValueError:
                    break
                for i, (lo, hi) in enumerate(LIMITS):
                    q[i] = min(hi, max(lo, q[i] + min(0.18, max(-0.18, delta[i]))))
            pose = self.fk(q)
            distance = math.sqrt(sum((target[i] - pose[i][3]) ** 2 for i in range(3)))
            downward = -pose[2][2]
            alignment = 1.0
            if jaw_direction is not None:
                # Local Y is approximately the line between the SO-101 jaws.
                horizontal = math.hypot(pose[0][1], pose[1][1])
                alignment = abs((pose[0][1] * jaw_direction[0] + pose[1][1] * jaw_direction[1]) /
                                max(horizontal, 1e-9))
            score = (distance + 0.04 * (1.0 - downward) + 0.035 * (1.0 - alignment) +
                     0.002 * sum((q[i] - seed[i]) ** 2 for i in range(5)))
            if best is None or score < best[0]:
                best = (score, distance, downward, alignment, q)
        if best[1] > 0.008 or best[2] < 0.55 or best[3] < 0.75:
            raise RuntimeError(
                f"no safe IK solution (error={best[1]*1000:.1f} mm, down={best[2]:.2f}, jaw alignment={best[3]:.2f})")
        return best[4], best[1], best[2]


class ApproachNode(Node):
    def __init__(self):
        super().__init__("so101_approach_nut")
        self.positions = None
        self.references = None
        self.position_sequence = 0
        self.create_subscription(JointState, "/joint_states", self._joint_state, 10)
        self.create_subscription(
            JointTrajectoryControllerState,
            "/arm_controller/controller_state",
            self._controller_state,
            20,
        )
        self.client = ActionClient(self, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory")

    def _joint_state(self, msg):
        values = dict(zip(msg.name, msg.position))
        if all(name in values for name in JOINTS + ["gripper"]):
            self.positions = [values[name] for name in JOINTS + ["gripper"]]
            self.position_sequence += 1

    def _controller_state(self, msg):
        if len(msg.reference.positions) != len(msg.joint_names):
            return
        values = dict(zip(msg.joint_names, msg.reference.positions))
        if all(name in values for name in JOINTS + ["gripper"]):
            self.references = [values[name] for name in JOINTS + ["gripper"]]

    def send(
        self,
        start,
        arm_goal,
        gripper_goal,
        duration,
        samples=4,
        phase="motion",
        allow_contact=False,
        stop_on_gripper_contact=False,
        hold_gripper=False,
    ):
        """Execute one synchronized arm/gripper trajectory phase.

        A held gripper keeps the contact preload constant across phase
        boundaries. Contact detection ends a blocked close command once the
        measured jaw position has settled.
        """
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINTS + ["gripper"]
        destination = list(arm_goal) + [gripper_goal]
        if hold_gripper:
            # Preserve contact preload when this goal replaces the preceding
            # trajectory.
            point = JointTrajectoryPoint()
            point.positions = list(start[:5]) + [gripper_goal]
            goal.trajectory.points.append(point)
        for step in range(1, samples + 1):
            alpha = step / samples
            point = JointTrajectoryPoint()
            arm_positions = [start[i] + alpha * (arm_goal[i] - start[i]) for i in range(5)]
            gripper_position = (
                gripper_goal
                if hold_gripper
                else start[5] + alpha * (gripper_goal - start[5])
            )
            point.positions = arm_positions + [gripper_position]
            seconds = duration * alpha
            point.time_from_start.sec = int(seconds)
            point.time_from_start.nanosec = int((seconds - int(seconds)) * 1e9)
            goal.trajectory.points.append(point)
        future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        handle = future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError(f"controller rejected the {phase} trajectory")
        result_future = handle.get_result_async()
        if stop_on_gripper_contact:
            if gripper_goal >= start[5]:
                raise ValueError("gripper contact detection requires a closing trajectory")
            contact_window = deque()
            contact_detected = False
            contact_gripper_command = gripper_goal
            last_position_sequence = self.position_sequence
            wait_deadline = time.monotonic() + duration + 3.0
            while rclpy.ok() and not result_future.done() and time.monotonic() < wait_deadline:
                rclpy.spin_once(self, timeout_sec=0.02)
                if self.position_sequence == last_position_sequence:
                    continue
                last_position_sequence = self.position_sequence
                now = time.monotonic()
                current_gripper = self.positions[5]
                contact_window.append((now, current_gripper))
                while contact_window and contact_window[0][0] < now - 0.30:
                    contact_window.popleft()
                moved = start[5] - current_gripper
                remaining = current_gripper - gripper_goal
                stalled = (
                    len(contact_window) >= 5
                    and contact_window[-1][0] - contact_window[0][0] >= 0.25
                    and max(value for _, value in contact_window)
                    - min(value for _, value in contact_window) <= 0.015
                )
                if moved >= 0.20 and remaining >= 0.05 and stalled:
                    # Preserve the active preload without stepping directly to
                    # the hard-close limit.
                    if self.references is not None:
                        contact_gripper_command = min(
                            self.positions[5] - 0.02,
                            self.references[5],
                        )
                        contact_gripper_command = max(gripper_goal, contact_gripper_command)
                    cancel_future = handle.cancel_goal_async()
                    cancel_deadline = time.monotonic() + 1.0
                    while (rclpy.ok() and not cancel_future.done()
                           and time.monotonic() < cancel_deadline):
                        rclpy.spin_once(self, timeout_sec=0.02)
                    response = cancel_future.result() if cancel_future.done() else None
                    if response is not None and response.goals_canceling:
                        contact_detected = True
                        # Wait for the controller to release the canceled goal.
                        release_deadline = time.monotonic() + 0.5
                        while (rclpy.ok() and not result_future.done()
                               and time.monotonic() < release_deadline):
                            rclpy.spin_once(self, timeout_sec=0.02)
                        break
            if contact_detected:
                self.get_logger().info(
                    f"{phase}: detected stable contact at measured gripper="
                    f"{self.positions[5]:.3f} rad; holding {contact_gripper_command:.3f} rad "
                    "and beginning lift without goal-timeout pause")
                # Arm motion starts from feedback; the gripper retains its
                # contact-time command reference.
                return list(self.positions[:5]) + [contact_gripper_command]
            if not result_future.done():
                raise RuntimeError(f"{phase} trajectory did not finish within {duration + 3.0:.1f} seconds")
        else:
            rclpy.spin_until_future_complete(self, result_future)
        wrapped = result_future.result()
        if wrapped is None:
            raise RuntimeError(f"{phase} trajectory returned no result")
        if wrapped.status != 4 or wrapped.result.error_code != 0:
            # Object contact can prevent the position-controlled jaw from
            # reaching its empty-hand limit. Arm accuracy remains mandatory.
            arm_error = max(abs(self.positions[i] - arm_goal[i]) for i in range(5))
            gripper_moved = self.positions[5] < start[5] - 0.20
            contact_error = wrapped.result.error_code in (-4, -5)
            # Closing contact allows more arm deflection than transport phases.
            allowed_arm_error = 0.20 if phase == "gripper close" else 0.12
            if (allow_contact and contact_error and arm_error <= allowed_arm_error and
                    (phase != "gripper close" or gripper_moved)):
                self.get_logger().warning(
                    f"{phase}: accepting expected contact stop; arm error={arm_error:.3f} rad, "
                    f"measured gripper={self.positions[5]:.3f} rad (command={gripper_goal:.3f})")
                return list(self.positions[:5]) + [gripper_goal]
            raise RuntimeError(
                f"{phase} trajectory failed (action status={wrapped.status}, "
                f"controller code={wrapped.result.error_code}, detail={wrapped.result.error_string!r}, "
                f"max arm error={arm_error:.3f} rad, measured gripper={self.positions[5]:.3f} rad)")
        return destination


def main():
    parser = argparse.ArgumentParser(description="Approach and rim-grasp a nut at a known world XYZ")
    parser.add_argument("--x", type=float, default=0.10, help="nut centre X in world (m)")
    parser.add_argument("--y", type=float, default=0.13, help="nut centre Y in world (m)")
    parser.add_argument("--z", type=float, default=0.631, help="nut base Z in world (m)")
    parser.add_argument("--clearance", type=float, default=0.080,
                        help="pre-grasp tool height above the nut base (m)")
    parser.add_argument("--grasp-height", type=float, default=0.022,
                        help="tool height above nut base while closing (m)")
    parser.add_argument("--rim-offset", type=float, default=0.0,
                        help="offset from nut centre toward the robot (m); default targets the hole centre")
    parser.add_argument("--open", type=float, default=1.35, dest="open_angle",
                        help="open gripper joint angle (rad)")
    parser.add_argument("--close", type=float, default=-0.174, dest="close_angle",
                        help="closed gripper joint angle (rad)")
    parser.add_argument("--duration", type=float, default=6.0, help="trajectory duration in seconds")
    parser.add_argument("--approach-only", action="store_true", help="stop above the rim without descending or closing")
    parser.add_argument("--grasp-only", action="store_true", help="close on the nut but do not lift and place it")
    parser.add_argument("--place-x", type=float, default=0.08, help="centre X of the other red part (m)")
    parser.add_argument("--place-y", type=float, default=0.02, help="centre Y of the other red part (m)")
    parser.add_argument("--lift-height", type=float, default=0.120, help="tool lift above source nut base (m)")
    parser.add_argument("--urdf", type=Path, required=True, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.clearance < 0.065 or not 0.012 <= args.grasp_height <= 0.055 or args.duration < 2.0:
        parser.error("clearance must be >= 0.065 m, grasp-height 0.012..0.055 m, and duration >= 2 s")
    if not -0.010 <= args.rim_offset <= 0.040 or not -0.174 <= args.close_angle < args.open_angle <= 1.745:
        parser.error("rim-offset or gripper open/close angles are outside their safe ranges")
    if args.lift_height < 0.09:
        parser.error("lift-height must be at least 0.09 m")

    rclpy.init()
    node = ApproachNode()
    deadline = time.monotonic() + 8.0
    while rclpy.ok() and node.positions is None and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
    if node.positions is None:
        raise RuntimeError("no /joint_states received; start run_sim_rviz.sh first")
    if not node.client.wait_for_server(timeout_sec=5.0):
        raise RuntimeError("arm trajectory action is unavailable")

    # The SDF model pose is the nut centre. A positive rim offset moves from
    # that centre toward the robot; zero puts the gripper frame over the hole.
    dx, dy = BASE_XYZ[0] - args.x, BASE_XYZ[1] - args.y
    length = math.hypot(dx, dy)
    rim_x = args.x + args.rim_offset * dx / length
    rim_y = args.y + args.rim_offset * dy / length
    target = (rim_x, rim_y, args.z + args.clearance)
    kinematics = Kinematics(args.urdf)
    radial = (dx / length, dy / length)
    goal_q, error, downward = kinematics.ik(target, node.positions[:5], radial)
    node.get_logger().info(
        f"target world XYZ={target}; IK error={error*1000:.1f} mm, downward alignment={downward:.3f}")

    # Plan all Cartesian phases before execution so control remains continuous
    # throughout the recorded demonstration.
    if not args.approach_only:
        grasp_target = (rim_x, rim_y, args.z + args.grasp_height)
        grasp_q, error, downward = kinematics.ik(grasp_target, goal_q, radial)
        node.get_logger().info(
            f"planned grasp XYZ={grasp_target}; IK error={error*1000:.1f} mm, "
            f"downward alignment={downward:.3f}")
        if not args.grasp_only:
            lift_target = (args.x, args.y, args.z + args.lift_height)
            lift_q, error, downward = kinematics.ik(lift_target, grasp_q, radial)
            place_dx, place_dy = BASE_XYZ[0] - args.place_x, BASE_XYZ[1] - args.place_y
            place_length = math.hypot(place_dx, place_dy)
            place_radial = (place_dx / place_length, place_dy / place_length)
            transfer_target = (args.place_x, args.place_y, args.z + args.lift_height)
            transfer_q, error, downward = kinematics.ik(transfer_target, lift_q, place_radial)
            # Other part is 30 mm tall. Preserve the same tool-to-nut grasp
            # offset, placing the carried nut's base on its top surface.
            place_target = (args.place_x, args.place_y, args.z + 0.030 + args.grasp_height)
            place_q, error, downward = kinematics.ik(place_target, transfer_q, place_radial)
            retreat_q, error, downward = kinematics.ik(transfer_target, place_q, place_radial)
            node.get_logger().info("complete pick-and-place trajectory planned; beginning execution")

    state = node.send(node.positions, goal_q, args.open_angle, args.duration, phase="approach")
    if args.approach_only:
        print(f"Approach complete. Open gripper is above the nut target at {target} m.")
    else:
        node.get_logger().info(f"descending to {grasp_target}")
        state = node.send(state, grasp_q, args.open_angle, max(2.5, args.duration * 0.55), phase="descent")
        node.get_logger().info(f"closing gripper fully to {args.close_angle:.3f} rad across the nut rim")
        state = node.send(state, grasp_q, args.close_angle, 4.0, samples=8,
                          phase="gripper close", allow_contact=True,
                          stop_on_gripper_contact=True)
        if args.grasp_only:
            print("Grasp sequence complete: the gripper is centered over the nut and fully closed.")
        else:
            # First lift vertically; this is also the physical grasp test.
            carry_gripper_goal = state[5]
            node.get_logger().info(f"lifting grasped nut to {lift_target}")
            state = node.send(
                state, lift_q, carry_gripper_goal, 4.0, phase="lift",
                allow_contact=True, hold_gripper=True)

            node.get_logger().info(f"transferring above other part at {transfer_target}")
            state = node.send(
                state, transfer_q, carry_gripper_goal, 5.0, phase="transfer",
                allow_contact=True, hold_gripper=True)

            node.get_logger().info(f"lowering nut onto other part at {place_target}")
            state = node.send(
                state, place_q, carry_gripper_goal, 4.0, phase="place descent",
                allow_contact=True, hold_gripper=True)
            state = node.send(state, place_q, args.open_angle, 3.0, samples=6, phase="release")

            node.get_logger().info("released; retreating vertically")
            node.send(state, retreat_q, args.open_angle, 3.5, phase="retreat")
            print("Pick-and-place sequence complete. Verify that the nut followed the initial lift and remained on the other part.")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ET.ParseError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
