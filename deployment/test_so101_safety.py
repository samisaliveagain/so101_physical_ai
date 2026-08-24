#!/usr/bin/env python3
"""Non-hardware tests for the slow-rollout safety envelope."""

from __future__ import annotations

import unittest
from threading import Event

import numpy as np

from so101_safety import (
    SafetyLimits,
    evaluate_tracking,
    plan_quintic_trajectory,
    sanitize_first_action,
    validate_action_chunk,
)
from so101_http_slow_rollout import execute_and_verify


def limits() -> SafetyLimits:
    return SafetyLimits.from_config(
        {
            "joint_names": ["a", "b", "c", "d", "e", "f"],
            "joint_min": [-100] * 6,
            "joint_max": [100] * 6,
            "max_model_delta": [5] * 6,
            "max_chunk_delta": [3] * 6,
            "max_velocity": [5] * 6,
            "max_acceleration": [12] * 6,
            "tracking_tolerance": [1] * 6,
            "minimum_commanded_motion": [0.25] * 6,
            "minimum_tracking_fraction": 0.25,
            "control_hz": 30,
            "min_move_duration_s": 2,
            "max_move_duration_s": 8,
            "settle_timeout_s": 4,
            "settle_required_samples": 3,
            "max_response_age_ms": 3000,
        }
    )


class SafetyTests(unittest.TestCase):
    def test_accepts_bounded_chunk(self) -> None:
        chunk = np.asarray([[1] * 6, [2] * 6, [3] * 6], dtype=float)
        np.testing.assert_array_equal(validate_action_chunk(chunk, np.zeros(6), limits()), chunk)

    def test_rejects_nonfinite_action(self) -> None:
        chunk = np.zeros((7, 6))
        chunk[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_action_chunk(chunk, np.zeros(6), limits())

    def test_rejects_joint_limit_violation(self) -> None:
        chunk = np.zeros((7, 6))
        chunk[-1, 2] = 101
        with self.assertRaisesRegex(ValueError, "joint limits"):
            validate_action_chunk(chunk, np.zeros(6), limits())

    def test_rejects_first_action_jump(self) -> None:
        chunk = np.zeros((7, 6))
        chunk[:, 0] = 5.1
        with self.assertRaisesRegex(ValueError, "first model action"):
            validate_action_chunk(chunk, np.zeros(6), limits())

    def test_rejects_within_chunk_jump(self) -> None:
        chunk = np.zeros((7, 6))
        chunk[1:, 1] = 3.1
        with self.assertRaisesRegex(ValueError, "consecutive-action"):
            validate_action_chunk(chunk, np.zeros(6), limits())

    def test_sanitizer_clips_first_position_and_delta(self) -> None:
        chunk = np.zeros((7, 6))
        chunk[0, 0] = 120
        chunk[0, 1] = 7
        target, report = sanitize_first_action(chunk, np.zeros(6), limits())
        self.assertEqual(target[0], 5)
        self.assertEqual(target[1], 5)
        self.assertIn("a", report["position_clipped_joints"])
        self.assertEqual(set(report["delta_clipped_joints"]), {"a", "b"})
        self.assertTrue(report["clipped"])

    def test_sanitizer_warns_but_ignores_unused_future_violation(self) -> None:
        chunk = np.zeros((7, 6))
        chunk[-1, 2] = 120
        target, report = sanitize_first_action(chunk, np.zeros(6), limits())
        np.testing.assert_array_equal(target, np.zeros(6))
        self.assertEqual(report["future_position_violation_count"], 1)
        self.assertGreater(report["future_transition_violation_count"], 0)
        self.assertFalse(report["clipped"])

    def test_sanitizer_still_rejects_nonfinite_output(self) -> None:
        chunk = np.zeros((7, 6))
        chunk[-1, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            sanitize_first_action(chunk, np.zeros(6), limits())

    def test_trajectory_respects_discrete_rate_limits(self) -> None:
        safe = limits()
        start = np.zeros(6)
        trajectory = plan_quintic_trajectory(start, np.full(6, 4.0), safe)
        full = np.vstack([start, trajectory])
        dt = 1 / safe.control_hz
        self.assertTrue(np.all(np.abs(np.diff(full, axis=0)) / dt <= safe.max_velocity + 1e-9))
        self.assertTrue(
            np.all(np.abs(np.diff(full, n=2, axis=0)) / dt**2 <= safe.max_acceleration + 1e-9)
        )
        np.testing.assert_allclose(trajectory[-1], np.full(6, 4.0))

    def test_trajectory_refuses_impossible_duration(self) -> None:
        safe = limits()
        impossible = SafetyLimits(
            **{**safe.__dict__, "max_velocity": np.full(6, 0.01), "max_move_duration_s": 2.0}
        )
        with self.assertRaisesRegex(ValueError, "exceed"):
            plan_quintic_trajectory(np.zeros(6), np.ones(6), impossible)

    def test_tracking_requires_motion_and_target_accuracy(self) -> None:
        safe = limits()
        start = np.zeros(6)
        target = np.full(6, 4.0)
        success, report = evaluate_tracking(start, target, np.full(6, 3.5), safe)
        self.assertTrue(success)
        self.assertTrue(report["success"])
        success, _ = evaluate_tracking(start, target, np.zeros(6), safe)
        self.assertFalse(success)

    def test_tracking_rejects_no_visible_command(self) -> None:
        safe = limits()
        success, report = evaluate_tracking(
            np.zeros(6), np.full(6, 0.1), np.full(6, 0.1), safe
        )
        self.assertFalse(success)
        self.assertEqual(report["intended_joints"], [])


class FakeHardware:
    def __init__(self, state: np.ndarray, follows_commands: bool):
        self._state = state.copy()
        self.follows_commands = follows_commands
        self.command_count = 0

    def command(self, target: np.ndarray) -> np.ndarray:
        self.command_count += 1
        if self.follows_commands:
            self._state = target.copy()
        return target.copy()

    def state(self) -> np.ndarray:
        return self._state.copy()


class FeedbackExecutionTests(unittest.TestCase):
    def test_final_target_is_held_until_feedback_confirms_motion(self) -> None:
        safe = SafetyLimits(**{**limits().__dict__, "settle_required_samples": 2})
        hardware = FakeHardware(np.zeros(6), follows_commands=True)
        target = np.full(6, 2.0)
        measured, report = execute_and_verify(
            hardware, np.zeros(6), target, target[None, :], safe, Event()
        )
        np.testing.assert_allclose(measured, target)
        self.assertTrue(report["success"])
        self.assertGreaterEqual(hardware.command_count, 3)

    def test_stalled_robot_times_out_instead_of_claiming_success(self) -> None:
        safe = SafetyLimits(
            **{
                **limits().__dict__,
                "settle_timeout_s": 0.08,
                "settle_required_samples": 1,
            }
        )
        hardware = FakeHardware(np.zeros(6), follows_commands=False)
        target = np.full(6, 2.0)
        with self.assertRaisesRegex(RuntimeError, "did not visibly reach"):
            execute_and_verify(hardware, np.zeros(6), target, np.empty((0, 6)), safe, Event())


if __name__ == "__main__":
    unittest.main()
