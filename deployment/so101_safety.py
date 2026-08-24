"""Pure, hardware-independent safety checks for slow SO-101 rollouts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SafetyLimits:
    joint_names: tuple[str, ...]
    joint_min: np.ndarray
    joint_max: np.ndarray
    max_model_delta: np.ndarray
    max_chunk_delta: np.ndarray
    max_velocity: np.ndarray
    max_acceleration: np.ndarray
    tracking_tolerance: np.ndarray
    minimum_commanded_motion: np.ndarray
    minimum_tracking_fraction: float
    control_hz: float
    min_move_duration_s: float
    max_move_duration_s: float
    settle_timeout_s: float
    settle_required_samples: int
    max_response_age_ms: float

    @classmethod
    def from_config(cls, config: dict) -> "SafetyLimits":
        names = tuple(config["joint_names"])
        if len(names) != 6 or len(set(names)) != 6:
            raise ValueError("safety.joint_names must contain six unique names")

        def vector(key: str) -> np.ndarray:
            value = np.asarray(config[key], dtype=np.float64)
            if value.shape != (6,) or not np.isfinite(value).all():
                raise ValueError(f"safety.{key} must contain six finite values")
            return value

        limits = cls(
            joint_names=names,
            joint_min=vector("joint_min"),
            joint_max=vector("joint_max"),
            max_model_delta=vector("max_model_delta"),
            max_chunk_delta=vector("max_chunk_delta"),
            max_velocity=vector("max_velocity"),
            max_acceleration=vector("max_acceleration"),
            tracking_tolerance=vector("tracking_tolerance"),
            minimum_commanded_motion=vector("minimum_commanded_motion"),
            minimum_tracking_fraction=float(config["minimum_tracking_fraction"]),
            control_hz=float(config["control_hz"]),
            min_move_duration_s=float(config["min_move_duration_s"]),
            max_move_duration_s=float(config["max_move_duration_s"]),
            settle_timeout_s=float(config["settle_timeout_s"]),
            settle_required_samples=int(config["settle_required_samples"]),
            max_response_age_ms=float(config["max_response_age_ms"]),
        )
        if np.any(limits.joint_min >= limits.joint_max):
            raise ValueError("every joint_min must be below joint_max")
        for key in (
            "max_model_delta",
            "max_chunk_delta",
            "max_velocity",
            "max_acceleration",
            "tracking_tolerance",
            "minimum_commanded_motion",
        ):
            if np.any(getattr(limits, key) <= 0):
                raise ValueError(f"every safety.{key} value must be positive")
        if limits.control_hz < 10:
            raise ValueError("safety.control_hz must be at least 10 Hz")
        if not 0 < limits.min_move_duration_s <= limits.max_move_duration_s:
            raise ValueError("invalid safety move-duration range")
        if not 0 < limits.minimum_tracking_fraction <= 1:
            raise ValueError("safety.minimum_tracking_fraction must be in (0, 1]")
        if limits.settle_timeout_s <= 0 or limits.settle_required_samples < 1:
            raise ValueError("invalid safety settling configuration")
        if limits.max_response_age_ms <= 0:
            raise ValueError("safety.max_response_age_ms must be positive")
        return limits


def validate_action_chunk(chunk: np.ndarray, observed_state: np.ndarray, limits: SafetyLimits) -> np.ndarray:
    chunk = np.asarray(chunk, dtype=np.float64)
    state = np.asarray(observed_state, dtype=np.float64)
    if chunk.ndim != 2 or chunk.shape[0] < 1 or chunk.shape[1] != 6:
        raise ValueError(f"expected an Nx6 action chunk, got {chunk.shape}")
    if state.shape != (6,):
        raise ValueError(f"expected a six-joint state, got {state.shape}")
    if not np.isfinite(chunk).all() or not np.isfinite(state).all():
        raise ValueError("state and action chunk must be finite")
    if np.any(state < limits.joint_min) or np.any(state > limits.joint_max):
        raise ValueError("observed robot state lies outside configured joint limits")
    if np.any(chunk < limits.joint_min) or np.any(chunk > limits.joint_max):
        raise ValueError("model action lies outside configured joint limits")
    if np.any(np.abs(chunk[0] - state) > limits.max_model_delta):
        raise ValueError("first model action jumps farther than the configured per-joint limit")
    if chunk.shape[0] > 1 and np.any(np.abs(np.diff(chunk, axis=0)) > limits.max_chunk_delta):
        raise ValueError("model chunk contains an excessive consecutive-action jump")
    return chunk


def sanitize_first_action(
    chunk: np.ndarray, observed_state: np.ndarray, limits: SafetyLimits
) -> tuple[np.ndarray, dict]:
    """Clip the one executed action while treating unused future-step violations as warnings.

    Shape and non-finite values remain hard failures. The observed state must also be inside the
    configured envelope; clipping a target cannot make an already unsafe starting state safe.
    """
    chunk = np.asarray(chunk, dtype=np.float64)
    state = np.asarray(observed_state, dtype=np.float64)
    if chunk.ndim != 2 or chunk.shape[0] < 1 or chunk.shape[1] != 6:
        raise ValueError(f"expected an Nx6 action chunk, got {chunk.shape}")
    if state.shape != (6,):
        raise ValueError(f"expected a six-joint state, got {state.shape}")
    if not np.isfinite(chunk).all() or not np.isfinite(state).all():
        raise ValueError("state and action chunk must be finite")
    if np.any(state < limits.joint_min) or np.any(state > limits.joint_max):
        raise ValueError("observed robot state lies outside configured joint limits")

    original = chunk[0].copy()
    position_bounded = np.clip(original, limits.joint_min, limits.joint_max)
    position_clipped = position_bounded != original
    bounded_delta = np.clip(
        position_bounded - state,
        -limits.max_model_delta,
        limits.max_model_delta,
    )
    target = np.clip(state + bounded_delta, limits.joint_min, limits.joint_max)
    delta_clipped = target != position_bounded

    future = chunk[1:]
    future_position_violations = int(
        np.count_nonzero((future < limits.joint_min) | (future > limits.joint_max))
    )
    future_transition_violations = int(
        np.count_nonzero(np.abs(np.diff(chunk, axis=0)) > limits.max_chunk_delta)
    )
    report = {
        "original_target": original.tolist(),
        "accepted_target": target.tolist(),
        "position_clipped_joints": [
            limits.joint_names[index] for index in np.flatnonzero(position_clipped)
        ],
        "delta_clipped_joints": [
            limits.joint_names[index] for index in np.flatnonzero(delta_clipped)
        ],
        "future_position_violation_count": future_position_violations,
        "future_transition_violation_count": future_transition_violations,
        "clipped": bool(np.any(position_clipped) or np.any(delta_clipped)),
    }
    return target, report


def plan_quintic_trajectory(start: np.ndarray, target: np.ndarray, limits: SafetyLimits) -> np.ndarray:
    """Return a sampled minimum-jerk trajectory satisfying discrete velocity/acceleration caps."""
    start = np.asarray(start, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if start.shape != (6,) or target.shape != (6,) or not np.isfinite([start, target]).all():
        raise ValueError("trajectory endpoints must be finite six-joint vectors")
    if np.any(target < limits.joint_min) or np.any(target > limits.joint_max):
        raise ValueError("trajectory target lies outside configured joint limits")

    duration = limits.min_move_duration_s
    sample_period = 1.0 / limits.control_hz
    while duration <= limits.max_move_duration_s + 1e-9:
        count = max(2, int(np.ceil(duration * limits.control_hz)) + 1)
        phase = np.linspace(0.0, 1.0, count)
        blend = 10 * phase**3 - 15 * phase**4 + 6 * phase**5
        trajectory = start + blend[:, None] * (target - start)
        velocity = np.abs(np.diff(trajectory, axis=0)) / sample_period
        acceleration = np.abs(np.diff(trajectory, n=2, axis=0)) / sample_period**2
        velocity_ok = velocity.size == 0 or np.all(velocity <= limits.max_velocity + 1e-9)
        acceleration_ok = acceleration.size == 0 or np.all(acceleration <= limits.max_acceleration + 1e-9)
        if velocity_ok and acceleration_ok:
            return trajectory[1:]
        duration += sample_period
    raise ValueError("safe trajectory would exceed max_move_duration_s")


def evaluate_tracking(
    start: np.ndarray, target: np.ndarray, measured: np.ndarray, limits: SafetyLimits
) -> tuple[bool, dict]:
    """Report whether an intended move is visibly underway and has reached its target tolerance."""
    start = np.asarray(start, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    measured = np.asarray(measured, dtype=np.float64)
    if any(value.shape != (6,) for value in (start, target, measured)):
        raise ValueError("tracking vectors must each contain six joints")
    if not np.isfinite([start, target, measured]).all():
        raise ValueError("tracking vectors must be finite")

    commanded_motion = np.abs(target - start)
    observed_motion = np.abs(measured - start)
    target_error = np.abs(target - measured)
    intended = commanded_motion >= limits.minimum_commanded_motion
    moved_enough = observed_motion >= commanded_motion * limits.minimum_tracking_fraction
    within_tolerance = target_error <= limits.tracking_tolerance
    success = bool(np.any(intended) and np.all(~intended | (moved_enough & within_tolerance)))
    report = {
        "intended_joints": [limits.joint_names[index] for index in np.flatnonzero(intended)],
        "commanded_motion": commanded_motion.tolist(),
        "observed_motion": observed_motion.tolist(),
        "target_error": target_error.tolist(),
        "moved_enough": moved_enough.tolist(),
        "within_tolerance": within_tolerance.tolist(),
        "success": success,
    }
    return success, report
