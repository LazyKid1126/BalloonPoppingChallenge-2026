"""Scenario 1 GNC baseline agent.

The agent is intentionally self-contained so it can be loaded directly by the
official evaluator. It keeps the model-based path as the default and exposes
optional hooks for RL target selection and TVC residuals.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Callable

import numpy as np

from BalloonPoppingGymEnv.agents.base_agent import BaseAgent


def _norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector))


def _unit(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    length = _norm(vector)
    if length > 1e-9:
        return np.asarray(vector, dtype=float) / length
    if fallback is None:
        fallback = np.array([0.0, 0.0, 1.0])
    return np.asarray(fallback, dtype=float)


def _clip(value: float, low: float, high: float) -> float:
    return float(min(high, max(low, value)))


def rotate_body_to_launch(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a body-frame vector into the launch frame.

    The formula mirrors the quaternion-vector rotation used by the repository's
    NavigationAgent example.
    """

    q0 = float(quaternion[0])
    q_vec = np.asarray(quaternion[1:4], dtype=float)
    vec = np.asarray(vector, dtype=float)
    tmp = 2.0 * np.cross(q_vec, vec)
    return vec + q0 * tmp + np.cross(q_vec, tmp)


def rotate_launch_to_body(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a launch-frame vector into the body frame."""

    q_conj = np.array([quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]])
    return rotate_body_to_launch(q_conj, vector)


def integrate_quaternion(
    quaternion: np.ndarray,
    gyro_body_rad_s: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Integrate attitude using body-frame angular rate."""

    gyro_increment = np.asarray(gyro_body_rad_s, dtype=float) * float(dt)
    angle = _norm(gyro_increment)
    if angle > 1e-12:
        delta_vec = math.sin(angle / 2.0) * gyro_increment / angle
    else:
        delta_vec = np.zeros(3)
    delta_w = math.cos(angle / 2.0)

    e0, e1, e2, e3 = np.asarray(quaternion, dtype=float)
    dx, dy, dz = delta_vec

    new_q = np.array(
        [
            e0 * delta_w - e1 * dx - e2 * dy - e3 * dz,
            e0 * dx + e1 * delta_w + e2 * dz - e3 * dy,
            e0 * dy - e1 * dz + e2 * delta_w + e3 * dx,
            e0 * dz + e1 * dy - e2 * dx + e3 * delta_w,
        ],
        dtype=float,
    )
    return new_q / max(_norm(new_q), 1e-12)


def vector_to_inclination_heading(vector: np.ndarray) -> np.ndarray:
    """Convert launch-frame ENU vector to [inclination, heading] in degrees."""

    east, north, up = np.asarray(vector, dtype=float)
    horizontal = math.hypot(east, north)
    inclination = math.degrees(math.atan2(max(up, 0.0), horizontal))
    heading = math.degrees(math.atan2(east, north)) % 360.0
    if horizontal < 1e-9:
        heading = 0.0
    return np.array([_clip(inclination, 0.0, 90.0), heading], dtype=float)


def inclination_heading_to_vector(inclination_heading: np.ndarray) -> np.ndarray:
    """Convert [inclination, heading] in degrees to a launch-frame unit vector."""

    inclination = math.radians(float(inclination_heading[0]))
    heading = math.radians(float(inclination_heading[1]))
    horizontal = math.cos(inclination)
    return _unit(
        np.array(
            [
                horizontal * math.sin(heading),
                horizontal * math.cos(heading),
                math.sin(inclination),
            ],
            dtype=float,
        )
    )


def quaternion_from_inclination_heading(inclination_heading: np.ndarray) -> np.ndarray:
    """Return the initial attitude quaternion used for launch pointing."""

    inclination = float(inclination_heading[0])
    heading = float(inclination_heading[1])
    psi = math.radians(-heading)
    theta = math.radians(inclination - 90.0)
    phi = 0.0

    q = np.array(
        [
            math.cos(theta / 2.0) * math.cos((phi + psi) / 2.0),
            math.sin(theta / 2.0) * math.cos((phi - psi) / 2.0),
            math.sin(theta / 2.0) * math.sin((phi - psi) / 2.0),
            math.cos(theta / 2.0) * math.sin((phi + psi) / 2.0),
        ],
        dtype=float,
    )
    return q / max(_norm(q), 1e-12)


@dataclass
class NavigationState:
    time: float
    launched: bool
    position: np.ndarray
    velocity: np.ndarray
    quaternion: np.ndarray
    gyro: np.ndarray
    nose_direction: np.ndarray


@dataclass
class TargetCandidate:
    index: int
    score: float
    aim_point: np.ndarray
    predicted_position: np.ndarray
    predicted_velocity: np.ndarray
    t_go: float
    features: np.ndarray


class _NoOpModel:
    def predict(self, observation, deterministic=True):  # noqa: D401
        return None, None


class Scenario1GncAgent(BaseAgent):
    """Scenario 1 model-based GNC baseline with optional RL hooks."""

    def __init__(
        self,
        given_parameters,
        planner_mode: str = "cluster_predictive",
        pn_gain: float = 0.0,
        rl_selector_path: str | None = None,
        rl_residual_path: str | None = None,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__(given_parameters)

        self.planner_mode = str(planner_mode)
        self.pn_gain = float(pn_gain)
        self.debug = bool(debug)

        self.min_launch_time = float(kwargs.get("min_launch_time", 1.0))
        self.max_launch_time = float(kwargs.get("max_launch_time", 6.0))
        self.min_released = int(kwargs.get("min_released", 8))
        self.min_launch_inclination = float(kwargs.get("min_launch_inclination", 85.0))
        self.launch_prediction_time = float(kwargs.get("launch_prediction_time", 12.0))
        self.boost_min_time = float(kwargs.get("boost_min_time", 1.0))
        self.boost_max_time = float(kwargs.get("boost_max_time", 2.5))
        self.guidance_start_altitude = float(kwargs.get("guidance_start_altitude", 25.0))
        self.guidance_start_speed = float(kwargs.get("guidance_start_speed", 15.0))
        self.gravity_turn_altitude = float(kwargs.get("gravity_turn_altitude", 80.0))
        self.gravity_turn_speed = float(kwargs.get("gravity_turn_speed", 40.0))
        self.terminal_intercept_dist = float(kwargs.get("terminal_intercept_dist", 30.0))
        self.terminal_gain_mult = float(kwargs.get("terminal_gain_mult", 1.5))

        self.min_dwell = float(kwargs.get("min_dwell", 0.5))
        self.switch_margin = float(kwargs.get("switch_margin", 0.20))
        self.lookahead_grid = np.array(
            kwargs.get("lookahead_grid", [2.0, 4.0, 6.0, 8.0, 10.0]),
            dtype=float,
        )
        self.cluster_radius = float(kwargs.get("cluster_radius", 25.0))
        self.intercept_speed_floor = float(kwargs.get("intercept_speed_floor", 60.0))
        self.max_t_go = float(kwargs.get("max_t_go", 12.0))
        self.top_k = int(kwargs.get("top_k", 16))

        self.kp_tvc = float(kwargs.get("kp_tvc", 12.0))
        self.kd_tvc = float(kwargs.get("kd_tvc", 2.0))
        self.roll_kd = float(kwargs.get("roll_kd", 2.0))
        self.max_residual_deg = float(kwargs.get("max_residual_deg", 3.0))
        self.tvc_sign_x = float(kwargs.get("tvc_sign_x", 1.0))
        self.tvc_sign_y = float(kwargs.get("tvc_sign_y", 1.0))
        self.throttle = float(kwargs.get("throttle", 1.0))

        control_params = self.given_parameters["rocket"]["control"]
        self.gimbal_limit = float(control_params["gimbal_range"])
        self.roll_limit = float(control_params["max_roll_torque"])
        self.dt = 1.0 / float(self.given_parameters["rocket"]["sensors"]["sampling_rate"])
        self.elevation = float(self.given_parameters["environment"]["elevation"])

        self.rocket_launched = False
        self.launched_at = None
        self.launch_vector = np.array([0.0, 0.0, 1.0], dtype=float)
        self.launch_inclination_heading = np.array([90.0, 0.0], dtype=float)

        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.last_time = None
        self.last_selected_index = None
        self.target_selected_at = 0.0
        self.target_switch_count = 0
        self.dwell_history = []
        self.first_seen_time: dict[int, float] = {}
        self.last_candidates: list[TargetCandidate] = []
        self.last_target_index = None
        self.last_selector_observation = np.zeros((self.top_k, 8), dtype=float)
        self.last_residual_observation = np.zeros(16, dtype=float)

        self.selector_policy: Callable | None = kwargs.get("selector_policy")
        self.residual_policy: Callable | None = kwargs.get("residual_policy")
        self.external_selector_action = None
        self.external_residual_action = None
        self.rl_selector = self._load_optional_model(rl_selector_path)
        self.rl_residual = self._load_optional_model(rl_residual_path)

    def get_action(self, observation):
        t = float(observation["simulation_time"])
        self._update_first_seen(observation, t)
        nav = self._navigation_update(observation, t)

        if not self.rocket_launched:
            self.launch_vector = self._choose_launch_vector(observation, nav)
            self.launch_inclination_heading = vector_to_inclination_heading(
                self.launch_vector
            )
            launch = self._should_launch(observation, t)
            if launch:
                self.rocket_launched = True
                self.launched_at = t
                self.quaternion = quaternion_from_inclination_heading(
                    self.launch_inclination_heading
                )
            return {
                "launch": bool(launch),
                "launch_inclination_heading": self.launch_inclination_heading.copy(),
                "tvc": np.zeros(2, dtype=float),
                "roll": 0.0,
                "throttle": self.throttle,
            }

        candidates = self._build_candidates(observation, nav)
        target = self._select_target(candidates, observation["balloon_status"], t)
        if self._boost_guidance_active(nav):
            tvc, roll = self._compute_control(
                nav,
                None,
                desired_direction=self.launch_vector,
            )
        else:
            blended, in_terminal = self._gravity_turn_blend(nav, target, observation)
            if in_terminal:
                tvc, roll = self._compute_control(
                    nav, target, desired_direction=blended,
                    gain_mult=self.terminal_gain_mult,
                )
            else:
                tvc, roll = self._compute_control(nav, target, desired_direction=blended)

        return {
            "launch": True,
            "launch_inclination_heading": self.launch_inclination_heading.copy(),
            "tvc": tvc,
            "roll": roll,
            "throttle": self.throttle,
        }

    def selector_observation(self, observation) -> np.ndarray:
        t = float(observation["simulation_time"])
        nav = self._navigation_update(observation, t, mutate=False)
        candidates = self._build_candidates(observation, nav)
        return self._candidate_matrix(candidates)

    def residual_observation(self, observation) -> np.ndarray:
        t = float(observation["simulation_time"])
        nav = self._navigation_update(observation, t, mutate=False)
        candidates = self._build_candidates(observation, nav)
        target = candidates[0] if candidates else None
        return self._control_feature_vector(nav, target)

    def get_debug_stats(self) -> dict:
        dwell_values = list(self.dwell_history)
        if self.last_selected_index is not None and self.last_time is not None:
            dwell_values.append(max(0.0, float(self.last_time) - self.target_selected_at))
        mean_dwell = float(np.mean(dwell_values)) if dwell_values else 0.0
        return {
            "target_switch_count": int(self.target_switch_count),
            "mean_target_dwell_time": mean_dwell,
            "last_target_index": (
                int(self.last_target_index) if self.last_target_index is not None else -1
            ),
        }

    def _navigation_update(
        self,
        observation,
        time_sec: float,
        mutate: bool = True,
    ) -> NavigationState:
        sensors = np.asarray(observation["rocket_sensors"], dtype=float)
        gyro = np.zeros(3, dtype=float)
        position = np.array([0.0, 0.0, self.elevation], dtype=float)
        velocity = np.zeros(3, dtype=float)
        quaternion = self.quaternion.copy()

        if not np.isnan(sensors[:3]).any():
            gyro = sensors[:3].copy()
            dt = self.dt if self.last_time is None else max(1e-4, time_sec - self.last_time)
            quaternion = integrate_quaternion(quaternion, gyro, dt)
        if not np.isnan(sensors[6:12]).any():
            position = sensors[6:9].copy()
            velocity = sensors[9:12].copy()

        if not self.rocket_launched and np.isnan(sensors[6:12]).any():
            nose = _unit(self.launch_vector)
        else:
            nose = _unit(rotate_body_to_launch(quaternion, np.array([0.0, 0.0, 1.0])))

        if mutate:
            self.quaternion = quaternion
            self.last_time = time_sec

        return NavigationState(
            time=time_sec,
            launched=self.rocket_launched,
            position=position,
            velocity=velocity,
            quaternion=quaternion,
            gyro=gyro,
            nose_direction=nose,
        )

    def _should_launch(self, observation, time_sec: float) -> bool:
        statuses = np.asarray(observation["balloon_status"]).reshape(-1)
        released = int(np.sum(statuses == 1))
        if time_sec < self.min_launch_time:
            return False
        return released >= self.min_released or time_sec >= self.max_launch_time

    def _choose_launch_vector(self, observation, nav: NavigationState) -> np.ndarray:
        statuses = np.asarray(observation["balloon_status"]).reshape(-1)
        states = np.asarray(observation["balloon_states"], dtype=float)
        released_indices = np.flatnonzero(statuses == 1)
        if len(released_indices) == 0:
            active_indices = np.arange(states.shape[0])
        else:
            active_indices = released_indices

        positions = (
            states[active_indices, :3]
            + states[active_indices, 3:6] * self.launch_prediction_time
        )
        if positions.size == 0:
            return np.array([0.0, 0.0, 1.0], dtype=float)

        best_center = positions[0]
        best_score = -1e9
        for pos in positions:
            distances = np.linalg.norm(positions - pos, axis=1)
            density = float(np.sum(distances <= max(self.cluster_radius, 1.0)))
            horizontal = math.hypot(pos[0], pos[1])
            score = density - 0.01 * horizontal + 0.002 * (pos[2] - self.elevation)
            if score > best_score:
                best_score = score
                best_center = pos
        raw_vector = _unit(best_center - nav.position, np.array([0.0, 0.0, 1.0]))
        launch_angles = vector_to_inclination_heading(raw_vector)
        launch_angles[0] = max(
            float(launch_angles[0]),
            _clip(self.min_launch_inclination, 0.0, 90.0),
        )
        return inclination_heading_to_vector(launch_angles)

    def _boost_guidance_active(self, nav: NavigationState) -> bool:
        if self.launched_at is None:
            return False

        elapsed = nav.time - float(self.launched_at)
        if elapsed < self.boost_min_time:
            return True
        if elapsed >= self.boost_max_time:
            return False

        altitude_agl = float(nav.position[2] - self.elevation)
        speed = _norm(nav.velocity)
        return altitude_agl < self.guidance_start_altitude and speed < self.guidance_start_speed

    def _gravity_turn_blend(
        self,
        nav: NavigationState,
        target: TargetCandidate | None,
        observation=None,
    ) -> tuple[np.ndarray | None, bool]:
        """Blend between vertical climb and target pursuit.

        Returns (blended_direction, in_terminal).
        blended_direction is None when fully transitioned (alpha>=0.99).
        in_terminal is True when within terminal intercept distance.
        """
        altitude_agl = float(nav.position[2] - self.elevation)
        speed = _norm(nav.velocity)

        # Terminal intercept override: if closing on a target, go full authority
        in_terminal = False
        if target is not None and self.terminal_intercept_dist > 0:
            rel = target.aim_point - nav.position
            dist = _norm(rel)
            closing = float(np.dot(nav.velocity, _unit(rel))) if dist > 0.1 else 0.0
            if dist < self.terminal_intercept_dist and closing > 0:
                in_terminal = True
                return None, True  # full authority toward target

        alpha_alt = _clip(altitude_agl / max(self.gravity_turn_altitude, 1.0), 0.0, 1.0)
        alpha_speed = _clip(speed / max(self.gravity_turn_speed, 1.0), 0.0, 1.0)
        alpha = max(alpha_alt, alpha_speed)

        if alpha >= 0.99:
            return None, False

        up = np.array([0.0, 0.0, 1.0], dtype=float)
        if target is None:
            return up, False

        aim_direction = _unit(target.aim_point - nav.position, nav.nose_direction)
        blended = (1.0 - alpha) * up + alpha * aim_direction
        return _unit(blended, nav.nose_direction), False

    def _update_first_seen(self, observation, time_sec: float) -> None:
        statuses = np.asarray(observation["balloon_status"]).reshape(-1)
        for idx in np.flatnonzero(statuses == 1):
            self.first_seen_time.setdefault(int(idx), time_sec)

    def _build_candidates(
        self,
        observation,
        nav: NavigationState,
    ) -> list[TargetCandidate]:
        statuses = np.asarray(observation["balloon_status"]).reshape(-1)
        states = np.asarray(observation["balloon_states"], dtype=float)
        active = np.flatnonzero(statuses == 1)
        candidates = []
        if len(active) == 0:
            self.last_candidates = []
            self.last_selector_observation = np.zeros((self.top_k, 8), dtype=float)
            return []

        fallback_velocity = nav.nose_direction * self.intercept_speed_floor
        rocket_velocity = nav.velocity
        if _norm(rocket_velocity) < 1.0:
            rocket_velocity = fallback_velocity

        positions_at_grid = {
            float(t): states[active, :3] + states[active, 3:6] * float(t)
            for t in self.lookahead_grid
        }

        for idx in active:
            balloon_pos = states[idx, :3]
            balloon_vel = states[idx, 3:6]
            rel = balloon_pos - nav.position
            rel_vel = balloon_vel - rocket_velocity
            rel_vel_sq = max(float(np.dot(rel_vel, rel_vel)), 1e-6)
            t_ca = _clip(-float(np.dot(rel, rel_vel)) / rel_vel_sq, 0.0, self.max_t_go)
            range_now = _norm(rel)
            t_range = _clip(range_now / self.intercept_speed_floor, 0.0, self.max_t_go)
            candidate_times = np.unique(
                np.concatenate((self.lookahead_grid, np.array([t_ca, t_range])))
            )

            best_t = float(candidate_times[0])
            best_miss = float("inf")
            best_pred = balloon_pos.copy()
            best_range = range_now
            best_density = 0.0
            for t_go in candidate_times:
                target_pred = balloon_pos + balloon_vel * t_go
                rocket_pred = nav.position + rocket_velocity * t_go
                miss = _norm(target_pred - rocket_pred)
                density = self._cluster_density(target_pred, positions_at_grid, t_go)
                adjusted_miss = miss - 2.0 * density
                if adjusted_miss < best_miss:
                    best_miss = adjusted_miss
                    best_t = float(t_go)
                    best_pred = target_pred
                    best_range = _norm(target_pred - nav.position)
                    best_density = density

            los = _unit(best_pred - nav.position, nav.nose_direction)
            forward_alignment = float(np.dot(_unit(rocket_velocity), los))
            closing_speed = -float(np.dot(rel_vel, _unit(rel, nav.nose_direction)))
            altitude_error = float(best_pred[2] - nav.position[2])
            target_age = max(0.0, nav.time - self.first_seen_time.get(int(idx), nav.time))
            raw_miss = max(0.0, best_miss + 2.0 * best_density)

            features = np.array(
                [
                    raw_miss,
                    best_range,
                    closing_speed,
                    forward_alignment,
                    altitude_error,
                    best_density,
                    1.0,
                    target_age,
                ],
                dtype=float,
            )
            score = self._score_candidate(features)
            candidates.append(
                TargetCandidate(
                    index=int(idx),
                    score=score,
                    aim_point=best_pred.copy(),
                    predicted_position=best_pred.copy(),
                    predicted_velocity=balloon_vel.copy(),
                    t_go=best_t,
                    features=features,
                )
            )

        candidates.sort(key=lambda candidate: candidate.score)
        self.last_candidates = candidates
        self.last_selector_observation = self._candidate_matrix(candidates)
        return candidates

    def _cluster_density(
        self,
        target_pred: np.ndarray,
        positions_at_grid: dict[float, np.ndarray],
        t_go: float,
    ) -> float:
        nearest_time = min(positions_at_grid.keys(), key=lambda t: abs(t - t_go))
        positions = positions_at_grid[nearest_time]
        distances = np.linalg.norm(positions - target_pred, axis=1)
        return float(np.sum(distances <= self.cluster_radius))

    def _score_candidate(self, features: np.ndarray) -> float:
        miss, distance, closing, forward, altitude, density, _status, age = features
        behind_penalty = 80.0 if forward < -0.10 else 0.0
        low_altitude_penalty = 20.0 if altitude < -10.0 else 0.0

        if self.planner_mode == "predictive_nearest":
            return float(miss + 0.04 * distance - 8.0 * max(forward, 0.0))
        if self.planner_mode == "urgency_aware":
            urgency = max(0.0, closing) + 0.2 * age
            return float(
                miss
                + 0.035 * distance
                + behind_penalty
                + low_altitude_penalty
                - 0.8 * urgency
                - 4.0 * density
            )

        return float(
            miss
            + 0.03 * distance
            + behind_penalty
            + low_altitude_penalty
            - 5.0 * density
            - 12.0 * max(forward, 0.0)
            - 0.2 * age
        )

    def _candidate_matrix(self, candidates: list[TargetCandidate]) -> np.ndarray:
        matrix = np.zeros((self.top_k, 8), dtype=float)
        for row, candidate in enumerate(candidates[: self.top_k]):
            matrix[row, :] = self._normalize_candidate_features(candidate.features)
        return matrix

    def _normalize_candidate_features(self, features: np.ndarray) -> np.ndarray:
        scales = np.array([100.0, 500.0, 150.0, 1.0, 300.0, 20.0, 1.0, 30.0])
        clipped = np.asarray(features, dtype=float) / scales
        return np.clip(clipped, -5.0, 5.0)

    def _select_target(
        self,
        candidates: list[TargetCandidate],
        balloon_status,
        time_sec: float,
    ) -> TargetCandidate | None:
        if not candidates:
            self.last_target_index = None
            return None

        rl_choice = self._rl_select_index(candidates)
        if rl_choice is not None:
            chosen = rl_choice
        else:
            chosen = candidates[0]

        statuses = np.asarray(balloon_status).reshape(-1)
        current = next(
            (
                candidate
                for candidate in candidates
                if candidate.index == self.last_selected_index
            ),
            None,
        )
        current_valid = (
            current is not None
            and self.last_selected_index is not None
            and statuses[int(self.last_selected_index)] == 1
        )

        if current_valid:
            dwell = time_sec - self.target_selected_at
            if dwell < self.min_dwell:
                chosen = current
            else:
                improvement = current.score - chosen.score
                threshold = max(abs(current.score) * self.switch_margin, self.switch_margin)
                if improvement <= threshold:
                    chosen = current

        if self.last_selected_index != chosen.index:
            if self.last_selected_index is not None:
                self.dwell_history.append(max(0.0, time_sec - self.target_selected_at))
                self.target_switch_count += 1
            self.target_selected_at = time_sec
            self.last_selected_index = chosen.index

        self.last_target_index = chosen.index
        return chosen

    def _rl_select_index(
        self,
        candidates: list[TargetCandidate],
    ) -> TargetCandidate | None:
        action = self.external_selector_action
        self.external_selector_action = None

        if action is None and self.selector_policy is not None:
            action = self.selector_policy(self.last_selector_observation.copy())

        if action is None and self.rl_selector is not None:
            action = self._predict_model(self.rl_selector, self.last_selector_observation)

        if action is None:
            return None

        try:
            action_index = int(np.asarray(action).reshape(-1)[0])
        except (TypeError, ValueError):
            return None
        if action_index < 0 or action_index >= min(len(candidates), self.top_k):
            return None
        return candidates[action_index]

    def _compute_control(
        self,
        nav: NavigationState,
        target: TargetCandidate | None,
        desired_direction: np.ndarray | None = None,
        gain_mult: float = 1.0,
    ) -> tuple[np.ndarray, float]:
        if desired_direction is not None:
            desired_direction = _unit(desired_direction, nav.nose_direction)
        elif target is None:
            desired_direction = nav.nose_direction
        else:
            aim_point = target.aim_point.copy()
            if self.pn_gain != 0.0:
                rel = target.predicted_position - nav.position
                los = _unit(rel, nav.nose_direction)
                rel_vel = target.predicted_velocity - nav.velocity
                lateral_vel = rel_vel - float(np.dot(rel_vel, los)) * los
                aim_point = aim_point + 0.02 * self.pn_gain * lateral_vel
            desired_direction = _unit(aim_point - nav.position, nav.nose_direction)

        error_launch = np.cross(nav.nose_direction, desired_direction)
        error_body = rotate_launch_to_body(nav.quaternion, error_launch)

        kp = self.kp_tvc * gain_mult
        kd = self.kd_tvc * gain_mult
        tvc = np.array(
            [
                self.tvc_sign_x * (kp * error_body[0] - kd * nav.gyro[0]),
                self.tvc_sign_y * (kp * error_body[1] - kd * nav.gyro[1]),
            ],
            dtype=float,
        )
        residual = self._rl_residual(nav, target, error_body, desired_direction)
        tvc = tvc + residual
        tvc = np.nan_to_num(tvc, nan=0.0, posinf=self.gimbal_limit, neginf=-self.gimbal_limit)
        tvc = np.clip(tvc, -self.gimbal_limit, self.gimbal_limit)

        roll = _clip(-self.roll_kd * float(nav.gyro[2]), -self.roll_limit, self.roll_limit)
        return tvc.astype(float), roll

    def _rl_residual(
        self,
        nav: NavigationState,
        target: TargetCandidate | None,
        error_body: np.ndarray,
        desired_direction: np.ndarray,
    ) -> np.ndarray:
        features = self._control_feature_vector(nav, target, error_body, desired_direction)
        self.last_residual_observation = features

        action = self.external_residual_action
        self.external_residual_action = None

        if action is None and self.residual_policy is not None:
            action = self.residual_policy(features.copy())

        if action is None and self.rl_residual is not None:
            action = self._predict_model(self.rl_residual, features)

        if action is None:
            return np.zeros(2, dtype=float)
        residual = np.asarray(action, dtype=float).reshape(-1)
        if residual.size < 2:
            return np.zeros(2, dtype=float)
        residual = np.nan_to_num(residual[:2], nan=0.0)
        return np.clip(residual, -self.max_residual_deg, self.max_residual_deg)

    def _control_feature_vector(
        self,
        nav: NavigationState,
        target: TargetCandidate | None,
        error_body: np.ndarray | None = None,
        desired_direction: np.ndarray | None = None,
    ) -> np.ndarray:
        if target is None:
            rel = np.zeros(3, dtype=float)
            target_features = np.zeros(8, dtype=float)
        else:
            rel = target.predicted_position - nav.position
            target_features = self._normalize_candidate_features(target.features)

        if desired_direction is None:
            desired_direction = _unit(rel, nav.nose_direction)
        if error_body is None:
            error_body = rotate_launch_to_body(
                nav.quaternion,
                np.cross(nav.nose_direction, desired_direction),
            )

        features = np.concatenate(
            [
                np.clip(error_body, -1.0, 1.0),
                np.clip(nav.gyro / 5.0, -5.0, 5.0),
                np.clip(desired_direction, -1.0, 1.0),
                np.array([_norm(nav.velocity) / 300.0, _norm(rel) / 1000.0]),
                target_features[:4],
            ]
        )
        if features.size < 16:
            features = np.pad(features, (0, 16 - features.size))
        return features[:16].astype(float)

    def _load_optional_model(self, path: str | None):
        if not path:
            return None
        expanded = os.path.expanduser(os.fspath(path))
        if not os.path.exists(expanded):
            return None

        try:
            from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
        except ImportError:
            return None

        for model_class in (PPO, SAC, TD3, DDPG, A2C):
            try:
                return model_class.load(expanded)
            except Exception:
                continue
        return _NoOpModel()

    @staticmethod
    def _predict_model(model, observation):
        try:
            action, _state = model.predict(observation, deterministic=True)
            return action
        except Exception:
            return None
