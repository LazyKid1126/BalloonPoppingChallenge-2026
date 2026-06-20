import logging
import numpy as np


class Navigator:
    def __init__(self, given_parameters):
        self.logger = logging.getLogger(__name__)
        self.given_parameters = given_parameters

    def reset(self):
        pass

    def compute(self, balloon_state: np.ndarray | None, rocket_state: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Computes the desired angular rates and throttle command.

        Parameters
        ----------
        balloon_state : np.ndarray | None
            Shape (6,), target state [x, y, z, vx, vy, vz] in [m, m/s], or None.
        rocket_state : np.ndarray
            Shape (10,), navigation state [pos(3), vel(3), quat(4)] in [m, m/s, unitless].

        Returns
        -------
        desired_rates : np.ndarray
            Shape (3,), body angular rate commands [wx, wy, wz] in [rad/s].
        desired_throttle : float
            Thrust command bounded within [0.0, 1.0].
        """
        if balloon_state is None:
            desired_rates = np.array([0.0, 0.0, 0.0])
            desired_throttle = 0.15
            return desired_rates, desired_throttle

        rocket_pos = rocket_state[0:3]
        rocket_quat = rocket_state[6:10]

        balloon_pos = balloon_state[0:3]
        balloon_vel = balloon_state[3:6]

        rocket_to_balloon = balloon_pos - rocket_pos
        distance = np.linalg.norm(rocket_to_balloon)

        if distance < 1e-3:
            return np.array([0.0, 0.0, 0.0]), 1.0

        target_dir_world = rocket_to_balloon / distance

        # Transform target direction to rocket body frame
        qw, qx, qy, qz = rocket_quat
        q_vec = np.array([qx, qy, qz])
        t = 2.0 * np.cross(-q_vec, target_dir_world)
        target_dir_body = target_dir_world + qw * t + np.cross(-q_vec, t)

        # Geometric guidance mapping
        guidance_gain = 5.0
        desired_pitch_rate = -guidance_gain * target_dir_body[1]
        desired_yaw_rate = guidance_gain * target_dir_body[0]
        desired_roll_rate = 0.0

        desired_rates = np.array([desired_pitch_rate, desired_yaw_rate, desired_roll_rate])
        desired_throttle = 1.0

        return desired_rates, desired_throttle
