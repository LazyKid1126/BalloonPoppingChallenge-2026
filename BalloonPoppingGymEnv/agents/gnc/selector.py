import logging
import numpy as np
from BalloonPoppingGymEnv.utils.schema import Schema

class Selector:
    def __init__(self, given_parameters):
        self.logger = logging.getLogger(__name__)
        self.given_parameters = given_parameters

        self.current_target_idx = None

    def reset(self):
        pass

    def get_launch_time(self, observation: dict) -> float:
        """
        Returns the desired launch time in seconds.
        """
        return 1.0

    def get_launch_heading(self, observation: dict) -> np.ndarray:
        """
        Returns [inclination, heading] in degrees based on balloon positions.
        """
        return np.array([90.0, 0.0])

    def select(self, observation: dict, rocket_state: np.ndarray) -> np.ndarray | None:
        """
        Selects the target tracking balloon from the active environment cluster.

        Parameters
        ----------
        observation : dict
            Current telemetry dictionary structured as follows:
            - simulation_time : float -> [s]
            - balloon_status : np.ndarray -> shape (N, 1), binary flags [1: active, 0: popped]
            - balloon_states : np.ndarray -> shape (N, 6), tracking states [x, y, z, vx, vy, vz] in [m, m/s]
            - rocket_sensors : np.ndarray -> shape (12,), [gyro(3), acc(3), pos(3), vel(3)] in [rad/s, m/s², m, m/s]

        Returns
        -------
        balloon_state : np.ndarray or None
            A 6-element array containing the full target state vector [x, y, z, vx, vy, vz]
            of the selected balloon, or None if no active targets remain.
        """
        balloon_status = observation[Schema.Observation.BALLOON_STATUS].flatten()
        balloon_states = observation[Schema.Observation.BALLOON_STATES]
        rocket_pos = rocket_state[0:3]

        # Keep tracking current target if it remains active
        if self.current_target_idx is not None and balloon_status[self.current_target_idx] == 1:
            return balloon_states[self.current_target_idx]

        min_dist = float("inf")
        best_target_idx = None

        # Greedy search for the closest active balloon
        for i in range(len(balloon_status)):
            if balloon_status[i] == 1:
                balloon_pos = balloon_states[i, 0:3]
                dist = np.linalg.norm(balloon_pos - rocket_pos)
                if dist < min_dist:
                    min_dist = dist
                    best_target_idx = i

        # Update current target index
        if best_target_idx is not None:
            self.current_target_idx = best_target_idx
            return balloon_states[best_target_idx]

        self.current_target_idx = None
        return None
