import logging
import numpy as np
from BalloonPoppingGymEnv.utils.schema import Schema

class Controller:
    def __init__(self, given_parameters):
        self.logger = logging.getLogger(__name__)
        self.given_parameters = given_parameters

        control_cfg = given_parameters[Schema.Given.Section.ROCKET][Schema.Given.Rocket.CONTROL]
        self.max_gimbal = control_cfg[Schema.Given.Control.GIMBAL_RANGE]
        self.max_roll = control_cfg[Schema.Given.Control.MAX_ROLL_TORQUE]
        self.throttle_min = control_cfg[Schema.Given.Control.THROTTLE_RANGE][0]
        self.throttle_max = control_cfg[Schema.Given.Control.THROTTLE_RANGE][1]

    def reset(self):
        pass

    def compute(self, observation: dict, target_rates: np.ndarray, desired_throttle: float) -> tuple[np.ndarray, float, float]:
        """
        Returns (tvc [x, y], roll, throttle) clipped within actuator limits.
        """

        actual_rates = observation[Schema.Observation.ROCKET_SENSORS][0:3]

        # before launch
        if np.isnan(actual_rates).any():
            actual_rates = np.zeros(3)

        error = target_rates - actual_rates

        # p control
        kp_gimbal = 2.5
        kp_roll = 1.0

        # Compute raw control outputs
        raw_pitch_gimbal = kp_gimbal * error[0]
        raw_yaw_gimbal = kp_gimbal * error[1]
        raw_roll_torque = kp_roll * error[2]

        raw_tvc = np.array([raw_pitch_gimbal, raw_yaw_gimbal])

        tvc = np.clip(raw_tvc, -self.max_gimbal, self.max_gimbal)
        roll = np.clip(raw_roll_torque, -self.max_roll, self.max_roll)
        throttle = np.clip(desired_throttle, self.throttle_min, self.throttle_max)

        return tvc, roll, throttle
