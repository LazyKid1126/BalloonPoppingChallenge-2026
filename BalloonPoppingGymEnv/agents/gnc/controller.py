import numpy as np
from BalloonPoppingGymEnv.utils.schema import Schema

class Controller:
    def __init__(self, given_parameters):
        self.given_parameters = given_parameters

        control_cfg = given_parameters[Schema.Given.Section.ROCKET][Schema.Given.Rocket.CONTROL]
        self.max_gimbal = control_cfg[Schema.Given.Control.GIMBAL_RANGE]
        self.max_roll = control_cfg[Schema.Given.Control.MAX_ROLL_TORQUE]
        self.throttle_min = control_cfg[Schema.Given.Control.THROTTLE_RANGE][0]
        self.throttle_max = control_cfg[Schema.Given.Control.THROTTLE_RANGE][1]

    def reset(self):
        pass

    def compute(self, gyro: np.ndarray, desired_rates: np.ndarray, desired_throttle: float) -> tuple[np.ndarray, float, float]:
        """
        Returns (tvc [x, y], roll, throttle) clipped within actuator limits.
        """
        tvc = np.clip(np.zeros(2), -self.max_gimbal, self.max_gimbal)
        roll = np.clip(0.0, -self.max_roll, self.max_roll)
        throttle = np.clip(desired_throttle, self.throttle_min, self.throttle_max)

        return tvc, roll, throttle
