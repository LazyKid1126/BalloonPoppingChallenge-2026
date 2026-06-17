import numpy as np


class Controller:
    def __init__(self, given_parameters):
        self.given_parameters = given_parameters
        control_cfg = given_parameters["rocket"]["control"]
        self.max_gimbal = control_cfg["gimbal_range"]
        self.max_roll = control_cfg["max_roll_torque"]
        self.throttle_min = control_cfg["throttle_range"][0]
        self.throttle_max = control_cfg["throttle_range"][1]

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
