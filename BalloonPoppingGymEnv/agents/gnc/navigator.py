import numpy as np


class Navigator:
    def __init__(self, given_parameters):
        self.given_parameters = given_parameters

    def reset(self):
        pass

    def compute(self, rocket_state: np.ndarray, target: np.ndarray | None) -> tuple[np.ndarray, float]:
        """
        Returns (desired_rates [wx, wy, wz], desired_throttle) to steer toward target.
        """
        desired_rates = np.zeros(3)
        desired_throttle = 1.0
        return desired_rates, desired_throttle
