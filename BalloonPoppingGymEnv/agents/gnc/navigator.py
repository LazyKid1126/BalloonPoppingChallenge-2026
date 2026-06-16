import numpy as np


class Navigator:
    def __init__(self, given_parameters):
        self.given_parameters = given_parameters

    def reset(self):
        pass

    def compute(self, rocket_state: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Returns (desired_rates [wx, wy, wz], throttle) to steer toward target.
        """
        pass
