import numpy as np


class Controller:
    def __init__(self, given_parameters):
        self.given_parameters = given_parameters

    def reset(self):
        pass

    def compute(self, gyro: np.ndarray, desired_rates: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Returns (tvc [x, y], roll) from PID tracking of desired_rates.
        """
        pass
