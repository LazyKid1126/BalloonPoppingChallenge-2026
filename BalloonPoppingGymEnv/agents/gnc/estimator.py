import numpy as np


class Estimator:
    def __init__(self, given_parameters):
        self.given_parameters = given_parameters

    def reset(self):
        pass

    def update(self, rocket_sensors: np.ndarray) -> np.ndarray:
        """
        Returns state vector [pos(3), vel(3), quat(4)] estimated from IMU + GNSS.
        """
        pass
