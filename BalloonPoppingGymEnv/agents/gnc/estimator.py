import numpy as np


class Estimator:
    def __init__(self, given_parameters):
        self.given_parameters = given_parameters
        self.reset()

    def reset(self):
        self.states = np.zeros(10)
        self.states[2] = self.given_parameters["environment"]["elevation"]

    def update(self, rocket_sensors: np.ndarray) -> np.ndarray:
        """
        Returns state vector [pos(3), vel(3), quat(4)] estimated from IMU + GNSS.
        """
        if np.isnan(rocket_sensors).any():
            return self.states
        self.states[0:3] = rocket_sensors[6:9]
        self.states[3:6] = rocket_sensors[9:12]
        return self.states
