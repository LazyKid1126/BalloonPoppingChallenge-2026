import numpy as np


class Selector:
    def __init__(self, given_parameters):
        self.given_parameters = given_parameters

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
        Returns target balloon position [x, y, z], or None if no active balloon.
        """
        pass
