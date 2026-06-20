import logging
import numpy as np


class Estimator:
    def __init__(self, given_parameters):
        self.logger = logging.getLogger(__name__)
        self.given_parameters = given_parameters

        # Time step
        sampling_rate = given_parameters["rocket"]["sensors"]["sampling_rate"]
        self.dt = 1.0 / sampling_rate

        # Init states
        self.pos = np.zeros(3) # [x, y, z]
        self.vel = np.zeros(3) # [vx, vy, vz]
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])  # [qw, qx, qy, qz]
        self.states = np.concatenate([self.pos, self.vel, self.quat])

    def reset(self):
        """
        Resets estimator internal storage states.
        """
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.states = np.concatenate([self.pos, self.vel, self.quat])

    def update(self, observation: dict) -> np.ndarray:
        """
        Estimate the state vector based on IMU and GNSS measurements.

        Parameters
        ----------
        observation : dict
            A dictionary containing the sensor observations (e.g., IMU and GNSS data packets).

        Returns
        -------
        state : np.ndarray
            The estimated 10-dimensional state vector structured as [pos(3), vel(3), quat(4)]:
            - pos : Position coordinates (x, y, z).
            - vel : Velocity components (vx, vy, vz).
            - quat : Orientation quaternion (qw, qx, qy, qz).
        """
        rocket_sensors = observation["rocket_sensors"]

        # gyroscopes will be NaN before launch
        if np.isnan(rocket_sensors[:3]).any():
            return self.states

        # Parse sensor data
        gyro = rocket_sensors[0:3] # gyroscopes
        self.acc = rocket_sensors[3:6] # accelerometers
        self.pos = rocket_sensors[6:9] # GNSS position
        self.vel = rocket_sensors[9:12] # GNSS velocity

        delta_theta = gyro * self.dt
        theta_mag = np.linalg.norm(delta_theta)

        if theta_mag > 1e-8:
            # Generate delta rotation quaternion
            qw_d = np.cos(theta_mag / 2.0)
            qxyz_d = (delta_theta / theta_mag) * np.sin(theta_mag / 2.0)
            q_delta = np.array([qw_d, qxyz_d[0], qxyz_d[1], qxyz_d[2]])

            # Perform quaternion multiplication
            # quat = quat x q_delta
            qw, qx, qy, qz = self.quat
            dw, dx, dy, dz = q_delta

            new_qw = qw * dw - qx * dx - qy * dy - qz * dz
            new_qx = qw * dx + qx * dw + qy * dz - qz * dy
            new_qy = qw * dy - qx * dz + qy * dw + qz * dx
            new_qz = qw * dz + qx * dy - qy * dx + qz * dw

            self.quat = np.array([new_qw, new_qx, new_qy, new_qz])

            # Normalize to eliminate compounding numerical drift errors
            self.quat /= np.linalg.norm(self.quat)

        self.states = np.concatenate([self.pos, self.vel, self.quat])
        return self.states
