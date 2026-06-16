import numpy as np
from BalloonPoppingGymEnv.agents.base_agent import BaseAgent
from BalloonPoppingGymEnv.agents.gnc.estimator import Estimator
from BalloonPoppingGymEnv.agents.gnc.selector import Selector
from BalloonPoppingGymEnv.agents.gnc.navigator import Navigator
from BalloonPoppingGymEnv.agents.gnc.controller import Controller


class iTronAgent(BaseAgent):
    def __init__(self, given_parameters):
        super().__init__(given_parameters)

        # Initialize GNC components
        self.estimator = Estimator(given_parameters)
        self.selector = Selector(given_parameters)
        self.navigator = Navigator(given_parameters)
        self.controller = Controller(given_parameters)

        control_cfg = given_parameters["rocket"]["control"]
        self.max_gimbal = control_cfg["gimbal_range"]
        self.max_roll = control_cfg["max_roll_torque"]
        self.throttle_min = control_cfg["throttle_range"][0]
        self.throttle_max = control_cfg["throttle_range"][1]

    def reset(self) -> None:
        self.estimator.reset()
        self.selector.reset()
        self.navigator.reset()
        self.controller.reset()

    def get_action(self, observation: dict) -> dict:
        rocket_state = self.estimator.update(observation["rocket_sensors"])
        target = self.selector.select(observation, rocket_state)
        desired_rates, throttle = self.navigator.compute(rocket_state, target)
        tvc, roll = self.controller.compute(observation["rocket_sensors"][:3], desired_rates)

        # Set launch parameters
        t = observation["simulation_time"]
        is_launched = t >= self.selector.get_launch_time(observation)
        launch_inclination_heading = self.selector.get_launch_heading(observation)

        # Set control limits
        tvc = np.clip(tvc, -self.max_gimbal, self.max_gimbal)
        roll = np.clip(roll, -self.max_roll, self.max_roll)
        throttle = np.clip(throttle, self.throttle_min, self.throttle_max)

        return {
            "launch": is_launched,
            "launch_inclination_heading": launch_inclination_heading,
            "tvc": tvc,
            "roll": roll,
            "throttle": throttle,
        }
