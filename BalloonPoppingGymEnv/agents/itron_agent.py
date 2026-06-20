import logging
from BalloonPoppingGymEnv.agents.base_agent import BaseAgent
from BalloonPoppingGymEnv.agents.gnc.estimator import Estimator
from BalloonPoppingGymEnv.agents.gnc.selector import Selector
from BalloonPoppingGymEnv.agents.gnc.navigator import Navigator
from BalloonPoppingGymEnv.agents.gnc.controller import Controller


class ITronAgent(BaseAgent):
    def __init__(self, given_parameters):
        """
        Parameters
        ----------
        given_parameters : dict
            A nested dictionary containing configuration metadata:
            - environment : dict
            - simulation : dict
            - balloon : dict
            - rocket : dict
        """
        super().__init__(given_parameters)
        self.logger = logging.getLogger(__name__)

        # Initialize GNC components
        self.estimator = Estimator(given_parameters)
        self.selector = Selector(given_parameters)
        self.navigator = Navigator(given_parameters)
        self.controller = Controller(given_parameters)

    def reset(self) -> None:
        self.estimator.reset()
        self.selector.reset()
        self.navigator.reset()
        self.controller.reset()

    def get_action(self, observation: dict) -> dict:
        rocket_state = self.estimator.update(observation)
        target = self.selector.select(observation, rocket_state)
        desired_rates, desired_throttle = self.navigator.compute(rocket_state, target)
        tvc, roll, throttle = self.controller.compute(observation["rocket_sensors"][:3], desired_rates, desired_throttle)

        # Set launch parameters
        t = observation["simulation_time"]
        is_launched = t >= self.selector.get_launch_time(observation)
        launch_inclination_heading = self.selector.get_launch_heading(observation)

        return {
            "launch": is_launched,
            "launch_inclination_heading": launch_inclination_heading,
            "tvc": tvc,
            "roll": roll,
            "throttle": throttle,
        }
