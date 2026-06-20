import numpy as np
import matplotlib.pyplot as plt
from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
from BalloonPoppingGymEnv.agents.itron_agent import ITronAgent
from BalloonPoppingGymEnv.utils.setup_logging import setup_logging

scenario_number = 0

def run_for_development():

    # Load scenario parameters
    scenario_parameters, given_parameters = load_scenario_parameters(scenario_number)

    # Create environment with scenario parameters turn off rendering to make own plots
    env = BalloonPoppingEnv(render_mode=None, parameters=scenario_parameters)

    # Instantiate agent with given parameters and any additional user kwargs
    agent = ITronAgent(given_parameters)

    # use seed=None to randomize environment
    observation, info = env.reset(seed=scenario_parameters["scenario"]["random_seed"])
    terminated = False

    angular_rates = np.full((3, 1), np.nan)
    time = np.full(1, np.nan)

    while not terminated:
        action = agent.get_action(observation)
        observation, reward, terminated, _, info = env.step(action)

        # ground truth angular rates, should not pass to agent
        angular_rates = np.append(angular_rates, info["rocket_states"][10:13].reshape(-1, 1), axis=1)
        time = np.append(time, observation["simulation_time"])

        print(f"simulation_time: {observation['simulation_time']:.2f} sec, reward: {info['popped_count']:.2f}", end='\r')

    plt.subplot(2, 1, 1)
    plt.plot(time, angular_rates[0], 'r-', label='x_rate')
    plt.plot(time, angular_rates[1], 'g-', label='y_rate')
    plt.plot(time, angular_rates[2], 'b-', label='z_rate')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Rates (rad/s)')
    plt.xlim(0, 30)
    plt.ylim(-0.1, 0.1)
    plt.legend()

    # TVC controller observed variables are tuples: (time, gimbal_x, gimbal_y)
    tvc = env._rocket_flight.rocket._controllers[0].observed_variables
    tvc_array = np.array(tvc, dtype=float)
    plt.subplot(2, 1, 2)
    plt.plot(tvc_array[:, 0], tvc_array[:, 1], 'r-', label='tvc_x')
    plt.plot(tvc_array[:, 0], tvc_array[:, 2], 'b-', label='tvc_y')
    plt.xlabel('Time (s)')
    plt.ylabel('TVC Gimbal Angle (deg)')
    plt.xlim(0, 30)
    plt.ylim(-0.1, 0.1)
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(f"Scenario {scenario_number} evaluation completed.")
    print(f"Total reward: {info['popped_count']}")

    # env._rocket_flight.all_info() # Uncomment to print all info from RocketPy

if __name__ == "__main__":
    setup_logging()
    run_for_development()
