import unittest

import numpy as np

from BalloonPoppingGymEnv.agents.gnc_agent import (
    Scenario1GncAgent,
    TargetCandidate,
    vector_to_inclination_heading,
)


def _given_parameters():
    return {
        "environment": {"elevation": 20.0},
        "rocket": {
            "sensors": {"sampling_rate": 100},
            "control": {
                "gimbal_range": 15.0,
                "max_roll_torque": 10.0,
            },
        },
    }


def _candidate(index, score):
    return TargetCandidate(
        index=index,
        score=score,
        aim_point=np.array([0.0, 0.0, 100.0]),
        predicted_position=np.array([0.0, 0.0, 100.0]),
        predicted_velocity=np.zeros(3),
        t_go=1.0,
        features=np.zeros(8),
    )


class TestScenario1GncAgent(unittest.TestCase):
    def test_target_hysteresis_keeps_target_before_min_dwell(self):
        agent = Scenario1GncAgent(_given_parameters(), min_dwell=0.5)
        statuses = np.array([[1], [1]])

        selected = agent._select_target(
            [_candidate(0, 1.0), _candidate(1, 100.0)],
            statuses,
            0.0,
        )
        self.assertEqual(selected.index, 0)

        selected = agent._select_target(
            [_candidate(1, 1.0), _candidate(0, 100.0)],
            statuses,
            0.1,
        )
        self.assertEqual(selected.index, 0)

    def test_target_hysteresis_switches_when_current_target_popped(self):
        agent = Scenario1GncAgent(_given_parameters(), min_dwell=0.5)
        agent._select_target(
            [_candidate(0, 1.0), _candidate(1, 100.0)],
            np.array([[1], [1]]),
            0.0,
        )

        selected = agent._select_target(
            [_candidate(1, 1.0)],
            np.array([[2], [1]]),
            0.1,
        )
        self.assertEqual(selected.index, 1)

    def test_vector_to_inclination_heading_uses_enu_convention(self):
        np.testing.assert_allclose(
            vector_to_inclination_heading(np.array([0.0, 0.0, 1.0])),
            np.array([90.0, 0.0]),
        )
        np.testing.assert_allclose(
            vector_to_inclination_heading(np.array([1.0, 0.0, 0.0])),
            np.array([0.0, 90.0]),
        )
        np.testing.assert_allclose(
            vector_to_inclination_heading(np.array([0.0, 1.0, 0.0])),
            np.array([0.0, 0.0]),
        )

    def test_launch_vector_clamps_low_scenario1_cluster_angle(self):
        agent = Scenario1GncAgent(
            _given_parameters(),
            min_launch_inclination=75.0,
            launch_prediction_time=8.0,
        )
        observation = {
            "simulation_time": 3.5,
            "balloon_status": np.array([[1], [1]]),
            "balloon_states": np.array(
                [
                    [-92.0, 79.0, 26.0, 0.0, 0.0, 6.0],
                    [-94.0, 82.0, 27.0, 0.0, 0.0, 6.0],
                ]
            ),
            "rocket_sensors": np.full(12, np.nan),
        }
        nav = agent._navigation_update(observation, 3.5, mutate=False)

        launch_vector = agent._choose_launch_vector(observation, nav)
        launch_angles = vector_to_inclination_heading(launch_vector)

        self.assertGreaterEqual(launch_angles[0], 75.0)
        self.assertTrue(np.all(np.isfinite(launch_vector)))

    def test_get_action_tvc_is_clipped_and_finite(self):
        agent = Scenario1GncAgent(_given_parameters(), kp_tvc=1000.0)
        agent.rocket_launched = True
        observation = {
            "simulation_time": 7.0,
            "balloon_status": np.array([[1]]),
            "balloon_states": np.array([[100.0, 0.0, 200.0, 0.0, 0.0, 0.0]]),
            "rocket_sensors": np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    20.0,
                    0.0,
                    0.0,
                    100.0,
                ]
            ),
        }

        action = agent.get_action(observation)
        self.assertTrue(np.all(np.isfinite(action["tvc"])))
        self.assertTrue(np.all(np.abs(action["tvc"]) <= 15.0))


if __name__ == "__main__":
    unittest.main()
