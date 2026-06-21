"""Detailed step-by-step diagnostic focusing on boost-to-guidance transition."""
import numpy as np
from BalloonPoppingGymEnv.agents.gnc_agent import Scenario1GncAgent, _norm
from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

SCENARIO = 1
SEED = 0

sp, gp = load_scenario_parameters(SCENARIO)
sp["scenario"]["random_seed"] = SEED
env = BalloonPoppingEnv(render_mode=None, parameters=sp)
agent = Scenario1GncAgent(gp, planner_mode="cluster_predictive", debug=True)

obs, info = env.reset(seed=SEED)
terminated = False
step = 0
launch_step = None
prev_in_boost = None

while not terminated:
    action = agent.get_action(obs)
    t = float(obs["simulation_time"])

    if action["launch"] and launch_step is None:
        launch_step = step
        print(f"LAUNCH at step={step} t={t:.2f}s inc={action['launch_inclination_heading'][0]:.1f}°")

    obs, _, terminated, _, info = env.step(action)
    step += 1

    rs = info["rocket_states"]
    if not np.isnan(rs[0]) and launch_step is not None:
        alt = rs[2] - 20.0
        speed = _norm(rs[3:6])
        
        # Check boost state
        from BalloonPoppingGymEnv.agents.gnc_agent import NavigationState
        nav = NavigationState(
            time=t, launched=True,
            position=rs[:3], velocity=rs[3:6],
            quaternion=rs[6:10], gyro=rs[10:13],
            nose_direction=np.array([0,0,1.0])  # dummy
        )
        in_boost = agent._boost_guidance_active(nav)
        
        # Print on transition or every 50 steps
        elapsed = t - agent.launched_at if agent.launched_at else 0
        if in_boost != prev_in_boost or (step - launch_step) % 50 == 0 or terminated:
            target_info = "BOOST (no target tracking)"
            if not in_boost and agent.last_target_index is not None:
                statuses = np.asarray(obs["balloon_status"]).reshape(-1)
                states = np.asarray(obs["balloon_states"], dtype=float)
                ti = agent.last_target_index
                if ti < len(states) and statuses[ti] == 1:
                    tpos = states[ti, :3]
                    target_info = f"GUIDANCE -> balloon[{ti}] at ({tpos[0]:.0f},{tpos[1]:.0f},{tpos[2]:.0f}) dist={_norm(tpos-rs[:3]):.0f}m"
                else:
                    target_info = "GUIDANCE (target lost)"
            
            print(f"  t={t:5.2f}s elapsed={elapsed:4.1f}s alt={alt:6.1f}m speed={speed:5.1f}m/s "
                  f"tvc=({action['tvc'][0]:6.2f},{action['tvc'][1]:6.2f}) "
                  f"vz={rs[5]:6.1f}m/s {target_info}")
            prev_in_boost = in_boost

        if step - launch_step > 1200:
            break

print(f"\nFinal: alt={rs[2]-20:.1f}m speed={_norm(rs[3:6]):.1f}m/s popped={info['popped_count']}")
