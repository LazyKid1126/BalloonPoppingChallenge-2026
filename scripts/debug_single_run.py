"""Extended single-seed diagnostic capturing detailed GNC telemetry.

Outputs per-step metrics: altitude, speed, min_balloon_dist, blend_alpha,
TVC saturation, etc.  Run on specific (variant, seed) pairs to understand
near-miss cases.
"""
from __future__ import annotations
import argparse, csv, sys
import numpy as np

from BalloonPoppingGymEnv.agents.gnc_agent import Scenario1GncAgent, _norm, _clip, _unit
from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


def run_diagnostic(scenario: int, seed: int, agent_kwargs: dict) -> dict:
    sp, gp = load_scenario_parameters(scenario)
    sp["scenario"]["random_seed"] = seed
    env = BalloonPoppingEnv(render_mode=None, parameters=sp)
    agent = Scenario1GncAgent(gp, **agent_kwargs)
    obs, info = env.reset(seed=seed)

    elevation = float(gp["environment"]["elevation"])
    gimbal_limit = float(gp["rocket"]["control"]["gimbal_range"])

    telemetry = {
        "max_alt_agl": 0.0,
        "max_speed": 0.0,
        "min_balloon_dist": float("inf"),
        "min_balloon_dist_idx": -1,
        "min_balloon_dist_time": 0.0,
        "flight_time": 0.0,
        "first_guidance_time": None,
        "blend_half_time": None,       # time when alpha first >= 0.5
        "tvc_saturated_steps": 0,
        "tvc_total_steps": 0,
        "steps": 0,
        "popped_count": 0,
    }
    terminated = False
    step = 0
    launched = False
    launch_time = None
    trajectory = []  # sampled every 10 steps

    while not terminated:
        action = agent.get_action(obs)
        t = float(obs["simulation_time"])

        if action["launch"] and not launched:
            launched = True
            launch_time = t

        obs, _, terminated, _, info = env.step(action)
        step += 1
        rs = info["rocket_states"]

        if launched and not np.isnan(rs[0]):
            alt_agl = float(rs[2]) - elevation
            speed = _norm(rs[3:6])
            telemetry["max_alt_agl"] = max(telemetry["max_alt_agl"], alt_agl)
            telemetry["max_speed"] = max(telemetry["max_speed"], speed)

            # Min balloon distance
            statuses = np.asarray(obs["balloon_status"]).reshape(-1)
            states = np.asarray(obs["balloon_states"], dtype=float)
            released = np.flatnonzero(statuses == 1)
            closest_dist = float("inf")
            closest_idx = -1
            if len(released) > 0:
                dists = np.linalg.norm(states[released, :3] - rs[:3], axis=1)
                ci = np.argmin(dists)
                closest_dist = float(dists[ci])
                closest_idx = int(released[ci])
                if closest_dist < telemetry["min_balloon_dist"]:
                    telemetry["min_balloon_dist"] = closest_dist
                    telemetry["min_balloon_dist_idx"] = closest_idx
                    telemetry["min_balloon_dist_time"] = t

            # Boost / guidance / blend state
            from BalloonPoppingGymEnv.agents.gnc_agent import NavigationState
            nav = NavigationState(
                time=t, launched=True, position=rs[:3], velocity=rs[3:6],
                quaternion=rs[6:10] if len(rs) > 9 else np.array([1,0,0,0.]),
                gyro=rs[10:13] if len(rs) > 12 else np.zeros(3),
                nose_direction=np.array([0,0,1.0])
            )
            in_boost = agent._boost_guidance_active(nav)

            if not in_boost and telemetry["first_guidance_time"] is None:
                telemetry["first_guidance_time"] = t

            # Compute blend alpha
            alpha_alt = _clip(alt_agl / max(agent.gravity_turn_altitude, 1.0), 0.0, 1.0)
            alpha_speed = _clip(speed / max(agent.gravity_turn_speed, 1.0), 0.0, 1.0)
            alpha = max(alpha_alt, alpha_speed)
            if alpha >= 0.5 and telemetry["blend_half_time"] is None:
                telemetry["blend_half_time"] = t

            # TVC saturation
            tvc = action["tvc"]
            tvc_mag = max(abs(float(tvc[0])), abs(float(tvc[1])))
            telemetry["tvc_total_steps"] += 1
            if tvc_mag >= gimbal_limit * 0.95:
                telemetry["tvc_saturated_steps"] += 1

            # Sample trajectory
            if step % 10 == 0:
                trajectory.append({
                    "t": t, "alt_agl": alt_agl, "speed": speed,
                    "closest_dist": closest_dist, "closest_idx": closest_idx,
                    "alpha": alpha, "in_boost": in_boost,
                    "tvc_mag": tvc_mag, "vz": float(rs[5]),
                })

    telemetry["flight_time"] = float(obs["simulation_time"])
    telemetry["steps"] = step
    telemetry["popped_count"] = int(info["popped_count"])
    telemetry["tvc_saturation_ratio"] = (
        telemetry["tvc_saturated_steps"] / max(telemetry["tvc_total_steps"], 1)
    )
    telemetry["trajectory"] = trajectory
    telemetry["agent_stats"] = agent.get_debug_stats()
    return telemetry


def print_summary(name: str, seed: int, t: dict):
    print(f"\n{'='*70}")
    print(f"  {name}  seed={seed}  popped={t['popped_count']}")
    print(f"{'='*70}")
    print(f"  Flight time:       {t['flight_time']:.1f}s  ({t['steps']} steps)")
    print(f"  Max altitude AGL:  {t['max_alt_agl']:.1f}m")
    print(f"  Max speed:         {t['max_speed']:.1f}m/s")
    print(f"  Min balloon dist:  {t['min_balloon_dist']:.1f}m (balloon #{t['min_balloon_dist_idx']}) at t={t['min_balloon_dist_time']:.1f}s")
    fg = t['first_guidance_time']
    bh = t['blend_half_time']
    print(f"  First guidance:    t={fg:.1f}s" if fg else "  First guidance:    never")
    print(f"  Blend alpha=0.5:  t={bh:.1f}s" if bh else "  Blend alpha=0.5:  never")
    print(f"  TVC saturation:    {t['tvc_saturation_ratio']*100:.1f}% ({t['tvc_saturated_steps']}/{t['tvc_total_steps']} steps)")
    print(f"  Target switches:   {t['agent_stats']['target_switch_count']}")
    print(f"  Mean dwell time:   {t['agent_stats']['mean_target_dwell_time']:.2f}s")
    
    print(f"\n  Trajectory (sampled every 0.1s):")
    print(f"  {'t':>6s} {'alt':>7s} {'speed':>7s} {'closest':>8s} {'alpha':>6s} {'tvc':>6s} {'vz':>7s} {'phase':>8s}")
    for row in t["trajectory"]:
        phase = "BOOST" if row["in_boost"] else f"a={row['alpha']:.2f}"
        print(f"  {row['t']:6.1f} {row['alt_agl']:7.1f} {row['speed']:7.1f} "
              f"{row['closest_dist']:8.1f} {row['alpha']:6.2f} {row['tvc_mag']:6.2f} "
              f"{row['vz']:7.1f} {phase:>8s}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, default=1)
    parser.add_argument("--seeds", default="19,40,2,52,67,96")
    parser.add_argument("--planner-mode", default="predictive_nearest")
    parser.add_argument("--pn-gain", type=float, default=0.0)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    kwargs = {"planner_mode": args.planner_mode, "pn_gain": args.pn_gain}

    for seed in seeds:
        t = run_diagnostic(args.scenario, seed, kwargs)
        print_summary(args.planner_mode, seed, t)


if __name__ == "__main__":
    main()
