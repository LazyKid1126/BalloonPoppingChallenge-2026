"""Monte Carlo evaluator for Scenario1GncAgent variants."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import yaml

from BalloonPoppingGymEnv.agents.gnc_agent import Scenario1GncAgent


DEFAULT_GRID = {
    "success_thresholds": [10, 20],
    "variants": [
        {
            "name": "predictive_nearest",
            "agent_kwargs": {"planner_mode": "predictive_nearest", "pn_gain": 0.0},
        },
        {
            "name": "urgency_aware",
            "agent_kwargs": {"planner_mode": "urgency_aware", "pn_gain": 0.0},
        },
        {
            "name": "cluster_predictive",
            "agent_kwargs": {"planner_mode": "cluster_predictive", "pn_gain": 0.0},
        },
        {
            "name": "cluster_predictive_pn3",
            "agent_kwargs": {"planner_mode": "cluster_predictive", "pn_gain": 3.0},
        },
    ],
}


def parse_seeds(seed_spec: str) -> list[int]:
    seeds = []
    for part in str(seed_spec).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            pieces = [p.strip() for p in part.split(":")]
            if len(pieces) not in (2, 3):
                raise ValueError(f"Invalid seed range: {part}")
            start = int(pieces[0])
            stop = int(pieces[1])
            step = int(pieces[2]) if len(pieces) == 3 else 1
            seeds.extend(range(start, stop, step))
        else:
            seeds.append(int(part))
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def load_variant_grid(path: str | None) -> dict:
    if not path:
        return DEFAULT_GRID
    with open(path, "r", encoding="utf-8-sig") as file:
        grid = yaml.safe_load(file) or {}
    if "variants" not in grid:
        raise ValueError(f"Variant grid {path} must define a variants list")
    grid.setdefault("success_thresholds", [10, 20])
    return grid


def run_single_case(case: dict) -> dict:
    start = time.perf_counter()
    scenario = int(case["scenario"])
    seed = int(case["seed"])
    variant_name = str(case["variant_name"])
    agent_kwargs = dict(case.get("agent_kwargs") or {})
    max_steps = case.get("max_steps")

    result = {
        "variant": variant_name,
        "seed": seed,
        "popped_count": 0,
        "terminated": False,
        "steps": 0,
        "simulation_time": 0.0,
        "target_switch_count": 0,
        "mean_target_dwell_time": 0.0,
        "last_target_index": -1,
        "max_altitude_agl": 0.0,
        "max_speed": 0.0,
        "min_balloon_dist": float("inf"),
        "min_balloon_dist_time": 0.0,
        "tvc_saturation_ratio": 0.0,
        "runtime_sec": 0.0,
        "error": "",
    }

    try:
        from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
        from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

        scenario_parameters, given_parameters = load_scenario_parameters(scenario)
        scenario_parameters["scenario"]["random_seed"] = seed
        env = BalloonPoppingEnv(render_mode=None, parameters=scenario_parameters)
        agent = Scenario1GncAgent(given_parameters, **agent_kwargs)

        observation, info = env.reset(seed=seed)
        terminated = False
        truncated = False
        steps = 0
        elevation = float(given_parameters["environment"]["elevation"])
        gimbal_limit = float(given_parameters["rocket"]["control"]["gimbal_range"])
        max_alt_agl = 0.0
        max_speed = 0.0
        min_bdist = float("inf")
        min_bdist_time = 0.0
        tvc_sat_steps = 0
        tvc_total_steps = 0
        launched = False

        while not (terminated or truncated):
            action = agent.get_action(observation)
            if action["launch"] and not launched:
                launched = True
            observation, _reward, terminated, truncated, info = env.step(action)
            steps += 1

            rs = info["rocket_states"]
            if launched and not np.isnan(rs[0]):
                alt_agl = float(rs[2]) - elevation
                spd = float(np.linalg.norm(rs[3:6]))
                max_alt_agl = max(max_alt_agl, alt_agl)
                max_speed = max(max_speed, spd)

                statuses = np.asarray(observation["balloon_status"]).reshape(-1)
                states = np.asarray(observation["balloon_states"], dtype=float)
                released = np.flatnonzero(statuses == 1)
                if len(released) > 0:
                    dists = np.linalg.norm(states[released, :3] - rs[:3], axis=1)
                    cd = float(np.min(dists))
                    if cd < min_bdist:
                        min_bdist = cd
                        min_bdist_time = float(observation["simulation_time"])

                tvc = action["tvc"]
                tvc_mag = max(abs(float(tvc[0])), abs(float(tvc[1])))
                tvc_total_steps += 1
                if tvc_mag >= gimbal_limit * 0.95:
                    tvc_sat_steps += 1

            if max_steps is not None and steps >= int(max_steps):
                truncated = True

        stats = agent.get_debug_stats()
        result.update(
            {
                "popped_count": int(info["popped_count"]),
                "terminated": bool(terminated),
                "steps": int(steps),
                "simulation_time": float(observation["simulation_time"]),
                "target_switch_count": int(stats.get("target_switch_count", 0)),
                "mean_target_dwell_time": float(
                    stats.get("mean_target_dwell_time", 0.0)
                ),
                "last_target_index": int(stats.get("last_target_index", -1)),
                "max_altitude_agl": round(max_alt_agl, 2),
                "max_speed": round(max_speed, 2),
                "min_balloon_dist": round(min_bdist, 2) if min_bdist < 1e6 else -1.0,
                "min_balloon_dist_time": round(min_bdist_time, 2),
                "tvc_saturation_ratio": round(
                    tvc_sat_steps / max(tvc_total_steps, 1), 4
                ),
            }
        )
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        result["runtime_sec"] = time.perf_counter() - start

    return result


def reduce_metrics(rows: list[dict], success_thresholds: list[int]) -> dict:
    by_variant: dict[str, list[dict]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)

    summary = {}
    for variant, variant_rows in sorted(by_variant.items()):
        counts = [int(row["popped_count"]) for row in variant_rows if not row.get("error")]
        if counts:
            sorted_counts = sorted(counts)
            bottom_n = max(1, int(np.ceil(0.10 * len(sorted_counts))))
            metrics = {
                "cases": len(variant_rows),
                "valid_cases": len(counts),
                "errors": len(variant_rows) - len(counts),
                "mean": float(statistics.fmean(counts)),
                "median": float(statistics.median(counts)),
                "std": float(statistics.pstdev(counts)) if len(counts) > 1 else 0.0,
                "min": int(min(counts)),
                "max": int(max(counts)),
                "bottom_10_percent_mean": float(
                    statistics.fmean(sorted_counts[:bottom_n])
                ),
                "target_switch_count_mean": float(
                    statistics.fmean(
                        float(row.get("target_switch_count", 0.0))
                        for row in variant_rows
                        if not row.get("error")
                    )
                ),
                "mean_target_dwell_time_mean": float(
                    statistics.fmean(
                        float(row.get("mean_target_dwell_time", 0.0))
                        for row in variant_rows
                        if not row.get("error")
                    )
                ),
                "max_altitude_agl_mean": float(
                    statistics.fmean(
                        float(row.get("max_altitude_agl", 0.0))
                        for row in variant_rows
                        if not row.get("error")
                    )
                ),
                "max_speed_mean": float(
                    statistics.fmean(
                        float(row.get("max_speed", 0.0))
                        for row in variant_rows
                        if not row.get("error")
                    )
                ),
                "min_balloon_dist_mean": float(
                    statistics.fmean(
                        float(row.get("min_balloon_dist", -1.0))
                        for row in variant_rows
                        if not row.get("error") and float(row.get("min_balloon_dist", -1.0)) >= 0
                    )
                ) if any(
                    float(row.get("min_balloon_dist", -1.0)) >= 0
                    for row in variant_rows if not row.get("error")
                ) else -1.0,
                "min_balloon_dist_median": float(
                    statistics.median(
                        float(row.get("min_balloon_dist", -1.0))
                        for row in variant_rows
                        if not row.get("error") and float(row.get("min_balloon_dist", -1.0)) >= 0
                    )
                ) if any(
                    float(row.get("min_balloon_dist", -1.0)) >= 0
                    for row in variant_rows if not row.get("error")
                ) else -1.0,
                "tvc_saturation_ratio_mean": float(
                    statistics.fmean(
                        float(row.get("tvc_saturation_ratio", 0.0))
                        for row in variant_rows
                        if not row.get("error")
                    )
                ),
            }
            for threshold in success_thresholds:
                metrics[f"success_rate@{threshold}"] = float(
                    sum(count >= int(threshold) for count in counts) / len(counts)
                )
        else:
            metrics = {
                "cases": len(variant_rows),
                "valid_cases": 0,
                "errors": len(variant_rows),
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0,
                "max": 0,
                "bottom_10_percent_mean": 0.0,
                "target_switch_count_mean": 0.0,
                "mean_target_dwell_time_mean": 0.0,
            }
            for threshold in success_thresholds:
                metrics[f"success_rate@{threshold}"] = 0.0
        summary[variant] = metrics

    return summary


def best_worst_lists(rows: list[dict], limit: int = 10) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["variant"]), []).append(row)

    output = {}
    for variant, variant_rows in sorted(grouped.items()):
        valid = [row for row in variant_rows if not row.get("error")]
        ordered = sorted(valid, key=lambda row: (int(row["popped_count"]), int(row["seed"])))
        output[variant] = {
            "worst": ordered[:limit],
            "best": list(reversed(ordered[-limit:])),
        }
    return output


def write_outputs(output_dir: Path, rows: list[dict], summary: dict, best_worst: dict):
    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "variant",
        "seed",
        "popped_count",
        "terminated",
        "steps",
        "simulation_time",
        "target_switch_count",
        "mean_target_dwell_time",
        "last_target_index",
        "runtime_sec",
        "error",
    ]
    with open(output_dir / "per_seed.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    with open(output_dir / "best_worst.json", "w", encoding="utf-8") as file:
        json.dump(best_worst, file, indent=2)


def build_cases(args, grid: dict, seeds: list[int]) -> list[dict]:
    cases = []
    for variant in grid["variants"]:
        name = str(variant["name"])
        agent_kwargs = dict(variant.get("agent_kwargs") or {})
        for seed in seeds:
            cases.append(
                {
                    "scenario": args.scenario,
                    "seed": seed,
                    "variant_name": name,
                    "agent_kwargs": agent_kwargs,
                    "max_steps": args.max_steps,
                }
            )
    return cases


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=int, default=1)
    parser.add_argument("--seeds", default="0:100")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--variant-grid", default="configs/gnc_s1_grid.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    seeds = parse_seeds(args.seeds)
    grid = load_variant_grid(args.variant_grid)
    cases = build_cases(args, grid, seeds)
    thresholds = [int(value) for value in grid.get("success_thresholds", [10, 20])]

    timestamp = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (
            Path("BalloonPoppingGymEnv")
            / "evaluation"
            / "results"
            / "gnc_mc"
            / timestamp
        )

    rows = []
    if args.workers <= 1:
        for case in cases:
            rows.append(run_single_case(case))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_single_case, case) for case in cases]
            for future in as_completed(futures):
                rows.append(future.result())

    rows.sort(key=lambda row: (str(row["variant"]), int(row["seed"])))
    summary = reduce_metrics(rows, thresholds)
    best_worst = best_worst_lists(rows)
    write_outputs(output_dir, rows, summary, best_worst)

    print(f"Wrote Monte Carlo results to {output_dir}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
