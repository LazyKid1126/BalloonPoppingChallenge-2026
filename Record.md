# GNC Scenario 1 — Experiment Record

> Single source of truth for all MC results, agent configuration snapshots, and strategic decisions.
> Updated: 2026-06-22

---

## Current Strategy

| Role | Variant | Config |
|---|---|---|
| **Baseline / fallback** | `predictive_nearest` + slow blend | `gravity_turn_altitude=80, gravity_turn_speed=40, kp_tvc=12, kd_tvc=2` |
| **Next candidate** | `low_gain_terminal_40m` | `predictive_nearest + kp_tvc=8 + kd_tvc=1.5 + terminal_intercept_dist=40 + terminal_gain_mult=2.0` |
| **Parked** | `cluster_predictive`, `pn3/pn5`, TVC sign flip, high-gain (kp≥18) | MC signal does not support |

---

## Full MC Runs

### Run 1 — `20260621T161955Z`  (Round 1)

| Item | Value |
|---|---|
| **Commit** | `3986852` (Gravity Turn Blend) |
| **Grid** | [gnc_s1_grid.yaml @ R1](configs/gnc_s1_grid.yaml) — 9 variants |
| **Seeds** | `0:100` (100 seeds) × 9 variants = 900 cases |
| **Workers** | 12 |
| **Results dir** | [`BalloonPoppingGymEnv/evaluation/results/gnc_mc/20260621T161955Z/`](BalloonPoppingGymEnv/evaluation/results/gnc_mc/20260621T161955Z/) |
| **Agent defaults** | `gravity_turn_altitude=80, gravity_turn_speed=40, terminal_intercept_dist=0 (not yet implemented)` |

#### Summary

| Variant | Mean | Max | Pops | Avg Flight | Switches |
|---|---|---|---|---|---|
| **predictive_nearest** | **0.04** | 1 | **4** (seeds 2,52,67,96) | 19.1s | 4.3 |
| cluster_predictive | 0.02 | 1 | 2 (seeds 19,40) | 19.6s | 6.4 |
| cluster_predictive_pn3 | 0.02 | 1 | 2 (seeds 19,40) | 19.7s | 6.5 |
| urgency_aware | 0.00 | 0 | 0 | 19.5s | 6.0 |
| cluster_predictive_high_gain | 0.00 | 0 | 0 | 20.1s | 6.7 |
| cluster_predictive_low_gain | 0.00 | 0 | 0 | 17.8s | 5.7 |
| conservative_boost | 0.00 | 0 | 0 | 12.4s | 3.9 |
| sign_x_neg | 0.00 | 0 | 0 | 5.8s | 1.7 |
| sign_y_neg | 0.00 | 0 | 0 | 5.7s | 1.6 |

#### Key Findings
- `predictive_nearest` is the clear winner (4 pops vs next-best 2).
- Negative TVC signs are fatal (5-6s flight time → crash).
- `conservative_boost` (old v2 params) confirms aggressive boost is better (12s vs 19s avg flight).
- `success_rate@10` and `@20` = 0.0 across all variants.

#### Extended Telemetry (predictive_nearest, selected seeds)

| Seed | Popped | Alt AGL | Speed | Min Dist | Min Dist Time | TVC Sat |
|---|---|---|---|---|---|---|
| 2 | **1** ✅ | 58.3m | 67.8m/s | **1.5m** | t=8.4s | 0% |
| 52 | **1** ✅ | 61.3m | 65.8m/s | **1.8m** | t=15.8s | 0% |
| 67 | **1** ✅ | 32.6m | 45.7m/s | **1.6m** | t=10.2s | 0% |
| 96 | **1** ✅ | 78.2m | 103.2m/s | **1.7m** | t=23.0s | 0% |
| 0 | 0 | 77.7m | 45.7m/s | 9.5m | t=11.6s | 0% |
| 1 | 0 | 178.5m | 121.4m/s | 7.0m | t=8.8s | 0% |
| 3 | 0 | 41.8m | 48.8m/s | 14.4m | t=10.8s | 0% |
| 4 | 0 | 109.1m | 57.8m/s | 5.2m | t=9.3s | 0% |

> Pop threshold: swept-path distance ≤ `balloon_radius = 1.5m`. Gap between pop/miss is typically 3-13m.

---

### Run 2 — `20260621T184547Z`  (Round 2)

| Item | Value |
|---|---|
| **Commit** | `3986852` (same, but with faster blend defaults in working tree) |
| **Grid** | gnc_s1_grid.yaml @ R2 — 11 variants |
| **Seeds** | `0:100` × 11 variants = 1100 cases |
| **Workers** | 12 |
| **Results dir** | [`BalloonPoppingGymEnv/evaluation/results/gnc_mc/20260621T184547Z/`](BalloonPoppingGymEnv/evaluation/results/gnc_mc/20260621T184547Z/) |
| **Agent defaults** | `gravity_turn_altitude=40, gravity_turn_speed=25` (fast blend — experimental) |

#### Summary

| Variant | Mean | Max | Pops | Avg Flight | Blend Params |
|---|---|---|---|---|---|
| **slow_blend_baseline** (80/40) | **0.04** | 1 | **4** (seeds 2,52,67,96) | 19.1s | 80/40 |
| low_gain (kp=8) | **0.03** | 1 | **3** (seeds 2,18,55) | 14.1s | 40/25 |
| urgency_aware | 0.02 | 1 | 2 (seeds 18,70) | 14.0s | 40/25 |
| pn5, fast_blend_high_gain_pn3, cluster_predictive | 0.01 | 1 | 1 each | — | 40/25 |
| **predictive_nearest** (new 40/25 default) | **0.00** | 0 | **0** ❌ | 14.6s | 40/25 |
| very_fast_blend (25/20) | 0.00 | 0 | 0 | 13.1s | 25/20 |
| high_gain (kp=18), very_high_gain (kp=24) | 0.00 | 0 | 0 | 15s | 40/25 |

#### Key Findings
- **Faster blend is counterproductive.** `slow_blend_baseline` (80/40) reproduced R1's 4 pops exactly. Fast blend (40/25) scored 0 for `predictive_nearest`.
- Root cause: fast blend turns rocket horizontal too early → max alt drops from 58m to 19m → no energy for second-pass intercept.
- **Low gain (kp=8) shows new signal**: 3 pops, including new seed 18 and 55 not seen in R1.
- Higher gains (kp≥18) produce 0 pops — aggressive steering bleeds energy.
- All `success_rate@10/@20` = 0.0.

#### Mechanism: Seed=2 Fast vs Slow

| Metric | Fast (40/25) | Slow (80/40) |
|---|---|---|
| Max alt AGL | 19m ❌ | 58m ✅ |
| Flight time | 11.8s | 24.3s |
| Min dist | 2.2m (miss) | 1.5m (**pop**) |

---

### Run 3 — Pending

| Item | Value |
|---|---|
| **Commit** | `32d0b3d` (Terminal intercept override) |
| **Grid** | [gnc_s1_grid.yaml @ R3](configs/gnc_s1_grid.yaml) — 13 variants |
| **New features** | `terminal_intercept_dist` (default 30m), `terminal_gain_mult` (default 1.5×) |
| **Agent defaults** | `gravity_turn_altitude=80, gravity_turn_speed=40` (reverted to slow) |
| **Sweep axes** | Low gain (kp 6/8/10) × terminal dist (0/20/30/40m) × terminal gain (1x/1.5x/2x) |
| **Status** | Awaiting execution |

---

## Smoke Tests (Development, Not for Strategy Decisions)

| Tag | Date | Purpose | Result |
|---|---|---|---|
| `smoke_test` | 2026-06-21 | First run, launch bug present | All 0 (crash at 354 steps) |
| `smoke_test_v2` | 2026-06-21 | After launch fix, no blend | All 0 (max alt 33m) |
| `smoke_test_v3` | 2026-06-21 | Gravity turn blend added | `predictive_nearest` seed=2 got 1 pop |

---

## Agent Architecture (as of `32d0b3d`)

```
Launch (t=0 → launch_time)
  └─ _should_launch(): wait for min_released=8 OR t≥6s
  └─ _choose_launch_vector(): cluster-based heading, clamped ≥85° inclination
       └─ launch_prediction_time=12s lookahead

Boost (launch → boost exit)
  └─ _boost_guidance_active(): min 1s, max 2.5s, exit at alt>25m OR speed>15m/s
  └─ TVC holds launch_vector direction

Guidance (boost exit → termination)
  └─ _gravity_turn_blend():
       ├─ Normal: α = max(alt/80m, speed/40m/s), blend up↑ + aim_direction
       └─ Terminal: if dist<30m AND closing → α=1.0, gain_mult=1.5×
  └─ _compute_control(): PD controller, kp×error - kd×gyro, clipped ±15°
  └─ _select_target(): hysteresis (min_dwell=0.5s, switch_margin=20%)
```

### Key Parameters Reference

| Parameter | Default | In Grid Sweep | Notes |
|---|---|---|---|
| `planner_mode` | `predictive_nearest` | ✅ | Winner of R1/R2 |
| `kp_tvc` / `kd_tvc` | 12.0 / 2.0 | ✅ (6-10 range) | Lower is better per R2 |
| `gravity_turn_altitude` | 80.0 | ✅ | Slow=better proven |
| `gravity_turn_speed` | 40.0 | ✅ | Slow=better proven |
| `terminal_intercept_dist` | 30.0 | ✅ (0-40m) | New in R3 |
| `terminal_gain_mult` | 1.5 | ✅ (1-2x) | New in R3 |
| `min_launch_inclination` | 85.0 | Fixed | Prevents near-horizontal launch |
| `pn_gain` | 0.0 | Parked | No clear signal |

---

## Metrics Gap

Current per-seed CSV only records: `popped_count, steps, simulation_time, target_switch_count, mean_target_dwell_time`.

**Missing (needed for next-level analysis):**
- `max_altitude_agl` — distinguishes energy-sufficient from energy-starved runs
- `min_balloon_distance` — the real performance proxy when pops are sparse
- `min_balloon_distance_time` — when the intercept window happens
- `tvc_saturation_ratio` — controller authority usage
- `first_guidance_time` — when boost exits
- `blend_alpha_half_time` — when blend reaches 0.5

> [!IMPORTANT]
> Adding `min_balloon_distance` to the MC CSV output is the single highest-value instrumentation change. With only 4% pop rate, `popped_count` is too sparse for meaningful grid comparison. `min_balloon_distance` is a continuous proxy that every seed produces.
