"""Simulator-only three-arm replay of one saved takeoff command sequence."""

from __future__ import annotations

import csv
import math
import os
import time
from pathlib import Path

from align.artifacts import write_json_atomic
from align.tasks.takeoff_replay import REPLAY_ARM_CHOICES, REPLAY_COLUMNS, WAKE_COLUMNS, load_trace
from align.tasks.wake_guard import guard_focal_command


def run_takeoff_replay(*, env, trace_path: Path, output: Path, arm: str, event) -> dict:
    """Replay open-loop world-velocity requests through the unchanged Lee controller."""
    if arm not in REPLAY_ARM_CHOICES or env.num_envs != 1 or env.construction_cfg.num_agents != 4:
        raise ValueError("takeoff replay requires one world, four drones, and a declared arm")
    import torch

    trace = load_trace(trace_path)
    focal_agent = 2
    device = env.device
    drone = env.drone
    reset_state = drone.get_state().clone()
    if arm == "isolated":
        positions = reset_state[..., :3].clone()
        remote = ((-4.0, -4.0, 0.06), (4.0, -4.0, 0.06), (4.0, 4.0, 0.06))
        for agent_id, point in zip((0, 1, 3), remote, strict=True):
            positions[0, agent_id] = torch.tensor(point, device=device)
        rotations = reset_state[..., 3:7].clone()
        ids = torch.tensor([0], device=device)
        drone.set_world_poses(positions + env.envs_positions.unsqueeze(1), rotations, ids)
        drone.set_velocities(torch.zeros(1, 4, 6, device=device), ids)
    if arm == "no_downwash":
        original_downwash = drone.downwash

        def disabled_downwash(p0, p1, p1_t, **kwargs):
            return torch.zeros_like(original_downwash(p0, p1, p1_t, **kwargs))

        drone.downwash = disabled_downwash
    state = drone.get_state().clone()
    focal_reset_error = float(
        (state[0, focal_agent, :13] - reset_state[0, focal_agent, :13]).abs().max()
    )
    if focal_reset_error > 1e-4:
        raise RuntimeError("focal state changed during diagnostic scene intervention")
    event("takeoff_replay_started", arm=arm, steps=len(trace), focal_reset_error=focal_reset_error)
    start = time.perf_counter()
    first_airborne = None
    first_contact = None
    guard_active_steps = 0
    guard_unresolved_steps = 0
    with (output / "trajectory.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=WAKE_COLUMNS if arm == "wake_guard" else REPLAY_COLUMNS
        )
        writer.writeheader()
        with torch.no_grad():
            for step, source_rows in enumerate(trace):
                requested = [
                    [float(source_rows[agent][f"command_v{axis}_m_s"]) for axis in "xyz"]
                    for agent in range(4)
                ]
                phase = source_rows[0]["phase"]
                commands = [vector.copy() for vector in requested]
                pre_positions = tuple(
                    tuple(float(value) for value in point)
                    for point in state[0, :, :3].cpu().tolist()
                )
                guard_details = {"active": False, "unresolved": False}
                if arm == "wake_guard" and phase == "takeoff":
                    guarded, guard_details = guard_focal_command(
                        pre_positions,
                        tuple(tuple(vector) for vector in requested),
                        focal_agent,
                    )
                    commands[focal_agent] = list(guarded)
                    guard_active_steps += int(guard_details["active"])
                    guard_unresolved_steps += int(guard_details["unresolved"])
                if arm == "isolated":
                    for agent in (0, 1, 3):
                        commands[agent] = [0.0, 0.0, 0.0]
                target_vel = torch.tensor([commands], dtype=torch.float32, device=device)
                raw = env.controller.compute(
                    state[..., :13], target_vel=target_vel, target_yaw=env.target_yaw
                )
                if phase == "ground":
                    raw.fill_(-1.0)
                elif arm == "isolated":
                    raw[:, [0, 1, 3]] = -1.0
                if not torch.isfinite(raw).all():
                    raise RuntimeError(f"nonfinite controller result at step {step}")
                applied = raw.clamp(-1.0, 1.0)
                drone.apply_action(applied)
                model_force = drone.forces.clone()
                rotor_thrust = drone.thrusts[..., 2].sum(dim=-1).clone()
                throttle = drone.throttle.clone()
                env.sim.step(False)
                state = drone.get_state().clone()
                contact = (
                    env.base_contact_view.get_net_contact_forces().reshape(1, 4, -1).norm(dim=-1)
                )
                if not torch.isfinite(state).all() or not torch.isfinite(contact).all():
                    raise RuntimeError(f"nonfinite physics state at step {step}")
                state_cpu = state[0].cpu().tolist()
                raw_cpu = raw[0].cpu().tolist()
                applied_cpu = applied[0].cpu().tolist()
                throttle_cpu = throttle[0].cpu().tolist()
                force_cpu = model_force[0].cpu().tolist()
                thrust_cpu = rotor_thrust[0].cpu().tolist()
                contact_cpu = contact[0].cpu().tolist()
                for agent_id in range(4):
                    vector = state_cpu[agent_id]
                    row = {
                        "step": step,
                        "phase": phase,
                        "agent_id": agent_id,
                        **{
                            f"command_v{axis}_m_s": commands[agent_id][index]
                            for index, axis in enumerate("xyz")
                        },
                        **{
                            name: vector[index]
                            for index, name in enumerate(
                                (
                                    "x_m",
                                    "y_m",
                                    "z_m",
                                    "qw",
                                    "qx",
                                    "qy",
                                    "qz",
                                    "vx_m_s",
                                    "vy_m_s",
                                    "vz_m_s",
                                )
                            )
                        },
                        **{f"rotor_raw_{index}": raw_cpu[agent_id][index] for index in range(4)},
                        **{
                            f"rotor_applied_{index}": applied_cpu[agent_id][index]
                            for index in range(4)
                        },
                        **{
                            f"rotor_throttle_{index}": throttle_cpu[agent_id][index]
                            for index in range(4)
                        },
                        "rotor_thrust_sum_n": thrust_cpu[agent_id],
                        **{
                            f"model_force_{axis}_n": force_cpu[agent_id][index]
                            for index, axis in enumerate("xyz")
                        },
                        "contact_force_n": contact_cpu[agent_id],
                    }
                    if arm == "wake_guard":
                        row.update(
                            {
                                **{
                                    f"requested_v{axis}_m_s": requested[agent_id][index]
                                    for index, axis in enumerate("xyz")
                                },
                                **{
                                    f"pre_{axis}_m": pre_positions[agent_id][index]
                                    for index, axis in enumerate("xyz")
                                },
                                "guard_active": int(guard_details["active"])
                                if agent_id == focal_agent
                                else 0,
                                "guard_unresolved": int(guard_details["unresolved"])
                                if agent_id == focal_agent
                                else 0,
                            }
                        )
                    if not all(
                        math.isfinite(float(value)) for key, value in row.items() if key != "phase"
                    ):
                        raise RuntimeError(f"nonfinite replay row at step {step}")
                    writer.writerow(row)
                focal_z = state_cpu[focal_agent][2]
                if first_airborne is None and focal_z >= 0.25:
                    first_airborne = step
                if (
                    first_airborne is not None
                    and first_contact is None
                    and contact_cpu[focal_agent] > 0.01
                ):
                    first_contact = step
                if step % 100 == 0:
                    stream.flush()
        stream.flush()
        os.fsync(stream.fileno())
    metrics = {
        "status": "passed",
        "arm": arm,
        "steps": len(trace),
        "agent_rows": len(trace) * 4,
        "focal_agent": focal_agent,
        "focal_reset_error": focal_reset_error,
        "focal_first_airborne_step": first_airborne,
        "focal_first_post_airborne_contact_step": first_contact,
        "physics_seconds": len(trace) * env.construction_cfg.physics_dt,
        "elapsed_seconds": time.perf_counter() - start,
        "controller": "LeePositionController target_vel world frame, one update per physics step",
        "downwash_override": arm == "no_downwash",
        "guard_active_steps": guard_active_steps,
        "guard_unresolved_steps": guard_unresolved_steps,
        "neighbor_placement": "three passive drones at remote corners"
        if arm == "isolated"
        else "source ground line",
        "training_performed": False,
    }
    write_json_atomic(output / "metrics.json", metrics, mode=0o644)
    event("takeoff_replay_finished", arm=arm, contact_step=first_contact)
    return metrics
