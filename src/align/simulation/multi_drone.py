"""Actual four-drone construction probe; simulator imports occur only inside run()."""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import time
import traceback
from dataclasses import asdict
from importlib import metadata
from itertools import combinations
from pathlib import Path

from align.artifacts import as_ist, utc_now, write_json_atomic
from align.formations import evaluate_formation
from align.simulation.assets import localize_materials
from align.simulation.multi_drone_contract import (
    MultiDroneConfig,
    build_group_layout,
    evaluate,
    phase_at,
    targets_for_phase,
    velocity_actions,
    velocity_commands,
)
from align.simulation.multi_drone_report import save_plots

COLUMNS = [
    "repeat",
    "step",
    "t",
    "phase",
    "agent_id",
    "slot_index",
    "x",
    "y",
    "z",
    "qw",
    "qx",
    "qy",
    "qz",
    "vx",
    "vy",
    "vz",
    "wx",
    "wy",
    "wz",
    "target_x",
    "target_y",
    "target_z",
    "target_vx",
    "target_vy",
    "target_vz",
    "a0",
    "a1",
    "a2",
    "a3",
    "raw_u0",
    "raw_u1",
    "raw_u2",
    "raw_u3",
    "u0",
    "u1",
    "u2",
    "u3",
    "rotor0",
    "rotor1",
    "rotor2",
    "rotor3",
    "contact_force_n",
]


def save_json(path, value):
    """Export probe records for the ordinary host user reading a root container."""
    write_json_atomic(path, value, mode=0o644)


def run(config: MultiDroneConfig, output: Path):
    started = time.perf_counter()
    result = {
        "status": "running",
        "started_at_utc": utc_now(),
        "python": platform.python_version(),
        "shutdown_mode": "fast",
        "rendering_validated": False,
        "drone_physics_tested": False,
        "multi_drone_physics_tested": False,
    }
    result["started_at_ist"] = as_ist(result["started_at_utc"])
    app = None
    rows, resets, episodes = [], [], []

    def event(phase, **details):
        item = {"phase": phase, "timestamp_utc": utc_now(), **details}
        item["timestamp_ist"] = as_ist(item["timestamp_utc"])
        with (output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(item, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(f"{item['timestamp_ist']} IST {phase} {details}", flush=True)

    try:
        event("starting")
        from isaacsim import SimulationApp

        app = SimulationApp({"headless": True, "multi_gpu": False, "fast_shutdown": True})
        result["startup_seconds"] = time.perf_counter() - started
        event("application_started")

        import numpy as np
        import torch
        from omni.isaac.core.objects import GroundPlane
        from omni.isaac.core.simulation_context import SimulationContext
        from omni_drones.controllers import LeePositionController
        from omni_drones.robots.drone import Hummingbird
        from pxr import UsdUtils

        if torch.__version__ != "2.2.2+cu118" or torch.version.cuda != "11.8":
            raise RuntimeError("Vendor PyTorch/CUDA changed")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Expected exactly one allocated GPU")
        torch.set_num_threads(4)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)
        result["versions"] = {
            name: metadata.version(name)
            for name in (
                "torch",
                "torchrl",
                "tensordict",
                "numpy",
                "cloudpickle",
                "omni-drones",
                "align",
            )
        }
        result["torch_path"] = torch.__file__
        result["gpu"] = torch.cuda.get_device_name(0)
        result["packages"] = sorted(
            (
                {"name": dist.metadata["Name"], "version": dist.version}
                for dist in metadata.distributions()
            ),
            key=lambda item: item["name"].lower(),
        )
        sim_params = {
            "gravity": [0, 0, -9.81],
            "use_gpu_pipeline": True,
            "use_gpu": True,
            "use_flatcache": True,
            "enable_stabilization": True,
            "solver_type": 1,
        }
        sim = SimulationContext(
            stage_units_in_meters=1.0,
            physics_dt=config.physics_dt,
            rendering_dt=config.physics_dt,
            sim_params=sim_params,
            backend="torch",
            device="cuda:0",
        )
        layout = build_group_layout(config)
        drone = Hummingbird()
        source_asset = Path(drone.usd_path)
        local_asset = output / "hummingbird-local.usda"
        result["visual_asset_adaptation"] = localize_materials(source_asset, local_asset)
        drone.usd_path = str(local_asset)
        drone.spawn(translations=layout.ground_positions_m)
        GroundPlane("/World/ground", size=20.0)
        sim.reset()
        drone.initialize(track_contact_forces=True)
        if tuple(drone.shape) != (1, config.num_agents):
            raise RuntimeError(
                f"Expected one environment and {config.num_agents} drones, got {drone.shape}"
            )
        controller = LeePositionController(9.81, drone.params).to("cuda:0")
        result["controller"] = {
            key: value.detach().cpu().tolist() for key, value in controller.state_dict().items()
        }
        result["model"] = {
            "parameters": drone.params,
            "base_mass_kg": drone.masses.cpu().tolist(),
            "base_inertias_kg_m2": drone.inertias.cpu().tolist(),
            "rotor_tau_up_per_step": drone.tau_up.cpu().tolist(),
            "rotor_tau_down_per_step": drone.tau_down.cpu().tolist(),
            "drag_coef_actual": drone.drag_coef.cpu().tolist(),
        }
        result["sim_params"] = sim_params
        result["robot_config"] = {
            "rigid_body": vars(drone.rigid_props),
            "articulation": vars(drone.articulation_props),
        }
        result["group_layout"] = asdict(layout)
        result["tensor_contract"] = {
            "state": [1, config.num_agents, 23],
            "high_level_action": [1, config.num_agents, 4],
            "target_world_velocity": [1, config.num_agents, 3],
            "rotor_action": [1, config.num_agents, 4],
            "contact_force": [1, config.num_agents, 3],
        }

        assets = output / "assets"
        assets.mkdir()
        layers, asset_paths, unresolved = UsdUtils.ComputeAllDependencies(drone.usd_path)
        if unresolved:
            raise RuntimeError(f"Unresolved model assets: {unresolved}")
        paths = {Path(layer.realPath) for layer in layers} | {Path(path) for path in asset_paths}
        paths.add(Path(drone.param_path))
        paths.add(source_asset)
        result["assets"] = []
        for index, path in enumerate(sorted(paths)):
            if not path.is_file():
                raise RuntimeError(f"Nonlocal model asset: {path}")
            target = assets / f"{index:02d}-{path.name}"
            shutil.copy2(path, target)
            result["assets"].append(
                {
                    "source": str(path),
                    "saved": str(target.relative_to(output)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        sim.stage.Flatten().Export(str(output / "scene.usda"))
        save_json(output / "runtime.json", result)

        device = torch.device("cuda:0")
        env_ids = torch.tensor([0], device=device)
        zero_velocities = torch.zeros(1, config.num_agents, 6, device=device)
        zero_joints = torch.zeros_like(drone.get_joint_positions())
        reset_positions = torch.tensor(
            [layout.ground_positions_m],
            device=device,
            dtype=torch.float32,
        )
        reset_rotations = torch.zeros(1, config.num_agents, 4, device=device)
        reset_rotations[..., 0] = 1.0
        target_yaw = torch.zeros(1, config.num_agents, 1, device=device)
        physics_started = time.perf_counter()
        with (output / "trajectory.csv").open("x", newline="") as stream, torch.no_grad():
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            for repeat in range(config.repetitions):
                torch.manual_seed(config.seed)
                drone._reset_idx(env_ids, train=False)
                drone.throttle.zero_()
                drone.set_world_poses(reset_positions, reset_rotations)
                drone.set_velocities(zero_velocities)
                drone.set_joint_positions(zero_joints)
                drone.set_joint_velocities(zero_joints)
                drone.forces.zero_()
                state = drone.get_state().clone()
                reset_error = max(
                    (state[..., :3] - reset_positions).abs().max().item(),
                    (state[..., 3:7] - reset_rotations).abs().max().item(),
                    state[..., 7:13].abs().max().item(),
                    drone.throttle.abs().max().item(),
                    drone.get_joint_positions().abs().max().item(),
                    drone.get_joint_velocities().abs().max().item(),
                )
                resets.append(
                    {
                        "repeat": repeat,
                        "max_reset_error": reset_error,
                        "state": state.cpu().tolist(),
                        "throttle": drone.throttle.cpu().tolist(),
                        "controller_state": "stateless; fixed gains, no integrator",
                    }
                )
                save_json(output / "resets.json", {"resets": resets})
                event("episode_start", repeat=repeat, reset_error=reset_error)

                dwell_steps = 0
                termination_reason = "timeout"
                step_count = 0
                for step in range(config.maximum_steps):
                    phase = phase_at(config, step)
                    targets = targets_for_phase(layout, phase)
                    if phase == "ground":
                        velocities = ((0.0, 0.0, 0.0),) * config.num_agents
                        actions = ((0.0, 0.0, 0.0, 0.0),) * config.num_agents
                        raw = torch.full(
                            (1, config.num_agents, 4),
                            -1.0,
                            device=device,
                        )
                    else:
                        positions = tuple(tuple(point) for point in state[0, :, :3].cpu().tolist())
                        velocities = velocity_commands(
                            positions,
                            targets,
                            position_gain_s_inv=config.position_gain_s_inv,
                            max_speed_m_s=config.max_speed_m_s,
                        )
                        actions = velocity_actions(velocities, config.max_speed_m_s)
                        target_velocity = torch.tensor(
                            [velocities],
                            device=device,
                            dtype=torch.float32,
                        )
                        raw = controller.compute(
                            state[..., :13],
                            target_vel=target_velocity,
                            target_yaw=target_yaw,
                        )
                    if not torch.isfinite(state).all() or not torch.isfinite(raw).all():
                        termination_reason = "nonfinite"
                        break
                    applied = raw.clamp(-1, 1)
                    drone.apply_action(applied)
                    sim.step(render=False)
                    state = drone.get_state().clone()
                    contact = (
                        drone.base_link.get_net_contact_forces()
                        .reshape(1, config.num_agents, -1)
                        .norm(dim=-1)
                    )
                    if not torch.isfinite(state).all() or not torch.isfinite(contact).all():
                        termination_reason = "nonfinite"
                        break

                    positions = tuple(tuple(point) for point in state[0, :, :3].cpu().tolist())
                    formation = evaluate_formation(positions, targets)
                    speed = state[..., 7:10].norm(dim=-1)
                    action_tensor = torch.tensor([actions], device=device)
                    target_tensor = torch.tensor([targets], device=device)
                    target_velocity_tensor = torch.tensor([velocities], device=device)
                    for agent in range(config.num_agents):
                        values = {
                            "repeat": repeat,
                            "step": step,
                            "t": (step + 1) * config.physics_dt,
                            "phase": phase,
                            "agent_id": agent,
                            "slot_index": layout.assignment.slot_for_agent[agent],
                            "contact_force_n": contact[0, agent].item(),
                        }
                        for offset, key in enumerate(
                            (
                                "x",
                                "y",
                                "z",
                                "qw",
                                "qx",
                                "qy",
                                "qz",
                                "vx",
                                "vy",
                                "vz",
                                "wx",
                                "wy",
                                "wz",
                            )
                        ):
                            values[key] = state[0, agent, offset].item()
                        for axis, key in enumerate(("target_x", "target_y", "target_z")):
                            values[key] = target_tensor[0, agent, axis].item()
                        for axis, key in enumerate(("target_vx", "target_vy", "target_vz")):
                            values[key] = target_velocity_tensor[0, agent, axis].item()
                        for index in range(4):
                            values[f"a{index}"] = action_tensor[0, agent, index].item()
                            values[f"raw_u{index}"] = raw[0, agent, index].item()
                            values[f"u{index}"] = applied[0, agent, index].item()
                            values[f"rotor{index}"] = drone.throttle[0, agent, index].item()
                        writer.writerow(values)
                        rows.append(values)
                    step_count = step + 1
                    if step % 100 == 0:
                        stream.flush()

                    pair_separation = min(
                        math.dist(first, second) for first, second in combinations(positions, 2)
                    )
                    airborne_contact = any(
                        force > config.contact_force_threshold_n
                        and position[2] > config.airborne_height_m
                        for force, position in zip(
                            contact[0].cpu().tolist(),
                            positions,
                            strict=True,
                        )
                    )
                    outside_envelope = any(
                        abs(position[0]) > config.safety_xy_limit_m
                        or abs(position[1]) > config.safety_xy_limit_m
                        or position[2] < -0.05
                        or position[2] > config.safety_z_limit_m
                        for position in positions
                    )
                    if pair_separation < config.minimum_separation_m:
                        termination_reason = "separation_violation"
                        break
                    if airborne_contact:
                        termination_reason = "airborne_contact"
                        break
                    if outside_envelope:
                        termination_reason = "safety_envelope"
                        break

                    settled = (
                        phase == "formation"
                        and formation.assigned_root_mean_squared_error_m
                        <= config.formation_rmse_tolerance_m
                        and formation.pairwise_root_mean_squared_error_m
                        <= config.pairwise_rmse_tolerance_m
                        and speed.max().item() <= config.speed_tolerance_m_s
                    )
                    dwell_steps = dwell_steps + 1 if settled else 0
                    if dwell_steps >= config.dwell_steps:
                        termination_reason = "success"
                        break

                episode = {
                    "repeat": repeat,
                    "steps": step_count,
                    "termination_reason": termination_reason,
                    "dwell_steps": dwell_steps,
                    "maximum_steps": config.maximum_steps,
                }
                episodes.append(episode)
                save_json(output / "episodes.json", {"episodes": episodes})
                stream.flush()
                os.fsync(stream.fileno())
                event(
                    "episode_finished",
                    repeat=repeat,
                    steps=step_count,
                    termination_reason=termination_reason,
                )

        torch.cuda.synchronize()
        result["physics_loop_wall_seconds"] = time.perf_counter() - physics_started
        result["physics_samples"] = len(rows)
        result["simulated_agent_seconds"] = len(rows) * config.physics_dt
        result["simulated_group_seconds"] = (
            sum(episode["steps"] for episode in episodes) * config.physics_dt
        )
        result["agent_samples_per_wall_second_including_logging"] = (
            len(rows) / result["physics_loop_wall_seconds"]
        )
        result["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["drone_physics_tested"] = True
        result["multi_drone_physics_tested"] = True
        assessment = evaluate(rows, resets, episodes, config)
        save_json(output / "metrics.json", assessment)
        save_plots(rows, output)
        result["status"] = assessment["status"]
        event("checks_finished", status=result["status"])
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        event("failed", error=result["error"])
    finally:
        result["phase"] = "before_close"
        result["finished_at_utc"] = utc_now()
        result["finished_at_ist"] = as_ist(result["finished_at_utc"])
        result["elapsed_seconds"] = time.perf_counter() - started
        save_json(output / "probe-result.json", result)
        event("before_close", status=result["status"])
        print("ALIGN_MULTI_DRONE_RESULT=" + json.dumps(result, allow_nan=False), flush=True)
        if app is not None:
            app.close()
    return 0 if result["status"] == "passed" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-root", action="store_true")
    args = parser.parse_args(argv)
    config = MultiDroneConfig.from_dict(json.loads(args.config.read_text()))
    return run(config, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
