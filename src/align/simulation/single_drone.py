"""Actual OmniDrones physics probe; simulator imports occur only inside run()."""

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
from importlib import metadata
from pathlib import Path

from align.artifacts import as_ist, utc_now, write_json_atomic
from align.simulation.assets import localize_materials
from align.simulation.contract import DroneCheckConfig, cases, command_at, evaluate

COLUMNS = [
    "case",
    "repeat",
    "step",
    "t",
    "command_active",
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
]


def save_json(path, value):
    """Export probe records for the ordinary host user reading a root container."""
    write_json_atomic(path, value, mode=0o644)


def save_plots(rows, output):
    """Use vendor matplotlib; retain CSV as the source of every plotted point."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(len(cases()), 2, figsize=(12, 14), constrained_layout=True)
    for row_index, (name, _, _) in enumerate(cases()):
        samples = [r for r in rows if r["case"] == name and r["repeat"] == 0]
        t = [r["t"] for r in samples]
        for axis in "xyz":
            axes[row_index, 0].plot(t, [r[axis] for r in samples], label=axis)
            axes[row_index, 1].plot(t, [r["v" + axis] for r in samples], label="v" + axis)
            axes[row_index, 1].plot(t, [r["target_v" + axis] for r in samples], "--", alpha=0.6)
        axes[row_index, 0].set(title=name + " position", ylabel="m", xlabel="simulation seconds")
        axes[row_index, 1].set(
            title=name + " velocity / dashed target", ylabel="m/s", xlabel="simulation seconds"
        )
        for ax in axes[row_index]:
            ax.grid(alpha=0.3)
            ax.legend()
    fig.savefig(output / "trajectory.png", dpi=150)
    fig.savefig(output / "trajectory.pdf")
    plt.close(fig)


def run(config, output):
    started = time.perf_counter()
    result = {
        "status": "running",
        "started_at_utc": utc_now(),
        "python": platform.python_version(),
        "shutdown_mode": "fast",
        "rendering_validated": False,
        "drone_physics_tested": False,
    }
    result["started_at_ist"] = as_ist(result["started_at_utc"])
    app = None
    rows, resets = [], []

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
            ({"name": d.metadata["Name"], "version": d.version} for d in metadata.distributions()),
            key=lambda d: d["name"].lower(),
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
        drone = Hummingbird()
        source_asset = Path(drone.usd_path)
        local_asset = output / "hummingbird-local.usda"
        result["visual_asset_adaptation"] = localize_materials(source_asset, local_asset)
        drone.usd_path = str(local_asset)
        drone.spawn(translations=[(0.0, 0.0, config.initial_height_m)])
        GroundPlane("/World/ground", size=20.0)
        sim.reset()
        drone.initialize()
        if tuple(drone.shape) != (1, 1):
            raise RuntimeError(f"Expected one environment and one drone, got {drone.shape}")
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
        assets = output / "assets"
        assets.mkdir()
        layers, asset_paths, unresolved = UsdUtils.ComputeAllDependencies(drone.usd_path)
        if unresolved:
            raise RuntimeError(f"Unresolved model assets: {unresolved}")
        paths = {Path(layer.realPath) for layer in layers} | {Path(p) for p in asset_paths}
        paths.add(Path(drone.param_path))
        paths.add(source_asset)
        result["assets"] = []
        for i, path in enumerate(sorted(paths)):
            if not path.is_file():
                raise RuntimeError(f"Nonlocal model asset: {path}")
            target = assets / f"{i:02d}-{path.name}"
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
        ids = torch.tensor([0], device=device)
        zero_vel = torch.zeros(1, 1, 6, device=device)
        zero_joints = torch.zeros_like(drone.get_joint_positions())
        expected_throttle = torch.sqrt(drone.gravity / drone.KF.sum(-1, keepdim=True)).expand_as(
            drone.throttle
        )
        physics_started = time.perf_counter()
        with (output / "trajectory.csv").open("x", newline="") as stream, torch.no_grad():
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            for repeat in range(config.repetitions):
                for name, axis, yaw in cases():
                    # Reset the existing instance, including actuation memory and joint phase.
                    torch.manual_seed(config.seed)
                    drone._reset_idx(ids, train=False)
                    pos = torch.tensor([[[0.0, 0.0, config.initial_height_m]]], device=device)
                    if axis is None:
                        pos += torch.tensor([[[0.15, -0.10, 0.08]]], device=device)
                    rot = torch.tensor(
                        [[[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]]], device=device
                    )
                    drone.set_world_poses(pos, rot)
                    drone.set_velocities(zero_vel)
                    drone.set_joint_positions(zero_joints)
                    drone.set_joint_velocities(zero_joints)
                    drone.forces.zero_()
                    state = drone.get_state().clone()
                    error = max(
                        (state[..., :3] - pos).abs().max().item(),
                        (state[..., 3:7] - rot).abs().max().item(),
                        state[..., 7:13].abs().max().item(),
                        (drone.throttle - expected_throttle).abs().max().item(),
                        drone.get_joint_positions().abs().max().item(),
                        drone.get_joint_velocities().abs().max().item(),
                    )
                    resets.append(
                        {
                            "case": name,
                            "repeat": repeat,
                            "max_reset_error": error,
                            "state": state.cpu().tolist(),
                            "throttle": drone.throttle.cpu().tolist(),
                            "controller_state": "stateless; fixed gains, no integrator",
                        }
                    )
                    save_json(output / "resets.json", {"resets": resets})
                    event("episode_start", case=name, repeat=repeat, reset_error=error)
                    duration = (
                        config.hover_seconds
                        if axis is None
                        else (config.settle_seconds + config.command_seconds + config.brake_seconds)
                    )
                    hold = torch.tensor([[[0.0, 0.0, config.initial_height_m]]], device=device)
                    target_yaw = torch.tensor([[[yaw]]], device=device)
                    for step in range(round(duration / config.physics_dt)):
                        action, velocity, active = command_at(config, axis, step)
                        target_vel = torch.tensor(velocity, device=device).reshape(1, 1, 3)
                        # Position feedback only for the explicit hover test. Velocity-only
                        # cases mirror the student's current-position target convention.
                        raw = controller.compute(
                            state[..., :13],
                            target_pos=hold if axis is None else None,
                            target_vel=target_vel,
                            target_yaw=target_yaw,
                        )
                        if not torch.isfinite(state).all() or not torch.isfinite(raw).all():
                            raise RuntimeError("Nonfinite observation/controller output")
                        applied = raw.clamp(-1, 1)
                        drone.apply_action(applied)
                        sim.step(render=False)
                        state = drone.get_state().clone()
                        if not torch.isfinite(state).all():
                            raise RuntimeError("Nonfinite physics observation")
                        values = (
                            [name, repeat, step, (step + 1) * config.physics_dt, int(active)]
                            + state[..., :13].flatten().cpu().tolist()
                            + list(velocity)
                            + action
                            + raw.flatten().cpu().tolist()
                            + applied.flatten().cpu().tolist()
                            + drone.throttle.flatten().cpu().tolist()
                        )
                        row = dict(zip(COLUMNS, values, strict=True))
                        writer.writerow(row)
                        rows.append(row)
                        if step % 100 == 0:
                            stream.flush()
                        if state[..., 2].item() < 0.2 or state[..., :3].abs().max().item() > 10:
                            raise RuntimeError("Flight left the test safety envelope")
                    stream.flush()
                    os.fsync(stream.fileno())
                    event("episode_finished", case=name, repeat=repeat)
        torch.cuda.synchronize()
        result["physics_loop_wall_seconds"] = time.perf_counter() - physics_started
        result["physics_samples"] = len(rows)
        result["simulated_seconds"] = len(rows) * config.physics_dt
        result["samples_per_wall_second_including_logging"] = (
            len(rows) / result["physics_loop_wall_seconds"]
        )
        result["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["drone_physics_tested"] = True
        assessment = evaluate(rows, resets, config)
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
        print("ALIGN_DRONE_RESULT=" + json.dumps(result, allow_nan=False), flush=True)
        if app is not None:
            app.close()
    return 0 if result["status"] == "passed" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-root", action="store_true")
    args = parser.parse_args(argv)
    return run(DroneCheckConfig.from_dict(json.loads(args.config.read_text())), args.output)


if __name__ == "__main__":
    raise SystemExit(main())
