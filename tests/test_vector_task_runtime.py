"""Host checks for vector task launch acceptance and saved-trajectory auditing."""

import ast
import csv
import json
import shutil
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from align.runtime.vector_task_runtime import valid_scenario
from align.simulation.multi_drone_contract import MultiDroneConfig
from align.simulation.vector_task import TRAJECTORY_COLUMNS, build_isaac_config
from align.simulation.vector_task_report import audit_saved_scenario
from align.tasks.environment import TaskEnvironmentConfig
from align.tasks.observation import ObservationConfig
from align.tasks.reward import RewardConfig


def resolved_config():
    return {
        "construction": MultiDroneConfig().to_dict(),
        "observation": ObservationConfig().to_dict(),
        "reward": RewardConfig().to_dict(),
        "task": TaskEnvironmentConfig().to_dict(),
    }


def trajectory_rows():
    rows = []
    for env_id in range(2):
        for agent_id in range(4):
            rows.append(
                {
                    "global_step": 0,
                    "env_id": env_id,
                    "episode_step": 1,
                    "phase": "ground",
                    "agent_id": agent_id,
                    "x": float(agent_id),
                    "y": 0.0,
                    "z": 0.06,
                    "vx": 0.0,
                    "vy": 0.0,
                    "vz": 0.0,
                    "target_x": float(agent_id),
                    "target_y": 0.0,
                    "target_z": 0.06,
                    "a0": 0.0,
                    "a1": 0.0,
                    "a2": 0.0,
                    "a3": 0.0,
                    "contact_force_n": 1.0,
                    "reward": 0.0,
                    "terminated": False,
                    "truncated": env_id == 1,
                    "reason": "time_limit" if env_id == 1 else "",
                }
            )
    return rows


class VectorTaskRuntimeTests(unittest.TestCase):
    def test_combined_runtime_patches_preserve_cloned_view_contract(self):
        root = Path(__file__).resolve().parents[1]
        source = root / ".runtime/sources/OmniDrones"
        if not source.is_dir():
            self.skipTest("pinned source cache is unavailable")
        patches = sorted((root / "runtime/patches").glob("*.patch"))
        with TemporaryDirectory() as directory:
            destination = Path(directory)
            # Apply every build patch together without touching the source cache.
            for patch in patches:
                for line in patch.read_text().splitlines():
                    if line.startswith("+++ b/"):
                        relative = line.removeprefix("+++ b/")
                        target = destination / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source / relative, target)
            for patch in patches:
                applied = subprocess.run(
                    ["git", "apply", str(patch)],
                    cwd=destination,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(applied.returncode, 0, applied.stderr)
            model = destination / "omni_drones/robots/drone/multirotor.py"
            tree = ast.parse(model.read_text())
            initialize = next(
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "initialize"
            )
            self.assertIn("prepare_contact_sensors", [arg.arg for arg in initialize.args.args])
            self.assertIn("disable_stablization", [arg.arg for arg in initialize.args.args])
            views = [
                node
                for node in ast.walk(initialize)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "RigidPrimView"
            ]
            self.assertEqual(len(views), 2)
            for view in views:
                options = {item.arg: item.value for item in view.keywords}
                # These options prevent post-clone scene rewrites. Native physics
                # validity must still be established by the GPU acceptance run.
                self.assertIs(ast.literal_eval(options["reset_xform_properties"]), False)
                self.assertEqual(options["prepare_contact_sensors"].id, "prepare_contact_sensors")
                self.assertEqual(options["disable_stablization"].id, "disable_stablization")

    def test_simulator_task_prepares_contacts_before_clone_and_uses_read_only_view(self):
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / "src/align/simulation/vector_task.py").read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]

        drone_initialize = next(
            node
            for node in calls
            if isinstance(node.func, ast.Attribute)
            and node.func.attr == "initialize"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "drone"
        )
        drone_options = {item.arg: item.value for item in drone_initialize.keywords}
        self.assertIs(ast.literal_eval(drone_options["track_contact_forces"]), False)
        self.assertIs(ast.literal_eval(drone_options["prepare_contact_sensors"]), False)
        self.assertIs(ast.literal_eval(drone_options["disable_stablization"]), False)

        contact_view = next(
            node
            for node in calls
            if isinstance(node.func, ast.Name) and node.func.id == "RigidContactView"
        )
        contact_options = {item.arg: item.value for item in contact_view.keywords}
        self.assertEqual(ast.literal_eval(contact_options["filter_paths_expr"]), [])
        self.assertIs(ast.literal_eval(contact_options["prepare_contact_sensors"]), False)
        self.assertIs(ast.literal_eval(contact_options["disable_stablization"]), False)

        attributes = {node.func.attr for node in calls if isinstance(node.func, ast.Attribute)}
        self.assertIn("CreateThresholdAttr", attributes)
        self.assertIn("CreateSleepThresholdAttr", attributes)

    def test_runtime_configuration_covers_pinned_isaac_env_contract(self):
        construction = MultiDroneConfig()
        task = TaskEnvironmentConfig()
        config = build_isaac_config(construction, task, num_envs=4)
        self.assertEqual(config.sim.device, "cuda:0")
        self.assertEqual(config.sim.dt, construction.physics_dt)
        self.assertEqual(config.sim.substeps, 1)
        self.assertTrue(config.sim.replicate_physics)
        self.assertEqual(config.env.num_envs, 4)
        self.assertEqual(config.env.max_episode_length, task.max_episode_steps)
        self.assertEqual(config.env.env_spacing, 8)
        self.assertEqual(config.viewer.eye, [8.0, 8.0, 6.0])
        self.assertEqual(config.viewer.lookat, [0.0, 0.0, 1.0])
        self.assertEqual(config.viewer.resolution, [1280, 720])

    def test_pass_requires_physics_container_metrics_and_host_audit(self):
        probe = {
            "status": "passed",
            "phase": "before_close",
            "drone_physics_tested": True,
            "vector_task_physics_tested": True,
        }
        metrics = {"status": "passed", "checks": {"finite": True}}
        audit = {"status": "passed", "checks": {"raw": True}}
        self.assertTrue(valid_scenario(0, probe, metrics, audit))
        self.assertFalse(valid_scenario(1, probe, metrics, audit))
        self.assertFalse(
            valid_scenario(
                0,
                {**probe, "vector_task_physics_tested": False},
                metrics,
                audit,
            )
        )
        self.assertFalse(
            valid_scenario(0, probe, {"status": "failed", "checks": {"x": False}}, audit)
        )
        self.assertFalse(
            valid_scenario(0, probe, metrics, {"status": "failed", "checks": {"x": False}})
        )

    def test_host_audit_requires_complete_finite_unique_groups(self):
        with TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "config.json").write_text(json.dumps(resolved_config()))
            rows = trajectory_rows()
            (run / "metrics.json").write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "checks": {"probe": True},
                        "agent_steps": len(rows),
                    }
                )
            )
            with (run / "trajectory.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=TRAJECTORY_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            result = audit_saved_scenario(run, 2)
            self.assertEqual(result["status"], "passed")
            self.assertTrue(all(result["checks"].values()))

            rows[-1]["agent_id"] = 2
            with (run / "trajectory.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=TRAJECTORY_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            result = audit_saved_scenario(run, 2)
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["checks"]["complete_agent_groups"])
            self.assertFalse(result["checks"]["unique_agent_rows"])


if __name__ == "__main__":
    unittest.main()
