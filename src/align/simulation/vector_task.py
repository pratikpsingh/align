"""Vectorized OmniDrones task probe; simulator imports occur only inside run()."""

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
from dataclasses import replace
from importlib import metadata
from pathlib import Path

from align.artifacts import as_ist, utc_now, write_json_atomic
from align.formations import ShapeTransitionConfig
from align.formations.transition_report import make_transition_report
from align.learning.collector_config import CollectorProbeConfig
from align.learning.critic_calibration_config import CriticCalibrationConfig
from align.learning.critic_normalization_config import CriticNormalizationConfig
from align.learning.ppo_config import RecurrentPPOConfig
from align.learning.recovery_config import RecoveryConfig
from align.learning.rollout import RolloutConfig
from align.learning.stability_config import StabilityConfig
from align.learning.training_config import TaskTrainingConfig
from align.policies.config import RecurrentPolicyConfig
from align.simulation.assets import localize_materials
from align.simulation.multi_drone_contract import MultiDroneConfig, build_group_layout
from align.tasks.environment import TaskEnvironmentConfig
from align.tasks.evaluation_timing import EvaluationTimingConfig, apply_evaluation_timing
from align.tasks.formation_schedule import FormationScheduleConfig
from align.tasks.observation import ObservationConfig, build_observations
from align.tasks.reward import RewardConfig, RewardMemory, compute_step_reward
from align.tasks.waypoints import WaypointRouteConfig, make_waypoint_plan

REASON_NAMES = {
    0: None,
    1: "success",
    2: "separation_violation",
    3: "flight_contact",
    4: "safety_envelope",
    5: "nonfinite",
    6: "time_limit",
}

GROUND_PRIM_PATH = "/World/ground"
GROUND_SIZE_M = 20.0


class _AttributeDictionary(dict):
    """Dictionary with attribute reads for the pinned IsaacEnv configuration API."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


TRAJECTORY_COLUMNS = (
    "global_step",
    "env_id",
    "episode_step",
    "phase",
    "agent_id",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "target_x",
    "target_y",
    "target_z",
    "a0",
    "a1",
    "a2",
    "a3",
    "contact_force_n",
    "reward",
    "terminated",
    "truncated",
    "reason",
)


def save_json(path, value):
    write_json_atomic(path, value, mode=0o644)


def load_bundle(path):
    values = json.loads(path.read_text())
    required = {"construction", "observation", "reward", "task"}
    collector_sections = {"policy", "rollout", "collector"}
    training_sections = {"policy", "rollout", "ppo", "recovery", "training"}
    normalized_training_sections = training_sections | {"critic_normalization"}
    stability_sections = training_sections | {"stability"}
    normalized_stability_sections = normalized_training_sections | {"stability"}
    critic_calibration_sections = training_sections | {"critic_calibration"}
    normalized_critic_calibration_sections = normalized_training_sections | {"critic_calibration"}
    learner_sections = (
        training_sections,
        normalized_training_sections,
        stability_sections,
        normalized_stability_sections,
        critic_calibration_sections,
        normalized_critic_calibration_sections,
    )
    known = (
        required
        | collector_sections
        | normalized_stability_sections
        | normalized_critic_calibration_sections
        | {"formation_schedule"}
    )
    missing = required - set(values)
    unknown = set(values) - known
    supplied_optional = set(values) - required
    learner_optional = supplied_optional - {"formation_schedule"}
    valid_optional = learner_optional in (
        set(),
        collector_sections,
        *learner_sections,
    )
    if missing or unknown or not valid_optional:
        raise ValueError(
            "resolved vector task config sections mismatch; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}, "
            f"optional_sections={sorted(supplied_optional)}"
        )
    construction = MultiDroneConfig.from_dict(values["construction"])
    observation = ObservationConfig.from_dict(values["observation"])
    reward = RewardConfig.from_dict(values["reward"])
    task = TaskEnvironmentConfig.from_dict(values["task"])
    task.validate_compatibility(construction, observation, reward)
    formation_schedule = (
        FormationScheduleConfig.from_dict(values["formation_schedule"])
        if "formation_schedule" in values
        else FormationScheduleConfig(kinds=(construction.formation_kind,), seed=construction.seed)
    )
    formation_schedule.assignments(task.num_envs, 0)
    policy = RecurrentPolicyConfig.from_dict(values["policy"]) if learner_optional else None
    rollout = RolloutConfig.from_dict(values["rollout"]) if learner_optional else None
    collector = (
        CollectorProbeConfig.from_dict(values["collector"])
        if learner_optional == collector_sections
        else None
    )
    ppo = (
        RecurrentPPOConfig.from_dict(values["ppo"])
        if learner_optional in learner_sections
        else None
    )
    recovery = (
        RecoveryConfig.from_dict(values["recovery"])
        if learner_optional in learner_sections
        else None
    )
    training = (
        TaskTrainingConfig.from_dict(values["training"])
        if learner_optional in learner_sections
        else None
    )
    stability = (
        StabilityConfig.from_dict(values["stability"])
        if learner_optional in (stability_sections, normalized_stability_sections)
        else None
    )
    critic_calibration = (
        CriticCalibrationConfig.from_dict(values["critic_calibration"])
        if learner_optional in (critic_calibration_sections, normalized_critic_calibration_sections)
        else None
    )
    critic_normalization = (
        CriticNormalizationConfig.from_dict(values["critic_normalization"])
        if "critic_normalization" in supplied_optional
        else CriticNormalizationConfig()
    )
    return (
        construction,
        observation,
        reward,
        task,
        policy,
        rollout,
        collector,
        ppo,
        recovery,
        training,
        values,
        stability,
        critic_calibration,
        critic_normalization,
        formation_schedule,
    )


def build_isaac_config(construction, task, num_envs):
    """Build the complete configuration consumed by the pinned IsaacEnv."""
    return _AttributeDictionary(
        sim=_AttributeDictionary(
            {
                "dt": construction.physics_dt,
                "substeps": 1,
                "gravity": [0, 0, -9.81],
                "replicate_physics": True,
                "use_flatcache": True,
                "use_gpu_pipeline": True,
                "device": "cuda:0",
                "solver_type": 1,
                "use_gpu": True,
                "bounce_threshold_velocity": 0.2,
                "friction_offset_threshold": 0.04,
                "friction_correlation_distance": 0.025,
                "enable_stabilization": True,
                "gpu_max_rigid_contact_count": 524288,
                "gpu_max_rigid_patch_count": 163840,
                "gpu_found_lost_pairs_capacity": 4194304,
                "gpu_found_lost_aggregate_pairs_capacity": 33554432,
                "gpu_total_aggregate_pairs_capacity": 4194304,
                "gpu_max_soft_body_contacts": 1048576,
                "gpu_max_particle_contacts": 1048576,
                "gpu_heap_capacity": 67108864,
                "gpu_temp_buffer_capacity": 16777216,
                "gpu_max_num_partitions": 8,
            }
        ),
        env=_AttributeDictionary(
            {
                "num_envs": num_envs,
                "env_spacing": 8,
                "max_episode_length": task.max_episode_steps,
            }
        ),
        viewer=_AttributeDictionary(
            {
                "eye": [8.0, 8.0, 6.0],
                "lookat": [0.0, 0.0, 1.0],
                "resolution": [1280, 720],
            }
        ),
    )


def phase_name(index):
    return ("ground", "takeoff", "formation")[index]


def run(
    config_path: Path,
    output: Path,
    scenario: str,
    num_envs: int,
    *,
    logical_run_id: str | None = None,
    attempt_id: str | None = None,
    checkpoint_directory: Path | None = None,
    resume: bool = False,
    source_identity: str | None = None,
    runtime_identity: str | None = None,
    evaluation_update: int | None = None,
    policy_telemetry: bool = False,
    evaluation_timing_config: Path | None = None,
    shape_transition_config: Path | None = None,
    waypoint_route_config: Path | None = None,
    training_start_update: int = 0,
    training_stop_update: int | None = None,
    fault_at_update: int | None = None,
    fault_after_rollout_step: int | None = None,
):
    started = time.perf_counter()
    (
        construction,
        observation,
        reward,
        task,
        policy,
        rollout,
        collector,
        ppo,
        recovery,
        training,
        resolved_config,
        stability,
        critic_calibration,
        critic_normalization,
        formation_schedule,
    ) = load_bundle(config_path)
    if num_envs not in (1, task.num_envs):
        raise ValueError("num_envs must be one or the configured probe batch")
    if scenario == "single" and num_envs != 1:
        raise ValueError("single scenario requires one environment")
    if (
        scenario
        in (
            "batch",
            "collector",
            "training",
            "stability",
            "evaluation",
            "reference",
            "critic-calibration",
        )
        and num_envs != task.num_envs
    ):
        raise ValueError(f"{scenario} scenario requires the configured environment count")
    if scenario == "collector" and any(value is None for value in (policy, rollout, collector)):
        raise ValueError("collector scenario requires policy, rollout, and collector sections")
    if scenario in ("training", "stability", "evaluation", "critic-calibration") and any(
        value is None for value in (policy, rollout, ppo, recovery, training)
    ):
        raise ValueError(
            f"{scenario} scenario requires policy, rollout, PPO, recovery, and training"
        )
    if scenario in ("stability", "evaluation") and stability is None:
        raise ValueError(f"{scenario} scenario requires the stability section")
    if scenario == "critic-calibration" and critic_calibration is None:
        raise ValueError("critic-calibration scenario requires the critic_calibration section")
    if evaluation_update is not None and (
        scenario != "evaluation" or type(evaluation_update) is not int or evaluation_update < 0
    ):
        raise ValueError("evaluation_update must be a nonnegative integer used only for evaluation")
    if policy_telemetry and scenario != "evaluation":
        raise ValueError("policy telemetry is valid only for evaluation")
    if evaluation_timing_config is not None and scenario not in ("evaluation", "reference"):
        raise ValueError("evaluation timing override is valid only for evaluation or reference")
    if shape_transition_config is not None and scenario != "reference":
        raise ValueError("shape transition command is currently supported only for reference")
    if waypoint_route_config is not None and scenario != "reference":
        raise ValueError("waypoint route is currently supported only for reference")
    if shape_transition_config is not None and waypoint_route_config is not None:
        raise ValueError("shape transition and waypoint route cannot be combined")
    timing_details = None
    if evaluation_timing_config is not None:
        timing = EvaluationTimingConfig.from_dict(json.loads(evaluation_timing_config.read_text()))
        construction, task, stability, timing_details = apply_evaluation_timing(
            timing, construction, task, stability
        )
        task.validate_compatibility(construction, observation, reward)
    transition_config = None
    transition_plan = None
    if shape_transition_config is not None:
        transition_config = ShapeTransitionConfig.from_dict(
            json.loads(shape_transition_config.read_text())
        )
        if formation_schedule.kinds != (transition_config.source_kind,):
            raise ValueError("shape transition requires one matching source formation kind")
        transition_plan = make_transition_report(construction, task, transition_config)
    route_config = None
    route_plan = None
    if waypoint_route_config is not None:
        route_config = WaypointRouteConfig.from_dict(json.loads(waypoint_route_config.read_text()))
        if formation_schedule.kinds != (construction.formation_kind,):
            raise ValueError("waypoint route requires one matching formation kind")
        route_plan = make_waypoint_plan(construction, task, route_config)
    if scenario != "stability" and (training_start_update != 0 or training_stop_update is not None):
        raise ValueError("training update ranges are valid only for stability")
    if (fault_at_update is None) != (fault_after_rollout_step is None):
        raise ValueError("fault update and rollout step must be declared together")
    if scenario != "stability" and fault_at_update is not None:
        raise ValueError("fault injection is valid only for stability training")
    if scenario in ("training", "stability", "evaluation", "critic-calibration") and any(
        value is None
        for value in (
            logical_run_id,
            attempt_id,
            checkpoint_directory,
            source_identity,
            runtime_identity,
        )
    ):
        raise ValueError(
            f"{scenario} scenario requires run, attempt, checkpoint, and identity values"
        )
    if scenario not in (
        "collector",
        "training",
        "stability",
        "evaluation",
        "reference",
        "critic-calibration",
    ) and any(value is not None for value in (policy, rollout, collector, ppo, recovery, training)):
        raise ValueError("learner sections are valid only for collector or training scenarios")

    result = {
        "status": "running",
        "started_at_utc": utc_now(),
        "python": platform.python_version(),
        "scenario": scenario,
        "num_envs": num_envs,
        "shutdown_mode": "fast",
        "rendering_validated": False,
        "video_validated": False,
        "drone_physics_tested": False,
        "vector_task_physics_tested": False,
    }
    result["started_at_ist"] = as_ist(result["started_at_utc"])
    result["evaluation_timing"] = timing_details
    result["shape_transition_plan"] = transition_plan
    result["shape_transition_physics_tested"] = False
    result["waypoint_route_plan"] = route_plan.to_dict() if route_plan is not None else None
    result["waypoint_route_physics_tested"] = False
    app = None

    def event(name, **details):
        item = {"event": name, "timestamp_utc": utc_now(), **details}
        item["timestamp_ist"] = as_ist(item["timestamp_utc"])
        with (output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(item, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(f"{item['timestamp_ist']} IST {name} {details}", flush=True)

    try:
        event("starting", scenario=scenario, num_envs=num_envs)
        from isaacsim import SimulationApp

        app = SimulationApp({"headless": True, "multi_gpu": False, "fast_shutdown": True})
        result["startup_seconds"] = time.perf_counter() - started
        event("application_started")

        import numpy as np
        import torch
        from omni.isaac.core.objects import GroundPlane
        from omni.isaac.core.prims import RigidContactView
        from omni_drones.controllers import LeePositionController
        from omni_drones.envs.isaac_env import IsaacEnv
        from omni_drones.robots.drone import Hummingbird
        from omni_drones.utils.torchrl import AgentSpec
        from pxr import PhysxSchema, UsdPhysics, UsdUtils
        from tensordict import TensorDict
        from torchrl.data import (
            BoundedTensorSpec,
            CompositeSpec,
            UnboundedContinuousTensorSpec,
        )

        if torch.__version__ != "2.2.2+cu118" or torch.version.cuda != "11.8":
            raise RuntimeError("Vendor PyTorch/CUDA changed")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Expected exactly one allocated GPU")
        torch.set_num_threads(4)
        torch.manual_seed(construction.seed)
        np.random.seed(construction.seed)
        random.seed(construction.seed)

        class AlignVectorTask(IsaacEnv):
            def __init__(self, cfg):
                self.output = output
                self.construction_cfg = construction
                self.observation_cfg = observation
                self.reward_cfg = reward
                self.task_cfg = task
                self.layout = build_group_layout(construction)
                self.formation_schedule = formation_schedule
                self.formation_kinds = formation_schedule.ordered_kinds
                self.formation_layouts = tuple(
                    build_group_layout(replace(construction, formation_kind=kind))
                    for kind in self.formation_kinds
                )
                super().__init__(cfg, headless=True)
                self.shape_transition_config = transition_config
                self.shape_transition_plan = transition_plan
                self.transition_destination = (
                    torch.tensor(
                        transition_plan["assignment"]["assigned_destination_m"],
                        device=self.device,
                        dtype=torch.float32,
                    )
                    if transition_plan is not None
                    else None
                )
                self.transition_issued = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
                self.transition_total_commands = 0
                self.last_transition_active = torch.zeros(
                    num_envs, dtype=torch.bool, device=self.device
                )
                self.waypoint_route_config = route_config
                self.waypoint_route_plan = route_plan
                self.waypoint_target_bank = (
                    torch.tensor(
                        route_plan.assigned_targets_m, device=self.device, dtype=torch.float32
                    )
                    if route_plan is not None
                    else None
                )
                self.waypoint_index = torch.zeros(num_envs, dtype=torch.long, device=self.device)
                self.waypoint_dwell = torch.zeros(num_envs, dtype=torch.long, device=self.device)
                self.waypoint_complete = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
                self.waypoint_issued = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
                self.last_waypoint_active = torch.zeros(
                    num_envs, dtype=torch.bool, device=self.device
                )
                self.last_waypoint_index = torch.zeros(
                    num_envs, dtype=torch.long, device=self.device
                )
                self.waypoint_advanced_this_step = torch.zeros(
                    num_envs, dtype=torch.bool, device=self.device
                )
                self.waypoint_settled_this_step = torch.zeros(
                    num_envs, dtype=torch.bool, device=self.device
                )
                self.waypoint_total_commands = 0
                self.waypoint_total_advances = 0
                self.waypoint_total_completions = 0
                self.template_target_bank = torch.tensor(
                    [
                        [
                            layout.ground_positions_m,
                            layout.takeoff_positions_m,
                            layout.assigned_target_positions_m,
                        ]
                        for layout in self.formation_layouts
                    ],
                    device=self.device,
                    dtype=torch.float32,
                )
                self.template_index = torch.zeros(num_envs, dtype=torch.long, device=self.device)
                self.template_batch_index = 0
                self.template_reset_count = torch.zeros(
                    num_envs, dtype=torch.long, device=self.device
                )
                self.template_kind_to_index = {
                    kind: index for index, kind in enumerate(self.formation_kinds)
                }
                for env_path in self.envs_prim_paths:
                    for template_body in self.scene_setup[
                        "contact_reports_prepared_before_cloning"
                    ]:
                        relative = template_body.removeprefix(self.template_env_ns)
                        body = self.sim.stage.GetPrimAtPath(env_path + relative)
                        if not body.IsValid() or not body.HasAPI(PhysxSchema.PhysxContactReportAPI):
                            raise RuntimeError(
                                f"Missing pre-clone contact schema at {env_path + relative}"
                            )
                        if not body.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                            raise RuntimeError(
                                "Missing pre-clone PhysX rigid-body schema at "
                                f"{env_path + relative}"
                            )
                        sleep_threshold = (
                            PhysxSchema.PhysxRigidBodyAPI(body).GetSleepThresholdAttr().Get()
                        )
                        if sleep_threshold != 0.0:
                            raise RuntimeError(
                                f"Expected zero sleep threshold at {env_path + relative}"
                            )
                event("cloned_scene_ready", num_envs=num_envs)
                self.drone.initialize(
                    track_contact_forces=False,
                    prepare_contact_sensors=False,
                    disable_stablization=False,
                )
                event("drone_views_ready", shape=list(self.drone.shape))
                self.base_contact_view = RigidContactView(
                    prim_paths_expr=f"{self.drone.prim_paths_expr}/base_link",
                    name="align_base_contacts",
                    filter_paths_expr=[],
                    prepare_contact_sensors=False,
                    disable_stablization=False,
                )
                self.base_contact_view.initialize()
                event("contact_view_ready")
                if tuple(self.drone.shape) != (num_envs, construction.num_agents):
                    raise RuntimeError(
                        f"Expected {(num_envs, construction.num_agents)}, got {self.drone.shape}"
                    )
                self.controller = LeePositionController(9.81, self.drone.params).to(self.device)
                self.target_yaw = torch.zeros(
                    num_envs, construction.num_agents, 1, device=self.device
                )
                self.reward_distance_memory = torch.zeros(
                    num_envs, construction.num_agents, device=self.device
                )
                self.reward_command_memory = torch.zeros(
                    num_envs, construction.num_agents, 3, device=self.device
                )
                self.reward_memory_valid = torch.zeros(
                    num_envs, dtype=torch.bool, device=self.device
                )
                self.reward_memory_phase = torch.full(
                    (num_envs,), -1, dtype=torch.long, device=self.device
                )
                self.reward_memory_transition_active = torch.zeros(
                    num_envs, dtype=torch.bool, device=self.device
                )
                self.success_dwell = torch.zeros(num_envs, dtype=torch.long, device=self.device)
                self.ever_airborne = torch.zeros(
                    num_envs, construction.num_agents, dtype=torch.bool, device=self.device
                )
                self.state = None
                self.contact_force = None
                self.last_actions = None
                self.commanded_velocities = None
                self.last_reward_components = None
                self.last_reward_raw_components = None
                self.last_team_reward = None
                self.last_targets = None
                self.last_reward_phase = None
                self.last_reason_code = None
                self.last_done = None
                self.last_reset_errors = {}

            def _design_scene(self):
                self.drone = Hummingbird()
                self.source_asset = Path(self.drone.usd_path)
                self.local_asset = self.output / "hummingbird-local.usda"
                self.asset_adaptation = localize_materials(self.source_asset, self.local_asset)
                self.drone.usd_path = str(self.local_asset)
                spawned = self.drone.spawn(translations=self.layout.ground_positions_m)
                contact_paths = []
                for drone_prim in spawned:
                    body_path = str(drone_prim.GetPath()) + "/base_link"
                    body = self.sim.stage.GetPrimAtPath(body_path)
                    if not body.IsValid() or not body.HasAPI(UsdPhysics.RigidBodyAPI):
                        raise RuntimeError(f"Expected drone rigid body at {body_path}")
                    contact = PhysxSchema.PhysxContactReportAPI.Apply(body)
                    contact.CreateThresholdAttr().Set(0.0)
                    rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(body)
                    rigid.CreateSleepThresholdAttr().Set(0.0)
                    contact_paths.append(body_path)
                self.scene_setup = {
                    "ground_implementation": "omni.isaac.core.objects.GroundPlane",
                    "ground_prim_path": GROUND_PRIM_PATH,
                    "ground_size_m": GROUND_SIZE_M,
                    "replicate_physics": bool(self.cfg.sim.replicate_physics),
                    "reset_xform_properties": False,
                    "contact_reports_prepared_before_cloning": contact_paths,
                    "contact_report_threshold": 0.0,
                    "rigid_body_sleep_threshold": 0.0,
                    "prepare_contact_sensors_during_view_initialization": False,
                    "disable_stabilization_during_view_initialization": False,
                    "contact_view_implementation": "omni.isaac.core.prims.RigidContactView",
                    "motion_view_tracks_contacts": False,
                }
                self.ground = GroundPlane(GROUND_PRIM_PATH, size=GROUND_SIZE_M)
                save_json(self.output / "scene-setup.json", self.scene_setup)
                event("source_scene_ready", contact_bodies=len(contact_paths))
                return [GROUND_PRIM_PATH]

            def _set_specs(self):
                agents = construction.num_agents
                self.observation_spec = (
                    CompositeSpec(
                        {
                            "agents": {
                                "observation": UnboundedContinuousTensorSpec(
                                    (agents, observation.actor_dimension), device=self.device
                                ),
                                "state": UnboundedContinuousTensorSpec(
                                    (observation.critic_dimension,), device=self.device
                                ),
                            }
                        }
                    )
                    .expand(num_envs)
                    .to(self.device)
                )
                self.action_spec = (
                    CompositeSpec(
                        {
                            "agents": {
                                "action": BoundedTensorSpec(
                                    -1.0, 1.0, (agents, 4), device=self.device
                                )
                            }
                        }
                    )
                    .expand(num_envs)
                    .to(self.device)
                )
                self.reward_spec = (
                    CompositeSpec(
                        {
                            "agents": {
                                "reward": UnboundedContinuousTensorSpec(
                                    (agents, 1), device=self.device
                                )
                            }
                        }
                    )
                    .expand(num_envs)
                    .to(self.device)
                )
                self.agent_spec["drone"] = AgentSpec(
                    "drone",
                    agents,
                    observation_key=("agents", "observation"),
                    action_key=("agents", "action"),
                    reward_key=("agents", "reward"),
                    state_key=("agents", "state"),
                )

            def phase_indices(self, progress=None):
                steps = self.progress_buf.long() if progress is None else progress.long()
                takeoff_start = construction.ground_steps
                formation_start = construction.ground_steps + construction.takeoff_steps
                return torch.where(
                    steps < takeoff_start,
                    torch.zeros_like(steps),
                    torch.where(
                        steps < formation_start,
                        torch.ones_like(steps),
                        torch.full_like(steps, 2),
                    ),
                )

            def targets_for_progress(self):
                phase = self.phase_indices()
                base = self.template_target_bank[self.template_index, phase]
                if self.waypoint_route_config is not None:
                    active = (phase == 2) & (
                        self.progress_buf >= self.waypoint_route_config.command_step
                    )
                    selected = self.waypoint_target_bank[self.waypoint_index]
                    return torch.where(active[:, None, None], selected, base)
                if self.shape_transition_config is None:
                    return base
                active = (phase == 2) & (
                    self.progress_buf >= self.shape_transition_config.command_step
                )
                return torch.where(active[:, None, None], self.transition_destination, base)

            def set_template_batch(self, batch_index):
                assignments = self.formation_schedule.assignments(num_envs, batch_index)
                self.template_batch_index = batch_index
                self.template_reset_count.zero_()
                self.template_index.copy_(
                    torch.tensor(
                        [self.template_kind_to_index[kind] for kind in assignments],
                        dtype=torch.long,
                        device=self.device,
                    )
                )
                event("template_batch_selected", batch_index=batch_index, assignments=assignments)

            def template_names(self):
                initial = tuple(
                    self.formation_kinds[index] for index in self.template_index.cpu().tolist()
                )
                if self.shape_transition_config is None:
                    return initial
                return tuple(
                    self.shape_transition_config.destination_kind if active else kind
                    for active, kind in zip(
                        self.last_transition_active.cpu().tolist(), initial, strict=True
                    )
                )

            def _reset_idx(self, env_ids):
                self.drone._reset_idx(env_ids, train=False)
                for env_id in env_ids.cpu().tolist():
                    assignments = self.formation_schedule.assignments(
                        num_envs, self.template_batch_index + int(self.template_reset_count[env_id])
                    )
                    self.template_index[env_id] = self.template_kind_to_index[assignments[env_id]]
                self.template_reset_count[env_ids] += 1
                count = len(env_ids)
                local = torch.tensor(
                    self.layout.ground_positions_m, device=self.device, dtype=torch.float32
                ).expand(count, -1, -1)
                world = local + self.envs_positions[env_ids].unsqueeze(1)
                rotations = torch.zeros(count, construction.num_agents, 4, device=self.device)
                rotations[..., 0] = 1.0
                velocities = torch.zeros(count, construction.num_agents, 6, device=self.device)
                joints = torch.zeros_like(self.drone.get_joint_positions()[env_ids])
                self.drone.throttle[env_ids] = 0.0
                self.drone.thrusts[env_ids] = 0.0
                self.drone.torques[env_ids] = 0.0
                self.drone.forces[env_ids] = 0.0
                self.drone.set_world_poses(world, rotations, env_ids)
                self.drone.set_velocities(velocities, env_ids)
                self.drone.set_joint_positions(joints, env_ids)
                self.drone.set_joint_velocities(joints, env_ids)
                self.reward_distance_memory[env_ids] = 0.0
                self.reward_command_memory[env_ids] = 0.0
                self.reward_memory_valid[env_ids] = False
                self.reward_memory_phase[env_ids] = -1
                self.reward_memory_transition_active[env_ids] = False
                self.transition_issued[env_ids] = False
                self.last_transition_active[env_ids] = False
                self.waypoint_index[env_ids] = 0
                self.waypoint_dwell[env_ids] = 0
                self.waypoint_complete[env_ids] = False
                self.waypoint_issued[env_ids] = False
                self.last_waypoint_active[env_ids] = False
                self.last_waypoint_index[env_ids] = 0
                self.waypoint_advanced_this_step[env_ids] = False
                self.waypoint_settled_this_step[env_ids] = False
                self.success_dwell[env_ids] = 0
                self.ever_airborne[env_ids] = False
                state = self.drone.get_state().clone()
                reset_error = torch.stack(
                    (
                        (state[env_ids, :, :3] - local).abs().amax(),
                        (state[env_ids, :, 3:7] - rotations).abs().amax(),
                        state[env_ids, :, 7:13].abs().amax(),
                        self.drone.throttle[env_ids].abs().amax(),
                        self.drone.get_joint_positions()[env_ids].abs().amax(),
                        self.drone.get_joint_velocities()[env_ids].abs().amax(),
                    )
                ).amax()
                for env_id in env_ids.cpu().tolist():
                    self.last_reset_errors[int(env_id)] = float(reset_error.item())

            def reset_mask(self, mask):
                return self._reset(
                    TensorDict(
                        {"_reset": mask.reshape(num_envs, 1)},
                        [num_envs],
                        device=self.device,
                    )
                )

            def step_actions(self, actions):
                return self._step(
                    TensorDict(
                        {"agents": {"action": actions}},
                        [num_envs],
                        device=self.device,
                    )
                )

            def _pre_sim_step(self, tensordict):
                actions = tensordict[("agents", "action")]
                if actions.shape != (num_envs, construction.num_agents, 4):
                    raise ValueError(f"Wrong action shape {tuple(actions.shape)}")
                self.action_saturation_count = int((actions.abs() > 1.0).sum().item())
                actions = actions.clamp(-1.0, 1.0)
                direction = actions[..., :3]
                direction_norm = direction.norm(dim=-1, keepdim=True)
                unit = torch.where(
                    direction_norm > 1e-8,
                    direction / direction_norm.clamp_min(1e-8),
                    torch.zeros_like(direction),
                )
                speed = actions[..., 3:4].abs() * construction.max_speed_m_s
                self.commanded_velocities = unit * speed
                self.last_actions = actions
                raw = self.controller.compute(
                    self.state[..., :13],
                    target_vel=self.commanded_velocities,
                    target_yaw=self.target_yaw,
                )
                ground = self.last_reward_phase == 0
                raw[ground] = -1.0
                self.raw_rotor_action = raw
                self.applied_rotor_action = raw.clamp(-1.0, 1.0)
                self.drone.apply_action(self.applied_rotor_action)

            def _step(self, tensordict):
                self.last_reward_phase = self.phase_indices().clone()
                if self.waypoint_route_config is not None:
                    self.last_waypoint_active = (self.last_reward_phase == 2) & (
                        self.progress_buf >= self.waypoint_route_config.command_step
                    )
                    self.last_waypoint_index = self.waypoint_index.clone()
                    newly_issued = self.last_waypoint_active & ~self.waypoint_issued
                    if bool(newly_issued.any().item()):
                        env_ids = newly_issued.nonzero().squeeze(-1).cpu().tolist()
                        self.waypoint_total_commands += len(env_ids)
                        self.waypoint_issued |= newly_issued
                        self.reward_memory_valid[newly_issued] = False
                        self.success_dwell[newly_issued] = 0
                        event(
                            "waypoint_route_command_issued",
                            env_ids=env_ids,
                            episode_step=self.waypoint_route_config.command_step,
                            legs=len(self.waypoint_route_plan.centers_m),
                        )
                if self.shape_transition_config is not None:
                    self.last_transition_active = (self.last_reward_phase == 2) & (
                        self.progress_buf >= self.shape_transition_config.command_step
                    )
                    newly_issued = self.last_transition_active & ~self.transition_issued
                    if bool(newly_issued.any().item()):
                        env_ids = newly_issued.nonzero().squeeze(-1).cpu().tolist()
                        self.transition_total_commands += len(env_ids)
                        self.success_dwell[newly_issued] = 0
                        self.transition_issued |= newly_issued
                        event(
                            "shape_command_issued",
                            env_ids=env_ids,
                            episode_step=self.shape_transition_config.command_step,
                            destination_kind=self.shape_transition_config.destination_kind,
                        )
                self.last_targets = self.targets_for_progress()
                self._pre_sim_step(tensordict)
                for substep in range(self.substeps):
                    self.sim.step(self._should_render(substep))
                self.state = self.drone.get_state().clone()
                self.contact_force = (
                    self.base_contact_view.get_net_contact_forces()
                    .reshape(num_envs, construction.num_agents, -1)
                    .norm(dim=-1)
                )
                self.progress_buf += 1
                reward_and_done = self._compute_reward_and_done()
                output = self._compute_state_and_obs()
                output.update(reward_and_done)
                return output

            def _compute_state_and_obs(self):
                self.state = self.drone.get_state().clone()
                positions = self.state[..., :3]
                velocities = self.state[..., 7:10]
                targets = self.targets_for_progress()

                relative_position = positions.unsqueeze(1) - positions.unsqueeze(2)
                relative_velocity = velocities.unsqueeze(1) - velocities.unsqueeze(2)
                distances = relative_position.norm(dim=-1)
                identity = torch.eye(
                    construction.num_agents, dtype=torch.bool, device=self.device
                ).unsqueeze(0)
                candidate = (distances <= observation.neighbor_radius_m) & ~identity
                sortable = torch.where(candidate, distances, torch.full_like(distances, torch.inf))
                order = torch.argsort(sortable, dim=-1, stable=True)
                selected_count = min(observation.max_neighbors, construction.num_agents - 1)
                selected = order[..., :selected_count]
                gather3 = selected.unsqueeze(-1).expand(-1, -1, -1, 3)
                selected_position = torch.gather(relative_position, 2, gather3)
                selected_velocity = torch.gather(relative_velocity, 2, gather3)
                selected_mask = torch.gather(candidate, 2, selected)
                neighbor_features = torch.cat(
                    (
                        (selected_position / observation.neighbor_radius_m).clamp(-1.0, 1.0),
                        (selected_velocity / observation.relative_velocity_scale_m_s).clamp(
                            -1.0, 1.0
                        ),
                    ),
                    dim=-1,
                ) * selected_mask.unsqueeze(-1)
                if selected_count < observation.max_neighbors:
                    pad = observation.max_neighbors - selected_count
                    neighbor_features = torch.cat(
                        (
                            neighbor_features,
                            torch.zeros(
                                num_envs,
                                construction.num_agents,
                                pad,
                                6,
                                device=self.device,
                            ),
                        ),
                        dim=2,
                    )
                    selected_mask = torch.cat(
                        (
                            selected_mask,
                            torch.zeros(
                                num_envs,
                                construction.num_agents,
                                pad,
                                dtype=torch.bool,
                                device=self.device,
                            ),
                        ),
                        dim=2,
                    )
                own = torch.cat(
                    (
                        (velocities / observation.own_velocity_scale_m_s).clamp(-1.0, 1.0),
                        ((targets - positions) / observation.target_distance_scale_m).clamp(
                            -1.0, 1.0
                        ),
                    ),
                    dim=-1,
                )
                actor = torch.cat(
                    (
                        own,
                        neighbor_features.flatten(start_dim=-2),
                        selected_mask.float(),
                    ),
                    dim=-1,
                )

                critic_rows = torch.cat(
                    (
                        (positions / observation.critic_position_scale_m).clamp(-1.0, 1.0),
                        (velocities / observation.own_velocity_scale_m_s).clamp(-1.0, 1.0),
                        (targets / observation.critic_position_scale_m).clamp(-1.0, 1.0),
                    ),
                    dim=-1,
                )
                critic_mask = torch.ones(num_envs, construction.num_agents, device=self.device)
                if construction.num_agents < observation.critic_capacity:
                    pad = observation.critic_capacity - construction.num_agents
                    critic_rows = torch.cat(
                        (
                            critic_rows,
                            torch.zeros(num_envs, pad, 9, device=self.device),
                        ),
                        dim=1,
                    )
                    critic_mask = torch.cat(
                        (critic_mask, torch.zeros(num_envs, pad, device=self.device)),
                        dim=1,
                    )
                critic = torch.cat((critic_rows.flatten(start_dim=1), critic_mask), dim=-1)
                self.last_actor_observation = actor
                self.last_critic_observation = critic
                self.last_observation_targets = targets
                return TensorDict(
                    {"agents": {"observation": actor, "state": critic}},
                    [num_envs],
                    device=self.device,
                )

            def _compute_reward_and_done(self):
                positions = self.state[..., :3]
                velocities = self.state[..., 7:10]
                targets = self.last_targets
                agents = construction.num_agents
                pair_indices = torch.triu_indices(agents, agents, 1, device=self.device)
                actual_pair = torch.cdist(positions, positions)[:, pair_indices[0], pair_indices[1]]
                target_pair = torch.cdist(targets, targets)[:, pair_indices[0], pair_indices[1]]
                target_diameter_sq = (
                    torch.cdist(targets, targets).amax(dim=(-2, -1)).square().clamp_min(1e-12)
                )
                formation = -(
                    (actual_pair - target_pair).square().mean(dim=-1) / target_diameter_sq
                )
                target_distances = (positions - targets).norm(dim=-1)
                speeds = velocities.norm(dim=-1)
                separation_matrix = torch.cdist(positions, positions)
                identity = torch.eye(agents, dtype=torch.bool, device=self.device).unsqueeze(0)
                minimum_separation = separation_matrix.masked_fill(identity, torch.inf).amin(dim=-1)
                self.ever_airborne |= positions[..., 2] > task.airborne_height_m
                flight_contact = (
                    self.ever_airborne | (self.last_reward_phase.unsqueeze(-1) == 2)
                ) & (self.contact_force > task.contact_force_threshold_n)

                same_phase = self.reward_memory_phase == self.last_reward_phase
                same_command = self.reward_memory_transition_active == self.last_transition_active
                memory_valid = self.reward_memory_valid & same_phase & same_command
                progress = torch.where(
                    memory_valid.unsqueeze(-1),
                    (self.reward_distance_memory - target_distances)
                    .div(reward.distance_scale_m)
                    .clamp(-1.0, 1.0),
                    torch.zeros_like(target_distances),
                )
                separation_intrusion = (
                    (reward.minimum_separation_m - minimum_separation) / reward.minimum_separation_m
                ).clamp_min(0.0)
                proximity = (1.0 - target_distances / reward.settling_radius_m).clamp_min(0.0)
                smoothness = torch.where(
                    memory_valid.unsqueeze(-1),
                    -(self.commanded_velocities - self.reward_command_memory).square().sum(dim=-1)
                    / (3.0 * reward.max_speed_m_s**2),
                    torch.zeros_like(target_distances),
                )
                effort = -(self.commanded_velocities.norm(dim=-1) / reward.max_speed_m_s).square()
                raw = torch.stack(
                    (
                        formation.unsqueeze(-1).expand(-1, agents),
                        -(target_distances / reward.distance_scale_m).square(),
                        progress,
                        -separation_intrusion.square(),
                        -flight_contact.float(),
                        -proximity * (speeds / reward.max_speed_m_s).square(),
                        smoothness,
                        effort,
                    ),
                    dim=-1,
                )
                weights = torch.tensor(
                    (
                        reward.formation_weight,
                        reward.tracking_weight,
                        reward.progress_weight,
                        reward.separation_weight,
                        reward.contact_weight,
                        reward.settling_weight,
                        reward.smoothness_weight,
                        reward.effort_weight,
                    ),
                    device=self.device,
                )
                time_scales = torch.tensor(
                    (
                        reward.control_dt_seconds,
                        reward.control_dt_seconds,
                        1.0,
                        reward.control_dt_seconds,
                        reward.control_dt_seconds,
                        reward.control_dt_seconds,
                        1.0,
                        reward.control_dt_seconds,
                    ),
                    device=self.device,
                )
                weighted = raw * weights * time_scales
                per_agent_reward = weighted.sum(dim=-1)
                self.last_reward_raw_components = raw
                self.last_reward_components = weighted
                self.last_team_reward = per_agent_reward.mean(dim=-1)
                self.reward_distance_memory[:] = target_distances
                self.reward_command_memory[:] = self.commanded_velocities
                self.reward_memory_valid[:] = True
                self.reward_memory_phase[:] = self.last_reward_phase
                self.reward_memory_transition_active[:] = self.last_transition_active

                assigned_rmse = (positions - targets).square().sum(dim=-1).mean(dim=-1).sqrt()
                pairwise_rmse = (actual_pair - target_pair).square().mean(dim=-1).sqrt()
                settled = (
                    (self.last_reward_phase == 2)
                    & (assigned_rmse <= task.formation_rmse_tolerance_m)
                    & (pairwise_rmse <= task.pairwise_rmse_tolerance_m)
                    & (speeds.amax(dim=-1) <= task.speed_tolerance_m_s)
                )
                if self.shape_transition_config is not None:
                    settled &= self.last_transition_active
                if self.waypoint_route_config is not None:
                    route = self.waypoint_route_config
                    arrival = (
                        self.last_waypoint_active
                        & ~self.waypoint_complete
                        & (target_distances.amax(dim=-1) <= route.arrival_radius_m)
                        & (minimum_separation.amin(dim=-1) >= route.minimum_arrival_separation_m)
                        & (speeds.amax(dim=-1) <= route.maximum_arrival_speed_m_s)
                    )
                    self.waypoint_settled_this_step = arrival
                    self.waypoint_dwell[:] = torch.where(
                        arrival, self.waypoint_dwell + 1, torch.zeros_like(self.waypoint_dwell)
                    )
                    reached = arrival & (self.waypoint_dwell >= route.arrival_dwell_steps)
                    final = reached & (
                        self.waypoint_index == len(self.waypoint_route_plan.centers_m) - 1
                    )
                    advance = reached & ~final
                    self.waypoint_advanced_this_step = advance
                    self.waypoint_index[advance] += 1
                    self.waypoint_complete |= final
                    self.waypoint_dwell[reached] = 0
                    self.reward_memory_valid[advance] = False
                    self.success_dwell[reached] = 0
                    self.waypoint_total_advances += int(advance.sum().item())
                    self.waypoint_total_completions += int(final.sum().item())
                    if bool(reached.any().item()):
                        event(
                            "waypoint_gate_passed",
                            advanced_env_ids=advance.nonzero().squeeze(-1).cpu().tolist(),
                            completed_env_ids=final.nonzero().squeeze(-1).cpu().tolist(),
                        )
                    settled &= self.waypoint_complete & ~advance
                self.success_dwell[:] = torch.where(
                    settled, self.success_dwell + 1, torch.zeros_like(self.success_dwell)
                )
                success = self.success_dwell >= task.success_dwell_steps
                separation_failure = minimum_separation.amin(dim=-1) < task.terminal_separation_m
                contact_failure = flight_contact.any(dim=-1)
                envelope_failure = (
                    (positions[..., :2].abs() > task.safety_xy_limit_m).any(dim=(-2, -1))
                    | (positions[..., 2] < task.crash_height_m).any(dim=-1)
                    | (positions[..., 2] > task.safety_z_limit_m).any(dim=-1)
                )
                nonfinite = (
                    ~torch.isfinite(self.state).all(dim=(-2, -1))
                    | ~torch.isfinite(self.contact_force).all(dim=-1)
                    | ~torch.isfinite(per_agent_reward).all(dim=-1)
                )
                reason = torch.zeros(num_envs, dtype=torch.long, device=self.device)
                reason = torch.where(success, torch.ones_like(reason), reason)
                reason = torch.where(separation_failure, torch.full_like(reason, 2), reason)
                reason = torch.where(contact_failure, torch.full_like(reason, 3), reason)
                reason = torch.where(envelope_failure, torch.full_like(reason, 4), reason)
                reason = torch.where(nonfinite, torch.full_like(reason, 5), reason)
                terminated = reason != 0
                truncated = (self.progress_buf >= task.max_episode_steps) & ~terminated
                reason = torch.where(truncated, torch.full_like(reason, 6), reason)
                done = terminated | truncated
                self.last_reason_code = reason
                self.last_done = done
                self.last_assigned_rmse = assigned_rmse
                self.last_pairwise_rmse = pairwise_rmse
                self.last_minimum_separation = minimum_separation.amin(dim=-1)
                return TensorDict(
                    {
                        "agents": {"reward": per_agent_reward.unsqueeze(-1)},
                        "done": done.unsqueeze(-1),
                        "terminated": terminated.unsqueeze(-1),
                        "truncated": truncated.unsqueeze(-1),
                    },
                    [num_envs],
                    device=self.device,
                )

        cfg = build_isaac_config(construction, task, num_envs)
        env = AlignVectorTask(cfg)
        result["formation_schedule"] = formation_schedule.to_dict()
        result["template_kinds"] = list(env.formation_kinds)
        result["template_assigned_targets_m"] = {
            kind: layout.assigned_target_positions_m
            for kind, layout in zip(env.formation_kinds, env.formation_layouts, strict=True)
        }
        result["versions"] = {
            name: metadata.version(name)
            for name in (
                "torch",
                "torchrl",
                "tensordict",
                "numpy",
                "omni-drones",
                "align",
            )
        }
        result["torch_path"] = torch.__file__
        result["gpu"] = torch.cuda.get_device_name(0)
        result["visual_asset_adaptation"] = env.asset_adaptation
        result["scene"] = env.scene_setup
        result["tensor_contract"] = {
            "actor_observation": [num_envs, construction.num_agents, observation.actor_dimension],
            "critic_state": [num_envs, observation.critic_dimension],
            "high_level_action": [num_envs, construction.num_agents, 4],
            "reward": [num_envs, construction.num_agents, 1],
            "done": [num_envs, 1],
        }
        result["controller"] = {
            key: value.detach().cpu().tolist() for key, value in env.controller.state_dict().items()
        }
        result["model"] = {
            "parameters": env.drone.params,
            "base_mass_kg": env.drone.masses.cpu().tolist(),
            "base_inertias_kg_m2": env.drone.inertias.cpu().tolist(),
        }

        assets = output / "assets"
        assets.mkdir()
        layers, asset_paths, unresolved = UsdUtils.ComputeAllDependencies(env.drone.usd_path)
        if unresolved:
            raise RuntimeError(f"Unresolved model assets: {unresolved}")
        paths = {Path(layer.realPath) for layer in layers} | {Path(path) for path in asset_paths}
        paths.update((Path(env.drone.param_path), env.source_asset, env.local_asset))
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
        env.sim.stage.Flatten().Export(str(output / "scene.usda"))
        save_json(output / "runtime.json", result)

        all_mask = torch.ones(num_envs, dtype=torch.bool, device=env.device)
        env.set_template_batch(0)
        initial = env.reset_mask(all_mask)
        reset_positions = torch.tensor(env.layout.ground_positions_m, device=env.device).expand(
            num_envs, -1, -1
        )
        reset_error = (env.state[..., :3] - reset_positions).abs().amax().item()
        result["initial_reset_error"] = reset_error
        result["initial_observation_finite"] = bool(
            torch.isfinite(initial[("agents", "observation")]).all()
            and torch.isfinite(initial[("agents", "state")]).all()
        )
        event("initial_reset", reset_error=reset_error)

        if scenario == "critic-calibration":
            from align.simulation.task_training import run_training_attempt

            if training.policy_seed not in critic_calibration.policy_seeds:
                raise ValueError("training policy seed is not in critic calibration seeds")
            metrics = run_training_attempt(
                env=env,
                initial=initial,
                output=output,
                checkpoint_directory=checkpoint_directory,
                logical_run_id=logical_run_id,
                attempt_id=attempt_id,
                resume=False,
                resolved_config=resolved_config,
                config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                source_identity=source_identity,
                runtime_identity=runtime_identity,
                policy_config=policy,
                rollout_config=rollout,
                ppo_config=ppo,
                recovery_config=recovery,
                training_config=training,
                critic_normalization_config=critic_normalization,
                event=event,
                critic_calibration_config=critic_calibration,
            )
            calibration = metrics["critic_calibration"]
            save_json(output / "metrics.json", metrics)
            result.update(
                status=metrics["status"],
                drone_physics_tested=True,
                vector_task_physics_tested=True,
                critic_calibration_tested=calibration["status"] == "passed",
                optimizer_updates=1,
                agent_steps=rollout.horizon * rollout.num_envs * rollout.num_agents,
                torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            )
            event(
                "critic_calibration_finished",
                status=result["status"],
                checks=calibration["checks"],
            )
            return 0 if result["status"] == "passed" else 1

        if scenario == "stability":
            from align.simulation.stability_training import run_stability_training

            metrics = run_stability_training(
                env=env,
                initial=initial,
                output=output,
                checkpoint_directory=checkpoint_directory,
                logical_run_id=logical_run_id,
                resolved_config=resolved_config,
                config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                source_identity=source_identity,
                runtime_identity=runtime_identity,
                policy_config=policy,
                rollout_config=rollout,
                ppo_config=ppo,
                recovery_config=recovery,
                training_config=training,
                critic_normalization_config=critic_normalization,
                stability_config=stability,
                event=event,
                start_update=training_start_update,
                stop_update=training_stop_update,
                fault_at_update=fault_at_update,
                fault_after_rollout_step=fault_after_rollout_step,
            )
            save_json(output / "metrics.json", metrics)
            result.update(
                status=metrics["status"],
                drone_physics_tested=True,
                vector_task_physics_tested=True,
                training_stability_tested=True,
                optimizer_updates=metrics["updates"],
                agent_steps=metrics["segment_agent_transitions"],
                torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            )
            event("stability_training_finished", status=result["status"], checks=metrics["checks"])
            return 0 if result["status"] == "passed" else 1

        if scenario == "reference":
            from align.simulation.reference_evaluation import run_reference_evaluation

            metrics = run_reference_evaluation(
                env=env,
                output=output,
                evaluation_steps=stability.evaluation_steps
                if stability
                else task.max_episode_steps,
                event=event,
            )
            save_json(output / "metrics.json", metrics)
            result.update(
                status=metrics["status"],
                drone_physics_tested=True,
                vector_task_physics_tested=True,
                reference_evaluation_tested=True,
                shape_transition_physics_tested=(
                    transition_config is not None and metrics["status"] == "passed"
                ),
                waypoint_route_physics_tested=(
                    route_config is not None and metrics["status"] == "passed"
                ),
                optimizer_updates=0,
                agent_steps=metrics["telemetry_rows"],
                torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            )
            event(
                "reference_evaluation_finished", status=result["status"], checks=metrics["checks"]
            )
            return 0 if result["status"] == "passed" else 1

        if scenario == "evaluation":
            from align.simulation.policy_evaluation import run_policy_evaluation

            metrics = run_policy_evaluation(
                env=env,
                initial=initial,
                output=output,
                checkpoint_directory=checkpoint_directory,
                logical_run_id=logical_run_id,
                resolved_config=resolved_config,
                config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                policy_config=policy,
                ppo_config=ppo,
                rollout_config=rollout,
                training_config=training,
                critic_normalization_config=critic_normalization,
                stability_config=stability,
                event=event,
                checkpoint_update=evaluation_update,
                telemetry=policy_telemetry,
            )
            save_json(output / "metrics.json", metrics)
            result.update(
                status=metrics["status"],
                drone_physics_tested=True,
                vector_task_physics_tested=True,
                deterministic_evaluation_tested=True,
                optimizer_updates=0,
                agent_steps=metrics["raw_rows"] * construction.num_agents,
                torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            )
            event("policy_evaluation_finished", status=result["status"], checks=metrics["checks"])
            return 0 if result["status"] == "passed" else 1

        if scenario == "training":
            from align.simulation.task_training import run_training_attempt

            metrics = run_training_attempt(
                env=env,
                initial=initial,
                output=output,
                checkpoint_directory=checkpoint_directory,
                logical_run_id=logical_run_id,
                attempt_id=attempt_id,
                resume=resume,
                resolved_config=resolved_config,
                config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                source_identity=source_identity,
                runtime_identity=runtime_identity,
                policy_config=policy,
                rollout_config=rollout,
                ppo_config=ppo,
                recovery_config=recovery,
                training_config=training,
                critic_normalization_config=critic_normalization,
                event=event,
            )
            save_json(output / "metrics.json", metrics)
            result.update(
                status=metrics["status"],
                drone_physics_tested=True,
                vector_task_physics_tested=True,
                task_training_tested=True,
                optimizer_updates=1,
                agent_steps=rollout.horizon * rollout.num_envs * rollout.num_agents,
                torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            )
            event("training_attempt_finished", status=result["status"], checks=metrics["checks"])
            return 0 if result["status"] == "passed" else 1

        if scenario == "collector":
            from align.simulation.recurrent_collector import collect_live_rollout

            metrics = collect_live_rollout(
                env=env,
                initial=initial,
                output=output,
                policy_config=policy,
                rollout_config=rollout,
                collector_config=collector,
                event=event,
            )
            save_json(output / "metrics.json", metrics)
            result.update(
                status=metrics["status"],
                drone_physics_tested=True,
                vector_task_physics_tested=True,
                recurrent_collector_tested=True,
                agent_steps=metrics["agent_transitions"],
                rollout_tensor_bytes=metrics["rollout_tensor_bytes"],
                torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            )
            event("collector_checks_finished", status=result["status"], checks=metrics["checks"])
            return 0 if result["status"] == "passed" else 1

        partial_reset_isolated = scenario == "single"
        forced_termination_observed = scenario == "single"
        truncation_observed = False
        partial_reset_progress = None
        reset_events = []
        done_events = []
        parity = {}
        ground_contact_peaks = torch.zeros(num_envs, device=env.device)
        physics_started = time.perf_counter()
        max_global_steps = task.max_episode_steps
        with (output / "trajectory.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=TRAJECTORY_COLUMNS)
            writer.writeheader()
            with torch.no_grad():
                for global_step in range(max_global_steps):
                    if scenario == "batch" and global_step == 40:
                        before = env.drone.get_state().clone()
                        mask = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
                        mask[0] = True
                        env.reset_mask(mask)
                        after = env.drone.get_state().clone()
                        unaffected_error = (after[1:] - before[1:]).abs().amax().item()
                        partial_reset_isolated = unaffected_error <= 1e-7
                        partial_reset_progress = env.progress_buf.cpu().tolist()
                        reset_events.append(
                            {
                                "global_step": global_step,
                                "env_ids": [0],
                                "unaffected_state_max_error": unaffected_error,
                                "progress": partial_reset_progress,
                            }
                        )
                        event(
                            "partial_reset",
                            env_ids=[0],
                            unaffected_state_max_error=unaffected_error,
                            progress=partial_reset_progress,
                        )

                    if scenario == "batch" and global_step == 70:
                        world_pos, world_rot = env.drone.get_world_poses(clone=True)
                        world_pos = world_pos.reshape(num_envs, construction.num_agents, 3)
                        world_rot = world_rot.reshape(num_envs, construction.num_agents, 4)
                        world_pos[1, :, 0] = env.envs_positions[1, 0] + task.safety_xy_limit_m + 1.0
                        env.drone.set_world_poses(
                            world_pos[1:2], world_rot[1:2], torch.tensor([1], device=env.device)
                        )
                        env.drone.set_velocities(
                            torch.zeros(1, construction.num_agents, 6, device=env.device),
                            torch.tensor([1], device=env.device),
                        )
                        env.state = env.drone.get_state().clone()
                        event("forced_safety_state", env_id=1)

                    targets = env.targets_for_progress()
                    positions = env.state[..., :3]
                    error = targets - positions
                    command = error * construction.position_gain_s_inv
                    speed = command.norm(dim=-1, keepdim=True)
                    command = command * torch.minimum(
                        torch.ones_like(speed),
                        torch.full_like(speed, construction.max_speed_m_s) / speed.clamp_min(1e-8),
                    )
                    direction = torch.where(
                        speed > 1e-8,
                        command / command.norm(dim=-1, keepdim=True).clamp_min(1e-8),
                        torch.zeros_like(command),
                    )
                    action = torch.cat(
                        (
                            direction,
                            (command.norm(dim=-1, keepdim=True) / construction.max_speed_m_s).clamp(
                                0.0, 1.0
                            ),
                        ),
                        dim=-1,
                    )
                    action[env.phase_indices() == 0] = 0.0
                    output_td = env.step_actions(action)
                    terminated = output_td["terminated"].squeeze(-1)
                    truncated = output_td["truncated"].squeeze(-1)
                    done = output_td["done"].squeeze(-1)
                    truncation_observed |= bool(truncated.any().item())
                    codes = env.last_reason_code.cpu().tolist()
                    for env_id in done.nonzero().squeeze(-1).cpu().tolist():
                        item = {
                            "global_step": global_step,
                            "env_id": env_id,
                            "episode_step": int(env.progress_buf[env_id].item()),
                            "terminated": bool(terminated[env_id].item()),
                            "truncated": bool(truncated[env_id].item()),
                            "reason": REASON_NAMES[codes[env_id]],
                        }
                        done_events.append(item)
                        event("episode_done", **item)
                    if scenario == "batch" and global_step == 70:
                        forced_termination_observed = bool(terminated[1].item() and codes[1] == 4)

                    ground_contact_peaks = torch.maximum(
                        ground_contact_peaks,
                        torch.where(
                            env.last_reward_phase == 0,
                            env.contact_force.amax(dim=-1),
                            torch.zeros_like(ground_contact_peaks),
                        ),
                    )
                    states_cpu = env.state.cpu()
                    targets_cpu = env.last_targets.cpu()
                    actions_cpu = env.last_actions.cpu()
                    contacts_cpu = env.contact_force.cpu()
                    rewards_cpu = output_td[("agents", "reward")].squeeze(-1).cpu()
                    for env_id in range(num_envs):
                        reason = REASON_NAMES[codes[env_id]]
                        for agent in range(construction.num_agents):
                            position = states_cpu[env_id, agent, :3].tolist()
                            velocity = states_cpu[env_id, agent, 7:10].tolist()
                            target = targets_cpu[env_id, agent].tolist()
                            act = actions_cpu[env_id, agent].tolist()
                            writer.writerow(
                                {
                                    "global_step": global_step,
                                    "env_id": env_id,
                                    "episode_step": int(env.progress_buf[env_id].item()),
                                    "phase": phase_name(int(env.last_reward_phase[env_id].item())),
                                    "agent_id": agent,
                                    "x": position[0],
                                    "y": position[1],
                                    "z": position[2],
                                    "vx": velocity[0],
                                    "vy": velocity[1],
                                    "vz": velocity[2],
                                    "target_x": target[0],
                                    "target_y": target[1],
                                    "target_z": target[2],
                                    "a0": act[0],
                                    "a1": act[1],
                                    "a2": act[2],
                                    "a3": act[3],
                                    "contact_force_n": float(contacts_cpu[env_id, agent].item()),
                                    "reward": float(rewards_cpu[env_id, agent].item()),
                                    "terminated": bool(terminated[env_id].item()),
                                    "truncated": bool(truncated[env_id].item()),
                                    "reason": reason or "",
                                }
                            )
                    if global_step == 0:
                        cpu_reward, _ = compute_step_reward(
                            states_cpu[0, :, :3].tolist(),
                            targets_cpu[0].tolist(),
                            states_cpu[0, :, 7:10].tolist(),
                            actions_cpu[0].tolist(),
                            contacts_cpu[0].tolist(),
                            [
                                value > task.airborne_height_m
                                for value in states_cpu[0, :, 2].tolist()
                            ],
                            reward,
                            RewardMemory(),
                        )
                        reward_error = max(
                            abs(expected - actual)
                            for expected, actual in zip(
                                cpu_reward.total_by_agent,
                                rewards_cpu[0].tolist(),
                                strict=True,
                            )
                        )
                        observation_targets = env.last_observation_targets[0].cpu().tolist()
                        cpu_observation = build_observations(
                            env.layout.agent_ids,
                            states_cpu[0, :, :3].tolist(),
                            states_cpu[0, :, 7:10].tolist(),
                            observation_targets,
                            observation,
                        )
                        actor_expected = [list(item.flat()) for item in cpu_observation.actors]
                        actor_actual = env.last_actor_observation[0].cpu().tolist()
                        actor_error = max(
                            abs(expected - actual)
                            for expected_row, actual_row in zip(
                                actor_expected, actor_actual, strict=True
                            )
                            for expected, actual in zip(expected_row, actual_row, strict=True)
                        )
                        critic_error = max(
                            abs(expected - actual)
                            for expected, actual in zip(
                                cpu_observation.critic.flat(),
                                env.last_critic_observation[0].cpu().tolist(),
                                strict=True,
                            )
                        )
                        parity = {
                            "reward_max_abs_error": reward_error,
                            "actor_observation_max_abs_error": actor_error,
                            "critic_observation_max_abs_error": critic_error,
                        }

                    if scenario == "batch" and global_step == 70:
                        mask = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
                        mask[1] = True
                        env.reset_mask(mask)
                        reset_events.append(
                            {
                                "global_step": global_step,
                                "env_ids": [1],
                                "reason": "after_forced_termination",
                                "progress": env.progress_buf.cpu().tolist(),
                            }
                        )
                    stream.flush()
                    if done.all():
                        break

        torch.cuda.synchronize()
        physics_seconds = time.perf_counter() - physics_started
        total_agent_steps = sum(1 for _ in open(output / "trajectory.csv")) - 1
        checks = {
            "reset_exact": reset_error <= construction.reset_tolerance,
            "ground_contacts_observed": bool(
                (ground_contact_peaks > task.contact_force_threshold_n).all().item()
            ),
            "finite_initial_observations": result["initial_observation_finite"],
            "tensor_shapes": (
                tuple(env.last_actor_observation.shape)
                == (num_envs, construction.num_agents, observation.actor_dimension)
                and tuple(env.last_critic_observation.shape)
                == (num_envs, observation.critic_dimension)
            ),
            "bounded_actions": bool((env.last_actions.abs() <= 1.0).all().item()),
            "torch_cpu_reward_parity": parity.get("reward_max_abs_error", math.inf) <= 1e-5,
            "torch_cpu_actor_parity": (
                parity.get("actor_observation_max_abs_error", math.inf) <= 1e-6
            ),
            "torch_cpu_critic_parity": (
                parity.get("critic_observation_max_abs_error", math.inf) <= 1e-6
            ),
            "truncation_observed": truncation_observed,
            "partial_reset_isolated": partial_reset_isolated,
            "forced_true_termination_observed": forced_termination_observed,
        }
        if scenario == "single":
            checks.pop("partial_reset_isolated")
            checks.pop("forced_true_termination_observed")
        metrics = {
            "status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
            "parity": parity,
            "reset_events": reset_events,
            "done_events": done_events,
            "physics_loop_wall_seconds": physics_seconds,
            "agent_steps": total_agent_steps,
            "agent_steps_per_wall_second": total_agent_steps / physics_seconds,
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "partial_reset_progress": partial_reset_progress,
            "ground_contact_peaks": ground_contact_peaks.cpu().tolist(),
        }
        save_json(output / "metrics.json", metrics)
        result.update(
            status=metrics["status"],
            physics_loop_wall_seconds=physics_seconds,
            agent_steps=total_agent_steps,
            agent_steps_per_wall_second=metrics["agent_steps_per_wall_second"],
            torch_peak_allocated_bytes=metrics["torch_peak_allocated_bytes"],
            drone_physics_tested=True,
            vector_task_physics_tested=True,
        )
        event("checks_finished", status=result["status"], checks=checks)
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
        print("ALIGN_VECTOR_TASK_RESULT=" + json.dumps(result, allow_nan=False), flush=True)
        if app is not None:
            app.close()
    return 0 if result["status"] == "passed" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scenario",
        choices=(
            "single",
            "batch",
            "collector",
            "training",
            "stability",
            "evaluation",
            "reference",
            "critic-calibration",
        ),
        required=True,
    )
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--logical-run-id")
    parser.add_argument("--attempt-id")
    parser.add_argument("--checkpoint-directory", type=Path)
    parser.add_argument("--evaluation-update", type=int)
    parser.add_argument("--policy-telemetry", action="store_true")
    parser.add_argument("--evaluation-timing-config", type=Path)
    parser.add_argument("--shape-transition-config", type=Path)
    parser.add_argument("--waypoint-route-config", type=Path)
    parser.add_argument("--training-start-update", type=int, default=0)
    parser.add_argument("--training-stop-update", type=int)
    parser.add_argument("--fault-at-update", type=int)
    parser.add_argument("--fault-after-rollout-step", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--source-identity")
    parser.add_argument("--runtime-identity")
    parser.add_argument("--allow-root", action="store_true")
    args = parser.parse_args(argv)
    return run(
        args.config,
        args.output,
        args.scenario,
        args.num_envs,
        logical_run_id=args.logical_run_id,
        attempt_id=args.attempt_id,
        checkpoint_directory=args.checkpoint_directory,
        resume=args.resume,
        source_identity=args.source_identity,
        runtime_identity=args.runtime_identity,
        evaluation_update=args.evaluation_update,
        policy_telemetry=args.policy_telemetry,
        evaluation_timing_config=args.evaluation_timing_config,
        shape_transition_config=args.shape_transition_config,
        waypoint_route_config=args.waypoint_route_config,
        training_start_update=args.training_start_update,
        training_stop_update=args.training_stop_update,
        fault_at_update=args.fault_at_update,
        fault_after_rollout_step=args.fault_after_rollout_step,
    )


if __name__ == "__main__":
    raise SystemExit(main())
