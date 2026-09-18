"""Check the declared long-schedule training/evaluation geometry."""

import json
import unittest
from pathlib import Path

from align.learning.learning_curve_config import LearningCurveConfig
from align.learning.rollout import RolloutConfig
from align.runtime.stability_runtime import validate_stability_resolved_config
from align.simulation.multi_drone_contract import MultiDroneConfig
from align.tasks.environment import TaskEnvironmentConfig
from align.tasks.evaluation_timing import EvaluationTimingConfig, apply_evaluation_timing

ROOT = Path(__file__).resolve().parents[1] / "configs"


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


class FeasiblePlaneProtocolTests(unittest.TestCase):
    def test_training_schedule_matches_accepted_reference_timing(self):
        baseline_construction = MultiDroneConfig.from_dict(read("multi-drone-construction.json"))
        baseline_task = TaskEnvironmentConfig.from_dict(read("recurrent-stability-task.json"))
        baseline_curve = LearningCurveConfig.from_dict(
            read("learning-curve-multi-template-probe.json")
        )
        timing = EvaluationTimingConfig.from_dict(read("evaluation-timing-extended.json"))
        construction, task, stability, details = apply_evaluation_timing(
            timing, baseline_construction, baseline_task, baseline_curve.to_stability_config()
        )
        self.assertEqual(
            MultiDroneConfig.from_dict(read("feasible-plane-construction.json")), construction
        )
        self.assertEqual(TaskEnvironmentConfig.from_dict(read("feasible-plane-task.json")), task)
        curve = LearningCurveConfig.from_dict(read("learning-curve-feasible-plane-probe.json"))
        self.assertEqual(curve.evaluation_steps, stability.evaluation_steps)
        self.assertEqual(curve.evaluation_milestones, (0, 2, 4))
        self.assertEqual(details["effective_evaluation_steps"], 2350)
        rollout = RolloutConfig.from_dict(read("feasible-plane-rollout.json"))
        self.assertEqual(rollout.horizon, 2304)
        self.assertLess(2104, rollout.horizon)
        self.assertLess(rollout.horizon, task.max_episode_steps)
        self.assertEqual(read("critic-normalization-feasible-plane.json")["warmup_steps"], 2304)
        self.assertEqual(
            validate_stability_resolved_config(
                {
                    "construction": construction.to_dict(),
                    "task": task.to_dict(),
                    "rollout": rollout.to_dict(),
                    "critic_normalization": read("critic-normalization-feasible-plane.json"),
                },
                curve.to_stability_config(),
            ),
            1550,
        )


if __name__ == "__main__":
    unittest.main()
