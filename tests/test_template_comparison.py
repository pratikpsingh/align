"""Checks for matched-budget comparison and misleading-result rejection."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from align.runtime.template_comparison import audit_raw_csv, compare_runs


class TemplateComparisonTests(unittest.TestCase):
    def test_long_budget_uses_validated_segment_contract(self):
        from align.learning.learning_curve_config import LearningCurveConfig

        root = Path(__file__).resolve().parents[1]
        values = json.loads((root / "configs/learning-curve-template-comparison.json").read_text())
        config = LearningCurveConfig.from_dict(values)
        self.assertEqual(config.policy_seeds, (41, 73))
        self.assertEqual(config.segment_ranges(), ((0, 4), (4, 8), (8, 12)))
        self.assertEqual(config.evaluation_milestones, (0, 4, 8, 12))

    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.curve = {
            "policy_seeds": [41, 73],
            "updates_per_seed": 4,
            "evaluation_milestones": [0, 2, 4],
            "evaluation_steps": 800,
        }
        self.plane = self._write_run("plane", ["plane"])
        self.generalist = self._write_run("generalist", ["cube", "sphere", "pyramid", "plane"])

    @staticmethod
    def _template(kind, seed, milestone, kinds):
        rows = 3200 // len(kinds)
        return {
            "rows": rows,
            "formation_rows": rows // 4,
            "outcome_counts": {str(code): int(code == 6) for code in range(1, 7)},
            "team_reward_mean": -0.1 + milestone / 100,
            "assigned_rmse_mean_m": 1.0 + seed / 1000 + milestone / 100 + len(kinds) / 10,
            "pairwise_rmse_mean_m": 0.2,
            "minimum_separation_m": 0.8,
        }

    def _write_run(self, name, kinds):
        path = self.root / name
        path.mkdir()
        config = {"formation_schedule": {"kinds": kinds}, "policy": {"width": 64}}
        seeds = []
        for seed in self.curve["policy_seeds"]:
            training = {
                "metrics": {
                    "measurements": [
                        {
                            "completed_update": update,
                            "template_measurements": {
                                kind: {"rows": 3072 // len(kinds), "formation_rows": 200}
                                for kind in kinds
                            },
                        }
                        for update in range(1, 5)
                    ]
                }
            }
            evaluations = []
            for milestone in self.curve["evaluation_milestones"]:
                templates = {kind: self._template(kind, seed, milestone, kinds) for kind in kinds}
                evaluations.append(
                    {
                        "exit_code": 0,
                        "completed_update": milestone,
                        "metrics": {
                            "status": "passed",
                            "completed_updates": milestone,
                            "requested_completed_update": milestone,
                            "raw_rows": 3200,
                            "outcome_counts": {
                                str(code): sum(
                                    row["outcome_counts"][str(code)] for row in templates.values()
                                )
                                for code in range(1, 7)
                            },
                            "template_measurements": templates,
                        },
                    }
                )
            seeds.append({"policy_seed": seed, "train": training, "evaluations": evaluations})
        report = {
            "status": "passed",
            "deterministic_evaluation_performed": True,
            "image_id": "sha256:fixture",
            "source": {"package_sha256": "source-fixture"},
            "optimizer_updates": 8,
            "evaluation_containers": 6,
            "seeds": seeds,
            "summary": {"template_evaluation_trends": []},
        }
        documents = {
            "report.json": report,
            "config.json": config,
            "learning-curve-config.json": self.curve,
            "build-report.json": {"status": "built", "image_id": "sha256:fixture"},
        }
        for filename, value in documents.items():
            (path / filename).write_text(json.dumps(value))
        return path

    def _edit(self, path, filename, change):
        target = path / filename
        value = json.loads(target.read_text())
        change(value)
        target.write_text(json.dumps(value))

    def test_paired_plane_deltas_keep_seed_and_checkpoint_identity(self):
        result = compare_runs(self.plane, self.generalist)
        self.assertEqual(result["image_id"], "sha256:fixture")
        self.assertEqual(len(result["paired_plane"]), 6)
        self.assertEqual([row["completed_update"] for row in result["plane_summary"]], [0, 2, 4])
        self.assertAlmostEqual(
            result["paired_plane"][0]["generalist_minus_baseline_assigned_rmse_mean_m"],
            0.3,
        )
        self.assertEqual(result["plane_summary"][2]["baseline_successes"], 0)
        self.assertEqual(result["plane_summary"][2]["generalist_completed_episodes"], 2)
        self.assertEqual(result["training_exposure"][0]["baseline_plane_training_rows"], 12288)
        self.assertEqual(result["training_exposure"][0]["generalist_plane_training_rows"], 3072)
        self.assertIn("same total updates, unequal plane exposure", result["comparison_scope"])

    def test_rejects_unequal_budget_source_or_image(self):
        for filename, mutation, expected in (
            (
                "learning-curve-config.json",
                lambda v: v.update(evaluation_steps=700),
                "different seeds, budgets, or milestones",
            ),
            ("report.json", lambda v: v.update(image_id="sha256:other"), "image identity"),
            (
                "report.json",
                lambda v: v["source"].update(package_sha256="other"),
                "same host package source",
            ),
            (
                "config.json",
                lambda v: v["policy"].update(width=128),
                "beyond the formation schedule",
            ),
        ):
            with self.subTest(filename=filename, expected=expected):
                target = self.generalist / filename
                original = target.read_text()
                self._edit(self.generalist, filename, mutation)
                with self.assertRaisesRegex(ValueError, expected):
                    compare_runs(self.plane, self.generalist)
                target.write_text(original)

    def test_rejects_missing_template_or_wrong_checkpoint(self):
        original = (self.generalist / "report.json").read_text()
        self._edit(
            self.generalist,
            "report.json",
            lambda v: v["seeds"][0]["evaluations"][0]["metrics"]["template_measurements"].pop(
                "cube"
            ),
        )
        with self.assertRaisesRegex(ValueError, "template coverage"):
            compare_runs(self.plane, self.generalist)
        (self.generalist / "report.json").write_text(original)
        self._edit(
            self.generalist,
            "report.json",
            lambda v: v["seeds"][0]["evaluations"][0]["metrics"].update(completed_updates=2),
        )
        with self.assertRaisesRegex(ValueError, "requested checkpoint"):
            compare_runs(self.plane, self.generalist)

    def test_raw_csv_audit_recomputes_metrics_and_rejects_tampering(self):
        path = self.root / "evaluation.csv"
        fields = (
            "formation_kind",
            "phase",
            "team_reward",
            "assigned_rmse_m",
            "pairwise_rmse_m",
            "minimum_separation_m",
            "reason_code",
        )
        rows = [
            ("plane", "ground", "0.1", "0.2", "0.3", "0.9", "0"),
            ("plane", "formation", "0.3", "0.4", "0.5", "0.8", "6"),
        ]
        path.write_text(",".join(fields) + "\n" + "\n".join(",".join(row) for row in rows) + "\n")
        expected = {
            "plane": {
                "rows": 2,
                "formation_rows": 1,
                "outcome_counts": {str(code): int(code == 6) for code in range(1, 7)},
                "team_reward_mean": 0.2,
                "assigned_rmse_mean_m": 0.3,
                "pairwise_rmse_mean_m": 0.4,
                "minimum_separation_m": 0.8,
            }
        }
        self.assertEqual(audit_raw_csv(path, expected, evaluation=True), 2)
        path.write_text(path.read_text().replace("0.3,0.4,0.5", "0.9,0.4,0.5"))
        with self.assertRaisesRegex(ValueError, "team_reward_mean differs"):
            audit_raw_csv(path, expected, evaluation=True)

    def test_rejects_inconsistent_raw_coverage(self):
        self._edit(
            self.generalist,
            "report.json",
            lambda v: v["seeds"][0]["evaluations"][0]["metrics"].update(raw_rows=3199),
        )
        with self.assertRaisesRegex(ValueError, "template rows differ"):
            compare_runs(self.plane, self.generalist)


if __name__ == "__main__":
    unittest.main()
