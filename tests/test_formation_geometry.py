"""Formation geometry tests remain independent of simulator and learning packages."""

import itertools
import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from align.formations.geometry import (
    SUPPORTED_KINDS,
    assign_agents_to_slots,
    evaluate_formation,
    generate_template,
    place_template,
)
from align.formations.report import build_report, load_config, main


def squared_distance(first, second):
    return sum((left - right) ** 2 for left, right in zip(first, second, strict=True))


def minimum_spacing(points):
    return min(math.dist(first, second) for first, second in itertools.combinations(points, 2))


class TemplateTests(unittest.TestCase):
    def test_all_templates_have_count_centroid_and_enforced_spacing(self):
        for kind in SUPPORTED_KINDS:
            with self.subTest(kind=kind):
                template = generate_template(kind, 8, 0.75)
                self.assertEqual(len(template.points_m), 8)
                for coordinate in range(3):
                    self.assertAlmostEqual(
                        sum(point[coordinate] for point in template.points_m), 0.0
                    )
                self.assertAlmostEqual(minimum_spacing(template.points_m), 0.75)
                self.assertGreaterEqual(template.diameter_m, 0.75)

    def test_cube_eight_is_the_cube_vertices(self):
        points = set(generate_template("cube", 8).points_m)
        expected = set(itertools.product((-0.5, 0.5), repeat=3))
        self.assertEqual(points, expected)

    def test_plane_is_horizontal_and_pyramid_has_apex(self):
        plane = generate_template("plane", 8).points_m
        self.assertTrue(all(point[2] == 0.0 for point in plane))

        pyramid = generate_template("pyramid", 5).points_m
        levels = sorted({round(point[2], 12) for point in pyramid})
        self.assertEqual(len(levels), 2)
        self.assertEqual(sum(point[2] == levels[-1] for point in pyramid), 1)

    def test_sphere_is_not_the_cube_under_pairwise_distance_signature(self):
        cube = generate_template("cube", 8).points_m
        sphere = generate_template("sphere", 8).points_m

        def signature(points):
            return sorted(
                round(math.dist(first, second), 10)
                for first, second in itertools.combinations(points, 2)
            )

        self.assertNotEqual(signature(cube), signature(sphere))
        self.assertGreater(max(point[2] for point in sphere), min(point[2] for point in sphere))

    def test_generation_is_deterministic_and_rejects_invalid_inputs(self):
        for kind in SUPPORTED_KINDS:
            self.assertEqual(generate_template(kind, 13), generate_template(kind, 13))
        for arguments in (("ring", 8, 1.0), ("cube", 0, 1.0), ("cube", 8, 0.0)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                generate_template(*arguments)

    def test_placement_uses_positive_yaw_about_world_z_then_translation(self):
        template = generate_template("plane", 4, 1.0)
        placed = place_template(template, (2.0, 3.0, 4.0), math.pi / 2.0)
        for original, transformed in zip(template.points_m, placed, strict=True):
            self.assertAlmostEqual(transformed[0], 2.0 - original[1])
            self.assertAlmostEqual(transformed[1], 3.0 + original[0])
            self.assertAlmostEqual(transformed[2], 4.0)


class AssignmentTests(unittest.TestCase):
    def test_reversed_targets_are_recovered_without_error(self):
        targets = generate_template("sphere", 8).points_m
        assignment = assign_agents_to_slots(tuple(reversed(targets)), targets)
        self.assertEqual(assignment.slot_for_agent, tuple(reversed(range(8))))
        self.assertAlmostEqual(assignment.total_squared_distance_m2, 0.0)

    def test_hungarian_result_matches_brute_force_optimum(self):
        agents = ((0.1, 0.0, 0.0), (2.2, 0.0, 0.0), (0.0, 2.8, 0.0), (2.4, 2.1, 0.0))
        targets = ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (2.0, 2.0, 0.0))
        result = assign_agents_to_slots(agents, targets)
        brute_force = min(
            sum(squared_distance(agents[index], targets[slot]) for index, slot in enumerate(order))
            for order in itertools.permutations(range(4))
        )
        self.assertAlmostEqual(result.total_squared_distance_m2, brute_force)
        self.assertEqual(len(set(result.slot_for_agent)), 4)

    def test_ties_are_deterministic_and_counts_must_match(self):
        agents = ((0.0, 0.0, 0.0),) * 3
        targets = ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 2.0, 0.0))
        first = assign_agents_to_slots(agents, targets)
        self.assertEqual(first, assign_agents_to_slots(agents, targets))
        with self.assertRaises(ValueError):
            assign_agents_to_slots(agents, targets[:2])


class MetricTests(unittest.TestCase):
    def test_common_translation_changes_tracking_but_not_shape(self):
        targets = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        agents = tuple((x + 1.0, y, z) for x, y, z in targets)
        metrics = evaluate_formation(agents, targets)
        self.assertAlmostEqual(metrics.assigned_mean_squared_error_m2, 1.0)
        self.assertAlmostEqual(metrics.pairwise_mean_squared_error_m2, 0.0)

    def test_mean_does_not_grow_with_agent_count_like_sum(self):
        for count in (2, 8):
            targets = tuple((float(index), 0.0, 0.0) for index in range(count))
            agents = tuple((x, y + 1.0, z) for x, y, z in targets)
            metrics = evaluate_formation(agents, targets)
            self.assertAlmostEqual(metrics.assigned_sum_squared_error_m2, float(count))
            self.assertAlmostEqual(metrics.assigned_mean_squared_error_m2, 1.0)
            self.assertAlmostEqual(metrics.assigned_root_mean_squared_error_m, 1.0)

    def test_known_pairwise_distortion_and_normalization(self):
        targets = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        agents = ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))
        metrics = evaluate_formation(agents, targets)
        self.assertAlmostEqual(metrics.pairwise_mean_squared_error_m2, 1.0)
        self.assertAlmostEqual(metrics.pairwise_normalized_mean_squared_error, 1.0)
        self.assertAlmostEqual(metrics.minimum_actual_separation_m, 2.0)


class ReportTests(unittest.TestCase):
    def test_report_covers_each_shape_and_records_contract(self):
        config = {
            "schema_version": 1,
            "num_agents": 8,
            "minimum_spacing_m": 1.0,
            "center_m": [0.0, 0.0, 2.0],
            "yaw_rad": 0.0,
            "kinds": list(SUPPORTED_KINDS),
        }
        report = build_report(config, config_path=Path("config.json"), command="inspect")
        self.assertTrue(report["passed"])
        self.assertEqual(
            [formation["kind"] for formation in report["formations"]],
            list(SUPPORTED_KINDS),
        )
        self.assertEqual(report["coordinate_contract"]["axes"]["+z"], "up")
        self.assertTrue(all(all(item["template_checks"].values()) for item in report["formations"]))
        self.assertIn("+05:30", report["created_at_ist"])

    def test_configuration_rejects_implicit_or_malformed_values(self):
        base = {
            "schema_version": 1,
            "num_agents": 8,
            "minimum_spacing_m": 1.0,
            "center_m": [0.0, 0.0, 2.0],
            "yaw_rad": 0.0,
            "kinds": list(SUPPORTED_KINDS),
        }
        for key, value in (
            ("num_agents", True),
            ("minimum_spacing_m", "1.0"),
            ("center_m", [0.0, 0.0]),
            ("yaw_rad", math.inf),
            ("kinds", [["cube"]]),
        ):
            with self.subTest(key=key), TemporaryDirectory() as directory:
                malformed = {**base, key: value}
                path = Path(directory) / "config.json"
                path.write_text(json.dumps(malformed))
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_command_saves_atomic_report_and_logs(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "runs"
            self.assertEqual(
                main(
                    [
                        "--config",
                        str(Path("configs/formations/eight-drone-suite.json").resolve()),
                        "--output-dir",
                        str(output),
                    ]
                ),
                0,
            )
            (run_dir,) = output.iterdir()
            report = json.loads((run_dir / "report.json").read_text())
            self.assertEqual(report["status"], "passed")
            self.assertGreaterEqual(report["duration_seconds"], 0.0)
            self.assertIn("report_completed", (run_dir / "events.jsonl").read_text())
            self.assertIn("passed", (run_dir / "formation.log").read_text())


if __name__ == "__main__":
    unittest.main()
