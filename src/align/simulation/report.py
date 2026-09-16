"""Re-evaluate a saved flight without Isaac Sim or PyTorch."""

import argparse
import csv
import json
from pathlib import Path

from align.artifacts import create_run_directory, write_json_atomic
from align.simulation.contract import DroneCheckConfig, evaluate


def assess_saved_run(run):
    config = DroneCheckConfig.from_dict(json.loads((run / "config.json").read_text()))
    with (run / "trajectory.csv").open(newline="") as stream:
        rows = [
            {key: value if key == "case" else float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]
    resets = json.loads((run / "resets.json").read_text())["resets"]
    return evaluate(rows, resets, config), rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--plots", action="store_true", help="Requires the reporting extra")
    args = parser.parse_args(argv)
    run = args.run.resolve()
    output = create_run_directory(run.parent.parent / "single-drone-reports")
    result, rows = assess_saved_run(run)
    result["source_run"] = str(run)
    write_json_atomic(output / "metrics.json", result)
    if args.plots:
        from align.simulation.single_drone import save_plots

        save_plots(rows, output)
    print(f"{result['status']}: {output}")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
