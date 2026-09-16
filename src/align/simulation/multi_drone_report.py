"""Re-evaluate and plot a saved multi-drone construction run without Isaac Sim."""

import argparse
import csv
import json
from pathlib import Path

from align.artifacts import create_run_directory, write_json_atomic
from align.formations import evaluate_formation
from align.simulation.multi_drone_contract import MultiDroneConfig, evaluate


def assess_saved_run(run):
    config = MultiDroneConfig.from_dict(json.loads((run / "config.json").read_text()))
    with (run / "trajectory.csv").open(newline="") as stream:
        rows = [
            {key: value if key == "phase" else float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]
    resets = json.loads((run / "resets.json").read_text())["resets"]
    episodes = json.loads((run / "episodes.json").read_text())["episodes"]
    return evaluate(rows, resets, episodes, config), rows


def save_plots(rows, output):
    """Plot raw positions and independently recomputed group metrics."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    samples = [row for row in rows if int(row["repeat"]) == 0]
    agent_ids = sorted({int(row["agent_id"]) for row in samples})
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for agent in agent_ids:
        agent_rows = [row for row in samples if int(row["agent_id"]) == agent]
        axes[0, 0].plot(
            [row["x"] for row in agent_rows],
            [row["y"] for row in agent_rows],
            label=f"agent {agent}",
        )
        axes[0, 0].scatter(
            agent_rows[-1]["target_x"],
            agent_rows[-1]["target_y"],
            marker="x",
        )
        axes[0, 1].plot(
            [row["t"] for row in agent_rows],
            [row["z"] for row in agent_rows],
            label=f"agent {agent}",
        )

    by_step = {}
    for row in samples:
        by_step.setdefault(int(row["step"]), []).append(row)
    times, assigned, pairwise, separation, contacts = [], [], [], [], []
    for step in sorted(by_step):
        step_rows = sorted(by_step[step], key=lambda row: int(row["agent_id"]))
        positions = tuple((row["x"], row["y"], row["z"]) for row in step_rows)
        targets = tuple((row["target_x"], row["target_y"], row["target_z"]) for row in step_rows)
        metrics = evaluate_formation(positions, targets)
        times.append(step_rows[0]["t"])
        assigned.append(metrics.assigned_root_mean_squared_error_m)
        pairwise.append(metrics.pairwise_root_mean_squared_error_m)
        separation.append(metrics.minimum_actual_separation_m)
        contacts.append(max(row["contact_force_n"] for row in step_rows))

    axes[1, 0].plot(times, assigned, label="assigned RMSE")
    axes[1, 0].plot(times, pairwise, label="pairwise RMSE")
    axes[1, 1].plot(times, separation, label="minimum separation")
    axes[1, 1].plot(times, contacts, label="maximum contact force")
    axes[0, 0].set(title="XY construction paths and final targets", xlabel="X (m)", ylabel="Y (m)")
    axes[0, 0].axis("equal")
    axes[0, 1].set(title="Altitude", xlabel="simulation seconds", ylabel="Z (m)")
    axes[1, 0].set(title="Formation errors", xlabel="simulation seconds", ylabel="metres")
    axes[1, 1].set(
        title="Safety measurements",
        xlabel="simulation seconds",
        ylabel="metres / newtons",
    )
    for axis in axes.flat:
        axis.grid(alpha=0.3)
        axis.legend()
    fig.savefig(output / "trajectory.png", dpi=150)
    fig.savefig(output / "trajectory.pdf")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--plots", action="store_true", help="Requires the reporting extra")
    args = parser.parse_args(argv)
    run = args.run.resolve()
    output = create_run_directory(run.parent.parent / "multi-drone-reports")
    result, rows = assess_saved_run(run)
    result["source_run"] = str(run)
    write_json_atomic(output / "metrics.json", result)
    if args.plots:
        save_plots(rows, output)
    print(f"{result['status']}: {output}")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
