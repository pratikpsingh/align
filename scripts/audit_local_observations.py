"""Thin entrypoint for auditing bounded local observations on saved trajectories."""

from align.tasks.observation_report import main

if __name__ == "__main__":
    raise SystemExit(main())
