"""Thin entrypoint for simulator-independent flight reporting."""

from align.simulation.report import main

if __name__ == "__main__":
    raise SystemExit(main())
