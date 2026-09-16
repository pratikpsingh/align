"""Thin host entrypoint for bounded task-connected recurrent training."""

from align.runtime.training_runtime import run_main

if __name__ == "__main__":
    raise SystemExit(run_main())
