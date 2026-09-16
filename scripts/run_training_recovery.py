"""Thin host entrypoint for the training-recovery acceptance probe."""

from align.runtime.recovery_runtime import run_main

if __name__ == "__main__":
    raise SystemExit(run_main())
