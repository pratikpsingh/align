"""Thin host entrypoint for vectorized OmniDrones task acceptance."""

from align.runtime.vector_task_runtime import run_main

if __name__ == "__main__":
    raise SystemExit(run_main())
