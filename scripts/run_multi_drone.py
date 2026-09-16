"""Thin host entrypoint for the deterministic multi-drone construction check."""

from align.runtime.multi_drone_runtime import run_main

if __name__ == "__main__":
    raise SystemExit(run_main())
