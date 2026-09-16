"""Thin host entrypoint for the deterministic OmniDrones acceptance check."""

from align.runtime.drone_runtime import run_main

if __name__ == "__main__":
    raise SystemExit(run_main())
