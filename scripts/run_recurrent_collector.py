"""Thin host entrypoint for the live recurrent rollout collector."""

from align.runtime.collector_runtime import run_main

if __name__ == "__main__":
    raise SystemExit(run_main())
