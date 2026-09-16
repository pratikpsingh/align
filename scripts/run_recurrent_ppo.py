"""Thin host entrypoint for the recurrent MAPPO optimizer acceptance probe."""

from align.runtime.ppo_runtime import run_main

if __name__ == "__main__":
    raise SystemExit(run_main())
