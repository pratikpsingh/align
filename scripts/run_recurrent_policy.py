"""Thin host entrypoint for the recurrent-policy tensor acceptance probe."""

from align.runtime.policy_runtime import run_main

if __name__ == "__main__":
    raise SystemExit(run_main())
