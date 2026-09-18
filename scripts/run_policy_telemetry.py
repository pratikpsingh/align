#!/usr/bin/env python3
"""Thin entry point for frozen-policy per-drone telemetry."""

from align.runtime.policy_telemetry_runtime import run_main

if __name__ == "__main__":
    raise SystemExit(run_main())
