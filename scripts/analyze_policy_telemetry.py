#!/usr/bin/env python3
"""Thin entry point for CPU-only policy telemetry analysis."""

from align.runtime.policy_telemetry_analysis import run_main

if __name__ == "__main__":
    raise SystemExit(run_main())
