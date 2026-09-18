#!/usr/bin/env python3
"""Thin entry point for CPU-only timing comparison audit."""

from align.runtime.policy_timing_analysis import run_main

if __name__ == "__main__":
    raise SystemExit(run_main())
