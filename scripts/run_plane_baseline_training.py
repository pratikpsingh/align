#!/usr/bin/env python3
"""Thin entry point for a plane-only matched training arm."""

from align.runtime.learning_curve_runtime import run_main

if __name__ == "__main__":
    raise SystemExit(
        run_main(
            default_curve_config="learning-curve-template-comparison.json",
            run_category="plane-baseline-training",
            default_formation_schedule_config="formation-schedule-plane.json",
        )
    )
