#!/usr/bin/env python3
"""Thin entry point for target-conditioned four-template training acceptance."""

from align.runtime.learning_curve_runtime import run_main

if __name__ == "__main__":
    raise SystemExit(
        run_main(
            default_curve_config="learning-curve-multi-template-probe.json",
            run_category="multi-template-training",
            default_formation_schedule_config="formation-schedule-four-templates.json",
        )
    )
