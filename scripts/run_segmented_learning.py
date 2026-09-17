#!/usr/bin/env python3
"""Thin entry point for segmented fresh-process checkpoint learning."""

from align.runtime.learning_curve_runtime import run_main

if __name__ == "__main__":
    raise SystemExit(
        run_main(
            default_curve_config="learning-curve-segmented.json",
            run_category="segmented-learning",
        )
    )
