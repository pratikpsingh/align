#!/usr/bin/env python3
"""Thin entry point for CPU-only analysis of frozen formation episodes."""

from align.runtime.formation_diagnosis import main

if __name__ == "__main__":
    raise SystemExit(main())
