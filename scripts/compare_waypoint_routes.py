#!/usr/bin/env python3
"""Thin entrypoint for matched direct-versus-waypoint flight reports."""

from align.runtime.waypoint_comparison import main

if __name__ == "__main__":
    raise SystemExit(main())
