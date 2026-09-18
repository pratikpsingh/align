#!/usr/bin/env python3
"""Thin entrypoint for a verified actor-only vendor-PyTorch export."""

from align.runtime.actor_export_runtime import run_main

if __name__ == "__main__":
    raise SystemExit(run_main())
