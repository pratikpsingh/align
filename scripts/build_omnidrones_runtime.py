"""Thin host entrypoint for the pinned derived-image build."""

from align.runtime.drone_runtime import build_main

if __name__ == "__main__":
    raise SystemExit(build_main())
