"""Thin entrypoint for the read-only vendor inventory."""

from align.runtime.vendor_inventory import main

if __name__ == "__main__":
    raise SystemExit(main())
