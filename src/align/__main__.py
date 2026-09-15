"""Support python -m align using the same entrypoint as the align command."""

from align.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
