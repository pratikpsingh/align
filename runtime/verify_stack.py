"""Fail if vendor libraries changed or the audited dependency closure is incomplete."""

import json
import sys
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement

stack = json.loads(Path("/opt/align-runtime/stack.json").read_text())
if list(sys.version_info[:2]) != stack["python_minor"]:
    raise RuntimeError(f"Unexpected Python: {sys.version}")
for name, expected in {**stack["vendor_required"], **stack["additions"]}.items():
    actual = metadata.version(name)
    if actual != expected:
        raise RuntimeError(f"{name}: expected {expected}, got {actual}")
for name in stack["additions"]:
    for requirement in metadata.requires(name) or []:
        req = Requirement(requirement)
        if req.marker and not req.marker.evaluate({"extra": ""}):
            continue
        actual = metadata.version(req.name)
        if actual not in req.specifier:
            raise RuntimeError(f"{name} requires {req}; installed {actual}")
print("Audited runtime dependency closure and vendor versions verified.", flush=True)
