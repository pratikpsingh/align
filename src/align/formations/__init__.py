"""Simulator-independent formation geometry, assignment, and metrics."""

from align.formations.geometry import (
    SUPPORTED_KINDS,
    Assignment,
    FormationMetrics,
    FormationTemplate,
    assign_agents_to_slots,
    evaluate_formation,
    generate_template,
    place_template,
)

__all__ = [
    "SUPPORTED_KINDS",
    "Assignment",
    "FormationMetrics",
    "FormationTemplate",
    "assign_agents_to_slots",
    "evaluate_formation",
    "generate_template",
    "place_template",
]
