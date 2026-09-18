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
from align.formations.transition import (
    ShapeTransitionConfig,
    TransitionAssignment,
    assign_transition_slots,
    linear_path_clearance,
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
    "ShapeTransitionConfig",
    "TransitionAssignment",
    "assign_transition_slots",
    "linear_path_clearance",
]
