"""
Phase 4 experimentation utilities for the LeIA project.

This package hosts the orchestration scripts used to compare larger
instruction-tuned models (4B–8B) while preserving the Freirean tutoring
approach implemented in earlier phases of the project.
"""

from .runner import HYPOTHESIS, MODELS, Phase4ExperimentRunner  # noqa: F401
from .analyzer import Phase4LogAnalyzer  # noqa: F401

