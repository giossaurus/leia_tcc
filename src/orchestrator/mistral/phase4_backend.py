"""
Phase 4 loader for Mistral instruction-tuned checkpoints.

This module provides a thin wrapper to keep provider-specific customisations
isolated from the shared experiment runner.
"""

from __future__ import annotations

from typing import Optional

from src.orchestrator.phase4.backends import GenerationConfig, PipelineFactory, build_hf_pipeline


def build_pipeline(repo_id: str, config: Optional[GenerationConfig] = None) -> PipelineFactory:
    cfg = config or GenerationConfig()
    return build_hf_pipeline(repo_id, cfg)

