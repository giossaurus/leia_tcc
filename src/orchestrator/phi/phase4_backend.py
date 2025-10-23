"""
Phase 4 loader for Phi instruction-tuned checkpoints.

The Phi models share the same Hugging Face loader surface used in the other
providers, but keeping a dedicated module makes it easier to expand with
provider-specific tweaks if necessary.
"""

from __future__ import annotations

from typing import Optional

from src.orchestrator.phase4.backends import GenerationConfig, PipelineFactory, build_hf_pipeline


def build_pipeline(repo_id: str, config: Optional[GenerationConfig] = None) -> PipelineFactory:
    cfg = config or GenerationConfig()
    return build_hf_pipeline(repo_id, cfg)

