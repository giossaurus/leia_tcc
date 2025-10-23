"""
Phase 4 loader for Meta Llama instruction-tuned checkpoints.

This module reuses the shared Hugging Face pipeline builder so that the runner
can dynamically import providers based on the experiment configuration.
"""

from __future__ import annotations

from typing import Optional

from src.orchestrator.phase4.backends import GenerationConfig, PipelineFactory, build_hf_pipeline


def build_pipeline(repo_id: str, config: Optional[GenerationConfig] = None) -> PipelineFactory:
    cfg = config or GenerationConfig()
    return build_hf_pipeline(repo_id, cfg)

