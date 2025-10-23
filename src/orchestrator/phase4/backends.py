"""
Shared helpers for building Hugging Face generation pipelines used in Phase 4.

All large-model experiments rely on the same configuration surface so that
different providers (Llama, Mistral, Phi) can reuse the loader logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Mapping

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


PipelineFactory = Callable[[str], List[Mapping[str, str]]]


@dataclass
class GenerationConfig:
    max_new_tokens: int = 240
    temperature: float = 0.7
    top_p: float = 0.92
    top_k: int = 50
    repetition_penalty: float = 1.05


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(device: str) -> torch.dtype:
    if device in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def build_hf_pipeline(repo_id: str, config: GenerationConfig) -> PipelineFactory:
    """
    Carrega um modelo instruction-tuned via Hugging Face e retorna uma função
    `pipeline` pronta para ser utilizada pelo orquestrador da fase 4.

    A função garante que o modelo seja movido para o melhor dispositivo
    disponível (CUDA, MPS ou CPU) e aplica parâmetros de geração consistentes.
    """
    device = _resolve_device()
    dtype = _resolve_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if device != "cpu":
        model.to(device)

    if device == "cuda":
        pipeline_device = 0
    elif device == "mps":
        pipeline_device = torch.device("mps")
    else:
        pipeline_device = -1

    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=pipeline_device,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        repetition_penalty=config.repetition_penalty,
        do_sample=True,
        return_full_text=False,
    )
    return text_pipeline
