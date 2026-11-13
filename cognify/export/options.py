"""Shared export options (kept separate to avoid import cycles)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ExportOptions:
    """Knobs for the export bundle (mirrors the C# ExportOptions)."""
    model_name: Optional[str] = None
    description: str = ""
    vocab_size: int = 4096
    quantization_type: str = "Q4_K_M"
    context_length: int = 512
    batch_size: int = 8
    learning_rate: float = 1e-4
    epochs: int = 3
    threads: int = 4
    save_every: int = 20
    # Number of safetensors shards for the bundle (1 = single file). Values
    # above the tensor count are clamped — shards always hold whole tensors.
    safetensors_split: int = 1
