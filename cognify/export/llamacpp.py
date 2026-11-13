"""llama.cpp training / fine-tuning / quantization config set.

Writes the usual companion files for an exported model:

* ``train_config.json`` — hyperparameters in the llama.cpp finetune style
* ``finetune.sh`` — ready-to-run ``llama-finetune`` (LoRA adapter) command
* ``quantize.sh`` — ``llama-quantize`` pipeline to Q4_K_M / Q5_K_M / Q8_0
* ``training_data.txt`` / ``val_data.txt`` — the mind's corpus, ~90/10 split
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

from .options import ExportOptions


def write(directory: str, corpus: Sequence[str], model_name: str,
          options: ExportOptions) -> List[str]:
    """Writes the config set; returns the list of files produced."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    # llama.cpp can only load/run the gpt2-architecture export — NOT the
    # 'cognify' archive GGUF. Finetuning and quantization must therefore
    # target the *-llamacpp.gguf file from the bundle.
    model_gguf = f"{model_name}-llamacpp.gguf"
    lora_out = f"{model_name}-lora.gguf"
    quantized = f"{model_name}.{options.quantization_type}.gguf"

    # ---- train_config.json -------------------------------------------------
    config: Dict[str, object] = {
        "model": model_gguf,
        "lora_out": lora_out,
        "lora_init": True,
        "train_data": "training_data.txt",
        "val_data": "val_data.txt",
        "sample_start": "<|im_start|>user\n",
        "sample_end": "<|im_end|>",
        "context_length": options.context_length,
        "batch_size": options.batch_size,
        "gradient_clipping": 1.0,
        "learning_rate": options.learning_rate,
        "epochs": options.epochs,
        "use_adam": True,
        "weight_decay": 0.0,
        "warmup_batches": 20,
        "threads": options.threads,
        "save_every": options.save_every,
        "checkpoint_out": f"{model_name}-checkpoint.gguf",
        "quantization": {
            "source": model_gguf,
            "tool": "llama-quantize",
            "types": ["Q4_K_M", "Q5_K_M", "Q8_0"],
            "target": quantized,
        },
    }
    path = out / "train_config.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    written.append(str(path))

    # ---- finetune.sh -------------------------------------------------------
    finetune = f"""#!/usr/bin/env bash
# Fine-tune the exported {model_name} weights with llama.cpp (LoRA adapter).
# Prereq: llama.cpp built from source (https://github.com/ggml-org/llama.cpp).
set -euo pipefail

MODEL="{model_gguf}"
TRAIN_DATA="training_data.txt"
VAL_DATA="val_data.txt"
LORA_OUT="{lora_out}"

./llama-finetune \\
  --model "$MODEL" \\
  --train-data "$TRAIN_DATA" \\
  --val-data "$VAL_DATA" \\
  --lora-out "$LORA_OUT" \\
  --save-every {options.save_every} \\
  --threads {options.threads} \\
  --ctx {options.context_length} \\
  --batch {options.batch_size} \\
  --lr {options.learning_rate} \\
  --epochs {options.epochs} \\
  --use-adam \\
  --warmup-batches 20

echo "LoRA adapter written to $LORA_OUT"
echo "Apply it at generation time with: llama-cli --model $MODEL --lora $LORA_OUT"
"""
    path = out / "finetune.sh"
    path.write_text(finetune, encoding="utf-8")
    written.append(str(path))

    # ---- quantize.sh -------------------------------------------------------
    quantize = f"""#!/usr/bin/env bash
# Quantize {model_gguf} (the llama.cpp-runnable gpt2 export, all F32).
set -euo pipefail

SRC="{model_gguf}"

./llama-quantize "$SRC" "{model_name}.Q4_K_M.gguf" Q4_K_M
./llama-quantize "$SRC" "{model_name}.Q5_K_M.gguf" Q5_K_M
./llama-quantize "$SRC" "{model_name}.Q8_0.gguf" Q8_0

echo "Quantized variants written. Typical choices: Q4_K_M (fast, small), Q8_0 (near-lossless)."
"""
    path = out / "quantize.sh"
    path.write_text(quantize, encoding="utf-8")
    written.append(str(path))

    # ---- training / validation data (~90/10 split) -------------------------
    lines: List[str] = []
    for line in corpus:
        cleaned = " ".join(str(line).replace("\n", " ").replace("\r", " ").split())
        if cleaned and cleaned not in lines:
            lines.append(cleaned)
    split = max(1, int(len(lines) * 0.9)) if lines else 0

    path = out / "training_data.txt"
    path.write_text("\n".join(lines[:split]) + ("\n" if lines[:split] else ""), encoding="utf-8")
    written.append(str(path))

    path = out / "val_data.txt"
    path.write_text("\n".join(lines[split:]) + ("\n" if lines[split:] else ""), encoding="utf-8")
    written.append(str(path))

    return written
