# Chat template fixtures

Real chat templates as served by llama.cpp `--jinja`, copied verbatim from
HuggingFace (fetched 2026-11-16):

| file | source | family |
| --- | --- | --- |
| `chatml.jinja` | teknium/OpenHermes-2.5-Mistral-7B `tokenizer_config.json` → `chat_template` | ChatML |
| `qwen25.jinja` | Qwen/Qwen2.5-7B-Instruct `tokenizer_config.json` → `chat_template` | Qwen2.5 (tools) |
| `qwen3.jinja` | Qwen/Qwen3-32B `tokenizer_config.json` → `chat_template` | Qwen3 (thinking + tools) |
| `zephyr.jinja` | TinyLlama/TinyLlama-1.1B-Chat-v1.0 `tokenizer_config.json` → `chat_template` | Zephyr |
| `llama31.jinja` | unsloth/Llama-3.1-8B-Instruct `chat_template.jinja` (Meta Llama 3.1 mirror) | Llama-3.1 (tools, ipython) |

The `../goldens/*.txt` files are the rendered outputs for the canonical
contexts defined in `test_jinja.py` (and mirrored in the C#
`ChatJinjaTests`) — the CognifyPy and Cognify engines must both reproduce
them byte-for-byte.
