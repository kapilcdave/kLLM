# Test fixtures

Reference outputs recorded from llama.cpp (build 12099, commit `bee23322f`,
CPU backend) by `scripts/make_fixtures.py`, plus a text excerpt for
perplexity checks.

| File | Contents |
|---|---|
| `*.tokens.json` | `llama-tokenize` output for tricky strings |
| `*.greedy.json` | `llama-server` greedy completions: token IDs and top-10 log-probs per step |
| `perplexity.json` | `llama-perplexity -c 256 --chunks 8` on `wiki.test.head.txt`, flash attention on and off |
| `wiki.test.head.txt` | First 16 kB of the WikiText-2 raw test split |

`wiki.test.head.txt` is an excerpt of WikiText-2 (Merity et al., 2016), derived
from Wikipedia and licensed under CC BY-SA 3.0.

The models these fixtures describe are not checked in. Put (or symlink) them
in `models/` or point `KLLM_TEST_MODELS` at a directory holding them:

- `SmolLM2-135M-Instruct-{f16,Q8_0,Q4_0}.gguf` from `bartowski/SmolLM2-135M-Instruct-GGUF`
- `Qwen3-0.6B-{Q4_K_M,Q8_0}.gguf` from `unsloth/Qwen3-0.6B-GGUF`
