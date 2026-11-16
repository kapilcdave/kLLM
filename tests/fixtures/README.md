# Test fixtures

Reference outputs recorded from llama.cpp (build 12099, commit `bee23322f`,
CPU backend) by `scripts/make_fixtures.py` and
`scripts/make_eval_fixtures.py`, plus a text excerpt to score.

| File | Contents |
|---|---|
| `*.tokens.json` | `llama-tokenize` output for tricky strings |
| `*.greedy.json` | `llama-server` greedy completions: token IDs and top-10 log-probs per step |
| `perplexity.json` | `llama-perplexity -c 256 --chunks 8` on `wiki.test.head.txt`, flash attention on and off |
| `divergence.json` | `llama-perplexity --kl-divergence -c 512 --chunks 4` against each family's reference model |
| `wiki.test.head.txt` | First 16 kB of the WikiText-2 raw test split |

`wiki.test.head.txt` is an excerpt of WikiText-2 (Merity et al., 2016), derived
from Wikipedia and licensed under CC BY-SA 3.0.

Perplexity and KL divergence answer different questions, so both are
recorded. Perplexity says whether a model predicts text well; it is an
average over a corpus, which is what makes it robust to the summation-order
noise that 8-bit activations introduce. KL divergence says how far a
quantization moved the distribution away from the full-precision model, which
an average over a corpus can hide: a kernel can keep perplexity and still
reorder the top of the distribution.

## Models

The models these fixtures describe are not checked in. Put (or symlink) them
in `models/`, or point `KLLM_TEST_MODELS` at a directory holding them. A test
whose model is absent passes with a notice; set `KLLM_REQUIRE_MODELS=1` to
turn absence into failure.

Downloaded:

- `SmolLM2-135M-Instruct-{f16,Q8_0,Q4_0}.gguf` from `bartowski/SmolLM2-135M-Instruct-GGUF`
- `Qwen3-0.6B-{Q4_K_M,Q8_0}.gguf` from `unsloth/Qwen3-0.6B-GGUF`

Derived from the f16 above with `llama-quantize --pure`, to give every
supported weight type coverage on a real model's weights rather than on
random bytes:

```sh
for t in Q4_1 Q5_0 Q5_1 Q2_K Q5_K_M Q6_K BF16; do
  llama-quantize --pure SmolLM2-135M-Instruct-f16.gguf \
    "SmolLM2-135M-Instruct-$t.gguf" $t 4
done
```

Q3_K needs a model whose rows are a multiple of 256, which SmolLM2 is not,
so its coverage comes from Qwen3-0.6B requantized:

```sh
llama-quantize --allow-requantize Qwen3-0.6B-Q8_0.gguf \
  Qwen3-0.6B-Q3_K_M.gguf Q3_K_M 4
```

Quantizing something already quantized makes the absolute perplexity
meaningless, and leaves the agreement with llama.cpp on the same file
exactly as meaningful as it was.

`--pure` stops llama.cpp mixing types per tensor, but a K-quant still needs
rows that are a multiple of 256 wide: SmolLM2 is 576 wide, so in the three
K-quant builds only `ffn_down` (1536 wide) carries the K-quant and the rest
falls back to a legacy type. Qwen3-0.6B (1024 wide) is K-quantized
throughout, which is why it carries the Q4_K and Q6_K coverage.
