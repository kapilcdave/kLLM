# kLLM

kLLM (Kapil LLM) is a memory-first LLM inference engine in Rust for quantized
models on older, memory-constrained machines. Its reference machine is a Dell
Precision 5520 (Xeon E3-1505M v6, 16 GiB RAM, 4 GiB Quadro M1200).

It reads GGUF files in place (memory-mapped, never copied), plans every large
allocation before making it, and runs Llama-family decoders on the CPU with
AVX2 kernels whose output is checked against llama.cpp.

```sh
cargo build --release
./target/release/kllm run model.gguf --chat --prompt "Explain RoPE in one paragraph." -n 200
```

## What works today

| Area | Status |
|---|---|
| GGUF parsing | Full metadata and tensor directory; every length is bounds-checked against the file before allocation; payloads are memory-mapped |
| Architectures | `llama` (Llama 1–3, Mistral, SmolLM, TinyLlama), `qwen2`, `qwen3` |
| Weight types | Validated end to end against llama.cpp: F16, Q8_0, Q4_0, Q4_1, Q4_K, Q6_K. Kernel-tested only (no real-model check yet): F32, BF16, Q5_0, Q5_1, Q5_K, Q2_K |
| Tokenizers | GPT-2 byte-level BPE (gpt2, smollm, llama3, qwen2 pre-tokenizers) and SentencePiece; special tokens; byte-exact with llama.cpp on the fixtures |
| Execution | CPU: batched prefill, f16 KV cache, AVX2/FMA/F16C kernels with a portable scalar fallback |
| Sampling | Greedy, temperature, top-k, top-p, min-p, repetition penalty, seeded |
| Planning | Per-layer VRAM/RAM placement from real tensor sizes and KV costs |
| CUDA | Not yet (milestone M5) |

## Commands

```text
kllm inspect MODEL.gguf [--tensors]
kllm plan-model MODEL.gguf --vram-gib N [--context N]
kllm run MODEL.gguf --prompt TEXT [-n N] [--chat] [--system TEXT] [--temp T] ...
kllm perplexity MODEL.gguf --file TEXT [--context 512] [--chunks N]
kllm bench MODEL.gguf [-p 128] [-n 64] [--profile]
```

`run` prints its memory plan before allocating anything and refuses a context
that does not fit in available RAM. `plan-model` works for any GGUF, including
architectures kLLM cannot run yet:

```text
$ kllm plan-model Qwen3.5-9B-Q4_K_M.gguf --vram-gib 3.7 --context 8192
  GPU layers:             20 (blocks 12..31)
  CPU layers:             12
  GPU weights:          2.48 GiB
  GPU KV cache:         0.62 GiB
  ...
```

## Correctness

The test-suite compares kLLM with llama.cpp (build 12099) on recorded
fixtures; see [tests/fixtures/README.md](tests/fixtures/README.md).

Perplexity on WikiText-2 (40 chunks x 512 tokens, 10,200 scored tokens):

| Model | llama.cpp | kLLM | Difference |
|---|---|---|---|
| SmolLM2-135M F16 | 17.6105 | 17.6099 | 0.003% |
| SmolLM2-135M Q8_0 | 17.7186 | 17.6975 | 0.12% |
| SmolLM2-135M Q4_0 | 20.3947 | 20.3897 | 0.02% |
| Qwen3-0.6B Q4_K_M | 20.1831 | 20.2292 | 0.23% |

On F16, greedy decoding reproduces llama.cpp's tokens exactly, with a mean
top-10 log-probability difference of 0.003. Quantized paths quantize
activations to 8 bits (as llama.cpp does), which amplifies summation-order
differences: llama.cpp disagrees with *itself* by up to 0.4% perplexity when
only its flash-attention switch is flipped. kLLM's differences are within that
spread.

## Performance

Same machine, 4 threads, `kllm bench MODEL -p 128 -n 64 -t 4 -r 3` against
`llama-bench -t 4 -p 128 -n 64 -r 3` (llama.cpp b12099 CPU build), tokens per
second. Run-to-run noise on this laptop is about ±10%.

| Model (sha256 prefix) | Prefill (pp128) llama.cpp / kLLM | Decode (tg64) llama.cpp / kLLM |
|---|---|---|
| SmolLM2-135M F16 (`f535f83e`) | 613 / 311 | 82.0 / 82.6 |
| SmolLM2-135M Q8_0 (`5a139571`) | 688 / 352 | 146 / 117 |
| SmolLM2-135M Q4_0 (`bcc3af28`) | 760 / 333 | 199 / 135 |
| Qwen3-0.6B Q4_K_M (`ac2d9771`) | 230 / 120 | 52.5 / 41.1 |

Decode is at 68–100% of llama.cpp. Prefill is at about half: kLLM's batched
kernels unpack each weight block once per eight activation columns but still
reduce every (row, column) pair separately, where llama.cpp uses row-
interleaved GEMM tiles. `kllm bench --profile` prints a per-phase breakdown.

## Development

```sh
cargo test            # unit tests; model tests skip if models are absent
KLLM_REQUIRE_MODELS=1 cargo test   # fail instead of skipping
KLLM_NO_SIMD=1 cargo test          # exercise the portable scalar kernels
```

Model-backed tests look in `models/` (git-ignored; symlinks are fine) or in
`$KLLM_TEST_MODELS`. `scripts/make_fixtures.py` re-records the llama.cpp
references. `examples/` holds kernel micro-benchmarks.

See [SPEC.md](SPEC.md) for the specification and [ROADMAP.md](ROADMAP.md) for
the plan.
