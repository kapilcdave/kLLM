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
| Weight types | Validated end to end against llama.cpp on real models, by perplexity *and* KL divergence: F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q2_K, Q4_K, Q5_K, Q6_K. F32 is kernel-tested only |
| Tokenizers | GPT-2 byte-level BPE (gpt2, smollm, llama3, qwen2 pre-tokenizers) and SentencePiece; special tokens; byte-exact with llama.cpp on the fixtures |
| Execution | CPU: batched prefill, f16 KV cache, AVX2/FMA/F16C kernels with a portable scalar fallback |
| Sampling | Greedy, temperature, top-k, top-p, min-p, repetition penalty, seeded |
| Planning | Per-layer VRAM/RAM placement from real tensor sizes and KV costs |
| CUDA | Kernels for all 12 weight types plus norm/attention/RoPE, loaded as PTX through the driver API with no CUDA toolkit; tested against an exact reference on a Quadro M1200. Not yet wired into the forward pass (milestone M6) |

## Commands

```text
kllm inspect MODEL.gguf [--tensors]
kllm plan-model MODEL.gguf --vram-gib N [--context N]
kllm run MODEL.gguf --prompt TEXT [-n N] [--chat] [--system TEXT] [--temp T] ...
kllm perplexity MODEL.gguf --file TEXT [--context 512] [--chunks N]
kllm compare REFERENCE.gguf TEST.gguf --file TEXT [--context 512] [--chunks N]
kllm bench MODEL.gguf [-p 128] [-n 64] [--profile]
kllm devices
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
fixtures; see [tests/fixtures/README.md](tests/fixtures/README.md). It checks
two different things, because one of them is not enough:

- **Perplexity** says whether a model predicts text well. It is an average
  over a corpus, which is what makes it robust to the summation-order noise
  that 8-bit activations introduce.
- **KL divergence** (`kllm compare`) says how far a quantization moved the
  distribution away from the full-precision model. An average over a corpus
  can hide that: a kernel can hold perplexity and still reorder the top of the
  distribution.

Mean KL divergence against each family's reference model, kLLM against
`llama-perplexity --kl-divergence` on the same text (4 chunks x 512 tokens):

| Model | reference | llama.cpp | kLLM | top-1 agreement |
|---|---|---|---|---|
| SmolLM2-135M BF16 | f16 | 0.000081 | 0.000000 | 100.00% / 99.61% |
| SmolLM2-135M Q8_0 | f16 | 0.003777 | 0.003630 | 97.16% / 96.57% |
| SmolLM2-135M Q6_K | f16 | 0.008890 | 0.008736 | 93.73% / 94.12% |
| SmolLM2-135M Q5_K_M | f16 | 0.063889 | 0.063725 | 85.10% / 85.39% |
| SmolLM2-135M Q5_1 | f16 | 0.069482 | 0.070725 | 85.49% / 85.69% |
| SmolLM2-135M Q5_0 | f16 | 0.088250 | 0.087968 | 83.14% / 82.45% |
| SmolLM2-135M Q4_0 | f16 | 0.148069 | 0.146500 | 81.08% / 81.08% |
| SmolLM2-135M Q4_1 | f16 | 0.271327 | 0.270642 | 72.65% / 74.12% |
| SmolLM2-135M Q2_K | f16 | 2.130084 | 2.124023 | 30.10% / 30.69% |
| Qwen3-0.6B Q4_K_M | Q8_0 | 0.076557 | 0.076500 | 86.76% / 86.77% |

kLLM is within 2% of llama.cpp on every one, and lower on eight of the ten.
BF16 is exactly zero because it should be: SmolLM2 was trained in bf16, so its
f16 GGUF holds exactly representable bf16 values (f16 keeps 10 mantissa bits
to bf16's 7) and converting back is lossless — 100k weights checked, none
differ. llama.cpp's 8e-5 there is its own F16-vs-BF16 kernel noise.

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
`$KLLM_TEST_MODELS`. `scripts/make_fixtures.py` and
`scripts/make_eval_fixtures.py` re-record the llama.cpp references;
`scripts/eval_report.py` writes one report with perplexity, divergence, speed,
model hashes and hardware beside llama.cpp's numbers (see `evals/`).

`examples/` holds the micro-benchmarks that located the bottlenecks above, and
they are the right first stop before optimizing anything:

| Example | Answers |
|---|---|
| `kernel_bench` | matvec GB/s per weight type — is a shape bandwidth-bound? |
| `gemm_bench` | batched matmul GOP/s |
| `tile_bench` | splits a matmul into kernel / row-streaming / full path, and times `prepare` separately |
| `attn_bench` | attention primitives in GMAC/s, per head width |
| `scale_bench` | parallel speedup available on this machine, and pool dispatch latency |
| `gpu_bench` | CUDA matvec GB/s and kernel launch overhead |

See [SPEC.md](SPEC.md) for the specification and [ROADMAP.md](ROADMAP.md) for
the plan.
