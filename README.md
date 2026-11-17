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
| Architectures | `llama` (Llama 1–3, Mistral, SmolLM, TinyLlama), `qwen2`, `qwen3`, `qwen3moe`. A 30B mixture of experts runs on a 16 GiB machine at 6x llama.cpp: see [docs/MOE.md](docs/MOE.md) |
| Weight types | Validated end to end against llama.cpp on real models, by perplexity *and* KL divergence: F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q2_K, Q4_K, Q5_K, Q6_K. F32 is kernel-tested only |
| Tokenizers | GPT-2 byte-level BPE (gpt2, smollm, llama3, qwen2 pre-tokenizers) and SentencePiece; special tokens; byte-exact with llama.cpp on the fixtures |
| Execution | CPU: batched prefill, f16 KV cache, AVX2/FMA/F16C kernels, an AArch64 NEON Q4_0 kernel, and a portable scalar fallback |
| Sampling | Greedy, temperature, top-k, top-p, min-p, repetition penalty, seeded |
| Planning | Per-layer VRAM/RAM placement from real tensor sizes and KV costs |
| CUDA | Whole transformer blocks and the output head run on the GPU with `--gpu-layers N|auto` (M6): kernels for all 12 weight types, coalesced struct-of-arrays matvecs for Q4_0/Q8_0/Q4_K/Q6_K, KV cache in VRAM. Loaded as PTX through the driver API, so no CUDA toolkit is needed. Perplexity matches llama.cpp on the GPU as closely as on the CPU (`tests/gpu.rs`) |

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

`run`, `perplexity` and `bench` take `--gpu-layers N` (or `-ngl N`): the
last N blocks run on the GPU, and N at or above the block count also moves
the output head, as with llama.cpp. `--gpu-layers auto` lets the planner
place as many blocks as fit in the GPU's free memory. It is off by default.
On the reference laptop the M1200 is clock-limited to 287 MHz on its 65 W
charger, and the CPU is faster: SmolLM2-135M Q4_0 decodes at 215 tok/s on
the CPU against 37 on the GPU, and Qwen3-0.6B Q4_K_M at 46 against 10.5. At the
GPU's full 1150 MHz that balance should change. Re-measure with the 130 W
charger before relying on it.

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

## Models bigger than RAM

kLLM has a third weight tier below VRAM and RAM: the file itself. When a
model does not fit the RAM budget, it keeps the output head and a prefix of
blocks resident and streams the rest from disk every token. Each streamed
block is read ahead two blocks early (`MADV_WILLNEED`) and dropped from the
page cache right after use, so the stream never evicts the resident set,
which is what plain least-recently-used caching does on a pass larger
than RAM. Streamed blocks stay in their original row-major form, so they
take no heap memory. The budget defaults to what is available now, including
any cgroup limit, minus the session and a 1 GiB margin. Set it with
`--ram-gib N`, or turn streaming off with `--ram-gib all`.

Qwen3-14B Q4_K_M (9.0 GB) in a 6 GiB memory cgroup, cold page cache, 4
threads, on a 1.4 GB/s NVMe:

| Engine | Decode |
|---|---|
| llama.cpp b12099, default | killed: its CPU repack needs the whole model in heap |
| llama.cpp `--repack 0` (mmap) | 4.35 s/token |
| kLLM, mmap only (`KLLM_NO_REPACK=1 --ram-gib all`) | 4.58 s/token |
| kLLM, streaming (`--ram-gib 4.5`) | **3.00 s/token** |

Streaming is bounded by the disk: 3.98 GB streamed per token at 1.4 GB/s is
2.84 s. It only helps when the model really does not fit. Near the edge, the
kernel can often free enough other cache to hold the whole model, and
`--ram-gib all` is then faster.

## A 30B model on a 16 GiB laptop

A dense 30B needs all of itself for every token, so on a machine this size it
is hopeless. A mixture of experts needs a routed fraction -- 8 of 128 experts
for Qwen3-30B-A3B, about 6% of the expert weights -- and *which* fraction
repeats enough for the page cache to absorb it. That is the whole reason this
works:

```sh
kllm run Qwen3-30B-A3B-Q4_K_M.gguf --chat --prompt "..." -n 400 --context 4096
```

17.3 GiB of weights, 16 GiB of RAM, **10-11 tok/s** against llama.cpp's 1.81
on the same file and machine. Three things matter in practice:

- **Let the cache warm.** The first run of a model reads experts from disk and
  lands at 4-7 tok/s; by the second or third it is at 10-11. Switching
  between two 18 GiB models resets that, so expect a slow first answer.
- **Leave the budget alone.** `--ram-gib` defaults to something sensible, and
  for a routed model it only governs the non-expert weights (about 1 GiB);
  everything else is better spent on page cache for the experts. Forcing it
  smaller makes kLLM stream blocks it had no need to and costs 6x.
- **Prompt length picks the batch.** A long prompt gets a wide batch, which
  is what makes routed prefill fast; a short one gets a narrow batch so the
  session does not spend 155 MiB of logits scratch it will not use.

For a conversation, use `kllm chat` rather than repeated `run` calls. It keeps
the session alive, so each turn feeds only what is new: in a three-turn
exchange the third turn fed 18 tokens instead of re-processing all 174.
Re-sending the history every turn is what makes a one-shot CLI unusable on a
slow machine -- a thousand tokens of context costs minutes to re-read at
prefill speed, every single turn.

For a reasoning model such as the base Qwen3-30B-A3B, pass `--no-think`.
Otherwise it answers by thinking first, and at ten tokens a second a 100-token
budget is spent before it has said anything. Its own chat template suppresses
reasoning by pre-filling an empty `<think></think>`, and that is what the flag
does. Instruct builds that do not reason (Qwen3's `-2507-Instruct`, for one)
need nothing.

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
second, best of three. The full table with model hashes and the exact commands
is in [evals/](evals/); regenerate it with `scripts/eval_report.py`.

| Model | pp128 kLLM / llama.cpp | ratio | tg64 kLLM / llama.cpp | ratio |
|---|---|---|---|---|
| SmolLM2-135M Q4_0 | 1145 / 750 | 1.53x | 229 / 219 | 1.05x |
| SmolLM2-135M Q8_0 | 1060 / 658 | 1.61x | 165 / 140 | 1.18x |
| SmolLM2-135M Q6_K | 988 / 667 | 1.48x | 165 / 145 | 1.14x |
| SmolLM2-135M Q2_K | 1089 / 814 | 1.34x | 247 / 244 | 1.01x |
| SmolLM2-135M Q5_K_M | 462 / 352 | 1.31x | 157 / 167 | 0.94x |
| SmolLM2-135M f16 | 629 / 547 | 1.15x | 85.5 / 82.3 | 1.04x |
| Qwen3-0.6B Q4_K_M | 324 / 266 | 1.22x | 48.3 / 55.2 | 0.88x |
| Qwen3-0.6B Q8_0 | 284 / 184 | 1.54x | 36.4 / 37.0 | 0.98x |

Prefill is ahead of llama.cpp on every weight type, float ones included
(1.11-1.65x). Decode is parity or better on most types and trails on the
K-quants (Q4_K 0.88x); it is memory-bandwidth-bound, where prefill is
compute-bound, which is why the two respond to different things -- a register
tile doubled float prefill and *cost* float decode until it was gated on
batch size.

Two notes on reading these numbers. Prefill was at 65% of llama.cpp until the
real bottleneck turned up, and it was not the kernels: at *one* thread kLLM
already beat llama.cpp, and the packed Q4_0 kernel runs near the
`vpmaddubsw` port limit. Activation quantization was running on the calling
thread while the workers idled — 71% of a 576x1536 matmul. And this machine
usually has other work on it, so `kllm bench` reports the best run rather than
the mean (interference can only ever slow a run down) and comparisons against
llama.cpp are interleaved in the same window.

`kllm bench --profile` prints a per-phase breakdown.

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
| `bench_q4_0` | scalar or NEON Q4_0 matvec latency on AArch64 |
| `gemm_bench` | batched matmul GOP/s |
| `tile_bench` | splits a matmul into kernel / row-streaming / full path, and times `prepare` separately |
| `attn_bench` | attention primitives in GMAC/s, per head width |
| `scale_bench` | parallel speedup available on this machine, and pool dispatch latency |
| `gpu_bench` | CUDA matvec GB/s and kernel launch overhead |

See [SPEC.md](SPEC.md) for the specification and [ROADMAP.md](ROADMAP.md) for
the plan.
